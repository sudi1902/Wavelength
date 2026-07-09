"""'Find cleaner version': rank Freesound sounds by audio similarity to an
extracted effect, and import the one the user picks.

Flow: the effect's CLAP embedding (stored at labeling time, computed lazily
otherwise) → Freesound text search by label with a duration window → download
candidate previews (cached on disk) → embed them → cosine-rank against the
effect → top N. Import converts the chosen preview to 16-bit WAV in the
library with license + attribution recorded and a link back to the original
effect.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from wavelength.config import Settings
from wavelength.library.db import LibraryDB
from wavelength.pipeline.label import ClapLabeler, LabelingError
from wavelength.sources import freesound
from wavelength.sources.freesound import FreesoundError, FreesoundSound


class SimilarError(Exception):
    """Similarity search failed; message is user-facing."""


@dataclass
class Candidate:
    sound: FreesoundSound
    similarity: float  # cosine similarity in [-1, 1]
    preview_path: Path


@dataclass
class SimilarResult:
    queries: list[str]  # what was searched, for display + user steering
    candidates: list[Candidate]

    @property
    def weak(self) -> bool:
        """True when even the best match isn't audibly close — the UI tells
        the user to steer rather than presenting distant sounds as matches."""
        return not self.candidates or self.candidates[0].similarity < 0.4


def _previews_dir(settings: Settings) -> Path:
    return settings.cache_dir / "freesound-previews"


def _require_labeler(settings: Settings) -> ClapLabeler:
    labeler = ClapLabeler(settings)
    if not labeler.is_ready():
        raise SimilarError(
            "Similarity search needs the CLAP model. Set it up once with: "
            "wavelength setup clap"
        )
    return labeler


def effect_embedding(
    settings: Settings, db: LibraryDB, effect_row
) -> np.ndarray:
    """The effect's stored embedding, computed and persisted if missing."""
    blob = effect_row["embedding"]
    if blob:
        return np.frombuffer(blob, dtype=np.float32)
    wav = settings.library_dir / effect_row["path"]
    if not wav.is_file():
        raise SimilarError(f"Audio file missing: {effect_row['path']}")
    labeler = _require_labeler(settings)
    try:
        [blob] = labeler.embed([wav])
    except LabelingError as exc:
        raise SimilarError(str(exc)) from exc
    db.set_embedding(effect_row["id"], blob)
    return np.frombuffer(blob, dtype=np.float32)


def _closest_vocab_labels(
    settings: Settings, reference: np.ndarray, top_n: int = 3
) -> list[str]:
    """The SFX vocabulary terms whose text embeddings sit closest to this
    effect's audio embedding — i.e. what the sound *actually* sounds like,
    independent of the single stored label (which may be wrong)."""
    labeler = _require_labeler(settings)
    try:
        entries = labeler.text_embeddings()
    except LabelingError as exc:
        raise SimilarError(str(exc)) from exc
    scored = [
        (float(np.dot(reference, np.asarray(e["embedding"], dtype=np.float32))),
         e["label"])
        for e in entries if e["group"] == "sfx"
    ]
    scored.sort(reverse=True)
    return [label for _, label in scored[:top_n]]


def find_similar(
    settings: Settings, db: LibraryDB, effect_row, query: str | None = None
) -> SimilarResult:
    """``query`` overrides the automatic search terms (user-steered search)."""
    cfg = settings.similar
    reference = effect_embedding(settings, db, effect_row)

    if query:
        queries = [query.strip()]
    else:
        # A single (possibly wrong) label makes a bad candidate pool that
        # ranking can't rescue. Pool several angles: the user's own label
        # if set, plus the vocabulary terms closest to the actual audio.
        queries = []
        if effect_row["user_label"]:
            queries.append(effect_row["user_label"])
        queries.extend(_closest_vocab_labels(settings, reference))
        seen = set()
        queries = [
            q for q in queries
            if q and q != "unknown" and not (q in seen or seen.add(q))
        ][:4]
    if not queries:
        raise SimilarError(
            "Nothing to search by — rename the effect (double-click its "
            "label) or type a search term, then try again."
        )

    duration = float(effect_row["duration_s"])
    per_query = max(8, cfg.candidates // len(queries))
    sounds_by_id: dict[int, FreesoundSound] = {}
    errors: list[str] = []
    for q in queries:
        try:
            for sound in freesound.search(
                cfg.freesound_api_key,
                q,
                min_duration_s=max(0.05, duration * 0.25),
                max_duration_s=min(15.0, max(3.0, duration * 6.0)),
                licenses=cfg.licenses,
                limit=per_query,
            ):
                sounds_by_id.setdefault(sound.id, sound)
        except FreesoundError as exc:
            errors.append(str(exc))
    sounds = list(sounds_by_id.values())
    if not sounds:
        if errors:
            raise SimilarError(errors[0])
        raise SimilarError(
            f"Freesound has no {'/'.join(cfg.licenses)} results for "
            f"'{' / '.join(queries)}'. Try a different search term."
        )

    previews: list[tuple[FreesoundSound, Path]] = []
    for sound in sounds:
        dest = _previews_dir(settings) / f"{sound.id}.mp3"
        try:
            freesound.download_preview(sound, dest)
        except FreesoundError:
            continue  # skip broken previews, rank the rest
        previews.append((sound, dest))
    if not previews:
        raise SimilarError("No candidate previews could be downloaded.")

    labeler = _require_labeler(settings)
    try:
        blobs = labeler.embed([p for _, p in previews])
    except LabelingError as exc:
        raise SimilarError(str(exc)) from exc

    candidates = []
    for (sound, path), blob in zip(previews, blobs):
        emb = np.frombuffer(blob, dtype=np.float32)
        candidates.append(
            Candidate(
                sound=sound,
                similarity=float(np.dot(reference, emb)),
                preview_path=path,
            )
        )
    candidates.sort(key=lambda c: c.similarity, reverse=True)
    return SimilarResult(queries=queries, candidates=candidates[: cfg.results])


def import_sound(
    settings: Settings,
    db: LibraryDB,
    effect_row,
    sound: FreesoundSound,
    preview_path: Path,
    session_id: str | None = None,
) -> Path:
    """Convert a chosen Freesound preview to a 16-bit WAV in the library,
    with license/attribution recorded and a link back to the effect."""
    with tempfile.TemporaryDirectory(prefix="wavelength-import-") as tmp:
        wav_tmp = Path(tmp) / "sound.wav"
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(preview_path),
             "-ar", "44100", "-c:a", "pcm_s16le", str(wav_tmp)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise SimilarError(
                "Could not convert the preview to WAV: "
                + result.stderr.strip()[-200:]
            )
        audio, sr = sf.read(wav_tmp, dtype="int16", always_2d=True)

    content_hash = hashlib.sha256(audio.tobytes()).hexdigest()
    existing = db.find_effect_by_hash(content_hash, session_id)
    if existing is not None:
        raise SimilarError("This sound is already in your library.")

    source_key = f"freesound:{sound.id}"
    source = db.find_source_by_url(sound.page_url, session_id)
    source_id = source["id"] if source else db.add_source(
        sha256=hashlib.sha256(source_key.encode()).hexdigest(),
        original_path=sound.page_url,
        filename=f"{sound.id}.mp3",
        duration_s=sound.duration_s,
        engine="freesound",
        source_url=sound.page_url,
        title=sound.name,
        uploader=sound.username,
        session_id=session_id,
    )

    label = effect_row["user_label"] or effect_row["auto_label"] or "effect"
    from wavelength.pipeline.store import _slug

    from datetime import datetime

    now = datetime.now()
    subdir = settings.effects_dir / f"{now:%Y}" / f"{now:%m}"
    subdir.mkdir(parents=True, exist_ok=True)
    base = f"{now:%Y%m%d}-fs{sound.id}__{_slug(label)}-clean"
    path = subdir / f"{base}.wav"
    n = 1
    while path.exists():
        path = subdir / f"{base}-{n}.wav"
        n += 1
    sf.write(path, audio, sr, subtype="PCM_16")

    db.add_effect(
        source_id=source_id,
        path=str(path.relative_to(settings.library_dir)),
        content_sha256=content_hash,
        duration_s=audio.shape[0] / sr,
        sample_rate=sr,
        peak_db=None,
        start_in_source_s=None,
        auto_label=label,
        license=sound.license,
        attribution=sound.attribution,
        derived_from=effect_row["id"],
    )
    return path
