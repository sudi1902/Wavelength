"""Pipeline orchestrator: ingest → separate → segment → clean → store.

One public entry point, ``extract_video``, used by both the CLI and (in
Phase 3) the watcher/server. Progress is reported through a callback so
callers choose their own presentation.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import soundfile as sf

from wavelength.config import Settings
from wavelength.library.db import LibraryDB
from wavelength.pipeline import ingest
from wavelength.pipeline.clean import CleanedEffect, RejectReason, clean_segment
from wavelength.pipeline.segment import detect_segments
from wavelength.pipeline.separate import Separator, get_separator
from wavelength.pipeline.store import archive_source, store_effect


class PipelineError(Exception):
    """Extraction failed for this video; message is user-facing."""


@dataclass
class ExtractionResult:
    video: Path
    effect_paths: list[Path] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    already_processed: bool = False

    @property
    def effect_count(self) -> int:
        return len(self.effect_paths)


Progress = Callable[[str], None]


def extract_video(
    video: Path,
    settings: Settings,
    *,
    engine: str | None = None,
    force: bool = False,
    max_duration_override: bool = False,
    progress: Progress = lambda msg: None,
    separator: Separator | None = None,
    db: LibraryDB | None = None,
) -> ExtractionResult:
    """Run the full pipeline on one video.

    ``force`` re-processes a video that is already in the library.
    ``separator``/``db`` can be injected (batch runs reuse a loaded model).
    """
    video = video.expanduser().resolve()
    ingest.require_ffmpeg()

    owns_db = db is None
    db = db or LibraryDB(settings.db_path)
    try:
        return _run(
            video, settings, engine=engine, force=force,
            max_duration_override=max_duration_override,
            progress=progress, separator=separator, db=db,
        )
    finally:
        if owns_db:
            db.close()


def _run(
    video: Path, settings: Settings, *, engine: str | None, force: bool,
    max_duration_override: bool, progress: Progress,
    separator: Separator | None, db: LibraryDB,
) -> ExtractionResult:
    result = ExtractionResult(video=video)

    # -- ingest ---------------------------------------------------------------
    progress(f"Probing {video.name}")
    info = ingest.probe(video)
    if info.duration_s > settings.max_video_duration_s and not max_duration_override:
        raise PipelineError(
            f"{video.name} is {info.duration_s / 60:.1f} min long "
            f"(cap: {settings.max_video_duration_s / 60:.0f} min). "
            "Re-run with --force-long to process anyway."
        )

    video_hash = ingest.file_sha256(video)
    existing = db.find_source(video_hash)
    if existing is not None:
        if not force:
            result.already_processed = True
            progress(
                f"Already processed ({existing['effect_count']} effects); "
                "use --force to re-extract"
            )
            return result
        db.delete_source(existing["id"])

    separator = separator or get_separator(settings, engine)
    if not separator.is_ready():
        progress(f"Setting up '{separator.name}' engine (first run)")
        separator.ensure_ready()

    source_id = db.add_source(
        sha256=video_hash,
        original_path=str(video),
        filename=video.name,
        duration_s=info.duration_s,
        engine=separator.name,
    )

    try:
        # -- extract audio ----------------------------------------------------
        progress("Extracting audio")
        with tempfile.TemporaryDirectory(prefix="wavelength-") as tmp:
            wav_path = Path(tmp) / "audio.wav"
            ingest.extract_audio(video, wav_path, info)
            audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
            audio = audio.T  # (channels, samples)

        # -- separate ---------------------------------------------------------
        progress(f"Separating stems ({separator.name})")
        separation = separator.separate(audio, sr)
        effects, stem_sr = separation.effects, separation.sample_rate

        # -- segment ----------------------------------------------------------
        progress("Detecting sound events")
        segments = detect_segments(effects, stem_sr, settings.segmentation)
        progress(f"Found {len(segments)} candidate segment(s)")

        # -- clean + store ----------------------------------------------------
        index = 1
        for segment in segments:
            cleaned = clean_segment(effects, stem_sr, segment, settings.cleaning)
            if isinstance(cleaned, RejectReason):
                result.rejected[cleaned.value] = (
                    result.rejected.get(cleaned.value, 0) + 1
                )
                continue
            stored = store_effect(
                cleaned, settings=settings, db=db,
                source_id=source_id, source_hash=video_hash, index=index,
            )
            if stored is None:
                result.duplicates += 1
                continue
            result.effect_paths.append(stored.path)
            index += 1

        # -- archive ----------------------------------------------------------
        archived_video = archived_stem = None
        if settings.archive_sources:
            progress("Archiving source video and effects stem")
            v_path, s_path = archive_source(
                video, effects, stem_sr,
                settings=settings, source_hash=video_hash,
            )
            archived_video, archived_stem = str(v_path), str(s_path)

        db.finish_source(
            source_id,
            effect_count=result.effect_count,
            archived_video_path=archived_video,
            archived_stem_path=archived_stem,
        )
    except Exception as exc:
        db.finish_source(
            source_id, effect_count=result.effect_count,
            status="failed", error=str(exc)[:500],
        )
        raise

    return result
