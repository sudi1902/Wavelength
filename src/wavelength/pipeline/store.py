"""Store stage: write effect WAVs into the library and record metadata.

Files are 16-bit PCM WAV at the stem's sample rate, stored flat under
``effects/YYYY/MM/`` with human-readable names:

    20260708-a3f2c1__effect-01.wav

Renames/tags live in the database (Phase 3), so paths written here never
need to change. Dedup: the 16-bit PCM payload is hashed; an effect whose
audio already exists in the library (same effect from a re-encoded copy of
the video) is skipped.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from wavelength.config import Settings
from wavelength.library.db import LibraryDB
from wavelength.pipeline.clean import CleanedEffect


@dataclass
class StoredEffect:
    path: Path
    duplicate: bool


def _to_int16(audio: np.ndarray) -> np.ndarray:
    """(channels, samples) float in [-1, 1] -> (samples, channels) int16."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped.T * 32767.0).astype(np.int16)


def store_effect(
    effect: CleanedEffect,
    *,
    settings: Settings,
    db: LibraryDB,
    source_id: int,
    source_hash: str,
    index: int,
) -> StoredEffect | None:
    """Write one effect to the library. Returns None for duplicates."""
    pcm = _to_int16(effect.audio)
    content_hash = hashlib.sha256(pcm.tobytes()).hexdigest()
    if db.find_effect_by_hash(content_hash) is not None:
        return None

    now = datetime.now()
    subdir = settings.effects_dir / f"{now:%Y}" / f"{now:%m}"
    subdir.mkdir(parents=True, exist_ok=True)

    base = f"{now:%Y%m%d}-{source_hash[:6]}__effect-{index:02d}"
    path = subdir / f"{base}.wav"
    n = 1
    while path.exists():
        path = subdir / f"{base}-{n}.wav"
        n += 1

    sf.write(path, pcm, effect.sample_rate, subtype="PCM_16")

    db.add_effect(
        source_id=source_id,
        path=str(path.relative_to(settings.library_dir)),
        content_sha256=content_hash,
        duration_s=effect.duration_s,
        sample_rate=effect.sample_rate,
        peak_db=round(effect.peak_db, 2),
        start_in_source_s=round(effect.start_in_source_s, 3),
    )
    return StoredEffect(path=path, duplicate=False)


def archive_source(
    video: Path,
    effects_stem: np.ndarray,
    stem_sr: int,
    *,
    settings: Settings,
    source_hash: str,
) -> tuple[Path, Path]:
    """Copy the original video and write the full effects stem into
    sources/, enabling future re-processing with better models."""
    dest_dir = settings.sources_dir / source_hash[:12]
    dest_dir.mkdir(parents=True, exist_ok=True)

    video_dest = dest_dir / video.name
    if not video_dest.exists():
        shutil.copy2(video, video_dest)

    stem_dest = dest_dir / "effects_stem.wav"
    if not stem_dest.exists():
        sf.write(stem_dest, _to_int16(effects_stem), stem_sr, subtype="PCM_16")
    return video_dest, stem_dest
