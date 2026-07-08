"""Configuration for the wavelength pipeline.

Defaults live here; user overrides come from a TOML file at
``~/.config/wavelength/config.toml`` (any subset of keys may be present).

Sample-rate policy
------------------
Audio is extracted from videos at its native sample rate (floored at 44.1 kHz:
sources below that — rare — are upsampled to meet the export minimum).
Separation engines may impose their own rate (BandIt Plus is a 44.1 kHz
model); effects are stored at whatever rate the effects stem comes back in,
never resampled a second time. To force a uniform library rate later, add a
resample step in ``store.py`` keyed off a single ``target_sample_rate``
setting — that is the only place it would need to change.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = Path("~/.config/wavelength/config.toml").expanduser()

MIN_SAMPLE_RATE = 44_100
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}


@dataclass
class SegmentationConfig:
    """Parameters for splitting the effects stem into discrete events."""

    frame_length: int = 2048
    hop_length: int = 512
    # A frame is "active" when its RMS exceeds BOTH the absolute threshold and
    # the estimated noise floor plus the margin. Margin tuned down from 12
    # after Phase-1 field testing missed quieter effects on real clips.
    activity_threshold_db: float = -48.0
    noise_floor_margin_db: float = 10.0
    # Active regions separated by a gap shorter than this are merged.
    # 0.2 s keeps an effect's reverb tail attached instead of splitting it.
    merge_gap_s: float = 0.20
    min_segment_s: float = 0.1
    # Regions longer than this get onset-based splitting attempts.
    max_segment_s: float = 15.0
    onset_split: bool = True


@dataclass
class CleaningConfig:
    """Parameters for trimming/fading segments and rejecting junk."""

    # Post padding raised from 0.10 after Phase-1 field testing: tails were
    # getting clipped, which reads as "unclear" effects.
    pre_pad_s: float = 0.05
    post_pad_s: float = 0.15
    fade_in_s: float = 0.010
    fade_out_s: float = 0.030
    normalize_peak_db: float = -1.0
    # Junk filters (checked before normalization). min_peak lowered from
    # -40 so quiet-but-real effects survive; the quarantine stage is now the
    # backstop for marginal segments.
    min_duration_s: float = 0.1
    min_peak_db: float = -45.0
    # Static hiss: noise-like spectrum AND an almost-flat energy envelope.
    # (Pure tones have flatness ~0, so steady beeps are kept; real effects
    # measure 25+ dB of envelope range vs ~1 dB for hiss.)
    junk_spectral_flatness: float = 0.05
    junk_envelope_range_db: float = 6.0


@dataclass
class LabelingConfig:
    """CLAP zero-shot labeling and quarantine decisions."""

    enabled: bool = True
    # Below this probability the effect keeps the label "unknown".
    min_confidence: float = 0.15
    # A segment whose best speech/music/noise score beats its best SFX score
    # AND clears this floor goes to quarantine/ for review, not the library.
    quarantine_enabled: bool = True
    quarantine_min_confidence: float = 0.35


@dataclass
class Settings:
    library_dir: Path = field(
        default_factory=lambda: Path("~/SoundEffectsLibrary").expanduser()
    )
    cache_dir: Path = field(
        default_factory=lambda: Path("~/.cache/wavelength").expanduser()
    )
    engine: str = "bandit"  # bandit | demucs | none
    device: str = "auto"  # auto | cpu | mps | cuda
    max_video_duration_s: float = 600.0
    archive_sources: bool = True
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)

    @property
    def effects_dir(self) -> Path:
        return self.library_dir / "effects"

    @property
    def sources_dir(self) -> Path:
        return self.library_dir / "sources"

    @property
    def quarantine_dir(self) -> Path:
        return self.library_dir / "quarantine"

    @property
    def db_path(self) -> Path:
        return self.library_dir / "library.db"

    @property
    def models_dir(self) -> Path:
        return self.cache_dir / "models"


def _apply(dc, data: dict):
    """Apply a dict of overrides onto a dataclass instance, recursively."""
    for f in dataclasses.fields(dc):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(dc, f.name)
        if dataclasses.is_dataclass(current):
            _apply(current, value)
        elif isinstance(current, Path):
            setattr(dc, f.name, Path(value).expanduser())
        else:
            setattr(dc, f.name, type(current)(value))
    return dc


def load_settings(config_path: Path | None = None) -> Settings:
    settings = Settings()
    path = config_path or CONFIG_PATH
    if path.is_file():
        with open(path, "rb") as fh:
            _apply(settings, tomllib.load(fh))
    if env_lib := os.environ.get("WAVELENGTH_LIBRARY_DIR"):
        settings.library_dir = Path(env_lib).expanduser()
    return settings
