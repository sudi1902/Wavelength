"""Segmentation stage: find discrete sound events in the effects stem.

Two mechanisms, both operating on a mono mixdown:

1. RMS energy gating — frames are "active" when their RMS exceeds both an
   absolute threshold and the estimated noise floor plus a margin. Contiguous
   active frames become regions; regions separated by short gaps are merged.
2. Onset splitting — regions longer than ``max_segment_s`` (likely several
   effects blurred together, or an effect over ambience) are split at
   spectral-flux onsets.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from wavelength.config import SegmentationConfig


@dataclass
class Segment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class GateStats:
    """How the activity gate was set for a stem — surfaced by --debug so a
    missed effect can be traced to its numbers."""

    noise_floor_db: float
    threshold_db: float
    peak_frame_db: float


def gate_stats(
    y: np.ndarray, sr: int, cfg: SegmentationConfig | None = None
) -> GateStats:
    cfg = cfg or SegmentationConfig()
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    rms_db = _frame_rms_db(y, cfg)
    floor = _noise_floor_db(rms_db)
    return GateStats(
        noise_floor_db=floor,
        threshold_db=max(cfg.activity_threshold_db, floor + cfg.noise_floor_margin_db),
        peak_frame_db=float(np.max(rms_db)) if rms_db.size else -80.0,
    )


def _frame_rms_db(y: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    rms = librosa.feature.rms(
        y=y, frame_length=cfg.frame_length, hop_length=cfg.hop_length
    )[0]
    return librosa.amplitude_to_db(rms, ref=1.0, top_db=None)


def _noise_floor_db(rms_db: np.ndarray) -> float:
    """Estimate the noise floor as a low percentile of non-digital-silence
    frames. Digital silence (< -80 dB) is excluded so a stem that is mostly
    true zeros doesn't drag the floor to -infinity."""
    audible = rms_db[rms_db > -80.0]
    if audible.size == 0:
        return -80.0
    return float(np.percentile(audible, 10))


def _active_regions(
    active: np.ndarray, times: np.ndarray, merge_gap_s: float
) -> list[Segment]:
    """Turn a boolean per-frame activity mask into merged time regions."""
    regions: list[Segment] = []
    start = None
    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = times[i]
        elif not is_active and start is not None:
            regions.append(Segment(start, times[i]))
            start = None
    if start is not None:
        regions.append(Segment(start, float(times[-1])))

    merged: list[Segment] = []
    for region in regions:
        if merged and region.start_s - merged[-1].end_s < merge_gap_s:
            merged[-1] = Segment(merged[-1].start_s, region.end_s)
        else:
            merged.append(region)
    return merged


def _split_at_onsets(
    y: np.ndarray, sr: int, region: Segment, cfg: SegmentationConfig
) -> list[Segment]:
    """Split an over-long region at detected onsets. Sub-segments shorter
    than min_segment_s are merged into their predecessor rather than lost."""
    i0 = int(region.start_s * sr)
    i1 = int(region.end_s * sr)
    onset_times = librosa.onset.onset_detect(
        y=y[i0:i1],
        sr=sr,
        hop_length=cfg.hop_length,
        backtrack=True,
        units="time",
    )
    # Cut points strictly inside the region (an onset at ~0 is the region's
    # own start, not a split point).
    cuts = [region.start_s + t for t in onset_times if 0.05 < t]
    if not cuts:
        return [region]

    bounds = [region.start_s, *cuts, region.end_s]
    parts: list[Segment] = []
    for a, b in zip(bounds, bounds[1:]):
        seg = Segment(a, b)
        if parts and seg.duration_s < cfg.min_segment_s:
            parts[-1] = Segment(parts[-1].start_s, seg.end_s)
        else:
            parts.append(seg)
    return parts


def detect_segments(
    y: np.ndarray, sr: int, cfg: SegmentationConfig | None = None
) -> list[Segment]:
    """Detect discrete sound events. ``y`` may be mono (n,) or multi-channel
    (channels, n); detection runs on a mono mixdown but returned times apply
    to the original audio."""
    cfg = cfg or SegmentationConfig()
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    if y.size == 0:
        return []

    rms_db = _frame_rms_db(y, cfg)
    times = librosa.frames_to_time(
        np.arange(len(rms_db)), sr=sr, hop_length=cfg.hop_length
    )
    threshold = max(
        cfg.activity_threshold_db, _noise_floor_db(rms_db) + cfg.noise_floor_margin_db
    )
    regions = _active_regions(rms_db > threshold, times, cfg.merge_gap_s)

    segments: list[Segment] = []
    for region in regions:
        if region.duration_s < cfg.min_segment_s:
            continue
        if cfg.onset_split and region.duration_s > cfg.max_segment_s:
            segments.extend(_split_at_onsets(y, sr, region, cfg))
        else:
            segments.append(region)
    return segments
