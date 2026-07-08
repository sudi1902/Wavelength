"""Cleaning stage: turn a raw segment into a usable, click-free effect clip.

Per segment: pull padded audio from the full stem, trim, fade in/out,
peak-normalize — and reject junk (too short, too quiet, static hiss).
Junk checks run BEFORE normalization so "too quiet" means quiet in the
source, not after gain.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import librosa
import numpy as np

from wavelength.config import CleaningConfig
from wavelength.pipeline.segment import Segment


class RejectReason(str, Enum):
    TOO_SHORT = "too_short"
    TOO_QUIET = "too_quiet"
    NOISE = "noise"


@dataclass
class Rejection:
    reason: RejectReason
    detail: str  # measured values vs threshold, for --debug output


@dataclass
class CleanedEffect:
    audio: np.ndarray  # (channels, samples) float32 in [-1, 1]
    sample_rate: int
    start_in_source_s: float
    duration_s: float
    peak_db: float  # pre-normalization peak


def _to_2d(y: np.ndarray) -> np.ndarray:
    return y[np.newaxis, :] if y.ndim == 1 else y


def _peak_db(y: np.ndarray) -> float:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return librosa.amplitude_to_db(np.array([max(peak, 1e-10)]))[0]


def _is_static_noise(y_mono: np.ndarray, cfg: CleaningConfig) -> bool:
    """Static hiss has a noise-like spectrum AND an almost-flat energy
    envelope. Both conditions are required: whooshes are noise-like
    spectrally but have a strong envelope (25+ dB of range vs ~1 dB for
    hiss), and steady pure-tone beeps have near-zero flatness — neither
    must be rejected."""
    if not np.any(y_mono):
        return True
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y_mono)))
    if flatness <= cfg.junk_spectral_flatness:
        return False
    rms_db = librosa.amplitude_to_db(
        librosa.feature.rms(y=y_mono)[0], ref=1.0, top_db=None
    )
    envelope_range = float(np.percentile(rms_db, 90) - np.percentile(rms_db, 10))
    return envelope_range < cfg.junk_envelope_range_db


def _fade(y: np.ndarray, sr: int, fade_in_s: float, fade_out_s: float) -> np.ndarray:
    n = y.shape[1]
    n_in = min(int(fade_in_s * sr), n // 2)
    n_out = min(int(fade_out_s * sr), n // 2)
    if n_in > 0:
        y[:, :n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    if n_out > 0:
        y[:, -n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)
    return y


def clean_segment(
    stem: np.ndarray,
    sr: int,
    segment: Segment,
    cfg: CleaningConfig | None = None,
) -> CleanedEffect | Rejection:
    """Extract one padded, faded, normalized effect clip from the effects stem.

    Returns a CleanedEffect, or a Rejection (reason + measured values) if
    the segment is junk. ``stem`` may be mono (n,) or (channels, n).
    """
    cfg = cfg or CleaningConfig()
    stem = _to_2d(stem)
    n_total = stem.shape[1]

    if segment.duration_s < cfg.min_duration_s:
        return Rejection(
            RejectReason.TOO_SHORT,
            f"{segment.duration_s * 1000:.0f}ms < {cfg.min_duration_s * 1000:.0f}ms",
        )

    core = stem[:, int(segment.start_s * sr) : int(segment.end_s * sr)]
    if core.size == 0:
        return Rejection(RejectReason.TOO_SHORT, "empty slice")

    peak_db = _peak_db(core)
    if peak_db < cfg.min_peak_db:
        return Rejection(
            RejectReason.TOO_QUIET,
            f"peak {peak_db:.1f} dBFS < {cfg.min_peak_db:.0f} dBFS",
        )

    if _is_static_noise(np.mean(core, axis=0), cfg):
        return Rejection(
            RejectReason.NOISE, "flat spectrum with near-static envelope"
        )

    # Padding comes from the surrounding stem audio (natural tails), clamped
    # to the stem bounds.
    i0 = max(0, int((segment.start_s - cfg.pre_pad_s) * sr))
    i1 = min(n_total, int((segment.end_s + cfg.post_pad_s) * sr))
    clip = stem[:, i0:i1].astype(np.float32).copy()

    clip = _fade(clip, sr, cfg.fade_in_s, cfg.fade_out_s)

    # Peak-normalize to the target ceiling.
    peak = float(np.max(np.abs(clip)))
    if peak > 0:
        target = librosa.db_to_amplitude(cfg.normalize_peak_db)
        clip *= target / peak

    return CleanedEffect(
        audio=clip,
        sample_rate=sr,
        start_in_source_s=i0 / sr,
        duration_s=clip.shape[1] / sr,
        peak_db=peak_db,
    )
