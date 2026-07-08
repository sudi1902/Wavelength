import numpy as np

from wavelength.config import CleaningConfig
from wavelength.pipeline.clean import CleanedEffect, RejectReason, clean_segment
from wavelength.pipeline.segment import Segment

from conftest import SR, build_events_audio, tone_burst


def _stem_with_event() -> np.ndarray:
    return build_events_audio([(1.0, tone_burst(880, 0.5))], total_s=3.0)


def test_clean_produces_padded_faded_normalized_clip():
    stem = _stem_with_event()
    cfg = CleaningConfig()
    result = clean_segment(stem, SR, Segment(1.0, 1.5), cfg)
    assert isinstance(result, CleanedEffect)
    # Padding applied on both sides.
    assert result.duration_s > 0.5
    assert result.start_in_source_s < 1.0
    # Fades: endpoints are (near) zero.
    assert abs(result.audio[0, 0]) < 1e-4
    assert abs(result.audio[0, -1]) < 1e-4
    # Normalized to -1 dBFS.
    peak = np.max(np.abs(result.audio))
    assert 0.85 < peak <= 0.9


def test_rejects_too_short():
    stem = _stem_with_event()
    result = clean_segment(stem, SR, Segment(1.0, 1.05))
    assert result is RejectReason.TOO_SHORT


def test_rejects_too_quiet():
    stem = _stem_with_event() * 0.005  # peak ~ -46 dBFS
    result = clean_segment(stem, SR, Segment(1.0, 1.5))
    assert result is RejectReason.TOO_QUIET


def test_rejects_static_noise():
    rng = np.random.default_rng(7)
    stem = (rng.standard_normal(SR * 2) * 0.1).astype(np.float32)
    result = clean_segment(stem, SR, Segment(0.5, 1.5))
    assert result is RejectReason.NOISE


def test_keeps_whoosh_like_enveloped_noise():
    # A whoosh is spectrally noise-like but has a strong envelope — the
    # noise filter must NOT reject it.
    rng = np.random.default_rng(7)
    n = int(SR * 0.8)
    t = np.linspace(0, 1, n)
    envelope = np.sin(np.pi * t) ** 2
    whoosh = (rng.standard_normal(n) * envelope * 0.4).astype(np.float32)
    stem = np.zeros(SR * 2, dtype=np.float32)
    stem[int(0.5 * SR) : int(0.5 * SR) + n] = whoosh
    result = clean_segment(stem, SR, Segment(0.5, 1.3))
    assert isinstance(result, CleanedEffect)


def test_stereo_stem():
    stem = np.stack([_stem_with_event(), _stem_with_event()])
    result = clean_segment(stem, SR, Segment(1.0, 1.5))
    assert isinstance(result, CleanedEffect)
    assert result.audio.shape[0] == 2
