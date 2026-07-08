import numpy as np

from wavelength.config import SegmentationConfig
from wavelength.pipeline.segment import detect_segments

from conftest import SR, build_events_audio, tone_burst


def test_detects_three_distinct_events(three_event_audio):
    segments = detect_segments(three_event_audio, SR)
    assert len(segments) == 3
    starts = [s.start_s for s in segments]
    for expected, actual in zip([0.5, 1.8, 3.0], starts):
        assert abs(actual - expected) < 0.1


def test_silence_yields_no_segments():
    y = np.zeros(SR * 2, dtype=np.float32)
    assert detect_segments(y, SR) == []


def test_low_level_noise_yields_no_segments():
    rng = np.random.default_rng(42)
    y = (rng.standard_normal(SR * 2) * 1e-4).astype(np.float32)  # ~ -80 dBFS
    assert detect_segments(y, SR) == []


def test_close_events_are_merged():
    # Two bursts 50 ms apart: below merge_gap_s, should come out as one event.
    y = build_events_audio(
        [(0.5, tone_burst(880, 0.2)), (0.75, tone_burst(880, 0.2))], total_s=2.0
    )
    segments = detect_segments(y, SR)
    assert len(segments) == 1


def test_stereo_input_accepted(three_event_audio):
    stereo = np.stack([three_event_audio, three_event_audio])
    assert len(detect_segments(stereo, SR)) == 3


def test_long_region_is_split_at_onsets():
    # Six bursts spaced 0.4 s apart riding on a continuous bed loud enough to
    # keep the gate open — one long region, must be split by onsets.
    events = [(0.2 + i * 0.4, tone_burst(700 + 200 * i, 0.35, amplitude=0.6))
              for i in range(6)]
    y = build_events_audio(events, total_s=3.0)
    rng = np.random.default_rng(0)
    y += (rng.standard_normal(len(y)) * 0.02).astype(np.float32)  # bed ~ -34 dB
    cfg = SegmentationConfig(max_segment_s=1.0)
    segments = detect_segments(y, SR, cfg)
    assert len(segments) >= 3
