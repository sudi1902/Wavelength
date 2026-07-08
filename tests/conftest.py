"""Shared fixtures: synthetic audio with known sound events, and synthetic
videos built with ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wavelength.config import Settings

SR = 44_100


def tone_burst(
    freq: float, duration_s: float, sr: int = SR, amplitude: float = 0.5
) -> np.ndarray:
    """A short enveloped sine burst — a clean stand-in for a 'ding'."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    envelope = np.exp(-4.0 * t / duration_s)
    return (amplitude * envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def build_events_audio(
    events: list[tuple[float, np.ndarray]], total_s: float, sr: int = SR
) -> np.ndarray:
    """Place event arrays at given start times over silence."""
    y = np.zeros(int(total_s * sr), dtype=np.float32)
    for start_s, event in events:
        i0 = int(start_s * sr)
        i1 = min(i0 + len(event), len(y))
        y[i0:i1] += event[: i1 - i0]
    return y


@pytest.fixture
def three_event_audio() -> np.ndarray:
    """4 s of audio containing three distinct sound events."""
    return build_events_audio(
        [
            (0.5, tone_burst(880, 0.3)),
            (1.8, tone_burst(440, 0.4)),
            (3.0, tone_burst(1320, 0.25)),
        ],
        total_s=4.0,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.library_dir = tmp_path / "library"
    s.cache_dir = tmp_path / "cache"
    s.engine = "none"
    s.archive_sources = True
    return s


def make_video(path: Path, audio: np.ndarray, sr: int = SR) -> Path:
    """Mux synthetic audio with a black video track into an mp4."""
    wav = path.with_suffix(".src.wav")
    sf.write(wav, audio, sr)
    duration = len(audio) / sr
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={duration}",
            "-i", str(wav),
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True, capture_output=True,
    )
    wav.unlink()
    return path
