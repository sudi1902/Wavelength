"""Ingest stage: validate videos with ffprobe and extract their audio.

All demuxing/decoding goes through ffmpeg so anything the platforms produce
(.mp4/.mov/.webm, HEVC iPhone footage, ...) works uniformly.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from wavelength.config import MIN_SAMPLE_RATE


class IngestError(Exception):
    """A video cannot be processed; message is user-facing."""


@dataclass
class MediaInfo:
    path: Path
    duration_s: float
    sample_rate: int
    channels: int
    audio_codec: str
    container: str


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise IngestError(
                f"'{tool}' not found on PATH. Install it with: brew install ffmpeg"
            )


def probe(video: Path) -> MediaInfo:
    """Validate a media file and return its audio properties.

    Raises IngestError for missing files, non-media/corrupted files, and
    files without an audio stream.
    """
    if not video.is_file():
        raise IngestError(f"File not found: {video}")

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
        raise IngestError(f"Not a readable media file: {video.name} ({detail})")

    info = json.loads(result.stdout)
    audio_streams = [
        s for s in info.get("streams", []) if s.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise IngestError(f"No audio track in {video.name}")

    audio = audio_streams[0]
    fmt = info.get("format", {})
    duration = float(fmt.get("duration") or audio.get("duration") or 0.0)
    if duration <= 0:
        raise IngestError(f"Could not determine duration of {video.name}")

    return MediaInfo(
        path=video,
        duration_s=duration,
        sample_rate=int(audio.get("sample_rate") or 0),
        channels=int(audio.get("channels") or 0),
        audio_codec=audio.get("codec_name", "unknown"),
        container=fmt.get("format_name", "unknown"),
    )


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_audio(video: Path, out_wav: Path, info: MediaInfo) -> int:
    """Extract the audio track to a stereo float WAV at its native sample
    rate (floored at 44.1 kHz). Returns the sample rate of the output."""
    rate = max(info.sample_rate, MIN_SAMPLE_RATE)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v", "error",
            "-i", str(video),
            "-vn",
            "-ac", "2",
            "-ar", str(rate),
            "-c:a", "pcm_f32le",
            str(out_wav),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
        raise IngestError(f"Audio extraction failed for {video.name}: {detail}")
    return rate
