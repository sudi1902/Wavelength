import subprocess

import numpy as np
import pytest
import soundfile as sf

from wavelength.pipeline import ingest

from conftest import SR, make_video, three_event_audio  # noqa: F401


def test_probe_valid_video(tmp_path, three_event_audio):
    video = make_video(tmp_path / "clip.mp4", three_event_audio)
    info = ingest.probe(video)
    assert info.channels >= 1
    assert info.sample_rate == SR
    assert 3.5 < info.duration_s < 4.5


def test_probe_missing_file(tmp_path):
    with pytest.raises(ingest.IngestError, match="not found"):
        ingest.probe(tmp_path / "nope.mp4")


def test_probe_corrupted_file(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video" * 100)
    with pytest.raises(ingest.IngestError):
        ingest.probe(junk)


def test_probe_video_without_audio(tmp_path):
    video = tmp_path / "mute.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
            "-c:v", "libx264", "-preset", "ultrafast", str(video),
        ],
        check=True, capture_output=True,
    )
    with pytest.raises(ingest.IngestError, match="No audio"):
        ingest.probe(video)


def test_extract_audio_roundtrip(tmp_path, three_event_audio):
    video = make_video(tmp_path / "clip.mp4", three_event_audio)
    info = ingest.probe(video)
    out = tmp_path / "audio.wav"
    rate = ingest.extract_audio(video, out, info)
    assert rate == SR
    audio, sr = sf.read(out, always_2d=True)
    assert sr == SR
    assert audio.shape[1] == 2
    # Events survive the aac encode round-trip.
    assert np.max(np.abs(audio)) > 0.2


def test_file_sha256_stable(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    assert ingest.file_sha256(f) == ingest.file_sha256(f)
