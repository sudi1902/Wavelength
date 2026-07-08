"""Download module tests with yt-dlp mocked — no network involved.
The real download path gets validated manually on real TikTok/Instagram
links (platforms block datacenter traffic, so CI can't do it)."""

import numpy as np
import pytest
import soundfile as sf

import wavelength.pipeline.download as download_mod
from wavelength.pipeline.download import DownloadError, download_video, is_url


def test_is_url():
    assert is_url("https://www.tiktok.com/@user/video/123")
    assert is_url("http://example.com/v")
    assert not is_url("~/Downloads/clip.mp4")
    assert not is_url("clip.mp4")


class _FakeYDL:
    """Mimics yt_dlp.YoutubeDL: extract_info(download=False) returns
    metadata; download=True writes the file."""

    captured_opts = None
    info = {}

    def __init__(self, opts):
        type(self).captured_opts = opts
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        info = dict(type(self).info)
        if download:
            from pathlib import Path

            out = Path(self.opts["outtmpl"] % {"id": info["id"], "ext": "m4a"})
            sf.write(out, np.zeros(1000, dtype=np.float32), 44100, format="WAV")
            info["requested_downloads"] = [{"filepath": str(out)}]
        return info

    def prepare_filename(self, info):
        return self.opts["outtmpl"] % {"id": info["id"], "ext": "m4a"}


@pytest.fixture
def fake_ydl(monkeypatch):
    _FakeYDL.info = {
        "id": "vid123",
        "title": "funny cat video",
        "uploader": "catlady",
        "duration": 32.0,
        "webpage_url": "https://www.tiktok.com/@catlady/video/123",
    }
    monkeypatch.setattr(download_mod.yt_dlp, "YoutubeDL", _FakeYDL)
    return _FakeYDL


def test_download_returns_metadata_and_file(tmp_path, fake_ydl):
    result = download_video("https://vt.tiktok.com/short123/", tmp_path)
    assert result.path.is_file()
    assert result.title == "funny cat video"
    assert result.uploader == "catlady"
    # Canonical URL from the platform, not the short link passed in.
    assert result.url == "https://www.tiktok.com/@catlady/video/123"
    assert result.duration_s == 32.0


def test_download_rejects_over_long_video_before_fetching(tmp_path, fake_ydl):
    fake_ydl.info = {**fake_ydl.info, "duration": 3600.0}
    with pytest.raises(DownloadError, match="force-long"):
        download_video(
            "https://www.tiktok.com/@u/video/1", tmp_path, max_duration_s=600
        )


def test_download_browser_cookies_passed_through(tmp_path, fake_ydl):
    download_video("https://www.instagram.com/reel/x/", tmp_path, browser="chrome")
    assert fake_ydl.captured_opts["cookiesfrombrowser"] == ("chrome",)


def test_download_no_cookies_by_default(tmp_path, fake_ydl):
    download_video("https://www.tiktok.com/@u/video/1", tmp_path)
    assert "cookiesfrombrowser" not in fake_ydl.captured_opts


def test_download_failure_gives_actionable_message(tmp_path, monkeypatch):
    class _BoomYDL(_FakeYDL):
        def extract_info(self, url, download=False):
            raise download_mod.yt_dlp.utils.DownloadError(
                "ERROR: [Instagram] login required, rate-limit reached"
            )

    monkeypatch.setattr(download_mod.yt_dlp, "YoutubeDL", _BoomYDL)
    with pytest.raises(DownloadError) as exc:
        download_video("https://www.instagram.com/reel/abc/", tmp_path)
    message = str(exc.value)
    assert "--browser" in message
    assert "pip install -U yt-dlp" in message
