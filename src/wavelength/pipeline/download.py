"""Download stage: fetch videos from TikTok/Instagram/etc. URLs via yt-dlp.

This is a front door to the existing pipeline — a URL is downloaded to a
temp dir and then processed exactly like a local file. Only the audio
matters downstream, so the audio-only stream is preferred when the platform
offers one (smaller download, identical result).

Instagram frequently blocks anonymous access; pass ``browser`` (e.g.
"chrome", "safari") to borrow the logged-in session cookies from that
browser. This downloads only what the user could already watch — it does
not bypass logins beyond using their own session.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yt_dlp


class DownloadError(Exception):
    """Download failed; message is user-facing."""


@dataclass
class DownloadedVideo:
    path: Path
    url: str  # canonical URL (post-redirect), used for dedup
    title: str | None
    uploader: str | None
    duration_s: float | None


def is_url(target: str) -> bool:
    return target.startswith(("http://", "https://"))


def download_video(
    url: str,
    dest_dir: Path,
    *,
    browser: str | None = None,
    max_duration_s: float | None = None,
    progress: Callable[[str], None] = lambda msg: None,
) -> DownloadedVideo:
    """Download a video (audio stream preferred) into dest_dir."""
    opts = {
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        # Audio-only stream when available; whole video otherwise.
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }
    if browser:
        opts["cookiesfrombrowser"] = (browser,)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            # Two-phase: resolve metadata first so an over-long video is
            # rejected before any bytes are downloaded.
            info = ydl.extract_info(url, download=False)
            if info.get("_type") == "playlist" and info.get("entries"):
                info = info["entries"][0]

            duration = info.get("duration")
            if (
                max_duration_s is not None
                and duration is not None
                and duration > max_duration_s
            ):
                raise DownloadError(
                    f"Video is {duration / 60:.1f} min long "
                    f"(cap: {max_duration_s / 60:.0f} min). "
                    "Re-run with --force-long to process anyway."
                )

            title = info.get("title")
            progress(f"Downloading: {title or url}")
            info = ydl.extract_info(url, download=True)
            if info.get("_type") == "playlist" and info.get("entries"):
                info = info["entries"][0]

            path = _downloaded_path(ydl, info)
    except DownloadError:
        raise
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(_friendly_message(url, str(exc))) from exc
    except Exception as exc:  # noqa: BLE001 - yt-dlp raises many types
        raise DownloadError(_friendly_message(url, str(exc))) from exc

    if not path.is_file():
        raise DownloadError(f"Download reported success but no file found for {url}")

    return DownloadedVideo(
        path=path,
        url=info.get("webpage_url") or url,
        title=info.get("title"),
        uploader=info.get("uploader"),
        duration_s=float(duration) if duration else None,
    )


def _downloaded_path(ydl, info: dict) -> Path:
    downloads = info.get("requested_downloads") or []
    if downloads and downloads[0].get("filepath"):
        return Path(downloads[0]["filepath"])
    return Path(ydl.prepare_filename(info))


def _friendly_message(url: str, detail: str) -> str:
    detail = detail.strip().splitlines()[-1] if detail.strip() else "unknown error"
    hints = []
    lower = detail.lower()
    if "instagram" in url.lower() and (
        "login" in lower or "rate" in lower or "not available" in lower
    ):
        hints.append(
            "Instagram often requires a logged-in session: retry with "
            "--browser chrome (or safari/firefox) to use your browser's cookies."
        )
    if "private" in lower:
        hints.append("The post appears to be private.")
    hints.append(
        "Platforms change frequently; if this persists, update the downloader: "
        "pip install -U yt-dlp"
    )
    return f"Could not download {url}: {detail}\n" + "\n".join(hints)
