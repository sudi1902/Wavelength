"""Watch-folder mode: auto-extract new videos dropped into a folder.

Watches for supported video files (AirDrop landings, browser downloads,
Finder copies) and fires a callback once the file has finished writing —
downloads arrive in chunks, so a file is only considered ready after its
size has been stable for a quiet period. Already-processed videos are
skipped by the pipeline's own hash dedup, so re-firing on the same file
is harmless.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from wavelength.config import SUPPORTED_EXTENSIONS

# Browsers download via temp names; ignore their in-progress extensions.
IGNORED_SUFFIXES = {".part", ".crdownload", ".download", ".tmp"}


def wait_until_stable(
    path: Path, quiet_s: float = 2.0, timeout_s: float = 600.0,
    poll_s: float = 0.5,
) -> bool:
    """Block until the file's size has not changed for ``quiet_s`` seconds.
    Returns False if the file disappears or the timeout passes."""
    deadline = time.monotonic() + timeout_s
    last_size = -1
    stable_since = None
    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size == last_size and size > 0:
            if stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= quiet_s:
                return True
        else:
            stable_since = None
            last_size = size
        time.sleep(poll_s)
    return False


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher: "VideoWatcher"):
        self.watcher = watcher

    def on_created(self, event):
        if not event.is_directory:
            self.watcher._consider(Path(event.src_path))

    def on_moved(self, event):
        # Browser download completion is usually a rename from a temp name.
        if not event.is_directory:
            self.watcher._consider(Path(event.dest_path))


class VideoWatcher:
    """Watches ``folder`` and calls ``on_video(path)`` (from a worker
    thread) for each new, fully-written, supported video file."""

    def __init__(
        self,
        folder: Path,
        on_video: Callable[[Path], None],
        *,
        quiet_s: float = 2.0,
    ):
        self.folder = folder.expanduser()
        self.on_video = on_video
        self.quiet_s = quiet_s
        self._observer: Observer | None = None
        self._seen: set[Path] = set()
        self._lock = threading.Lock()

    def _consider(self, path: Path) -> None:
        if path.suffix.lower() in IGNORED_SUFFIXES:
            return
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        with self._lock:
            if path in self._seen:
                return
            self._seen.add(path)
        threading.Thread(
            target=self._process, args=(path,), daemon=True
        ).start()

    def _process(self, path: Path) -> None:
        if wait_until_stable(path, quiet_s=self.quiet_s):
            self.on_video(path)
        with self._lock:
            self._seen.discard(path)

    def start(self) -> None:
        if self._observer is not None:
            return
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Watch folder does not exist: {self.folder}")
        self._observer = Observer()
        self._observer.schedule(_Handler(self), str(self.folder), recursive=False)
        self._observer.start()

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    @property
    def running(self) -> bool:
        return self._observer is not None
