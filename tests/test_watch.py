"""Folder watcher: detection, write-stabilization, and filtering."""

import shutil
import threading
import time
from pathlib import Path

from wavelength.watch import VideoWatcher, wait_until_stable

from conftest import make_video, three_event_audio  # noqa: F401


def test_wait_until_stable_waits_for_growth_to_stop(tmp_path):
    f = tmp_path / "grow.mp4"
    f.write_bytes(b"x" * 100)

    def grow():
        for _ in range(3):
            time.sleep(0.3)
            with open(f, "ab") as fh:
                fh.write(b"x" * 100)

    t = threading.Thread(target=grow)
    start = time.monotonic()
    t.start()
    assert wait_until_stable(f, quiet_s=0.6, poll_s=0.1)
    elapsed = time.monotonic() - start
    t.join()
    assert elapsed >= 0.9  # had to outlast the growth phase
    assert f.stat().st_size == 400


def test_wait_until_stable_missing_file(tmp_path):
    assert not wait_until_stable(tmp_path / "ghost.mp4", quiet_s=0.2, poll_s=0.05)


def test_watcher_fires_for_new_video(tmp_path, three_event_audio):
    src = make_video(tmp_path / "made" / "clip.mp4", three_event_audio)
    watch_dir = tmp_path / "downloads"
    watch_dir.mkdir()

    seen: list[Path] = []
    fired = threading.Event()

    def on_video(path: Path):
        seen.append(path)
        fired.set()

    watcher = VideoWatcher(watch_dir, on_video, quiet_s=0.4)
    watcher.start()
    try:
        shutil.copy2(src, watch_dir / "incoming.mp4")
        assert fired.wait(timeout=10), "watcher never fired"
        assert seen[0].name == "incoming.mp4"
    finally:
        watcher.stop()


def test_watcher_ignores_non_video_and_partials(tmp_path):
    watch_dir = tmp_path / "downloads"
    watch_dir.mkdir()
    seen = []
    watcher = VideoWatcher(watch_dir, seen.append, quiet_s=0.2)
    watcher.start()
    try:
        (watch_dir / "notes.txt").write_text("hello")
        (watch_dir / "video.mp4.part").write_bytes(b"partial")
        time.sleep(1.2)
        assert seen == []
    finally:
        watcher.stop()
