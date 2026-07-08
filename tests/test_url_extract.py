"""URL-ingestion flow: download (faked) → full pipeline → library record
with URL metadata; URL-level dedup; DB migration from the pre-URL schema."""

import shutil
import sqlite3

import pytest

import wavelength.pipeline.runner as runner_mod
from wavelength.library.db import LibraryDB
from wavelength.pipeline.download import DownloadedVideo
from wavelength.pipeline.runner import extract_url

from conftest import make_video, settings, three_event_audio  # noqa: F401

URL = "https://www.tiktok.com/@catlady/video/123"


@pytest.fixture
def fake_download(tmp_path, three_event_audio, monkeypatch):
    """Replace download_video with one that produces a real local video."""
    src = make_video(tmp_path / "downloaded_src.mp4", three_event_audio)
    calls = []

    def _fake(url, dest_dir, *, browser=None, max_duration_s=None,
              progress=lambda m: None):
        calls.append(url)
        dest = dest_dir / "vid123.mp4"
        shutil.copy2(src, dest)
        return DownloadedVideo(
            path=dest, url=URL, title="funny cat video",
            uploader="catlady", duration_s=4.0,
        )

    monkeypatch.setattr(runner_mod, "download_video", _fake)
    return calls


def test_extract_url_runs_pipeline_and_records_metadata(settings, fake_download):
    result = extract_url(URL, settings)
    assert result.effect_count == 3
    with LibraryDB(settings.db_path) as db:
        row = db.find_source_by_url(URL)
        assert row is not None
        assert row["title"] == "funny cat video"
        assert row["uploader"] == "catlady"
        assert row["effect_count"] == 3
        # Library listing surfaces the title.
        assert db.list_effects()[0]["source_title"] == "funny cat video"


def test_extract_url_dedupes_by_url_without_downloading(settings, fake_download):
    extract_url(URL, settings)
    assert len(fake_download) == 1
    second = extract_url(URL, settings)
    assert second.already_processed
    assert len(fake_download) == 1  # no second download

    forced = extract_url(URL, settings, force=True)
    assert not forced.already_processed
    assert len(fake_download) == 2


def test_db_migration_adds_url_columns_to_old_library(tmp_path):
    """A library created before Phase 1.5 (no url/title/uploader columns)
    must open cleanly and accept URL ingests."""
    db_path = tmp_path / "old" / "library.db"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE source_videos (
            id INTEGER PRIMARY KEY,
            sha256 TEXT NOT NULL UNIQUE,
            original_path TEXT NOT NULL,
            filename TEXT NOT NULL,
            duration_s REAL NOT NULL,
            engine TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'processed',
            error TEXT,
            effect_count INTEGER NOT NULL DEFAULT 0,
            archived_video_path TEXT,
            archived_stem_path TEXT,
            ingested_at TEXT NOT NULL
        );
        INSERT INTO source_videos
            (sha256, original_path, filename, duration_s, engine, ingested_at)
        VALUES ('aa', '/x/v.mp4', 'v.mp4', 4.0, 'bandit', '2026-07-08');
        """
    )
    conn.commit()
    conn.close()

    with LibraryDB(db_path) as db:
        # Old row survives, new columns exist and are NULL.
        row = db.find_source("aa")
        assert row["source_url"] is None
        # URL insert works post-migration.
        db.add_source(
            sha256="bb", original_path=URL, filename="vid123.mp4",
            duration_s=4.0, engine="bandit",
            source_url=URL, title="t", uploader="u",
        )
        assert db.find_source_by_url(URL)["title"] == "t"
