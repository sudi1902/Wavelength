"""SQLite index for the sound library.

The database holds all metadata; audio files on disk are the source of
truth for audio. Paths are stored relative to the library dir so the whole
library folder can be moved.

Schema is Phase-1 minimal but forward-compatible: label/tag columns already
exist for Phase 2 (auto-labeling) and Phase 3 (UI tagging).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_videos (
    id INTEGER PRIMARY KEY,
    sha256 TEXT NOT NULL UNIQUE,
    original_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    duration_s REAL NOT NULL,
    engine TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'processed',  -- processed | failed
    error TEXT,
    effect_count INTEGER NOT NULL DEFAULT 0,
    archived_video_path TEXT,
    archived_stem_path TEXT,
    ingested_at TEXT NOT NULL,
    source_url TEXT,                           -- canonical URL for URL ingests
    title TEXT,                                -- post title from the platform
    uploader TEXT                              -- account the post came from
);

CREATE TABLE IF NOT EXISTS effects (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES source_videos(id),
    path TEXT NOT NULL,                        -- relative to library dir
    content_sha256 TEXT NOT NULL UNIQUE,       -- hash of 16-bit PCM payload
    duration_s REAL NOT NULL,
    sample_rate INTEGER NOT NULL,
    peak_db REAL,
    start_in_source_s REAL,
    auto_label TEXT,
    label_confidence REAL,
    user_label TEXT,
    tags TEXT NOT NULL DEFAULT '',             -- comma-separated
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'library',    -- library | quarantine
    quarantine_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_effects_source ON effects(source_id);
"""

# Columns added after the first release; applied to existing databases on
# open. SQLite's CREATE TABLE IF NOT EXISTS does not add columns to tables
# that already exist.
MIGRATIONS = [
    ("source_videos", "source_url", "TEXT"),
    ("source_videos", "title", "TEXT"),
    ("source_videos", "uploader", "TEXT"),
    # Phase 2: 'library' or 'quarantine' (speech/music bleed under review)
    ("effects", "status", "TEXT NOT NULL DEFAULT 'library'"),
    ("effects", "quarantine_reason", "TEXT"),
    # Phase 3
    ("effects", "favorite", "INTEGER NOT NULL DEFAULT 0"),
    # Phase 3.5: CLAP audio embedding (float32 bytes) for similarity search;
    # license/attribution for sounds imported from Freesound; derived_from
    # links an imported sound to the extracted effect it replaces/augments.
    ("effects", "embedding", "BLOB"),
    ("effects", "license", "TEXT"),
    ("effects", "attribution", "TEXT"),
    ("effects", "derived_from", "INTEGER"),
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LibraryDB:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        for table, column, col_type in MIGRATIONS:
            existing = {
                row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")
            }
            if column not in existing:
                self.conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
                )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "LibraryDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- source videos -------------------------------------------------------

    def find_source(self, sha256: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM source_videos WHERE sha256 = ?", (sha256,)
        ).fetchone()

    def find_source_by_url(self, source_url: str) -> sqlite3.Row | None:
        """URL-based dedup: platforms re-encode on every download, so the
        same post yields different file hashes — the URL is the stable key."""
        return self.conn.execute(
            "SELECT * FROM source_videos WHERE source_url = ?"
            " ORDER BY id DESC LIMIT 1",
            (source_url,),
        ).fetchone()

    def add_source(
        self, *, sha256: str, original_path: str, filename: str,
        duration_s: float, engine: str,
        source_url: str | None = None, title: str | None = None,
        uploader: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO source_videos"
            " (sha256, original_path, filename, duration_s, engine,"
            "  ingested_at, source_url, title, uploader)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sha256, original_path, filename, duration_s, engine, utcnow(),
             source_url, title, uploader),
        )
        self.conn.commit()
        return cur.lastrowid

    def finish_source(
        self, source_id: int, *, effect_count: int,
        archived_video_path: str | None = None,
        archived_stem_path: str | None = None,
        status: str = "processed", error: str | None = None,
    ) -> None:
        self.conn.execute(
            "UPDATE source_videos SET effect_count = ?, archived_video_path = ?,"
            " archived_stem_path = ?, status = ?, error = ? WHERE id = ?",
            (effect_count, archived_video_path, archived_stem_path, status,
             error, source_id),
        )
        self.conn.commit()

    def effects_for_source(self, source_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM effects WHERE source_id = ?", (source_id,)
        ).fetchall()

    def delete_source(self, source_id: int) -> None:
        """Remove a source and its effect rows (used to re-process).
        Callers that own the files should remove them first via
        effects_for_source — this only touches the database."""
        self.conn.execute("DELETE FROM effects WHERE source_id = ?", (source_id,))
        self.conn.execute("DELETE FROM source_videos WHERE id = ?", (source_id,))
        self.conn.commit()

    # -- effects ---------------------------------------------------------------

    def find_effect_by_hash(self, content_sha256: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM effects WHERE content_sha256 = ?", (content_sha256,)
        ).fetchone()

    def add_effect(
        self, *, source_id: int, path: str, content_sha256: str,
        duration_s: float, sample_rate: int, peak_db: float | None,
        start_in_source_s: float | None,
        auto_label: str | None = None, label_confidence: float | None = None,
        status: str = "library", quarantine_reason: str | None = None,
        embedding: bytes | None = None, license: str | None = None,
        attribution: str | None = None, derived_from: int | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO effects (source_id, path, content_sha256, duration_s,"
            " sample_rate, peak_db, start_in_source_s, created_at,"
            " auto_label, label_confidence, status, quarantine_reason,"
            " embedding, license, attribution, derived_from)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (source_id, path, content_sha256, duration_s, sample_rate,
             peak_db, start_in_source_s, utcnow(),
             auto_label, label_confidence, status, quarantine_reason,
             embedding, license, attribution, derived_from),
        )
        self.conn.commit()
        return cur.lastrowid

    def set_embedding(self, effect_id: int, embedding: bytes) -> None:
        self.conn.execute(
            "UPDATE effects SET embedding = ? WHERE id = ?",
            (embedding, effect_id),
        )
        self.conn.commit()

    _EFFECT_SELECT = (
        "SELECT e.*, s.filename AS source_filename, s.title AS source_title,"
        " s.uploader AS source_uploader, s.source_url AS source_url"
        " FROM effects e JOIN source_videos s ON s.id = e.source_id"
    )

    def list_effects(self, status: str = "library") -> list[sqlite3.Row]:
        return self.conn.execute(
            f"{self._EFFECT_SELECT} WHERE e.status = ?"
            " ORDER BY e.created_at DESC, e.id DESC",
            (status,),
        ).fetchall()

    def search_effects(self, query: str) -> list[sqlite3.Row]:
        """Case-insensitive substring match across labels, tags, filename,
        and source title/uploader. Library effects only."""
        like = f"%{query}%"
        return self.conn.execute(
            f"{self._EFFECT_SELECT} WHERE e.status = 'library' AND ("
            " e.auto_label LIKE ? OR e.user_label LIKE ? OR e.tags LIKE ?"
            " OR e.path LIKE ? OR s.title LIKE ? OR s.uploader LIKE ?"
            " OR s.filename LIKE ?)"
            " ORDER BY e.created_at DESC, e.id DESC",
            (like, like, like, like, like, like, like),
        ).fetchall()

    def get_effect(self, effect_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            f"{self._EFFECT_SELECT} WHERE e.id = ?", (effect_id,)
        ).fetchone()

    def set_user_label(self, effect_id: int, label: str) -> None:
        self.conn.execute(
            "UPDATE effects SET user_label = ? WHERE id = ?", (label, effect_id)
        )
        self.conn.commit()

    def set_tags(self, effect_id: int, tags: str) -> None:
        self.conn.execute(
            "UPDATE effects SET tags = ? WHERE id = ?", (tags, effect_id)
        )
        self.conn.commit()

    def set_status(
        self, effect_id: int, status: str, path: str | None = None
    ) -> None:
        if path is not None:
            self.conn.execute(
                "UPDATE effects SET status = ?, path = ? WHERE id = ?",
                (status, path, effect_id),
            )
        else:
            self.conn.execute(
                "UPDATE effects SET status = ? WHERE id = ?", (status, effect_id)
            )
        self.conn.commit()

    def set_favorite(self, effect_id: int, favorite: bool) -> None:
        self.conn.execute(
            "UPDATE effects SET favorite = ? WHERE id = ?",
            (1 if favorite else 0, effect_id),
        )
        self.conn.commit()

    def delete_effect(self, effect_id: int) -> None:
        self.conn.execute("DELETE FROM effects WHERE id = ?", (effect_id,))
        self.conn.commit()
