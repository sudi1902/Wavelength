"""FastAPI application: REST API + static frontend for the library UI.

Two modes:

- **Local** (default): single user on localhost. No sessions, full access —
  file-path extraction, watch folders, the works.
- **Public** (`wavelength serve --public`): safe to expose to the internet.
  Every browser gets an anonymous session cookie and its own private
  library; every effect operation is ownership-checked. File-path
  extraction and the watch API are disabled (they would read the host's
  filesystem), and extraction is rate-limited per session and per IP.

SQLite connections are opened per request (sqlite3 objects are not
thread-safe across FastAPI's threadpool).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from collections import defaultdict
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wavelength import __version__
from wavelength.config import Settings
from wavelength.library.db import LibraryDB
from wavelength.pipeline.download import is_url
from wavelength.pipeline.label import ClapLabeler
from wavelength.pipeline.separate import SeparationError, get_separator
from wavelength.server.jobs import JobQueue
from wavelength.watch import VideoWatcher

STATIC_DIR = Path(__file__).parent / "static"
SESSION_COOKIE = "wl_session"

# Public-mode extraction quotas (per calendar day).
SESSION_DAILY_LIMIT = 10
IP_DAILY_LIMIT = 30


def _effect_json(row) -> dict:
    return {
        "id": row["id"],
        "path": row["path"],
        "filename": Path(row["path"]).name,
        "label": row["user_label"] or row["auto_label"],
        "auto_label": row["auto_label"],
        "user_label": row["user_label"],
        "confidence": row["label_confidence"],
        "tags": [t for t in (row["tags"] or "").split(",") if t],
        "duration_s": row["duration_s"],
        "sample_rate": row["sample_rate"],
        "favorite": bool(row["favorite"]),
        "status": row["status"],
        "quarantine_reason": row["quarantine_reason"],
        "license": row["license"],
        "attribution": row["attribution"],
        "created_at": row["created_at"],
        "source": {
            "filename": row["source_filename"],
            "title": row["source_title"],
            "uploader": row["source_uploader"],
            "url": row["source_url"],
        },
    }


class ExtractRequest(BaseModel):
    target: str
    force: bool = False


class RenameRequest(BaseModel):
    label: str


class TagsRequest(BaseModel):
    tags: list[str]


class FavoriteRequest(BaseModel):
    favorite: bool


class WatchRequest(BaseModel):
    enabled: bool
    folder: str | None = None


class ImportRequest(BaseModel):
    freesound_id: int
    name: str
    username: str
    license: str
    duration_s: float
    page_url: str


class QuotaTracker:
    """In-memory daily counters. Single-process by design (the app runs one
    worker); resets naturally at midnight via the date in the key."""

    def __init__(self):
        self._counts: dict[tuple, int] = defaultdict(int)

    def check_and_count(self, session_id: str, ip: str) -> str | None:
        """Returns an error message when over quota; otherwise counts the
        extraction and returns None."""
        today = date.today().isoformat()
        if self._counts[("s", session_id, today)] >= SESSION_DAILY_LIMIT:
            return (
                f"Daily limit reached ({SESSION_DAILY_LIMIT} extractions). "
                "Come back tomorrow!"
            )
        if self._counts[("ip", ip, today)] >= IP_DAILY_LIMIT:
            return "Daily limit reached for this network. Come back tomorrow!"
        self._counts[("s", session_id, today)] += 1
        self._counts[("ip", ip, today)] += 1
        if len(self._counts) > 50_000:  # prune stale days
            self._counts = defaultdict(int, {
                k: v for k, v in self._counts.items() if k[2] == today
            })
        return None


def create_app(settings: Settings, public: bool = False) -> FastAPI:
    app = FastAPI(title="Wavelength", version=__version__)
    jobs = JobQueue(settings)
    quotas = QuotaTracker()
    watcher_state: dict = {"watcher": None, "folder": None}

    def _db() -> LibraryDB:
        return LibraryDB(settings.db_path)

    def _sid(request: Request) -> str | None:
        return getattr(request.state, "session_id", None) if public else None

    def _get_row(db: LibraryDB, effect_id: int, sid: str | None):
        row = db.get_effect(effect_id)
        if row is None:
            raise HTTPException(status_code=404, detail="No such effect")
        if public and row["session_id"] != sid:
            # Someone else's sound: indistinguishable from nonexistent.
            raise HTTPException(status_code=404, detail="No such effect")
        return row

    if public:
        @app.middleware("http")
        async def session_middleware(request: Request, call_next):
            sid = request.cookies.get(SESSION_COOKIE, "")
            fresh = not (len(sid) == 32 and sid.isalnum())
            if fresh:
                sid = uuid.uuid4().hex
            request.state.session_id = sid
            response: Response = await call_next(request)
            if fresh:
                response.set_cookie(
                    SESSION_COOKIE, sid,
                    max_age=180 * 24 * 3600, httponly=True, samesite="lax",
                )
            return response

    # -- library -----------------------------------------------------------

    @app.get("/api/effects")
    def list_effects(request: Request, q: str = "", status: str = "library"):
        sid = _sid(request)
        with _db() as db:
            rows = (
                db.search_effects(q, sid, scoped=public) if q
                else db.list_effects(status=status, session_id=sid, scoped=public)
            )
        return [_effect_json(r) for r in rows]

    @app.get("/api/effects/{effect_id}/audio")
    def effect_audio(request: Request, effect_id: int):
        with _db() as db:
            row = _get_row(db, effect_id, _sid(request))
        path = settings.library_dir / row["path"]
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Audio file missing")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/api/effects/{effect_id}/download")
    def effect_download(request: Request, effect_id: int, mp3: bool = False):
        with _db() as db:
            row = _get_row(db, effect_id, _sid(request))
        src = settings.library_dir / row["path"]
        if not src.is_file():
            raise HTTPException(status_code=404, detail="Audio file missing")
        name = row["user_label"] or Path(row["path"]).stem
        if not mp3:
            return FileResponse(src, media_type="audio/wav", filename=f"{name}.wav")
        tmp = Path(tempfile.mkdtemp(prefix="wavelength-dl-")) / f"{name}.mp3"
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src),
             "-c:a", "libmp3lame", "-b:a", "320k", str(tmp)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="MP3 encode failed")
        return FileResponse(tmp, media_type="audio/mpeg", filename=tmp.name)

    @app.post("/api/effects/{effect_id}/rename")
    def rename_effect(request: Request, effect_id: int, body: RenameRequest):
        with _db() as db:
            _get_row(db, effect_id, _sid(request))
            db.set_user_label(effect_id, body.label.strip()[:80])
            return _effect_json(db.get_effect(effect_id))

    @app.post("/api/effects/{effect_id}/tags")
    def set_tags(request: Request, effect_id: int, body: TagsRequest):
        with _db() as db:
            _get_row(db, effect_id, _sid(request))
            db.set_tags(
                effect_id,
                ",".join(t.strip()[:40] for t in body.tags if t.strip())[:400],
            )
            return _effect_json(db.get_effect(effect_id))

    @app.post("/api/effects/{effect_id}/favorite")
    def set_favorite(request: Request, effect_id: int, body: FavoriteRequest):
        with _db() as db:
            _get_row(db, effect_id, _sid(request))
            db.set_favorite(effect_id, body.favorite)
            return _effect_json(db.get_effect(effect_id))

    @app.post("/api/effects/{effect_id}/promote")
    def promote_effect(request: Request, effect_id: int):
        with _db() as db:
            row = _get_row(db, effect_id, _sid(request))
            if row["status"] != "quarantine":
                return _effect_json(row)
            rel = Path(row["path"])
            dest_rel = Path("effects").joinpath(*rel.parts[1:])
            dest = settings.library_dir / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(settings.library_dir / row["path"], dest)
            db.set_status(effect_id, "library", path=str(dest_rel))
            return _effect_json(db.get_effect(effect_id))

    @app.delete("/api/effects/{effect_id}")
    def delete_effect(request: Request, effect_id: int):
        with _db() as db:
            row = _get_row(db, effect_id, _sid(request))
            (settings.library_dir / row["path"]).unlink(missing_ok=True)
            db.delete_effect(effect_id)
        return {"deleted": effect_id}

    # -- find cleaner version -------------------------------------------------

    @app.get("/api/effects/{effect_id}/similar")
    def similar(request: Request, effect_id: int):
        from wavelength.pipeline.similar import SimilarError, find_similar

        with _db() as db:
            row = _get_row(db, effect_id, _sid(request))
            try:
                candidates = find_similar(settings, db, row)
            except SimilarError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        return [
            {
                "freesound_id": c.sound.id,
                "name": c.sound.name,
                "username": c.sound.username,
                "license": c.sound.license,
                "duration_s": c.sound.duration_s,
                "similarity": round(c.similarity, 3),
                "page_url": c.sound.page_url,
            }
            for c in candidates
        ]

    @app.get("/api/freesound-preview/{freesound_id}")
    def freesound_preview(freesound_id: int):
        path = settings.cache_dir / "freesound-previews" / f"{freesound_id}.mp3"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Preview not cached")
        return FileResponse(path, media_type="audio/mpeg")

    @app.post("/api/effects/{effect_id}/import-similar")
    def import_similar(request: Request, effect_id: int, body: ImportRequest):
        from wavelength.pipeline.similar import SimilarError, import_sound
        from wavelength.sources.freesound import FreesoundSound

        preview = (
            settings.cache_dir / "freesound-previews" / f"{body.freesound_id}.mp3"
        )
        if not preview.is_file():
            raise HTTPException(status_code=404, detail="Preview not cached")
        sound = FreesoundSound(
            id=body.freesound_id, name=body.name, username=body.username,
            license=body.license, duration_s=body.duration_s,
            preview_url="", page_url=body.page_url,
        )
        with _db() as db:
            row = _get_row(db, effect_id, _sid(request))
            try:
                path = import_sound(
                    settings, db, row, sound, preview,
                    session_id=_sid(request),
                )
            except SimilarError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        return {"imported": path.name}

    # -- extraction jobs ----------------------------------------------------

    @app.post("/api/jobs")
    def submit_job(request: Request, body: ExtractRequest):
        target = body.target.strip()
        if not target:
            raise HTTPException(status_code=400, detail="Empty target")
        if public:
            if not is_url(target):
                raise HTTPException(
                    status_code=400,
                    detail="Paste a video link (file paths only work in the "
                    "local app).",
                )
            ip = request.client.host if request.client else "unknown"
            error = quotas.check_and_count(_sid(request) or "anon", ip)
            if error:
                raise HTTPException(status_code=429, detail=error)
        job = jobs.submit(target, force=body.force, session_id=_sid(request))
        return job.to_dict()

    @app.get("/api/jobs")
    def list_jobs(request: Request):
        sid = _sid(request)
        return [
            j.to_dict() for j in jobs.list()
            if not public or getattr(j, "session_id", None) == sid
        ]

    @app.get("/api/jobs/{job_id}")
    def get_job(request: Request, job_id: int):
        job = jobs.get(job_id)
        if job is None or (
            public and getattr(job, "session_id", None) != _sid(request)
        ):
            raise HTTPException(status_code=404, detail="No such job")
        return job.to_dict()

    # -- watch folder (local mode only) ----------------------------------------

    @app.post("/api/watch")
    def set_watch(body: WatchRequest):
        if public:
            raise HTTPException(
                status_code=403, detail="Watch folders are a local-app feature."
            )
        watcher: VideoWatcher | None = watcher_state["watcher"]
        if not body.enabled:
            if watcher is not None:
                watcher.stop()
                watcher_state["watcher"] = None
            return {"enabled": False, "folder": watcher_state["folder"]}

        folder = Path(body.folder or "~/Downloads").expanduser()
        if watcher is not None:
            watcher.stop()
        try:
            watcher = VideoWatcher(folder, lambda p: jobs.submit(str(p)))
            watcher.start()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        watcher_state["watcher"] = watcher
        watcher_state["folder"] = str(folder)
        return {"enabled": True, "folder": str(folder)}

    @app.get("/api/watch")
    def get_watch():
        if public:
            return {"enabled": False, "folder": None}
        watcher: VideoWatcher | None = watcher_state["watcher"]
        return {
            "enabled": watcher is not None and watcher.running,
            "folder": watcher_state["folder"],
        }

    # -- status ---------------------------------------------------------------

    @app.get("/api/status")
    def status(request: Request):
        sid = _sid(request)
        with _db() as db:
            library_count = len(
                db.list_effects(session_id=sid, scoped=public)
            )
            quarantine_count = len(
                db.list_effects(status="quarantine", session_id=sid, scoped=public)
            )
        try:
            engine_ready = get_separator(settings).is_ready()
        except SeparationError:
            engine_ready = False
        return {
            "version": __version__,
            "public": public,
            "library_dir": str(settings.library_dir) if not public else None,
            "engine": settings.engine,
            "engine_ready": engine_ready,
            "labeler_ready": ClapLabeler(settings).is_ready(),
            "library_count": library_count,
            "quarantine_count": quarantine_count,
        }

    @app.exception_handler(Exception)
    async def on_error(request, exc):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
