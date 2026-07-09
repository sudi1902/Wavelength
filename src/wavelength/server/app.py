"""FastAPI application: REST API + static frontend for the library UI.

Run with `wavelength serve` — binds to localhost only; this is a personal
tool, not a network service. SQLite connections are opened per request
(sqlite3 objects are not thread-safe across FastAPI's threadpool).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wavelength import __version__
from wavelength.config import Settings
from wavelength.library.db import LibraryDB
from wavelength.pipeline.label import ClapLabeler
from wavelength.pipeline.separate import SeparationError, get_separator
from wavelength.server.jobs import JobQueue
from wavelength.watch import VideoWatcher

STATIC_DIR = Path(__file__).parent / "static"


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


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="Wavelength", version=__version__)
    jobs = JobQueue(settings)
    watcher_state: dict = {"watcher": None, "folder": None}

    def _db() -> LibraryDB:
        return LibraryDB(settings.db_path)

    def _get_row(db: LibraryDB, effect_id: int):
        row = db.get_effect(effect_id)
        if row is None:
            raise HTTPException(status_code=404, detail="No such effect")
        return row

    # -- library -----------------------------------------------------------

    @app.get("/api/effects")
    def list_effects(q: str = "", status: str = "library"):
        with _db() as db:
            rows = db.search_effects(q) if q else db.list_effects(status=status)
        return [_effect_json(r) for r in rows]

    @app.get("/api/effects/{effect_id}/audio")
    def effect_audio(effect_id: int):
        with _db() as db:
            row = _get_row(db, effect_id)
        path = settings.library_dir / row["path"]
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Audio file missing")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/api/effects/{effect_id}/download")
    def effect_download(effect_id: int, mp3: bool = False):
        with _db() as db:
            row = _get_row(db, effect_id)
        src = settings.library_dir / row["path"]
        if not src.is_file():
            raise HTTPException(status_code=404, detail="Audio file missing")
        name = row["user_label"] or Path(row["path"]).stem
        if not mp3:
            return FileResponse(
                src, media_type="audio/wav", filename=f"{name}.wav"
            )
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
    def rename_effect(effect_id: int, body: RenameRequest):
        with _db() as db:
            _get_row(db, effect_id)
            db.set_user_label(effect_id, body.label.strip())
            return _effect_json(db.get_effect(effect_id))

    @app.post("/api/effects/{effect_id}/tags")
    def set_tags(effect_id: int, body: TagsRequest):
        with _db() as db:
            _get_row(db, effect_id)
            db.set_tags(effect_id, ",".join(t.strip() for t in body.tags if t.strip()))
            return _effect_json(db.get_effect(effect_id))

    @app.post("/api/effects/{effect_id}/favorite")
    def set_favorite(effect_id: int, body: FavoriteRequest):
        with _db() as db:
            _get_row(db, effect_id)
            db.set_favorite(effect_id, body.favorite)
            return _effect_json(db.get_effect(effect_id))

    @app.post("/api/effects/{effect_id}/promote")
    def promote_effect(effect_id: int):
        with _db() as db:
            row = _get_row(db, effect_id)
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
    def delete_effect(effect_id: int):
        with _db() as db:
            row = _get_row(db, effect_id)
            (settings.library_dir / row["path"]).unlink(missing_ok=True)
            db.delete_effect(effect_id)
        return {"deleted": effect_id}

    # -- extraction jobs ----------------------------------------------------

    @app.post("/api/jobs")
    def submit_job(body: ExtractRequest):
        target = body.target.strip()
        if not target:
            raise HTTPException(status_code=400, detail="Empty target")
        job = jobs.submit(target, force=body.force)
        return job.to_dict()

    @app.get("/api/jobs")
    def list_jobs():
        return [j.to_dict() for j in jobs.list()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: int):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        return job.to_dict()

    # -- watch folder ---------------------------------------------------------

    @app.post("/api/watch")
    def set_watch(body: WatchRequest):
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
            watcher = VideoWatcher(
                folder, lambda p: jobs.submit(str(p))
            )
            watcher.start()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        watcher_state["watcher"] = watcher
        watcher_state["folder"] = str(folder)
        return {"enabled": True, "folder": str(folder)}

    @app.get("/api/watch")
    def get_watch():
        watcher: VideoWatcher | None = watcher_state["watcher"]
        return {
            "enabled": watcher is not None and watcher.running,
            "folder": watcher_state["folder"],
        }

    # -- status ---------------------------------------------------------------

    @app.get("/api/status")
    def status():
        with _db() as db:
            library_count = len(db.list_effects())
            quarantine_count = len(db.list_effects(status="quarantine"))
        try:
            engine_ready = get_separator(settings).is_ready()
        except SeparationError:
            engine_ready = False
        return {
            "version": __version__,
            "library_dir": str(settings.library_dir),
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
