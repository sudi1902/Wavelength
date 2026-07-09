"""Public (internet) mode: anonymous session isolation, filesystem
lockdown, and quotas. These are the security properties that make it safe
to expose the app — treat failures here as ship-blockers."""

import time

import pytest
from fastapi.testclient import TestClient

import wavelength.server.app as app_mod
from wavelength.server.app import create_app

from conftest import make_video, settings, three_event_audio  # noqa: F401


@pytest.fixture
def public_app(settings):
    return create_app(settings, public=True)


def _client(app):
    """A TestClient that keeps its own cookie jar = one anonymous visitor."""
    return TestClient(app)


def _extract_and_wait(client, target, timeout=60.0):
    job = client.post("/api/jobs", json={"target": target}).json()
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.2)
    raise TimeoutError


def test_session_cookie_issued(public_app):
    with _client(public_app) as c:
        c.get("/api/status")
        assert "wl_session" in c.cookies
        assert len(c.cookies["wl_session"]) == 32


def test_file_paths_rejected_in_public_mode(public_app, tmp_path):
    with _client(public_app) as c:
        res = c.post("/api/jobs", json={"target": "/Users/sudip/Movies/x.mov"})
        assert res.status_code == 400
        assert "link" in res.json()["detail"].lower()
        res = c.post("/api/jobs", json={"target": str(tmp_path / "v.mp4")})
        assert res.status_code == 400


def test_watch_api_disabled_in_public_mode(public_app, tmp_path):
    with _client(public_app) as c:
        res = c.post("/api/watch", json={"enabled": True, "folder": str(tmp_path)})
        assert res.status_code == 403
        assert c.get("/api/watch").json() == {"enabled": False, "folder": None}


def test_sessions_are_isolated(
    public_app, settings, tmp_path, three_event_audio, monkeypatch
):
    """Visitor A extracts a video; visitor B must see nothing of it."""
    # Public mode only takes URLs; fake the download to a local synthetic
    # video so the full pipeline runs without network.
    import wavelength.pipeline.runner as runner_mod
    from wavelength.pipeline.download import DownloadedVideo

    src = make_video(tmp_path / "clip.mp4", three_event_audio)

    def fake_download(url, dest_dir, **kwargs):
        import shutil
        dest = dest_dir / "vid.mp4"
        shutil.copy2(src, dest)
        return DownloadedVideo(dest, url, "cool video", "creator", 4.0)

    monkeypatch.setattr(runner_mod, "download_video", fake_download)

    with _client(public_app) as visitor_a, _client(public_app) as visitor_b:
        job = _extract_and_wait(visitor_a, "https://www.tiktok.com/@x/video/1")
        assert job["status"] == "done"
        assert job["effect_count"] == 3

        a_effects = visitor_a.get("/api/effects").json()
        assert len(a_effects) == 3
        assert visitor_a.get("/api/status").json()["library_count"] == 3

        # Visitor B sees an empty world.
        assert visitor_b.get("/api/effects").json() == []
        assert visitor_b.get("/api/status").json()["library_count"] == 0
        assert visitor_b.get("/api/jobs").json() == []

        # B cannot touch A's effects — reads, writes, or deletes.
        eid = a_effects[0]["id"]
        assert visitor_b.get(f"/api/effects/{eid}/audio").status_code == 404
        assert visitor_b.post(
            f"/api/effects/{eid}/rename", json={"label": "mine now"}
        ).status_code == 404
        assert visitor_b.delete(f"/api/effects/{eid}").status_code == 404
        assert visitor_b.get(f"/api/jobs/{job['id']}").status_code == 404

        # A still owns everything.
        assert visitor_a.get(f"/api/effects/{eid}/audio").status_code == 200

        # B extracting the SAME url is not blocked by A's dedup.
        job_b = _extract_and_wait(visitor_b, "https://www.tiktok.com/@x/video/1")
        assert job_b["status"] == "done"
        assert job_b["effect_count"] == 3
        assert visitor_b.get("/api/status").json()["library_count"] == 3


def test_session_daily_quota(public_app, monkeypatch):
    monkeypatch.setattr(app_mod, "SESSION_DAILY_LIMIT", 2)
    with _client(public_app) as c:
        for _ in range(2):
            res = c.post("/api/jobs", json={"target": "https://example.com/v"})
            assert res.status_code == 200
        res = c.post("/api/jobs", json={"target": "https://example.com/v"})
        assert res.status_code == 429
        assert "limit" in res.json()["detail"].lower()


def test_local_mode_unchanged(settings, tmp_path, three_event_audio):
    """Local mode keeps file paths, watch API, and an unscoped library."""
    app = create_app(settings, public=False)
    with TestClient(app) as c:
        video = make_video(tmp_path / "clip.mp4", three_event_audio)
        job = _extract_and_wait(c, str(video))
        assert job["status"] == "done" and job["effect_count"] == 3
        assert c.get("/api/status").json()["public"] is False
        assert len(c.get("/api/effects").json()) == 3
