"""API tests for the library server (TestClient, engine=none)."""

import time

import pytest
from fastapi.testclient import TestClient

from wavelength.server.app import create_app

from conftest import make_video, settings, three_event_audio  # noqa: F401


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def _extract_and_wait(client, video, timeout=60.0):
    job = client.post("/api/jobs", json={"target": str(video)}).json()
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.2)
    raise TimeoutError("job did not finish")


def test_status_endpoint(client):
    s = client.get("/api/status").json()
    assert s["engine"] == "none"
    assert s["engine_ready"] is True
    assert s["library_count"] == 0


def test_extract_job_and_effect_crud(client, tmp_path, three_event_audio):
    video = make_video(tmp_path / "clip.mp4", three_event_audio)
    job = _extract_and_wait(client, video)
    assert job["status"] == "done"
    assert job["effect_count"] == 3

    effects = client.get("/api/effects").json()
    assert len(effects) == 3
    eid = effects[0]["id"]

    # audio serving
    audio = client.get(f"/api/effects/{eid}/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert len(audio.content) > 1000

    # rename + search
    client.post(f"/api/effects/{eid}/rename", json={"label": "laser-zap"})
    found = client.get("/api/effects", params={"q": "laser-zap"}).json()
    assert len(found) == 1 and found[0]["label"] == "laser-zap"

    # tags
    updated = client.post(
        f"/api/effects/{eid}/tags", json={"tags": ["sci-fi", "ui"]}
    ).json()
    assert updated["tags"] == ["sci-fi", "ui"]

    # favorite
    updated = client.post(
        f"/api/effects/{eid}/favorite", json={"favorite": True}
    ).json()
    assert updated["favorite"] is True

    # wav download has attachment filename from user label
    dl = client.get(f"/api/effects/{eid}/download")
    assert "laser-zap.wav" in dl.headers["content-disposition"]

    # delete
    client.delete(f"/api/effects/{eid}")
    assert len(client.get("/api/effects").json()) == 2
    assert client.get(f"/api/effects/{eid}/audio").status_code == 404


def test_duplicate_job_reports_already_processed(client, tmp_path, three_event_audio):
    video = make_video(tmp_path / "clip.mp4", three_event_audio)
    first = _extract_and_wait(client, video)
    assert first["status"] == "done"
    second = _extract_and_wait(client, video)
    assert second["already_processed"] is True


def test_bad_target_job_errors_cleanly(client, tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    job = _extract_and_wait(client, junk)
    assert job["status"] == "error"
    assert "media" in job["error"].lower() or "readable" in job["error"].lower()


def test_watch_endpoint_lifecycle(client, tmp_path):
    state = client.post(
        "/api/watch", json={"enabled": True, "folder": str(tmp_path)}
    ).json()
    assert state["enabled"] is True
    assert client.get("/api/watch").json()["enabled"] is True
    state = client.post("/api/watch", json={"enabled": False}).json()
    assert state["enabled"] is False


def test_watch_missing_folder_400(client):
    res = client.post(
        "/api/watch", json={"enabled": True, "folder": "/nope/never/exists"}
    )
    assert res.status_code == 400


def test_index_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "wavelength" in res.text.lower()
