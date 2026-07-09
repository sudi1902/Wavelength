"""'Find cleaner version': Freesound client (HTTP mocked), similarity
ranking (embedder faked), and import (real ffmpeg conversion)."""

import json
import subprocess

import numpy as np
import pytest

import wavelength.pipeline.similar as similar_mod
import wavelength.sources.freesound as fs_mod
from wavelength.library.db import LibraryDB
from wavelength.pipeline.similar import (
    SimilarError,
    find_similar,
    import_sound,
)
from wavelength.sources.freesound import (
    FreesoundError,
    FreesoundSound,
    _short_license,
    search,
)

from conftest import SR, settings  # noqa: F401


# ---------------------------------------------------------------- freesound

class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


def test_search_requires_api_key():
    with pytest.raises(FreesoundError, match="freesound.org/apiv2/apply"):
        search("", "whoosh")


def test_search_builds_filters_and_parses(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured.update(params)
        return _FakeResponse(payload={"results": [
            {"id": 42, "name": "Clean Whoosh", "username": "prosound",
             "license": "http://creativecommons.org/publicdomain/zero/1.0/",
             "duration": 1.2, "url": "https://freesound.org/s/42/",
             "previews": {"preview-hq-mp3": "https://cdn/42.mp3"}},
            {"id": 43, "name": "No Preview", "username": "x",
             "license": "Attribution", "duration": 1.0, "url": "u",
             "previews": {}},
        ]})

    monkeypatch.setattr(fs_mod.requests, "get", fake_get)
    sounds = search("KEY", "whoosh", min_duration_s=0.3, max_duration_s=3.0,
                    licenses=["cc0", "by"])
    assert len(sounds) == 1  # preview-less result dropped
    assert sounds[0].license == "CC0"
    assert sounds[0].attribution.startswith('"Clean Whoosh" by prosound')
    assert "duration:[0.30 TO 3.00]" in captured["filter"]
    assert '"Creative Commons 0"' in captured["filter"]
    assert '"Attribution"' in captured["filter"]
    assert captured["token"] == "KEY"


def test_search_bad_key(monkeypatch):
    monkeypatch.setattr(
        fs_mod.requests, "get", lambda *a, **k: _FakeResponse(status_code=401)
    )
    with pytest.raises(FreesoundError, match="401"):
        search("BAD", "pop")


def test_short_license_mapping():
    assert _short_license("Attribution") == "CC-BY"
    assert _short_license("Attribution Noncommercial") == "CC-BY-NC"
    assert _short_license("http://creativecommons.org/licenses/by/4.0/") == "CC-BY"
    assert _short_license("http://creativecommons.org/licenses/by-nc/3.0/") == "CC-BY-NC"
    assert _short_license("Creative Commons 0") == "CC0"


# ------------------------------------------------------------------ ranking

def _make_mp3(path, freq=880.0, duration=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"sine=frequency={freq}:duration={duration}",
         "-c:a", "libmp3lame", "-b:a", "128k", str(path)],
        check=True, capture_output=True,
    )


class _FakeEmbedder:
    """Deterministic embeddings: filename-keyed unit vectors."""

    def __init__(self, mapping):
        self.mapping = mapping

    def is_ready(self):
        return True

    def embed(self, paths):
        out = []
        for p in paths:
            vec = np.zeros(4, dtype=np.float32)
            vec[self.mapping[p.name]] = 1.0
            out.append(vec.tobytes())
        return out


def _effect_row(db, settings, embedding=None, label="whoosh"):
    import soundfile as sf

    wav = settings.effects_dir / "e.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav, np.zeros(4410, dtype=np.float32), SR)
    sid = db.add_source(sha256="e" * 64, original_path="/v.mp4",
                        filename="v.mp4", duration_s=3.0, engine="none")
    eid = db.add_effect(
        source_id=sid, path="effects/e.wav", content_sha256="x" * 64,
        duration_s=1.0, sample_rate=SR, peak_db=-6.0, start_in_source_s=0.0,
        auto_label=label, embedding=embedding,
    )
    return db.get_effect(eid)


def test_find_similar_ranks_by_cosine(settings, monkeypatch, tmp_path):
    settings.similar.freesound_api_key = "KEY"

    sounds = [
        FreesoundSound(1, "far", "a", "CC0", 1.0, "http://cdn/1.mp3", "u1"),
        FreesoundSound(2, "close", "b", "CC0", 1.0, "http://cdn/2.mp3", "u2"),
    ]
    monkeypatch.setattr(similar_mod.freesound, "search", lambda *a, **k: sounds)

    def fake_download(sound, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mp3")
        return dest

    monkeypatch.setattr(similar_mod.freesound, "download_preview", fake_download)
    # Reference embedding = axis 0; "2.mp3" also axis 0 (identical), "1.mp3"
    # axis 1 (orthogonal).
    embedder = _FakeEmbedder({"1.mp3": 1, "2.mp3": 0})
    monkeypatch.setattr(similar_mod, "_require_labeler", lambda s: embedder)

    reference = np.zeros(4, dtype=np.float32)
    reference[0] = 1.0
    with LibraryDB(settings.db_path) as db:
        row = _effect_row(db, settings, embedding=reference.tobytes())
        candidates = find_similar(settings, db, row)

    assert [c.sound.id for c in candidates] == [2, 1]
    assert candidates[0].similarity == pytest.approx(1.0)
    assert candidates[1].similarity == pytest.approx(0.0)


def test_find_similar_requires_label(settings, monkeypatch):
    settings.similar.freesound_api_key = "KEY"
    with LibraryDB(settings.db_path) as db:
        row = _effect_row(db, settings, label=None)
        with pytest.raises(SimilarError, match="no label"):
            find_similar(settings, db, row)


def test_embedding_computed_and_persisted_when_missing(settings, monkeypatch):
    embedder = _FakeEmbedder({"e.wav": 2})
    monkeypatch.setattr(similar_mod, "_require_labeler", lambda s: embedder)
    with LibraryDB(settings.db_path) as db:
        row = _effect_row(db, settings, embedding=None)
        vec = similar_mod.effect_embedding(settings, db, row)
        assert vec[2] == 1.0
        # persisted
        assert db.get_effect(row["id"])["embedding"] is not None


# ------------------------------------------------------------------- import

def test_import_sound_stores_wav_with_attribution(settings, tmp_path):
    preview = tmp_path / "77.mp3"
    _make_mp3(preview)
    sound = FreesoundSound(
        77, "Studio Whoosh", "prosound", "CC-BY", 0.5, "",
        "https://freesound.org/s/77/",
    )
    with LibraryDB(settings.db_path) as db:
        row = _effect_row(db, settings)
        path = import_sound(settings, db, row, sound, preview)

        assert path.is_file()
        assert "whoosh-clean" in path.name
        imported = [e for e in db.list_effects() if e["id"] != row["id"]]
        assert len(imported) == 1
        rec = imported[0]
        assert rec["license"] == "CC-BY"
        assert "prosound" in rec["attribution"]
        assert rec["derived_from"] == row["id"]
        assert rec["source_title"] == "Studio Whoosh"

        # importing the same sound again is refused (content dedup)
        with pytest.raises(SimilarError, match="already"):
            import_sound(settings, db, row, sound, preview)
