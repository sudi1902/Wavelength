"""Phase 2 library management: labeled storage, quarantine routing,
search/rename/tag/delete, and the labeled end-to-end pipeline."""

import numpy as np
import soundfile as sf

import wavelength.pipeline.runner as runner_mod
from wavelength.library.db import LibraryDB
from wavelength.pipeline.clean import CleanedEffect
from wavelength.pipeline.label import LabelResult
from wavelength.pipeline.runner import extract_video
from wavelength.pipeline.store import store_effect

from conftest import SR, make_video, settings, three_event_audio, tone_burst  # noqa: F401


def _effect(freq=880.0):
    return CleanedEffect(
        audio=tone_burst(freq, 0.3)[np.newaxis, :], sample_rate=SR,
        start_in_source_s=1.0, duration_s=0.3, peak_db=-6.0,
    )


def _source(db, key="a"):
    return db.add_source(
        sha256=key * 64, original_path="/x/v.mp4", filename="v.mp4",
        duration_s=4.0, engine="none",
    )


def test_labeled_effect_gets_label_in_filename_and_db(settings):
    with LibraryDB(settings.db_path) as db:
        sid = _source(db)
        stored = store_effect(
            _effect(), settings=settings, db=db, source_id=sid,
            source_hash="a" * 64, index=1,
            label=LabelResult("whoosh", 0.8, False, None),
        )
        assert "__whoosh-01" in stored.path.name
        assert not stored.quarantined
        row = db.list_effects()[0]
        assert row["auto_label"] == "whoosh"
        assert row["label_confidence"] == 0.8
        assert row["status"] == "library"


def test_quarantined_effect_routed_to_quarantine_dir(settings):
    with LibraryDB(settings.db_path) as db:
        sid = _source(db)
        stored = store_effect(
            _effect(), settings=settings, db=db, source_id=sid,
            source_hash="a" * 64, index=1,
            label=LabelResult("pop", 0.2, True, "speech 0.61"),
        )
        assert stored.quarantined
        assert stored.path.is_relative_to(settings.quarantine_dir)
        assert db.list_effects(status="library") == []
        row = db.list_effects(status="quarantine")[0]
        assert row["quarantine_reason"] == "speech 0.61"


def test_search_rename_tag_delete(settings):
    with LibraryDB(settings.db_path) as db:
        sid = _source(db)
        stored = store_effect(
            _effect(), settings=settings, db=db, source_id=sid,
            source_hash="a" * 64, index=1,
            label=LabelResult("whoosh", 0.8, False, None),
        )
        eid = db.list_effects()[0]["id"]

        assert len(db.search_effects("whoosh")) == 1
        assert db.search_effects("nonexistent") == []

        db.set_user_label(eid, "epic-swipe")
        assert len(db.search_effects("epic-swipe")) == 1

        db.set_tags(eid, "transition,fast")
        assert len(db.search_effects("transition")) == 1

        db.delete_effect(eid)
        assert db.list_effects() == []
        assert stored.path.exists()  # file removal is the CLI's job


def test_promote_status_change(settings):
    with LibraryDB(settings.db_path) as db:
        sid = _source(db)
        store_effect(
            _effect(), settings=settings, db=db, source_id=sid,
            source_hash="a" * 64, index=1,
            label=LabelResult("pop", 0.2, True, "music 0.55"),
        )
        eid = db.list_effects(status="quarantine")[0]["id"]
        db.set_status(eid, "library", path="effects/2026/07/x.wav")
        assert db.list_effects(status="quarantine") == []
        assert db.list_effects()[0]["path"] == "effects/2026/07/x.wav"


class _FakeLabeler:
    """Labels everything 'ding' except quarantining the second effect."""

    def __init__(self, settings):
        self.settings = settings

    def is_ready(self):
        return True

    def label(self, wav_paths):
        out = []
        for i, _ in enumerate(wav_paths):
            if i == 1:
                out.append(LabelResult("pop", 0.2, True, "speech 0.70"))
            else:
                out.append(LabelResult("ding", 0.66, False, None))
        return out


def test_e2e_with_labeling_and_quarantine(
    tmp_path, settings, three_event_audio, monkeypatch
):
    monkeypatch.setattr(runner_mod, "ClapLabeler", _FakeLabeler)
    video = make_video(tmp_path / "clip.mp4", three_event_audio)
    result = extract_video(video, settings)

    assert result.labeled
    assert result.effect_count == 2
    assert len(result.quarantined_paths) == 1
    assert all("ding" in p.name for p in result.effect_paths)
    with LibraryDB(settings.db_path) as db:
        assert len(db.list_effects(status="library")) == 2
        assert len(db.list_effects(status="quarantine")) == 1


def test_e2e_without_labeler_stores_unlabeled(tmp_path, settings, three_event_audio):
    # Default ClapLabeler is not set up in tests -> graceful degradation.
    video = make_video(tmp_path / "clip.mp4", three_event_audio)
    result = extract_video(video, settings)
    assert not result.labeled
    assert result.effect_count == 3
    assert all("__effect-" in p.name for p in result.effect_paths)


def test_debug_mode_reports_verdicts(tmp_path, settings, three_event_audio):
    video = make_video(tmp_path / "clip.mp4", three_event_audio)
    messages = []
    extract_video(video, settings, debug=True, progress=messages.append)
    debug_lines = [m for m in messages if m.startswith("debug·")]
    assert any("gate:" in m for m in debug_lines)
    assert sum("kept" in m for m in debug_lines) == 3
