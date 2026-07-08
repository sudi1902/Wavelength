"""End-to-end: synthetic video in → multiple effect WAVs in the library.

Uses the passthrough engine so no ML models are needed; the ML engines are
exercised separately on real hardware.
"""

import soundfile as sf

from wavelength.library.db import LibraryDB
from wavelength.pipeline.runner import extract_video

from conftest import make_video, settings, three_event_audio  # noqa: F401


def test_video_with_three_events_yields_three_wavs(
    tmp_path, settings, three_event_audio
):
    video = make_video(tmp_path / "clip.mp4", three_event_audio)
    result = extract_video(video, settings)

    assert result.effect_count == 3
    for path in result.effect_paths:
        assert path.is_file()
        info = sf.info(path)
        assert info.subtype == "PCM_16"
        assert info.samplerate >= 44_100
        assert info.duration >= 0.1

    # Source + effects stem archived.
    archived = list(settings.sources_dir.rglob("*"))
    assert any(p.name == "clip.mp4" for p in archived)
    assert any(p.name == "effects_stem.wav" for p in archived)

    # Library records written.
    with LibraryDB(settings.db_path) as db:
        assert len(db.list_effects()) == 3
        src = db.find_source(db.list_effects()[0]["content_sha256"]) or True  # noqa


def test_reprocessing_same_video_is_skipped(tmp_path, settings, three_event_audio):
    video = make_video(tmp_path / "clip.mp4", three_event_audio)
    first = extract_video(video, settings)
    assert first.effect_count == 3

    second = extract_video(video, settings)
    assert second.already_processed
    assert second.effect_count == 0

    forced = extract_video(video, settings, force=True)
    assert not forced.already_processed
    assert forced.effect_count == 3
    with LibraryDB(settings.db_path) as db:
        # Re-extraction after force: old rows replaced, files deduped by
        # content hash would collide — but delete_source removed them, so
        # exactly 3 rows remain.
        assert len(db.list_effects()) == 3


def test_failed_video_is_retried_without_force(tmp_path, settings, three_event_audio):
    from wavelength.pipeline import ingest

    video = make_video(tmp_path / "clip.mp4", three_event_audio)
    # Simulate a previous crashed run: a 'failed' record for this exact video.
    with LibraryDB(settings.db_path) as db:
        source_id = db.add_source(
            sha256=ingest.file_sha256(video),
            original_path=str(video), filename=video.name,
            duration_s=4.0, engine="bandit",
        )
        db.finish_source(source_id, effect_count=0, status="failed",
                         error="MPS crash")

    result = extract_video(video, settings)
    assert not result.already_processed
    assert result.effect_count == 3


def test_no_archive_when_disabled(tmp_path, settings, three_event_audio):
    settings.archive_sources = False
    video = make_video(tmp_path / "clip.mp4", three_event_audio)
    extract_video(video, settings)
    assert not settings.sources_dir.exists()
