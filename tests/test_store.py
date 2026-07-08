import numpy as np
import soundfile as sf

from wavelength.library.db import LibraryDB
from wavelength.pipeline.clean import CleanedEffect
from wavelength.pipeline.store import store_effect

from conftest import SR, settings, tone_burst  # noqa: F401


def _effect(freq: float = 880.0) -> CleanedEffect:
    audio = tone_burst(freq, 0.3)[np.newaxis, :]
    return CleanedEffect(
        audio=audio, sample_rate=SR,
        start_in_source_s=1.0, duration_s=0.3, peak_db=-6.0,
    )


def test_store_writes_wav_and_db_row(settings):
    with LibraryDB(settings.db_path) as db:
        source_id = db.add_source(
            sha256="a" * 64, original_path="/x/v.mp4", filename="v.mp4",
            duration_s=4.0, engine="none",
        )
        stored = store_effect(
            _effect(), settings=settings, db=db,
            source_id=source_id, source_hash="a" * 64, index=1,
        )
        assert stored is not None
        assert stored.path.is_file()
        assert "effect-01" in stored.path.name

        audio, sr = sf.read(stored.path)
        assert sr == SR
        assert sf.info(stored.path).subtype == "PCM_16"

        rows = db.list_effects()
        assert len(rows) == 1
        assert rows[0]["sample_rate"] == SR
        # Path stored relative to the library dir.
        assert (settings.library_dir / rows[0]["path"]).is_file()


def test_duplicate_audio_is_skipped(settings):
    with LibraryDB(settings.db_path) as db:
        source_id = db.add_source(
            sha256="b" * 64, original_path="/x/v.mp4", filename="v.mp4",
            duration_s=4.0, engine="none",
        )
        first = store_effect(
            _effect(), settings=settings, db=db,
            source_id=source_id, source_hash="b" * 64, index=1,
        )
        second = store_effect(
            _effect(), settings=settings, db=db,
            source_id=source_id, source_hash="b" * 64, index=2,
        )
        assert first is not None
        assert second is None
        assert len(db.list_effects()) == 1


def test_distinct_audio_both_stored(settings):
    with LibraryDB(settings.db_path) as db:
        source_id = db.add_source(
            sha256="c" * 64, original_path="/x/v.mp4", filename="v.mp4",
            duration_s=4.0, engine="none",
        )
        a = store_effect(
            _effect(880), settings=settings, db=db,
            source_id=source_id, source_hash="c" * 64, index=1,
        )
        b = store_effect(
            _effect(440), settings=settings, db=db,
            source_id=source_id, source_hash="c" * 64, index=2,
        )
        assert a is not None and b is not None
        assert a.path != b.path
