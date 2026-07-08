import numpy as np
import pytest
import soundfile as sf

from wavelength.pipeline.separate import SeparationError, get_separator
from wavelength.pipeline.separate.bandit import BanditSeparator

from conftest import SR, settings  # noqa: F401


def test_get_separator_dispatch(settings):
    assert get_separator(settings, "none").name == "none"
    assert get_separator(settings, "bandit").name == "bandit"
    with pytest.raises(SeparationError, match="Unknown engine"):
        get_separator(settings, "spleeter")


def test_passthrough_returns_input_as_effects(settings):
    audio = np.zeros((2, SR), dtype=np.float32)
    result = get_separator(settings, "none").separate(audio, SR)
    assert result.effects is audio
    assert result.sample_rate == SR
    assert result.speech is None


def test_bandit_not_ready_without_setup(settings):
    separator = get_separator(settings, "bandit")
    assert not separator.is_ready()
    with pytest.raises(SeparationError, match="wavelength setup bandit"):
        separator.separate(np.zeros((2, SR), dtype=np.float32), SR)


def test_bandit_collect_stems_default_layout(tmp_path):
    # MSST default filename template: {file_name}/{instr} under store_dir.
    out = tmp_path / "mixture"
    out.mkdir()
    for name in ("speech", "music", "effects"):
        sf.write(out / f"{name}.wav", np.zeros((100, 2), np.float32), 44_100)
    stems = BanditSeparator._collect_stems(tmp_path)
    assert set(stems) == {"speech", "music", "effects"}
    assert stems["effects"].shape == (2, 100)


def test_bandit_collect_stems_flat_flac_sfx_alias(tmp_path):
    # Alternate layout: flat files, flac codec, 'sfx' stem name.
    for name in ("mixture_speech", "mixture_music", "mixture_sfx"):
        sf.write(tmp_path / f"{name}.flac", np.zeros((100, 2), np.float32), 44_100)
    stems = BanditSeparator._collect_stems(tmp_path)
    assert set(stems) == {"speech", "music", "effects"}
