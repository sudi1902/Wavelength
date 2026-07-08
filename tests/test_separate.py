import numpy as np
import pytest
import soundfile as sf

from wavelength.pipeline.separate import SeparationError, get_separator
from wavelength.pipeline.separate.bandit import (
    REQUIREMENTS_DENYLIST,
    BanditSeparator,
    _requirement_name,
)

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


class _FakeRun:
    """Records inference invocations and scripts their outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.commands = []

    def __call__(self, cmd, **kwargs):
        import subprocess

        self.commands.append(cmd)
        return subprocess.CompletedProcess(
            cmd, returncode=self.outcomes.pop(0), stdout="", stderr="boom"
        )


def test_bandit_auto_falls_back_to_cpu_when_gpu_fails(
    tmp_path, settings, monkeypatch
):
    import wavelength.pipeline.separate.bandit as bandit_mod

    settings.device = "auto"
    separator = BanditSeparator(settings)
    fake = _FakeRun([1, 0])  # gpu attempt fails, cpu attempt succeeds
    monkeypatch.setattr(bandit_mod.subprocess, "run", fake)

    separator._run_inference(tmp_path, tmp_path)

    assert len(fake.commands) == 2
    # First attempt goes through the MPS shim, no --force_cpu.
    assert any("mps_shim.py" in part for part in fake.commands[0])
    assert "--force_cpu" not in fake.commands[0]
    # Retry is CPU-only, no shim.
    assert "--force_cpu" in fake.commands[1]
    assert not any("mps_shim.py" in part for part in fake.commands[1])
    # Shim was written and is valid Python.
    shim = settings.cache_dir / "mps_shim.py"
    compile(shim.read_text(), str(shim), "exec")


def test_bandit_cpu_device_never_tries_gpu(tmp_path, settings, monkeypatch):
    import wavelength.pipeline.separate.bandit as bandit_mod

    settings.device = "cpu"
    separator = BanditSeparator(settings)
    fake = _FakeRun([0])
    monkeypatch.setattr(bandit_mod.subprocess, "run", fake)

    separator._run_inference(tmp_path, tmp_path)

    assert len(fake.commands) == 1
    assert "--force_cpu" in fake.commands[0]


def test_bandit_all_attempts_fail_raises(tmp_path, settings, monkeypatch):
    import wavelength.pipeline.separate.bandit as bandit_mod

    settings.device = "auto"
    separator = BanditSeparator(settings)
    fake = _FakeRun([1, 1])
    monkeypatch.setattr(bandit_mod.subprocess, "run", fake)

    with pytest.raises(SeparationError, match="BandIt inference failed"):
        separator._run_inference(tmp_path, tmp_path)


def test_requirement_name_parsing():
    assert _requirement_name("pyaudio") == "pyaudio"
    assert _requirement_name("wxpython==4.2.2") == "wxpython"
    assert _requirement_name("torch>=2.0.1") == "torch"
    assert _requirement_name("pedalboard~=0.8.1") == "pedalboard"
    assert _requirement_name("huggingface-hub>=0.23.0") == "huggingface-hub"


def test_filtered_requirements_drops_denylist(tmp_path, settings):
    # Simulate MSST's requirements.txt and confirm the denylist is removed
    # while real deps survive.
    separator = BanditSeparator(settings)
    separator.msst_dir.mkdir(parents=True)
    separator.venv_dir.mkdir(parents=True)
    (separator.msst_dir / "requirements.txt").write_text(
        "torch>=2.0.1\nasteroid==0.7.0\npyaudio\nwxpython==4.2.2\n"
        "keyboard\nspafe==0.3.2\nsageattention==1.0.6\n"
    )
    kept = (separator._filtered_requirements()).read_text().lower()
    for denied in REQUIREMENTS_DENYLIST:
        if denied in {"pyaudio", "wxpython", "keyboard", "sageattention"}:
            assert denied not in kept
    assert "torch" in kept
    assert "asteroid" in kept
    assert "spafe" in kept
