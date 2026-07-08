"""Labeling decision logic with the CLAP worker faked — model quality is
validated on real hardware; here we test the plumbing and thresholds."""

import json
import subprocess

import pytest

import wavelength.pipeline.label as label_mod
from wavelength.pipeline.label import CLAP_WORKER, ClapLabeler, LabelingError
from wavelength.pipeline.vocabulary import SFX_LABELS, build_prompts

from conftest import settings  # noqa: F401


def _make_ready(labeler: ClapLabeler):
    labeler.venv_python.parent.mkdir(parents=True, exist_ok=True)
    labeler.venv_python.touch()
    (labeler.venv_dir / ".model-ok").touch()


def _fake_worker(outputs):
    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, returncode=0, stdout=json.dumps(outputs) + "\n", stderr=""
        )

    return run


def test_worker_script_is_valid_python():
    compile(CLAP_WORKER, "clap_worker.py", "exec")


def test_vocabulary_prompts():
    prompts = build_prompts()
    groups = {p["group"] for p in prompts}
    assert groups == {"sfx", "other"}
    sfx_labels = [p["label"] for p in prompts if p["group"] == "sfx"]
    assert len(sfx_labels) == len(set(sfx_labels))  # no duplicates
    assert "whoosh" in sfx_labels and "pop" in sfx_labels
    other_labels = {p["label"] for p in prompts if p["group"] == "other"}
    assert {"speech", "music", "laughter"} <= other_labels


def test_confident_sfx_label(settings, monkeypatch, tmp_path):
    labeler = ClapLabeler(settings)
    _make_ready(labeler)
    monkeypatch.setattr(label_mod.subprocess, "run", _fake_worker([
        {"sfx": {"label": "whoosh", "prob": 0.72},
         "other": {"label": "music", "prob": 0.10}},
    ]))
    wav = tmp_path / "a.wav"
    wav.touch()
    [result] = labeler.label([wav])
    assert result.label == "whoosh"
    assert result.confidence == 0.72
    assert not result.quarantine


def test_speech_bleed_goes_to_quarantine(settings, monkeypatch, tmp_path):
    labeler = ClapLabeler(settings)
    _make_ready(labeler)
    monkeypatch.setattr(label_mod.subprocess, "run", _fake_worker([
        {"sfx": {"label": "pop", "prob": 0.20},
         "other": {"label": "speech", "prob": 0.61}},
    ]))
    wav = tmp_path / "a.wav"
    wav.touch()
    [result] = labeler.label([wav])
    assert result.quarantine
    assert result.quarantine_reason == "speech 0.61"


def test_weak_other_score_does_not_quarantine(settings, monkeypatch, tmp_path):
    # other > sfx but below the quarantine confidence floor -> keep.
    labeler = ClapLabeler(settings)
    _make_ready(labeler)
    monkeypatch.setattr(label_mod.subprocess, "run", _fake_worker([
        {"sfx": {"label": "pop", "prob": 0.10},
         "other": {"label": "noise", "prob": 0.15}},
    ]))
    wav = tmp_path / "a.wav"
    wav.touch()
    [result] = labeler.label([wav])
    assert not result.quarantine
    assert result.label == "unknown"  # 0.10 < min_confidence 0.15


def test_worker_failure_raises_labeling_error(settings, monkeypatch, tmp_path):
    labeler = ClapLabeler(settings)
    _make_ready(labeler)

    def boom(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="cuda oom")

    monkeypatch.setattr(label_mod.subprocess, "run", boom)
    wav = tmp_path / "a.wav"
    wav.touch()
    with pytest.raises(LabelingError, match="cuda oom"):
        labeler.label([wav])


def test_all_sfx_labels_slug_safely():
    from wavelength.pipeline.store import _slug

    for label in SFX_LABELS:
        slug = _slug(label)
        assert slug and all(c.isalnum() or c == "-" for c in slug)
