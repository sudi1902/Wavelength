"""Labeling stage: name each effect with CLAP zero-shot classification, and
flag separation bleed (speech/music) for quarantine.

CLAP needs PyTorch, so — like the BandIt engine — it lives in its own venv
under the wavelength cache, set up once via `wavelength setup clap`. A small
worker script runs inside that venv and returns JSON scores; the decision
logic (label vs quarantine) stays here where thresholds are configurable.

If the labeler isn't set up, the pipeline degrades gracefully: effects are
stored unlabeled as before.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from wavelength.config import Settings
from wavelength.pipeline.vocabulary import build_prompts


class LabelingError(Exception):
    """Labeling failed; message is user-facing."""


@dataclass
class LabelResult:
    label: str  # best SFX label, or "unknown" below the confidence floor
    confidence: float  # probability of the best SFX label
    quarantine: bool
    quarantine_reason: str | None  # e.g. "speech 0.62"


# The worker runs inside the clap venv. It loads the model once, embeds all
# prompts and all wav files, and prints per-file group-best scores as JSON.
# Softmax over similarity*100 follows the standard CLIP/CLAP zero-shot recipe.
CLAP_WORKER = '''\
import json
import sys
import warnings

warnings.filterwarnings("ignore")


def main():
    spec = json.load(open(sys.argv[1]))
    wavs = sys.argv[2:]

    import numpy as np
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()

    if wavs == ["--selftest"]:
        print(json.dumps({"ok": True}))
        return

    prompts = spec["prompts"]
    text_emb = model.get_text_embedding([p["text"] for p in prompts])
    audio_emb = model.get_audio_embedding_from_filelist(x=wavs, use_tensor=False)

    text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    audio_emb = audio_emb / np.linalg.norm(audio_emb, axis=1, keepdims=True)
    sims = audio_emb @ text_emb.T

    results = []
    for row in sims:
        logits = row * 100.0
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        best = {}
        for p, prob in zip(prompts, probs):
            group = p["group"]
            if group not in best or prob > best[group]["prob"]:
                best[group] = {"label": p["label"], "prob": float(prob)}
        results.append(best)
    print(json.dumps(results))


main()
'''


class ClapLabeler:
    name = "clap"

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def venv_dir(self) -> Path:
        return self.settings.cache_dir / "clap-venv"

    @property
    def venv_python(self) -> Path:
        return self.venv_dir / "bin" / "python"

    def is_ready(self) -> bool:
        return self.venv_python.is_file() and (self.venv_dir / ".model-ok").is_file()

    def ensure_ready(self) -> None:
        if self.is_ready():
            return
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.venv_python.is_file():
            print("[clap] Creating isolated venv ...")
            result = subprocess.run(
                [sys.executable, "-m", "venv", str(self.venv_dir)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise LabelingError(f"venv creation failed:\n{result.stderr.strip()}")
        if not (self.venv_dir / ".deps-ok").is_file():
            print("[clap] Installing CLAP and PyTorch (one-time, several "
                  "minutes) ...")
            # torch/torchaudio/torchvision are undeclared laion_clap deps —
            # it imports them but doesn't require them at install time.
            result = subprocess.run(
                [str(self.venv_python), "-m", "pip", "install", "--quiet",
                 "laion_clap", "torch", "torchaudio", "torchvision"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise LabelingError(
                    "Installing CLAP dependencies failed:\n"
                    + result.stderr.strip()[-2000:]
                )
            (self.venv_dir / ".deps-ok").touch()
        print("[clap] Downloading model checkpoint (~2 GB, one-time) ...")
        out = self._run_worker(["--selftest"])
        if not out.get("ok"):
            raise LabelingError("CLAP self-test returned an unexpected result")
        (self.venv_dir / ".model-ok").touch()
        print("[clap] Ready.")

    def label(self, wav_paths: list[Path]) -> list[LabelResult]:
        if not wav_paths:
            return []
        cfg = self.settings.labeling
        raw = self._run_worker([str(p) for p in wav_paths])
        results = []
        for best in raw:
            sfx = best.get("sfx", {"label": "unknown", "prob": 0.0})
            other = best.get("other", {"label": "", "prob": 0.0})
            quarantine = (
                cfg.quarantine_enabled
                and other["prob"] > sfx["prob"]
                and other["prob"] >= cfg.quarantine_min_confidence
            )
            label = sfx["label"] if sfx["prob"] >= cfg.min_confidence else "unknown"
            results.append(
                LabelResult(
                    label=label,
                    confidence=round(sfx["prob"], 3),
                    quarantine=quarantine,
                    quarantine_reason=(
                        f"{other['label']} {other['prob']:.2f}" if quarantine else None
                    ),
                )
            )
        return results

    def _run_worker(self, args: list[str]):
        worker_path = self.settings.cache_dir / "clap_worker.py"
        worker_path.parent.mkdir(parents=True, exist_ok=True)
        worker_path.write_text(CLAP_WORKER)
        prompts_path = self.settings.cache_dir / "clap_prompts.json"
        prompts_path.write_text(json.dumps({"prompts": build_prompts()}))

        result = subprocess.run(
            [str(self.venv_python), str(worker_path), str(prompts_path), *args],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            raise LabelingError(
                "CLAP worker failed:\n" + result.stderr.strip()[-2000:]
            )
        try:
            return json.loads(result.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise LabelingError(
                f"CLAP worker produced unparseable output: {result.stdout[-500:]}"
            ) from exc
