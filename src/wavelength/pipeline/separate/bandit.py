"""BandIt Plus engine — the primary separator.

Cinematic audio source separation (speech / music / effects) using the
BandIt Plus checkpoint (DnR test SDR 11.50) run through ZFTurbo's
Music-Source-Separation-Training (MSST) inference script.

MSST is research-grade code with a heavy dependency tree, so it is kept
fully isolated: cloned into the wavelength cache dir with its own venv,
and invoked via subprocess. First run: `wavelength setup bandit` (or any
extract, which will prompt) clones the repo, builds the venv, and downloads
the checkpoint (~500 MB total download, one time).

The model's native sample rate is 44.1 kHz; input is resampled to it and
stems come back at 44.1 kHz.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from wavelength.pipeline.separate.base import (
    SeparationError,
    SeparationResult,
    Separator,
)

MSST_REPO_URL = "https://github.com/ZFTurbo/Music-Source-Separation-Training.git"
MSST_RELEASE = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3"
CHECKPOINT_NAME = "model_bandit_plus_dnr_sdr_11.47.chpt"
CONFIG_NAME = "config_dnr_bandit_bsrnn_multi_mus64.yaml"
MODEL_SAMPLE_RATE = 44_100

# Stem-name aliases across DnR conventions.
STEM_ALIASES = {
    "speech": ("speech", "dialog", "dialogue", "vocals"),
    "music": ("music",),
    "effects": ("effects", "effect", "sfx"),
}

# BandIt registers some float64 buffers (filterbank tables); Apple's MPS
# backend only supports float32, so moving the model to the GPU crashes with
# "Cannot convert a MPS Tensor to float64". This shim wraps MSST's
# inference.py and casts the model to float32 whenever it is moved to MPS.
# Run as: python mps_shim.py <path/to/inference.py> <inference args...>
MPS_SHIM = '''\
import runpy
import sys

import torch.nn as nn

_orig_to = nn.Module.to


def _to_float32_on_mps(self, *args, **kwargs):
    device = args[0] if args else kwargs.get("device")
    if device is not None and "mps" in str(device):
        self.float()
    return _orig_to(self, *args, **kwargs)


nn.Module.to = _to_float32_on_mps

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''

# MSST's requirements.txt bundles packages that BandIt inference never
# imports and that break a clean install: pyaudio (needs the portaudio
# system library), a GUI toolkit, global keyboard hooks, and CUDA-only
# training kernels with no Apple Silicon wheels. We install everything
# except these. Validated: the model imports and builds with the remainder.
REQUIREMENTS_DENYLIST = {
    "pyaudio",
    "wxpython",
    "keyboard",
    "sageattention",
    "bitsandbytes",
    "moises-light",
    "wandb",
}


def _requirement_name(line: str) -> str:
    """Extract the distribution name from a requirements.txt line."""
    line = line.strip()
    for sep in ("==", ">=", "<=", "~=", ">", "<", "!=", "==", "@", "["):
        idx = line.find(sep)
        if idx != -1:
            line = line[:idx]
    return line.strip().lower()


class BanditSeparator(Separator):
    name = "bandit"

    @property
    def msst_dir(self) -> Path:
        return self.settings.cache_dir / "msst"

    @property
    def venv_dir(self) -> Path:
        return self.settings.cache_dir / "msst-venv"

    @property
    def venv_python(self) -> Path:
        return self.venv_dir / "bin" / "python"

    @property
    def checkpoint_path(self) -> Path:
        return self.settings.models_dir / "bandit" / CHECKPOINT_NAME

    @property
    def config_path(self) -> Path:
        return self.settings.models_dir / "bandit" / CONFIG_NAME

    def is_ready(self) -> bool:
        return (
            (self.msst_dir / "inference.py").is_file()
            and self.venv_python.is_file()
            and self.checkpoint_path.is_file()
            and self.config_path.is_file()
        )

    def ensure_ready(self) -> None:
        if self.is_ready():
            return
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        self._clone_msst()
        self._build_venv()
        self._download_model()

    # -- setup steps ---------------------------------------------------------

    def _clone_msst(self) -> None:
        if (self.msst_dir / "inference.py").is_file():
            return
        print("[bandit] Cloning Music-Source-Separation-Training ...")
        ref = os.environ.get("WAVELENGTH_MSST_REF")
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd += ["--branch", ref]
        cmd += [MSST_REPO_URL, str(self.msst_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SeparationError(
                f"Could not clone MSST repository:\n{result.stderr.strip()}"
            )

    def _filtered_requirements(self) -> Path:
        """Write MSST's requirements minus the denylisted packages to a file
        in the venv dir, and return its path."""
        src = (self.msst_dir / "requirements.txt").read_text().splitlines()
        kept = [
            line for line in src
            if line.strip()
            and _requirement_name(line) not in REQUIREMENTS_DENYLIST
        ]
        dest = self.venv_dir / "wavelength-requirements.txt"
        dest.write_text("\n".join(kept) + "\n")
        return dest

    def _build_venv(self) -> None:
        if self.venv_python.is_file() and (self.venv_dir / ".deps-ok").is_file():
            return
        print("[bandit] Creating isolated venv and installing dependencies "
              "(one-time, several minutes — includes PyTorch) ...")
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(self.venv_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise SeparationError(f"venv creation failed:\n{result.stderr.strip()}")
        # MSST's librosa imports pkg_resources at load time. That module is
        # provided by setuptools — but (a) Python 3.12+ venvs no longer
        # bundle setuptools, and (b) setuptools 81+ REMOVED pkg_resources.
        # Pin <81 so the module is actually present.
        result = subprocess.run(
            [
                str(self.venv_python), "-m", "pip", "install", "--quiet",
                "--upgrade", "pip", "wheel", "setuptools<81",
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise SeparationError(
                "Preparing the venv failed:\n" + result.stderr.strip()[-2000:]
            )
        result = subprocess.run(
            [
                str(self.venv_python), "-m", "pip", "install", "--quiet",
                "-r", str(self._filtered_requirements()),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise SeparationError(
                "Installing MSST dependencies failed:\n"
                + result.stderr.strip()[-2000:]
            )
        (self.venv_dir / ".deps-ok").touch()

    def _download_model(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        for name, dest in ((CONFIG_NAME, self.config_path),
                           (CHECKPOINT_NAME, self.checkpoint_path)):
            if dest.is_file():
                continue
            url = f"{MSST_RELEASE}/{name}"
            print(f"[bandit] Downloading {name} ...")
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                urllib.request.urlretrieve(url, tmp)
                tmp.rename(dest)
            except Exception as exc:  # noqa: BLE001 - report any download failure
                tmp.unlink(missing_ok=True)
                raise SeparationError(
                    f"Download failed for {url}: {exc}\n"
                    f"You can download it manually to {dest}"
                ) from exc

    # -- inference -----------------------------------------------------------

    def separate(self, audio: np.ndarray, sr: int) -> SeparationResult:
        if not self.is_ready():
            raise SeparationError(
                "BandIt engine is not set up. Run: wavelength setup bandit"
            )
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        if sr != MODEL_SAMPLE_RATE:
            import librosa

            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=MODEL_SAMPLE_RATE, axis=1
            )

        with tempfile.TemporaryDirectory(prefix="wavelength-bandit-") as tmp:
            in_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            sf.write(in_dir / "mixture.wav", audio.T, MODEL_SAMPLE_RATE,
                     subtype="FLOAT")

            self._run_inference(in_dir, out_dir)

            stems = self._collect_stems(out_dir)
            if "effects" not in stems:
                outputs = sorted(
                    str(p.relative_to(out_dir)) for p in out_dir.rglob("*.*")
                )
                raise SeparationError(
                    f"BandIt produced no effects stem (outputs: {outputs})"
                )

        return SeparationResult(
            speech=stems.get("speech"),
            music=stems.get("music"),
            effects=stems["effects"],
            sample_rate=MODEL_SAMPLE_RATE,
        )

    def _run_inference(self, in_dir: Path, out_dir: Path) -> None:
        """Run MSST inference, on GPU (MPS/CUDA) when available with an
        automatic CPU retry. device setting: auto = try GPU then fall back;
        cpu = CPU only; mps/cuda = GPU only, surface failures."""
        device = self.settings.device
        attempts = ["gpu", "cpu"] if device == "auto" else (
            ["cpu"] if device == "cpu" else ["gpu"]
        )

        inference_args = [
            str(self.msst_dir / "inference.py"),
            "--model_type", "bandit",
            "--config_path", str(self.config_path),
            "--start_check_point", str(self.checkpoint_path),
            "--input_folder", str(in_dir),
            "--store_dir", str(out_dir),
            # Force float WAV output; MSST otherwise switches to FLAC
            # when the estimate's peak is <= 1.0.
            "--pcm_type", "FLOAT",
        ]

        shim_path = self.settings.cache_dir / "mps_shim.py"
        shim_path.parent.mkdir(parents=True, exist_ok=True)
        shim_path.write_text(MPS_SHIM)

        last_error = ""
        for attempt in attempts:
            if attempt == "gpu":
                cmd = [str(self.venv_python), str(shim_path), *inference_args]
            else:
                cmd = [str(self.venv_python), *inference_args, "--force_cpu"]
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(self.msst_dir),
                timeout=1800,
            )
            if result.returncode == 0:
                return
            last_error = result.stderr.strip()[-2000:]
            if attempt == "gpu" and len(attempts) > 1:
                print("[bandit] GPU inference failed; retrying on CPU ...")

        raise SeparationError("BandIt inference failed:\n" + last_error)

    @staticmethod
    def _collect_stems(out_dir: Path) -> dict[str, np.ndarray]:
        stems: dict[str, np.ndarray] = {}
        # Default filename template is {file_name}/{instr}, i.e.
        # out/mixture/effects.wav — but be liberal about layout and codec.
        for f in sorted([*out_dir.rglob("*.wav"), *out_dir.rglob("*.flac")]):
            lower = f.stem.lower()
            for stem, aliases in STEM_ALIASES.items():
                if stem not in stems and any(a in lower for a in aliases):
                    data, _ = sf.read(f, dtype="float32", always_2d=True)
                    stems[stem] = data.T
                    break
        return stems
