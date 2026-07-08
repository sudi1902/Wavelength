"""Demucs fallback engine (htdemucs).

Demucs separates music stems (vocals / drums / bass / other), not cinematic
stems, so the mapping is approximate:

    speech  <- vocals
    music   <- drums + bass
    effects <- other

Known limitation (documented in PLAN.md): melodic instruments land in
"other" alongside sound effects, so background music with synths/guitars
bleeds into the effects stem. Use the bandit engine when possible; this
exists as a fallback because demucs is a mature pip package that reliably
installs everywhere.

Install with: pip install "wavelength[demucs]"
"""

from __future__ import annotations

import numpy as np

from wavelength.pipeline.separate.base import (
    SeparationError,
    SeparationResult,
    Separator,
)

DEMUCS_MODEL = "htdemucs"
MODEL_SAMPLE_RATE = 44_100


def _pick_device(preference: str) -> str:
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DemucsSeparator(Separator):
    name = "demucs"

    def __init__(self, settings):
        super().__init__(settings)
        self._separator = None

    def is_ready(self) -> bool:
        try:
            import demucs.api  # noqa: F401
        except ImportError:
            return False
        return True

    def ensure_ready(self) -> None:
        if not self.is_ready():
            raise SeparationError(
                "demucs is not installed. Install with: "
                'pip install "wavelength[demucs]"'
            )
        # Instantiating triggers the model weight download (~300 MB, cached
        # by torch hub) and validates the device.
        self._get_separator()

    def _get_separator(self):
        if self._separator is None:
            import demucs.api

            device = _pick_device(self.settings.device)
            try:
                self._separator = demucs.api.Separator(
                    model=DEMUCS_MODEL, device=device
                )
            except Exception as exc:  # noqa: BLE001 - surface any load failure
                raise SeparationError(
                    f"Could not load demucs model '{DEMUCS_MODEL}': {exc}"
                ) from exc
        return self._separator

    def separate(self, audio: np.ndarray, sr: int) -> SeparationResult:
        import torch

        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        if sr != MODEL_SAMPLE_RATE:
            import librosa

            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=MODEL_SAMPLE_RATE, axis=1
            )

        separator = self._get_separator()
        _, stems = separator.separate_tensor(
            torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        )
        stems = {k: v.cpu().numpy() for k, v in stems.items()}

        missing = {"vocals", "drums", "bass", "other"} - stems.keys()
        if missing:
            raise SeparationError(f"demucs returned incomplete stems: {stems.keys()}")

        return SeparationResult(
            speech=stems["vocals"],
            music=stems["drums"] + stems["bass"],
            effects=stems["other"],
            sample_rate=MODEL_SAMPLE_RATE,
        )
