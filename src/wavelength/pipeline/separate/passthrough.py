"""Passthrough engine: no separation — the whole mix IS the effects stem.

For videos known to contain no speech/music, and for testing the rest of
the pipeline without ML dependencies.
"""

from __future__ import annotations

import numpy as np

from wavelength.pipeline.separate.base import SeparationResult, Separator


class PassthroughSeparator(Separator):
    name = "none"

    def is_ready(self) -> bool:
        return True

    def ensure_ready(self) -> None:
        pass

    def separate(self, audio: np.ndarray, sr: int) -> SeparationResult:
        return SeparationResult(speech=None, music=None, effects=audio, sample_rate=sr)
