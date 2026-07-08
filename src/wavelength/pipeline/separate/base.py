"""Separator interface: audio in, speech/music/effects stems out.

Engines differ wildly in their runtime needs (BandIt runs in an isolated
venv via subprocess, demucs runs in-process, passthrough needs nothing), so
the interface is deliberately file-free: numpy in, numpy out. Engine-specific
setup lives behind ``ensure_ready()`` so the CLI can drive first-run
downloads with clear messaging.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from wavelength.config import Settings

STEMS = ("speech", "music", "effects")


class SeparationError(Exception):
    """Separation failed; message is user-facing."""


@dataclass
class SeparationResult:
    """Stems as (channels, samples) float arrays, all at ``sample_rate``
    (which may differ from the input rate — engines have native rates)."""

    speech: np.ndarray | None
    music: np.ndarray | None
    effects: np.ndarray
    sample_rate: int


class Separator(ABC):
    name: str = "base"

    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    def is_ready(self) -> bool:
        """True if models/dependencies are installed and usable."""

    @abstractmethod
    def ensure_ready(self) -> None:
        """Install/download whatever the engine needs (idempotent).
        Raises SeparationError with actionable instructions on failure."""

    @abstractmethod
    def separate(self, audio: np.ndarray, sr: int) -> SeparationResult:
        """Separate (channels, samples) audio into stems."""


def get_separator(settings: Settings, engine: str | None = None) -> Separator:
    engine = engine or settings.engine
    if engine == "bandit":
        from wavelength.pipeline.separate.bandit import BanditSeparator

        return BanditSeparator(settings)
    if engine == "demucs":
        from wavelength.pipeline.separate.demucs import DemucsSeparator

        return DemucsSeparator(settings)
    if engine == "none":
        from wavelength.pipeline.separate.passthrough import PassthroughSeparator

        return PassthroughSeparator(settings)
    raise SeparationError(
        f"Unknown engine '{engine}'. Available: bandit, demucs, none"
    )
