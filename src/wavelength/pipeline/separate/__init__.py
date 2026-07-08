"""Separation engines. All engines implement the Separator interface and
return speech / music / effects stems; the pipeline only keeps effects."""

from wavelength.pipeline.separate.base import (
    SeparationError,
    SeparationResult,
    Separator,
    get_separator,
)

__all__ = ["Separator", "SeparationResult", "SeparationError", "get_separator"]
