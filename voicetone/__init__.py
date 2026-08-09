"""AutoAce voice-tone and background-noise analysis.

Two stages, kept strictly apart (BUILD_SPEC.md 2):
  stage 1  predictors write continuous estimates into `Latents`
  stage 2  `derive()` turns those latents into the discrete output schema

Fields that share a latent parent are derived from that one value, so they
cannot contradict each other.
"""
from __future__ import annotations

from .audio import AudioContext, AudioLoadError, load
from .derive import Thresholds, derive, load_thresholds
from .latent import Latents
from .pipeline import (FileResult, analyze_batch, analyze_file, register,
                       registry)
from .schema import FIELDS, CallAnalysis

__version__ = "1.0.0"

__all__ = [
    "AudioContext", "AudioLoadError", "load",
    "Latents", "Thresholds", "derive", "load_thresholds",
    "CallAnalysis", "FIELDS",
    "FileResult", "analyze_file", "analyze_batch", "register", "registry",
    "__version__",
]
