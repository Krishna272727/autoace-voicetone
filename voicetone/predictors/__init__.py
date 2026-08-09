"""Stage-1 predictors.

Every predictor obeys one contract (BUILD_SPEC.md 2.3):

    class MyPredictor:
        name = "my_predictor"
        def __call__(self, ctx: AudioContext, lat: Latents) -> None:
            lat.some_latent = ...     # estimates only, never schema fields

Shared work goes in `ctx.cache` so it is computed once -- the VAD timeline is
reused by silence detection, noise estimation, role assignment and emotion.

Imports are lazy: pulling in `emotion` loads torch and transformers, which
costs seconds and hundreds of megabytes. A caller that only wants the DSP
stack should not pay for that.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .stub import StubPredictor

if TYPE_CHECKING:  # pragma: no cover
    from .emotion import EmotionPredictor
    from .noise import NoisePredictor
    from .overlap import OverlapPredictor
    from .quality import QualityPredictor
    from .roles import RolePredictor
    from .vad import VADPredictor

_LAZY = {
    "VADPredictor": ".vad",
    "QualityPredictor": ".quality",
    "NoisePredictor": ".noise",
    "OverlapPredictor": ".overlap",
    "RolePredictor": ".roles",
    "EmotionPredictor": ".emotion",
}

__all__ = ["StubPredictor", *_LAZY]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module
        mod = import_module(_LAZY[name], __name__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
