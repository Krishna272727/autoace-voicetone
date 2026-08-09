"""Predictor stack assembly, shared by the CLI and the dashboard.

Two things happen here that matter for cost and latency:

1. **Models load once.** The stack is memoised per configuration, so the
   dashboard builds it at first request and reuses it for every file after.
   Cold-loading Whisper and a SER checkpoint per request adds 10-30 s and would
   dominate the entire measurement (BUILD_SPEC.md 7).
2. **A predictor that cannot construct is skipped, not fatal.** A missing or
   gated checkpoint degrades the system by one latent -- `derive.py` sees None,
   answers from what it has, and charges confidence for the gap. It does not
   take the process down.
"""
from __future__ import annotations

import functools
import logging
import os

log = logging.getLogger("autoace.stack")

# Registration order is dependency order: VAD fills ctx.cache["vad"], which
# noise, overlap, roles and emotion all read; roles fills the customer mask
# that emotion runs on.
_PHASES: tuple[tuple[str, str], ...] = (
    ("vad", "VADPredictor"),
    ("quality", "QualityPredictor"),
    ("noise", "NoisePredictor"),
    ("overlap", "OverlapPredictor"),
    ("roles", "RolePredictor"),
    ("emotion", "EmotionPredictor"),
)

DEFAULT_STACK = tuple(name for name, _ in _PHASES)


def _requested(spec: str | None) -> tuple[str, ...]:
    """Parse a stack spec: "all", "none", or a comma-separated phase list.
    Unknown names are dropped with a warning rather than raising -- a typo in
    an env var should not take a deployment down."""
    if spec is None:
        spec = os.getenv("VOICETONE_STACK", "all")
    spec = spec.strip().lower()
    if spec in ("", "all", "full"):
        return DEFAULT_STACK
    if spec in ("none", "stub"):
        return ()
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    out = [w for w in wanted if w in DEFAULT_STACK]
    for bad in set(wanted) - set(out):
        log.warning("unknown stack phase %r ignored; known: %s",
                    bad, ", ".join(DEFAULT_STACK))
    # Keep dependency order regardless of the order the caller listed them in.
    return tuple(n for n in DEFAULT_STACK if n in out)


@functools.lru_cache(maxsize=4)
def build_predictors(spec: str | None = None) -> tuple:
    """Build the stack. Memoised, so models are constructed once per process."""
    from importlib import import_module

    from .predictors import StubPredictor

    out = []
    for phase in _requested(spec):
        cls_name = dict(_PHASES)[phase]
        try:
            mod = import_module(f".predictors.{phase}", __package__)
            out.append(getattr(mod, cls_name)())
            log.info("stack: %s ready", phase)
        except Exception as exc:                   # noqa: BLE001
            # Typically a checkpoint that would not download, or a gated repo.
            log.warning("stack: %s unavailable (%s: %s) -- continuing without "
                        "it; its latents stay None", phase, type(exc).__name__, exc)

    # The stub goes last and only fills latents nobody estimated, recording each
    # one in `Latents.stubbed` so confidence is charged for the guess.
    out.append(StubPredictor())
    return tuple(out)


def stack_names(spec: str | None = None) -> list[str]:
    return [getattr(p, "name", type(p).__name__) for p in build_predictors(spec)]
