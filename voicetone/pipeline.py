"""Pipeline orchestration.

Predictors are plain callables: ctx -> dict of partial fields.
They run in registration order and may read ctx.cache to share work.
A predictor that raises is logged and skipped; it never kills the file,
and a file that fails never kills the batch (spec section 7).
"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from . import audio
from .derive import Thresholds, derive
from .latent import Latents
from .schema import CallAnalysis

log = logging.getLogger("autoace")


class Predictor(Protocol):
    """Stage 1. Writes continuous estimates into `lat`; returns nothing.

    Predictors must NOT write schema fields directly -- all discrete outputs
    are derived in one place (derive.py) so coupled fields stay consistent.
    """
    name: str
    def __call__(self, ctx: audio.AudioContext, lat: Latents) -> None: ...


_REGISTRY: list[Predictor] = []


def register(p: Predictor) -> Predictor:
    _REGISTRY.append(p)
    return p


def registry() -> list[Predictor]:
    return list(_REGISTRY)


@dataclass
class FileResult:
    name: str
    status: str                       # "ok" | "failed"
    result: dict[str, Any] | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    audio_s: float = 0.0
    stage_times: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    latents: dict[str, Any] = field(default_factory=dict)

    @property
    def rtf(self) -> float | None:
        return self.elapsed_s / self.audio_s if self.audio_s else None


def analyze_file(path: str | Path,
                 predictors: list[Predictor] | None = None,
                 thresholds: Thresholds | None = None,
                 on_stage: Callable[[str, int, int], None] | None = None) -> FileResult:
    """Analyse one file. Never raises: a failure returns status="failed".

    `on_stage(name, index, total)` is called before each predictor runs, so a
    caller can report progress *within* a file. Without it, a one-file batch
    shows 0 of 1 for the entire run and looks hung -- which on the full stack
    means about forty seconds of apparently-frozen page.
    """
    path = Path(path)
    preds = predictors if predictors is not None else registry()
    t0 = time.perf_counter()
    res = FileResult(name=path.name, status="ok")

    try:
        ctx = audio.load(path)
    except Exception as exc:                       # noqa: BLE001
        res.status = "failed"
        res.error = f"{type(exc).__name__}: {exc}"
        res.elapsed_s = time.perf_counter() - t0
        log.warning("decode failed for %s: %s", path.name, res.error)
        return res

    res.audio_s = ctx.duration
    if ctx.n_channels >= 2 and not ctx.true_stereo:
        res.warnings.append(
            f"stereo container but channels are duplicated "
            f"(corr={ctx.channel_corr:.4f}) - treating as mono"
        )

    lat = Latents()
    for i, p in enumerate(preds):
        name = getattr(p, "name", p.__class__.__name__)
        if on_stage:
            try:
                on_stage(name, i, len(preds))
            except Exception:                      # noqa: BLE001
                pass                               # progress must never break a run
        ts = time.perf_counter()
        try:
            p(ctx, lat)                            # stage 1: estimate latents
        except Exception as exc:                   # noqa: BLE001
            msg = f"predictor '{name}' failed: {type(exc).__name__}: {exc}"
            res.warnings.append(msg)
            log.warning("%s | %s", path.name, msg)
            log.debug(traceback.format_exc())
        res.stage_times[name] = time.perf_counter() - ts

    # stage 2: one place where all coupling is resolved
    try:
        res.result = derive(lat, thresholds).to_flat()
    except Exception as exc:                       # noqa: BLE001
        res.warnings.append(f"derivation failed: {exc}; using defaults")
        res.result = CallAnalysis().to_flat()
    res.latents = lat.as_dict()
    if lat.missing():
        res.warnings.append(f"latents not estimated: {', '.join(lat.missing())}")

    res.elapsed_s = time.perf_counter() - t0
    return res


def analyze_batch(paths: list[str | Path],
                  predictors: list[Predictor] | None = None,
                  progress: Callable[[int, int, FileResult], None] | None = None,
                  thresholds: Thresholds | None = None) -> list[FileResult]:
    out: list[FileResult] = []
    total = len(paths)
    for i, p in enumerate(paths, 1):
        r = analyze_file(p, predictors, thresholds)
        out.append(r)
        if progress:
            progress(i, total, r)
    return out
