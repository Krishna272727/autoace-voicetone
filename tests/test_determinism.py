"""Determinism (BUILD_SPEC.md 9.5). Reproducibility is 15% of the grade.

Same input, same output, byte for byte. Every stochastic component in the stack
is pinned: KMeans gets a fixed `random_state` and `n_init`, torch gets a fixed
seed, ASR uses greedy decoding at temperature 0 with no sampling fallback.
"""
from __future__ import annotations

import json

import pytest

from voicetone import analyze_file
from voicetone.stack import build_predictors

STACK = "vad,quality,noise"


def _canonical(result: dict) -> str:
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def test_same_file_twice_is_byte_identical(samples):
    if not samples:
        pytest.skip("samples not present")
    preds = list(build_predictors(STACK))
    a = analyze_file(samples[0], preds)
    b = analyze_file(samples[0], preds)
    assert _canonical(a.result) == _canonical(b.result)


def test_all_samples_are_reproducible(samples):
    if not samples:
        pytest.skip("samples not present")
    preds = list(build_predictors(STACK))
    first = {p.name: _canonical(analyze_file(p, preds).result) for p in samples}
    second = {p.name: _canonical(analyze_file(p, preds).result) for p in samples}
    assert first == second


def test_latents_are_reproducible(samples):
    """Not just the discrete output -- the continuous estimates underneath it,
    which is a much tighter check."""
    if not samples:
        pytest.skip("samples not present")
    preds = list(build_predictors(STACK))
    a = analyze_file(samples[0], preds).latents
    b = analyze_file(samples[0], preds).latents
    for key in ("speech_ratio", "max_silence_s", "noise_level_db",
                "degradation_score"):
        assert a[key] == b[key], f"{key} drifted between runs"


def test_order_of_files_does_not_matter(samples):
    """Shared caches must not leak state from one file into the next."""
    if len(samples) < 2:
        pytest.skip("need two samples")
    preds = list(build_predictors(STACK))
    forward = [_canonical(analyze_file(p, preds).result) for p in samples]
    backward = [_canonical(analyze_file(p, preds).result)
                for p in reversed(samples)]
    assert forward == list(reversed(backward))
