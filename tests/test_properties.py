"""Property tests over the derivation layer (BUILD_SPEC.md 9.3).

These run on random latents rather than on audio, so they exercise the whole
decision surface -- including the regions the three sample files never reach.
Half the output space has no real example, so this is the only place some of
these guarantees get checked at all.
"""
from __future__ import annotations

import random

import pytest

from voicetone import Latents, Thresholds, derive
from voicetone.schema import CallAnalysis

N_DRAWS = 20_000


def _random_latents(rng: random.Random) -> Latents:
    def maybe(v):
        return v if rng.random() > 0.15 else None      # exercise the None paths
    return Latents(
        speech_ratio=rng.uniform(0, 1),
        max_silence_s=maybe(rng.uniform(0, 120)),
        overlap_ratio=maybe(rng.uniform(0, 1)),
        noise_level_db=maybe(rng.uniform(-90, 0)),
        noise_class_dist={rng.choice(["television", "static", "music",
                                      "office chatter", "road noise"]):
                          rng.uniform(0, 1)} if rng.random() > 0.2 else {},
        degradation_score=maybe(rng.uniform(0, 1)),
        role_confidence=maybe(rng.uniform(0, 1)),
        arousal=maybe(rng.uniform(-1, 1)),
        valence=maybe(rng.uniform(-1, 1)),
    )


@pytest.fixture(scope="module")
def draws():
    rng = random.Random(0)
    return [derive(_random_latents(rng)) for _ in range(N_DRAWS)]


def test_noise_absent_implies_none_and_empty(draws):
    for r in draws:
        if not r.background_noise_present:
            assert r.background_noise_severity.value == "none"
            assert r.background_noise_type == ""


def test_noise_present_implies_severity_and_type(draws):
    for r in draws:
        if r.background_noise_present:
            assert r.background_noise_severity.value != "none"
            assert r.background_noise_type != ""


def test_never_emits_unspecified(draws):
    """"unspecified" scores zero against any real label; a wrong guess from the
    right family earns partial credit (BUILD_SPEC 5, Phase 3)."""
    for r in draws:
        assert r.background_noise_type != "unspecified"


def test_neutral_is_never_high_intensity(draws):
    for r in draws:
        if r.emotional_tone.value == "neutral":
            assert r.emotional_intensity.value != "high"


def test_distressed_is_never_low_intensity(draws):
    for r in draws:
        if r.emotional_tone.value == "distressed":
            assert r.emotional_intensity.value != "low"


def test_confidence_in_range(draws):
    for r in draws:
        assert 0.0 <= r.confidence <= 1.0


def test_quality_is_invariant_to_noise_level():
    """The forbidden coupling (BUILD_SPEC 2.2).

    Sweep noise across its entire range with degradation_score pinned, and
    assert `audio_quality` never moves. The provided labels contain a call with
    "sharp static" and audio_quality=clear; a system that reads impairment off
    noise fails that case.
    """
    for deg in [i / 40 for i in range(41)]:
        answers = set()
        for db in range(-70, -7):
            lat = Latents(degradation_score=deg, noise_level_db=float(db),
                          arousal=0.0, valence=0.0, max_silence_s=0.0,
                          overlap_ratio=0.0)
            answers.add(derive(lat).audio_quality.value)
        assert len(answers) == 1, (
            f"audio_quality moved with noise at degradation_score={deg}: {answers}")


def test_severity_is_monotonic_in_noise_level():
    """Louder noise must never produce a *less* severe answer."""
    order = ["none", "low", "medium", "high"]
    prev = -1
    for db in range(-90, 0):
        lat = Latents(noise_level_db=float(db), noise_class_dist={"static": 1.0})
        idx = order.index(derive(lat).background_noise_severity.value)
        assert idx >= prev, f"severity went down at {db} dB"
        prev = idx


def test_quality_is_monotonic_in_degradation():
    order = ["clear", "slightly_impaired", "severely_impaired"]
    prev = -1
    for i in range(101):
        lat = Latents(degradation_score=i / 100)
        idx = order.index(derive(lat).audio_quality.value)
        assert idx >= prev
        prev = idx


def test_all_tone_classes_are_reachable():
    """Every enum value must be producible.

    `frustrated` and `distressed` have zero examples in the provided data, so
    without this test they could be dead code that never fires on the hidden
    set (BUILD_SPEC 3.1).
    """
    seen = set()
    for a in [i / 20 for i in range(-20, 21)]:
        for v in [i / 20 for i in range(-20, 21)]:
            seen.add(derive(Latents(arousal=a, valence=v)).emotional_tone.value)
    assert seen == {"neutral", "satisfied", "frustrated", "upset", "distressed"}


def test_all_severity_and_quality_classes_are_reachable():
    sev = {derive(Latents(noise_level_db=float(db),
                          noise_class_dist={"static": 1.0}
                          )).background_noise_severity.value
           for db in range(-90, 0)}
    assert sev == {"none", "low", "medium", "high"}
    qual = {derive(Latents(degradation_score=i / 100)).audio_quality.value
            for i in range(101)}
    assert qual == {"clear", "slightly_impaired", "severely_impaired"}


def test_missing_latents_lower_confidence():
    full = derive(Latents(max_silence_s=0.0, overlap_ratio=0.0,
                          noise_level_db=-60.0, degradation_score=0.0,
                          arousal=0.0, valence=0.0))
    empty = derive(Latents())
    assert empty.confidence < full.confidence


def test_stubbed_latents_are_penalised_like_missing():
    """A fallback constant is a real float but carries no evidence."""
    measured = Latents(max_silence_s=0.0, overlap_ratio=0.0,
                       noise_level_db=-60.0, degradation_score=0.0,
                       arousal=0.0, valence=0.0)
    stubbed = Latents(**{k: getattr(measured, k) for k in Latents.CORE})
    stubbed.stubbed = list(Latents.CORE)
    assert derive(stubbed).confidence < derive(measured).confidence


def test_schema_defaults_are_self_consistent():
    r = CallAnalysis()
    assert r.background_noise_type == ""
    assert r.background_noise_severity.value == "none"
    assert 0.0 <= r.confidence <= 1.0


def test_thresholds_load_from_yaml():
    from voicetone.derive import DEFAULT_CONFIG, load_thresholds
    t = load_thresholds(DEFAULT_CONFIG)
    assert isinstance(t, Thresholds)
    # Ordering constraints the derivation depends on.
    assert t.noise_present_db <= t.noise_medium_db <= t.noise_high_db
    assert t.quality_slight < t.quality_severe
    assert t.arousal_low < t.arousal_high


def test_missing_config_falls_back_to_defaults():
    from voicetone.derive import load_thresholds
    assert load_thresholds("/nonexistent/thresholds.yaml") == Thresholds()
