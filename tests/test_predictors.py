"""Unit tests: each predictor against input whose answer is known analytically
(BUILD_SPEC.md 9.1), plus the vocabulary layer.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from voicetone import Latents, load, noise_vocab
from voicetone.predictors.quality import (_bandwidth, _clipping, _dropouts,
                                          _level)

SR = 16_000


# --------------------------------------------------------------------------
# quality sub-scores -- constructed signals, known answers
# --------------------------------------------------------------------------

def test_clipping_zero_on_clean_sine():
    t = np.arange(SR) / SR
    assert _clipping(0.5 * np.sin(2 * np.pi * 200 * t)) == 0.0


def test_clipping_high_on_square_wave():
    t = np.arange(SR) / SR
    assert _clipping(np.sign(np.sin(2 * np.pi * 200 * t)) * 0.99) > 0.9


def test_clipping_ignores_intersample_overshoot():
    """A lossy codec reconstructs peaks above full scale. That is not clipping,
    and treating it as such flagged a `clear` sample call as impaired."""
    t = np.arange(SR) / SR
    loud = 1.4 * np.sin(2 * np.pi * 200 * t)     # peaks well over 1.0, not flat
    assert _clipping(loud) == 0.0


def test_clipping_empty_input():
    assert _clipping(np.zeros(0)) == 0.0


def test_level_flags_quiet_and_hot():
    t = np.arange(4 * SR) / SR
    speech = [(0.0, 4.0)]
    nominal = 0.1 * np.sin(2 * np.pi * 200 * t)
    quiet = 1e-4 * np.sin(2 * np.pi * 200 * t)
    hot = 0.98 * np.sin(2 * np.pi * 200 * t)
    assert _level(nominal, SR, speech)[0] == 0.0
    assert _level(quiet, SR, speech)[0] > 0.8
    assert _level(hot, SR, speech)[0] > 0.5


def test_dropouts_finds_inserted_holes():
    rng = np.random.default_rng(0)
    x = (0.1 * rng.normal(size=8 * SR)).astype(np.float32)
    speech = [(0.0, 8.0)]
    assert _dropouts(x, SR, speech) == 0.0
    holed = x.copy()
    for i in range(20):                            # 20 holes of 60 ms
        s = int((0.2 + i * 0.38) * SR)
        holed[s:s + int(0.06 * SR)] = 0.0
    assert _dropouts(holed, SR, speech) > 0.4


def test_dropouts_ignores_uniform_attenuation():
    """Scaling a file down must not manufacture packet loss."""
    rng = np.random.default_rng(1)
    x = (0.1 * rng.normal(size=6 * SR)).astype(np.float32)
    speech = [(0.0, 6.0)]
    assert _dropouts(x * 1e-3, SR, speech) == pytest.approx(
        _dropouts(x, SR, speech), abs=0.05)


def test_bandwidth_flags_8k_and_passes_wideband():
    freqs = np.linspace(0, 8000, 512)
    flat = np.ones_like(freqs)
    assert _bandwidth(flat, freqs, 16_000)[0] < 0.15     # wideband is fine
    assert _bandwidth(flat, freqs, 8_000)[0] > 0.4       # telephony is not


# --------------------------------------------------------------------------
# VAD -- known-length silence
# --------------------------------------------------------------------------

def test_vad_measures_inserted_silence(tmp_path):
    from voicetone.predictors.vad import VADPredictor, timeline
    rng = np.random.default_rng(0)
    # 4 s speech-like burst, 12 s digital silence, 4 s burst.
    t = np.arange(4 * SR) / SR
    burst = (0.2 * np.sin(2 * np.pi * 150 * t) *
             (1 + 0.5 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)
    x = np.concatenate([burst, np.zeros(12 * SR, np.float32), burst])
    p = tmp_path / "gap.wav"
    sf.write(p, x, SR)

    ctx = load(p)
    tl = timeline(ctx)
    assert tl.max_silence_s > 10.0, "a 12 s digital gap must be detected"
    lat = Latents()
    VADPredictor()(ctx, lat)
    assert lat.max_silence_s == pytest.approx(tl.max_silence_s)


def test_vad_on_pure_silence(tmp_path):
    from voicetone.predictors.vad import timeline
    p = tmp_path / "sil.wav"
    sf.write(p, np.zeros(5 * SR, np.float32), SR)
    tl = timeline(load(p))
    assert tl.speech_ratio == 0.0
    assert tl.max_silence_s == pytest.approx(5.0, abs=0.2)


def test_vad_on_clip_shorter_than_one_window(tmp_path):
    from voicetone.predictors.vad import timeline
    p = tmp_path / "tiny.wav"
    sf.write(p, np.zeros(200, np.float32), SR)     # < 512 samples
    tl = timeline(load(p))
    assert tl.speech == []


# --------------------------------------------------------------------------
# noise vocabulary
# --------------------------------------------------------------------------

@pytest.mark.parametrize("surface,canonical", [
    ("TV", "television"), ("tv", "television"), ("sharp static", "static"),
    ("hiss", "static"), ("loud TV", "television"), ("Babble", "office chatter"),
    ("keyboard", "keyboard typing"), ("faint hum", "hum"),
    ("background voices", "office chatter"), ("", ""),
])
def test_normalize_folds_surface_forms(surface, canonical):
    assert noise_vocab.normalize(surface) == canonical


def test_match_is_symmetric_and_bounded():
    for a in ("TV", "static", "music", ""):
        for b in ("television", "sharp static", "office chatter", ""):
            s = noise_vocab.match(a, b)
            assert 0.0 <= s <= 1.0
            assert s == noise_vocab.match(b, a)


def test_match_scores_label_variants_as_equal():
    assert noise_vocab.match("TV", "television") == 1.0
    assert noise_vocab.match("sharp static", "static") == 1.0


def test_empty_vs_present_scores_zero():
    assert noise_vocab.match("", "static") == 0.0
    assert noise_vocab.match("television", "") == 0.0
    assert noise_vocab.match("", "") == 1.0


def test_audioset_siblings_are_summed():
    """Car + Engine + Vehicle must outvote one spurious louder class."""
    scores = {"Car": 0.2, "Engine": 0.2, "Vehicle": 0.2, "Music": 0.45}
    out = noise_vocab.from_audioset(scores)
    assert out["road noise"] == pytest.approx(0.6)
    assert max(out, key=out.get) == "road noise"


def test_audioset_ignores_foreground_and_call_flow():
    scores = {"Speech": 0.9, "Dial tone": 0.5, "Sidetone": 0.4, "Radio": 0.07}
    out = noise_vocab.from_audioset(scores)
    assert "office chatter" not in out                # Speech is not noise
    assert out.get("television") == pytest.approx(0.07)
    assert max(out, key=out.get) == "television"


def test_emission_stays_in_the_closed_vocabulary():
    """Free text must never escape into `background_noise_type`."""
    out = noise_vocab.from_audioset({"Sine wave": 0.9, "Whoosh": 0.5})
    assert all(k in noise_vocab.CANONICAL for k in out)


# --------------------------------------------------------------------------
# scorer
# --------------------------------------------------------------------------

def test_scorer_gives_credit_for_label_variants():
    from voicetone.score import score
    truth = {"a.ogg": {"background_noise_type": "TV",
                       "background_noise_present": True}}
    pred = {"a.ogg": {"background_noise_type": "television",
                      "background_noise_present": True}}
    rep = score(pred, truth)
    assert rep["fields"]["background_noise_type"]["accuracy"] == 1.0


def test_scorer_reports_both_manifest_directions():
    from voicetone.score import score
    rep = score({"only_pred.ogg": {}}, {"only_truth.ogg": {}})
    assert rep["missing_from_predictions"] == ["only_truth.ogg"]
    assert rep["unlabelled_predictions"] == ["only_pred.ogg"]


def test_scorer_adjacent_accuracy_on_ordinal_fields():
    from voicetone.score import score
    truth = {"a": {"emotional_tone": "upset"}}
    pred = {"a": {"emotional_tone": "frustrated"}}      # one class away
    rep = score(pred, truth)["fields"]["emotional_tone"]
    assert rep["accuracy"] == 0.0
    assert rep["adjacent_accuracy"] == 1.0
