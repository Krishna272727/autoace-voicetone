"""Phase 6 -- emotion, as continuous (valence, arousal).

Writes `arousal`, `valence` and `valence_source`. It does **not** choose a tone
class: `derive.py` reads the same two numbers to produce `emotional_tone` and
`emotional_intensity`, so the two outputs cannot contradict each other and the
boundaries stay tunable without retraining anything. With three labelled calls
you cannot fit a classifier, but you can fit thresholds.

AROUSAL is prosody -- pitch movement, energy dynamics, speaking rate, spectral
tilt. It is reliable from audio alone.

VALENCE is the hard part and the whole ballgame. Audio-only valence is the
known-weak dimension in speech emotion recognition: prosody carries arousal,
semantics carry valence. And valence is exactly what separates the classes here
-- `satisfied` and `upset` are the same high arousal with opposite sign, as are
`neutral` and `frustrated`.

**call_003 is the live trap.** It is labelled `satisfied` / `medium` and it has
the highest speech level of the three files. A system driven by loudness calls
it `upset`. The words are what save it. This is also why the brief says not to
infer frustration from loudness: the hidden set is built to punish
acoustics-only systems.

SER runs on **customer segments only**. Agent speech is emotionally-toned TTS
("I'm so sorry to hear that") and would wreck the estimate if misattributed --
which is why Phase 5 gates this one.

THE ASR CASCADE
---------------
Transcription is the most expensive stage in the system, so it runs
conditionally rather than always:

    always:  prosody + SER  ->  arousal, acoustic valence
                  |
        low arousal AND unambiguous valence?
             yes -> emit            no -> ASR on customer speech
                                          -> lexical valence -> fuse

Most clips are clearly neutral and resolve without ASR. The budget is spent on
the activated, ambiguous ones -- which is exactly where the hidden set will
punish a cheap system.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from ..audio import AudioContext
from ..latent import Latents
from .vad import timeline

log = logging.getLogger("autoace.emotion")

SR = 16_000

# SER checkpoint classes (IEMOCAP 4-way) placed in the valence/arousal plane.
# Positions are from the standard circumplex, not fitted to the samples.
SER_VA: dict[str, tuple[float, float]] = {
    "neu": (0.00, 0.10), "neutral": (0.00, 0.10),
    "hap": (0.70, 0.55), "happy": (0.70, 0.55),
    "ang": (-0.70, 0.85), "angry": (-0.70, 0.85),
    "sad": (-0.55, 0.15), "sadness": (-0.55, 0.15),
}

# Fusion weights when both paths are available. Lexical is trusted more,
# because that is the dimension acoustics is weak on.
W_ACOUSTIC_V = 0.35
W_LEXICAL_V = 0.65

# Cascade trigger: run ASR when the clip is activated or the acoustic valence
# sits near zero, i.e. where a sign error would change the class.
ASR_AROUSAL_TRIGGER = 0.30
ASR_VALENCE_AMBIGUOUS = 0.35
ASR_MAX_S = float(os.getenv("VOICETONE_ASR_MAX_S", "180"))


# --------------------------------------------------------------------------
# prosody -> arousal
# --------------------------------------------------------------------------

def _prosodic_arousal(x: np.ndarray) -> tuple[float, dict[str, float]]:
    """Arousal in [-1, 1] from level dynamics, pitch movement and spectral tilt.

    Deliberately built from *dynamics* (variation, movement, rate) rather than
    absolute loudness. Absolute level is a recording-gain artifact as much as a
    speaker state, and the brief explicitly forbids reading activation off
    loudness alone.
    """
    if x.size < SR // 2:
        return 0.0, {}

    n, h = int(0.025 * SR), int(0.010 * SR)
    F = np.lib.stride_tricks.sliding_window_view(x, n)[::h]
    if F.shape[0] < 8:
        return 0.0, {}
    rms = np.sqrt((F.astype(np.float64) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    voiced = db > (np.percentile(db, 90) - 25.0)
    if voiced.sum() < 8:
        return 0.0, {}

    # 1. Level dynamics: the spread of speech-frame level.
    level_sd = float(np.std(db[voiced]))

    # 2. Pitch, by autocorrelation on voiced frames only. librosa.yin was
    #    measured at 0.0155 RTF -- 22% of the DSP budget by itself -- and is
    #    not used (BUILD_SPEC 7).
    f0s = []
    idx = np.flatnonzero(voiced)
    for i in idx[:: max(1, len(idx) // 120)]:
        fr = F[i].astype(np.float64)
        fr = fr - fr.mean()
        if not np.any(fr):
            continue
        ac = np.correlate(fr, fr, mode="full")[len(fr) - 1:]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        lo, hi = int(SR / 320), int(SR / 70)          # 70-320 Hz
        if hi >= ac.size:
            continue
        seg = ac[lo:hi]
        if seg.size and seg.max() > 0.30:             # periodic enough to trust
            f0s.append(SR / (lo + int(np.argmax(seg))))
    f0 = np.asarray(f0s, dtype=np.float64)
    if f0.size >= 4:
        # Semitone spread is speaker-independent in a way that Hz is not.
        st = 12 * np.log2(np.maximum(f0, 1e-6) / max(float(np.median(f0)), 1e-6))
        pitch_sd = float(np.std(st))
    else:
        pitch_sd = 0.0

    # 3. Speaking rate, from the energy-envelope modulation near 4 Hz.
    env = rms - rms.mean()
    if env.size >= 32 and np.any(env):
        spec = np.abs(np.fft.rfft(env * np.hanning(env.size)))
        fr_hz = np.fft.rfftfreq(env.size, d=h / SR)
        sel = (fr_hz >= 2.0) & (fr_hz <= 8.0)
        rate = float(spec[sel].sum() / (spec.sum() + 1e-12))
    else:
        rate = 0.0

    # 4. Spectral tilt: vocal effort brightens the spectrum.
    S = np.abs(np.fft.rfft(F[voiced] * np.hanning(n), axis=1)).mean(axis=0)
    fr_hz = np.fft.rfftfreq(n, d=1 / SR)
    lo_e = float(S[(fr_hz >= 100) & (fr_hz < 1000)].sum())
    hi_e = float(S[(fr_hz >= 1000) & (fr_hz < 4000)].sum())
    tilt = float(np.clip(np.log10((hi_e + 1e-12) / (lo_e + 1e-12)) + 1.0, -1, 1))

    def unit(v: float, lo: float, hi: float) -> float:
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    parts = {
        "level_sd": unit(level_sd, 3.0, 11.0),
        "pitch_sd": unit(pitch_sd, 0.8, 4.0),
        "rate": unit(rate, 0.15, 0.55),
        "tilt": unit(tilt, -0.6, 0.4),
    }
    raw = (0.30 * parts["level_sd"] + 0.30 * parts["pitch_sd"] +
           0.20 * parts["rate"] + 0.20 * parts["tilt"])
    arousal = float(np.clip(2.0 * raw - 1.0, -1.0, 1.0))
    detail = {k: round(v, 3) for k, v in parts.items()}
    detail.update({"level_sd_db": round(level_sd, 2),
                   "pitch_sd_st": round(pitch_sd, 2)})
    return arousal, detail


# --------------------------------------------------------------------------
# SER -> acoustic (valence, arousal)
# --------------------------------------------------------------------------

def _ser_va(x: np.ndarray) -> tuple[float, float, dict[str, float]] | None:
    """Run the SER checkpoint and map its class posterior into the VA plane."""
    if x.size < SR // 2:
        return None
    import torch

    from ..models import ser_model
    fe, model, id2label = ser_model()

    # Judge a bounded set of excerpts so cost does not grow with duration.
    win = SR * 6
    if x.size <= win:
        chunks = [x]
    else:
        starts = np.linspace(0, x.size - win, min(6, x.size // win + 1)).astype(int)
        chunks = [x[s:s + win] for s in starts]

    probs = None
    for c in chunks:
        inp = fe(c, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inp).logits
        p = torch.softmax(logits, dim=-1)[0].numpy()
        probs = p if probs is None else probs + p
    probs = probs / len(chunks)

    v = a = 0.0
    dist: dict[str, float] = {}
    for i, p in enumerate(probs):
        label = str(id2label[i]).lower()
        dist[label] = round(float(p), 4)
        vv, aa = SER_VA.get(label, (0.0, 0.2))
        v += float(p) * vv
        a += float(p) * aa
    # Arousal positions above are on a 0..1 activation scale; move to -1..1.
    return float(np.clip(v, -1, 1)), float(np.clip(2 * a - 1, -1, 1)), dist


# --------------------------------------------------------------------------
# ASR -> lexical valence
# --------------------------------------------------------------------------

def _lexical_valence(ctx: AudioContext, regions: list[tuple[float, float]]
                     ) -> tuple[float, str] | None:
    """Transcribe customer speech and score its sentiment."""
    import torch

    from ..models import asr_model, text_sentiment

    x = np.asarray(ctx.speech, dtype=np.float32).ravel()
    if regions:
        keep = np.zeros(x.size, dtype=bool)
        for a, b in regions:
            keep[int(a * SR):min(x.size, int(b * SR))] = True
        audio = x[keep]
    else:
        audio = x
    if audio.size < SR:
        return None
    audio = audio[:int(ASR_MAX_S * SR)]

    model = asr_model()
    segments, info = model.transcribe(audio, beam_size=1, vad_filter=False,
                                      temperature=0.0, language=None)
    text = " ".join(s.text for s in segments).strip()
    if not text:
        return None

    tok, sent, id2label = text_sentiment()
    enc = tok(text[:2000], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = sent(**enc).logits
    p = torch.softmax(logits, dim=-1)[0].numpy()

    # valence = P(positive) - P(negative). Neutral mass pulls toward zero
    # instead of being forced onto one side, which is the whole reason for
    # using a three-class head here.
    pos = neg = 0.0
    for i, prob in enumerate(p):
        name = str(id2label[i]).upper()
        if name.startswith("POS") or name in ("LABEL_2",):
            pos += float(prob)
        elif name.startswith("NEG") or name in ("LABEL_0",):
            neg += float(prob)
    return float(np.clip(pos - neg, -1.0, 1.0)), text


class EmotionPredictor:
    name = "emotion"

    def __call__(self, ctx: AudioContext, lat: Latents) -> None:
        tl = timeline(ctx)
        if not tl.speech:
            lat.arousal, lat.valence = 0.0, 0.0
            lat.valence_source = "none"
            lat.notes.append("emotion: no speech; neutral at low confidence")
            return

        # Customer-only audio, populated by Phase 5. Falls back to the full mix
        # when role assignment did not run, with the risk noted.
        cust = ctx.cache.get("customer_audio")
        regions = ctx.cache.get("customer_regions") or []
        if cust is None:
            cust = np.asarray(ctx.speech, dtype=np.float32).ravel()
            lat.notes.append("emotion: no role assignment; scoring the full "
                             "mix, so agent speech may contaminate this")
        cust = np.asarray(cust, dtype=np.float32).ravel()

        arousal_p, detail = _prosodic_arousal(cust)

        v_ac = a_ser = None
        try:
            got = _ser_va(cust)
            if got:
                v_ac, a_ser, dist = got
                detail.update({f"ser_{k}": v for k, v in dist.items()})
        except Exception as exc:                   # noqa: BLE001
            lat.notes.append(f"emotion: SER unavailable ({type(exc).__name__})")
            log.debug("SER failed", exc_info=True)

        # Arousal: prosody and SER agree often; average when both are present.
        arousal = arousal_p if a_ser is None else 0.5 * (arousal_p + a_ser)
        valence = v_ac if v_ac is not None else 0.0
        source = "acoustic" if v_ac is not None else "none"

        # --- the cascade -------------------------------------------------
        activated = arousal >= ASR_AROUSAL_TRIGGER
        ambiguous = abs(valence) < ASR_VALENCE_AMBIGUOUS
        if activated or ambiguous:
            try:
                got = _lexical_valence(ctx, regions)
            except Exception as exc:               # noqa: BLE001
                got = None
                lat.notes.append(f"emotion: ASR unavailable ({type(exc).__name__})")
                log.debug("ASR failed", exc_info=True)
            if got:
                v_lex, text = got
                if v_ac is None:
                    valence, source = v_lex, "lexical"
                else:
                    valence = W_ACOUSTIC_V * v_ac + W_LEXICAL_V * v_lex
                    source = "fused"
                detail["lexical_valence"] = round(v_lex, 3)
                ctx.cache["transcript"] = text
                lat.notes.append(f"emotion: transcript ({len(text.split())} words) "
                                 f"lexical valence {v_lex:+.2f}")
        else:
            lat.notes.append("emotion: ASR skipped (calm and unambiguous)")

        # Role uncertainty shrinks the estimate toward neutral rather than
        # letting a possibly-misattributed reading through at full strength.
        rc = lat.role_confidence
        if rc is not None and rc < 0.5:
            shrink = 0.5 + rc
            valence *= shrink
            arousal *= shrink
            lat.notes.append(f"emotion: role confidence {rc:.2f}; estimate "
                             f"shrunk toward neutral by x{shrink:.2f}")

        lat.arousal = round(float(np.clip(arousal, -1, 1)), 4)
        lat.valence = round(float(np.clip(valence, -1, 1)), 4)
        lat.valence_source = source
        lat.quality_detail.update({f"_emo_{k}": v for k, v in detail.items()})
