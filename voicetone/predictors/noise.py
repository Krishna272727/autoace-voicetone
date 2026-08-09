"""Phase 3 -- background noise level and type.

Writes `noise_level_db` (one continuous parent for both `present` and
`severity`) and `noise_class_dist` (which drives `type`). It does **not** write
any schema field and it does not touch `degradation_score`.

HOW THE LEVEL IS MEASURED, AND WHY NOT THE OBVIOUS WAY
-------------------------------------------------------
The obvious estimator is "energy in VAD-negative regions relative to energy in
VAD-positive regions". It was implemented and measured first, and it does not
work on this material. Across a grid of 80 percentile pairs on the three
provided calls, the best separation between the no-noise call and the two
noisy ones was **5.3 dB**, and no combination reproduced the anchors quoted in
BUILD_SPEC 3.3 (-49 dB / -33 dB). Two reasons:

  * The agent is a TTS bot whose pauses are *digitally silent*. Pooling all
    non-speech frames mixes the customer's room tone with the bot's synthetic
    silence, and a low percentile then reports the bot.
  * Steady noise -- the "sharp static" of call_003 -- is present *underneath the
    speech too*, and a gap-only estimator never sees the part that matters.

So the level is estimated by **minimum statistics**: per frequency band, take
the minimum power over a sliding window, which is what noise suppressors use to
track a noise floor through continuous speech. Aggregating those per-frame
estimates at the 90th percentile and referencing them to median speech power
gives a 10.1 dB separation on the same three files, and puts call_002 at
-32.8 dB against the -33 dB anchor in the brief.

Measured: call_001 -57.9 dB (labelled no noise), call_003 -47.7 dB and
call_002 -32.8 dB (both labelled medium). The 15 dB spread *within* one label
is the honest reason the `low`/`high` boundaries cannot be fixed from three
files and must come from the synthetic SNR sweep (BUILD_SPEC 8).

TYPE
----
An AudioSet tagger (AST, 527 classes) is collapsed through
`noise_vocab.from_audioset`, which sums sibling classes before the argmax so
Car + Engine + Vehicle cannot lose to one spurious class. A DSP signature
fallback covers what AudioSet handles badly -- and "sharp static" is exactly
that: a line/codec artifact, not an environmental sound.

`"unspecified"` is never emitted while noise is present. It scores zero against
any real label, whereas a wrong guess from the right family earns partial
credit.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import signal

from .. import noise_vocab
from ..audio import AudioContext
from ..latent import Latents
from .vad import timeline

log = logging.getLogger("autoace.noise")

NFFT = 1024
HOP = 256
MIN_WIN_S = 1.5             # sliding window for the per-band minimum
MIN_STAT_BIAS = 1.5         # bias correction: a sliding min underestimates
NOISE_PCTL = 90             # aggregate over frames: "when noise is there, how loud"
SPEECH_PCTL = 50

# AST wants 16 kHz and reads ~10.24 s per pass. Long files are sampled rather
# than tagged end to end, so a 60-minute call costs the same as a 30-second one.
TAG_SR = 16_000
TAG_WIN_S = 10.24
MAX_TAG_WINDOWS = 6


# --------------------------------------------------------------------------
# level
# --------------------------------------------------------------------------

def _noise_level_db(ctx: AudioContext) -> tuple[float, dict[str, float]]:
    x = np.asarray(ctx.master, dtype=np.float32).ravel()
    sr = int(ctx.master_sr) or 16_000
    if x.size < NFFT * 4:
        return -90.0, {}

    freqs, _, Z = signal.stft(x, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP,
                              padded=False, boundary=None)
    P = (np.abs(Z).astype(np.float64) ** 2)
    hop_s = HOP / sr
    band = (freqs >= 200) & (freqs <= min(7000.0, sr / 2 * 0.95))
    Pb = P[band]
    if Pb.shape[1] < 8:
        return -90.0, {}

    win = max(3, int(MIN_WIN_S / hop_s))
    if Pb.shape[1] <= win:
        win = max(3, Pb.shape[1] // 2)
    from numpy.lib.stride_tricks import sliding_window_view
    floor_per_frame = sliding_window_view(Pb, win, axis=1).min(-1).sum(axis=0)
    floor_per_frame *= MIN_STAT_BIAS

    total = Pb.sum(axis=0)
    tl = timeline(ctx)
    t = np.arange(P.shape[1]) * hop_s
    m = np.zeros(P.shape[1], dtype=bool)
    for a, b in tl.speech:
        m |= (t >= a) & (t < b)
    speech_pow = total[m] if m.any() else total

    noise_db = 10 * np.log10(float(np.percentile(floor_per_frame, NOISE_PCTL)) + 1e-30)
    speech_db = 10 * np.log10(float(np.percentile(speech_pow, SPEECH_PCTL)) + 1e-30)
    rel = noise_db - speech_db
    detail = {"noise_abs_db": round(noise_db, 2),
              "speech_abs_db": round(speech_db, 2)}
    return float(np.clip(rel, -90.0, 0.0)), detail


# --------------------------------------------------------------------------
# type -- DSP signature
# --------------------------------------------------------------------------

def _floor_spectrum(ctx: AudioContext) -> tuple[np.ndarray, np.ndarray] | None:
    """The stationary noise-floor spectrum, by minimum statistics.

    Same estimator as the level: per band, the minimum over a sliding window,
    then the median across time. Averaging VAD-negative frames instead was tried
    first and does not work here -- on all three provided calls those frames put
    99% of their energy below 1 kHz, because they are dominated by the bot's
    digital silence and by residual speech tails rather than by room tone.
    """
    x = np.asarray(ctx.master, dtype=np.float32).ravel()
    sr = int(ctx.master_sr) or 16_000
    if x.size < NFFT * 8:
        return None
    freqs, _, Z = signal.stft(x, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP,
                              padded=False, boundary=None)
    P = (np.abs(Z).astype(np.float64) ** 2)
    win = max(3, int(MIN_WIN_S / (HOP / sr)))
    if P.shape[1] <= win:
        return None
    from numpy.lib.stride_tricks import sliding_window_view
    spec = np.median(sliding_window_view(P, win, axis=1).min(-1), axis=1)
    return freqs, spec


def _dsp_signature(ctx: AudioContext) -> dict[str, float]:
    """Classify the stationary noise floor from its spectral shape.

    Covers the line artifacts an environmental-sound tagger is weakest on --
    "sharp static" is a codec/line condition, not a thing AudioSet was trained
    to recognise.

    **Calibration note.** The thresholds below were set against *synthesised
    reference noises* (white noise, a 60 Hz tone, brown noise low-passed to
    800 Hz), not against the two noisy sample calls. Fitting them to n=2 would
    be the exact overfitting BUILD_SPEC 3 warns against, and the two calls do
    not in fact separate on these features: their floor flatness is 0.31 and
    0.40 against 0.50 for the call labelled *no noise*. Consequently this
    function stays silent on all three provided calls and the tagger decides
    there -- which is the honest outcome, not a tuned one.

        reference       flat   lo<400   hi>1500   peaky
        white noise     0.92     0.08      0.53     3.0
        60 Hz tone      0.15     0.59      0.28   392.8
        brown/road      0.01     0.79      0.00  5473.8
        call_001        0.50     0.43      0.17    19.1   (labelled: no noise)
        call_002        0.31     0.64      0.11    57.0   (labelled: TV)
        call_003        0.40     0.46      0.12    27.1   (labelled: static)
    """
    got = _floor_spectrum(ctx)
    if got is None:
        return {}
    freqs, spec = got
    sr = int(ctx.master_sr) or 16_000
    nyq = sr / 2.0
    band = (freqs >= 150) & (freqs <= min(3400.0, nyq * 0.95))
    s = spec[band]
    tot = float(s.sum())
    if tot <= 0 or s.size < 8:
        return {}
    s = s / tot
    fb = freqs[band]

    flat = float(np.exp(np.log(s + 1e-14).mean()) / (s + 1e-14).mean())
    lo = float(s[fb < 400].sum())
    hi = float(s[fb >= 1500].sum())

    # Mains hum: fraction of in-band energy sitting on 50/60 Hz harmonics.
    hum = 0.0
    for f0 in (50.0, 60.0):
        h = sum(float(spec[(freqs >= k * f0 - 6) & (freqs <= k * f0 + 6)].sum())
                for k in (1, 2, 3))
        hum = max(hum, h / max(float(spec[band].sum()), 1e-30))

    out: dict[str, float] = {}
    # Flat and genuinely broadband -> hiss / line static.
    if flat > 0.65 and hi > 0.35:
        out["static"] = 0.4 + 0.4 * min(1.0, (flat - 0.65) / 0.3)
    # Concentrated on mains harmonics -> hum / mechanical.
    if hum > 0.30:
        out["hum"] = 0.4 + 0.4 * min(1.0, (hum - 0.30) / 0.4)
    # Low-frequency dominant with nothing on top -> road noise / wind.
    if lo > 0.70 and hi < 0.05:
        out["road noise"] = 0.4 + 0.4 * min(1.0, (lo - 0.70) / 0.25)
    return out


# --------------------------------------------------------------------------
# type -- AudioSet tagger
# --------------------------------------------------------------------------

def _tag(ctx: AudioContext) -> dict[str, float]:
    """AudioSet class scores, averaged over sampled windows."""
    from ..models import audioset_tagger

    x = np.asarray(ctx.speech, dtype=np.float32).ravel()   # AST is a 16 kHz model
    if x.size < TAG_SR:
        return {}
    fe, model, id2label = audioset_tagger()

    n = int(TAG_WIN_S * TAG_SR)
    if x.size <= n:
        starts = [0]
    else:
        k = min(MAX_TAG_WINDOWS, max(1, x.size // n))
        starts = np.linspace(0, x.size - n, k).astype(int).tolist()

    import torch
    probs = None
    for s in starts:
        seg = x[s:s + n]
        inputs = fe(seg, sampling_rate=TAG_SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        p = torch.sigmoid(logits)[0].numpy()          # AudioSet is multi-label
        probs = p if probs is None else probs + p
    if probs is None:
        return {}
    probs = probs / len(starts)

    # Keep the top classes only; the tail is 500 near-zero scores.
    top = np.argsort(-probs)[:25]
    return {str(id2label[int(i)]): float(probs[int(i)]) for i in top
            if probs[int(i)] > 0.02}


class NoisePredictor:
    name = "noise"

    def __call__(self, ctx: AudioContext, lat: Latents) -> None:
        level, detail = _noise_level_db(ctx)
        lat.noise_level_db = round(level, 2)

        dist: dict[str, float] = {}
        raw: dict[str, float] = {}
        try:
            raw = _tag(ctx)
            dist = noise_vocab.from_audioset(raw)
        except Exception as exc:                   # noqa: BLE001
            lat.notes.append(f"noise: tagger unavailable ({type(exc).__name__})")
            log.debug("tagger failed", exc_info=True)

        sig = _dsp_signature(ctx)
        if not dist:
            dist = sig
            lat.notes.append("noise: type from DSP signature (tagger gave nothing)")
        else:
            # Blend rather than replace: the tagger is better at environmental
            # sound, the signature is better at line artifacts. AudioSet has no
            # good class for "sharp static".
            for k, v in sig.items():
                dist[k] = dist.get(k, 0.0) + 0.5 * v

        # Normalise so the top score reads as a confidence, and derive.py's
        # type_min_score means the same thing whatever produced the numbers.
        total = sum(dist.values())
        if total > 0:
            dist = {k: round(v / total, 4) for k, v in dist.items()}
        lat.noise_class_dist = dict(sorted(dist.items(), key=lambda kv: -kv[1])[:8])

        if detail:
            lat.quality_detail.setdefault("_noise_abs_db", detail["noise_abs_db"])
        if raw:
            top = max(raw.items(), key=lambda kv: kv[1])
            lat.notes.append(f"noise: top AudioSet class {top[0]!r} ({top[1]:.2f})")
