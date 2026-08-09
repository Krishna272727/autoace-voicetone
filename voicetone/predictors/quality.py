"""Phase 2 -- technical audio quality.

Writes `degradation_score` in [0,1] and a per-sub-score breakdown in
`quality_detail`. Nothing else. In particular:

    THIS MODULE MUST NEVER READ ANY NOISE VARIABLE.

That is the forbidden coupling (BUILD_SPEC.md 2.2). The provided labels include
a call with `background_noise_type: "sharp static"` and `audio_quality: "clear"`
-- a television playing in the customer's room does not make the *line* bad. A
system that infers impairment from noise fails that case, and the property test
sweeps `noise_level_db` across its whole range with `degradation_score` fixed
and asserts the quality answer never moves.

Sample-rate discipline: the time-domain and spectral DSP run on `ctx.master` at
the **native** rate. Running them on the 16 kHz copy would discard everything
above 8 kHz, which is where band-limiting and codec artifacts are visible.

WHAT IS MEASURED, AND WHAT WAS DROPPED
--------------------------------------
Seven sub-scores were specified. Four survived measurement against a synthetic
degradation ladder (clipping, level, band-limiting, μ-law/GSM/low-bitrate Opus,
and ffmpeg-generated reverb); three did not and were removed rather than left
in looking authoritative:

  kept     clipping, dropouts, level, bandwidth  -- each moves monotonically
           with the fault it names, and each is verifiable by construction.
  dropped  spectral "roughness" as a distortion proxy. It ran *backwards*:
           0.83 on clean audio against 0.53 on band-limited audio, because it
           was dominated by the noise floor above the speech band.
  dropped  in-band spectral flatness as a "robotic"/codec proxy. Range across
           the entire ladder was 0.31-0.49 with clean files sitting mid-range:
           no discriminative power.
  dropped  envelope-decay slope as a reverb proxy. Artificially reverberated
           copies measured a *faster* decay than their dry source, and a file
           attenuated by 46 dB measured 829 dB/s from noise-floor artifacts.

SQUIM (torchaudio's reference-free STOI/PESQ/SI-SDR estimator) was evaluated as
a perceptual backbone, as BUILD_SPEC 5 Phase 2 suggests, and **rejected**. Its
licence was fine (CC-BY-4.0, not the NC licence 14 forbids). Three measurements
killed it:

  * **It is not stable under excerpt length.** On one clean call, STOI came back
    0.391 / 0.947 / 0.982 and PESQ 1.48 / 1.82 / 2.27 for 3 s / 5 s / 10 s
    excerpts of the same audio. A metric whose answer depends that strongly on
    how much of the file you show it cannot arbitrate "clear" vs "impaired".
  * **It is expensive and superlinear.** 0.132 RTF at 3 s rising to 0.198 RTF at
    10 s, single-core -- on its own several times the entire rest of the DSP
    stack, and the wrong shape for hour-long files.
  * **It responds to background noise.** SI-SDR on the clean-but-noisy call_002
    read -16.5 dB. Wiring that into `audio_quality` would rebuild precisely the
    noise/quality coupling that 2.2 forbids and that the "sharp static +
    clear" label punishes.

So quality is DSP-only, from four sub-scores that each move monotonically with
the fault they name and are verifiable by construction. The honest limitation:
reverberation and codec artifacting are **not** detected, and are listed as a
known failure mode rather than papered over.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import signal

from ..audio import AudioContext
from ..latent import Latents
from .vad import timeline

log = logging.getLogger("autoace.quality")

# One STFT, shared by every spectral sub-score. Recomputing it per feature was
# measured at 0.0137 RTF each time and is the easiest saving in the stack.
NFFT = 1024
HOP = 512

# How much each sub-score can contribute. A blown-out or gappy line is
# unusable; a band-limited one is merely telephony and stays "slightly".
WEIGHTS: dict[str, float] = {
    "clipping": 1.00,
    "dropouts": 0.90,
    "level": 0.60,
    "bandwidth": 0.55,
}

# Clipping is judged relative to the file's own peak, not to 1.0. A lossy codec
# reconstructs intersample peaks above full scale -- the provided Opus calls
# decode to |x| up to 1.41 -- so an absolute 0.985 rail flags ordinary loud
# speech as clipped. Measured on call_003 that produced clipping=0.34 and an
# `audio_quality` of slightly_impaired against a `clear` label.
CLIP_LEVEL = 0.985
# Clipping means *flat tops*. A lone sample at the rail is a peak; only runs of
# at least this many consecutive railed samples count as a clipped region.
CLIP_MIN_RUN = 3
# Maximum spread within a run, as a fraction of the peak, for it to count as a
# flat top. A few quantisation steps; well under the 1.5% a smooth peak spans.
CLIP_FLAT_TOL = 0.002
# A digital hole is defined relative to the speech level, not absolutely: a file
# uniformly attenuated by 30 dB has quiet passages below any fixed floor and
# would otherwise report continuous packet loss.
DROP_BELOW_SPEECH_DB = 55.0
DROP_FLOOR_MIN = 1e-6


def _ramp(x: float, lo: float, hi: float) -> float:
    """Linear 0->1 between lo and hi, clamped. Inverted when lo > hi."""
    if hi == lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


# --------------------------------------------------------------------------
# sub-scores -- each one testable in isolation against a constructed input
# --------------------------------------------------------------------------

def _clipping(x: np.ndarray) -> float:
    """Samples pinned at full scale. Consecutive runs matter more than isolated
    hits: one sample at 1.0 is a peak, 200 in a row is a square wave."""
    if x.size == 0:
        return 0.0
    peak = float(np.abs(x).max())
    if peak <= 0.0:
        return 0.0
    mag = np.abs(x)
    at_rail = mag >= CLIP_LEVEL * peak
    if not at_rail.any():
        return 0.0
    idx = np.flatnonzero(np.diff(np.concatenate(
        ([0], at_rail.view(np.int8), [0]))))
    starts, ends = idx[::2], idx[1::2]

    # Proximity to the peak is not clipping -- every smooth waveform spends
    # time near its own maximum. A 200 Hz sine scored 1.0 under a
    # proximity-only rule. What distinguishes clipping is that the top is
    # *flat*: consecutive samples pinned at the same value.
    #
    # Any run of a smooth signal that reaches the peak from the 0.985
    # threshold must span at least 1.5% of the peak; a digitally clipped run
    # spans a few quantisation steps at most.
    lengths = []
    for s, e in zip(starts, ends):
        n = e - s
        if n < CLIP_MIN_RUN:
            continue
        seg = mag[s:e]
        if float(seg.max() - seg.min()) <= CLIP_FLAT_TOL * peak:
            lengths.append(n)
    if not lengths:
        return 0.0
    flat = np.asarray(lengths)
    # Fraction of the file inside a flat-topped run, and the worst single run.
    frac = float(flat.sum()) / x.size
    longest = int(flat.max())
    return max(_ramp(frac, 0.0005, 0.04), _ramp(longest, 8, 100))


def _dropouts(x: np.ndarray, sr: int, speech: list[tuple[float, float]]) -> float:
    """Digital holes *inside* speech -- packet loss, not pauses.

    Only runs between 4 ms and 250 ms count. Shorter is a glottal closure,
    longer is somebody not talking; neither is a transmission fault.
    """
    if x.size == 0 or not speech:
        return 0.0
    # Floor tracks the speech level, so uniform attenuation does not manufacture
    # dropouts where there are none.
    m = np.zeros(x.size, dtype=bool)
    for a, b in speech:
        m[int(a * sr):min(x.size, int(b * sr))] = True
    ref = x[m] if m.any() else x
    speech_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    floor = max(DROP_FLOOR_MIN, speech_rms * 10 ** (-DROP_BELOW_SPEECH_DB / 20))

    quiet = np.abs(x) < floor
    idx = np.flatnonzero(np.diff(np.concatenate(([0], quiet.view(np.int8), [0]))))
    if idx.size == 0:
        return 0.0
    starts, ends = idx[::2], idx[1::2]
    lo, hi = max(1, int(0.004 * sr)), int(0.25 * sr)
    speech_s = sum(b - a for a, b in speech) or 1.0

    holes = 0
    for s, e in zip(starts, ends):
        if not (lo <= e - s <= hi):
            continue
        mid = (s + e) / 2 / sr
        if any(a <= mid <= b for a, b in speech):
            holes += 1
    return _ramp(holes / speech_s, 0.3, 5.0)      # dropouts per second of speech


def _bandwidth(P: np.ndarray, freqs: np.ndarray, sr: int) -> tuple[float, float]:
    """Band-limiting -> (score, hf/lf ratio in dB).

    Two independent signals, combined with max():

      * the sample rate itself -- an 8 kHz file cannot carry anything above
        4 kHz no matter what the spectrum says, and telephony band-limiting is
        a real (if mild) impairment;
      * the energy above 3.6 kHz relative to the 0.3-3.4 kHz speech band, which
        catches a 48 kHz file that has been low-passed somewhere upstream.

    This asks where the signal's *own* energy stops. It is not a noise
    measurement and does not become one when the room is loud.
    """
    nyq = sr / 2.0
    # 8 kHz telephony -> narrowband by construction; 16 kHz is wideband enough.
    sr_score = _ramp(-sr, -16000.0, -7900.0) * 0.55

    lf = float(P[(freqs >= 300) & (freqs <= 3400)].sum())
    top = min(7500.0, nyq * 0.95)
    ratio_db = -99.0
    ratio_score = 0.0
    # Needs a usable band above 3.6 kHz to measure at all. On an 8 kHz file the
    # window would be a 200 Hz sliver just under Nyquist, where the anti-alias
    # filter is already rolling off -- the ratio would measure the resampler,
    # not the content. There, sr_score alone is the honest answer.
    if top > 4200.0 and lf > 0:
        hf = float(P[(freqs >= 3600) & (freqs <= top)].sum())
        ratio_db = 10 * np.log10((hf + 1e-20) / (lf + 1e-20))
        # Measured: clean 48 kHz calls sit at -21 to -30 dB; a 3.4 kHz
        # low-passed copy at -41; a 1.8 kHz low-passed copy at -51.
        ratio_score = _ramp(-ratio_db, 32.0, 48.0)
    return max(sr_score, ratio_score), ratio_db


def _level(x: np.ndarray, sr: int, speech: list[tuple[float, float]]) -> tuple[float, float]:
    """Speech RMS relative to full scale -> (score, dBFS).

    Both directions are faults: too quiet loses consonants under the
    quantisation floor, too hot is on its way to clipping.
    """
    if x.size == 0:
        return 0.0, -120.0
    if speech:
        m = np.zeros(x.size, dtype=bool)
        for a, b in speech:
            m[int(a * sr):min(x.size, int(b * sr))] = True
        seg = x[m] if m.any() else x
    else:
        seg = x
    rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if seg.size else 0.0
    db = 20 * np.log10(max(rms, 1e-9))
    quiet = _ramp(-db, 42.0, 60.0)      # -42 dBFS fine ... -60 dBFS unusable
    hot = _ramp(db, -6.0, -1.5)         # -6 dBFS fine ... -1.5 dBFS on the rail
    return max(quiet, hot), db


def _combine(parts: dict[str, float]) -> float:
    """Noisy-OR over weighted sub-scores.

    Chosen over a mean because faults do not average: a file that is perfect
    except for being fully clipped is not "mostly fine". One severe fault
    carries the score, several mild ones still accumulate, and a plain max
    would throw the second fact away.
    """
    keep = 1.0
    for name, s in parts.items():
        keep *= (1.0 - WEIGHTS.get(name, 0.5) * float(np.clip(s, 0.0, 1.0)))
    return float(np.clip(1.0 - keep, 0.0, 1.0))


class QualityPredictor:
    name = "quality"

    def __call__(self, ctx: AudioContext, lat: Latents) -> None:
        x = np.asarray(ctx.master, dtype=np.float32).ravel()
        sr = int(ctx.master_sr) or 16_000
        if x.size == 0:
            lat.degradation_score = 1.0
            lat.quality_detail = {"empty": 1.0}
            lat.notes.append("quality: empty audio")
            return

        speech = timeline(ctx).speech      # speech structure only, never noise

        # --- one STFT, reused by every spectral sub-score ------------------
        nfft = int(min(NFFT, 1 << int(np.floor(np.log2(max(x.size, 256))))))
        nfft = max(nfft, 256)
        noverlap = nfft - HOP if nfft > HOP else nfft // 2
        freqs, _, Z = signal.stft(x, fs=sr, nperseg=nfft, noverlap=noverlap,
                                  padded=False, boundary=None)
        mag = np.abs(Z).astype(np.float32)
        hop_s = (nfft - noverlap) / sr

        # Speech-frame mask from the VAD timeline, mapped onto STFT frames.
        n_frames = mag.shape[1]
        voiced = np.zeros(n_frames, dtype=bool)
        for a, b in speech:
            i, j = int(a / hop_s), min(n_frames, int(np.ceil(b / hop_s)))
            if j > i:
                voiced[i:j] = True
        if not voiced.any():
            voiced[:] = True               # no speech: judge the whole clip
        power = (mag[:, voiced].astype(np.float64) ** 2).mean(axis=1)

        bandwidth, ratio_db = _bandwidth(power, freqs, sr)
        level, level_db = _level(x, sr, speech)

        parts = {
            "clipping": _clipping(x),
            "dropouts": _dropouts(x, sr, speech),
            "level": level,
            "bandwidth": bandwidth,
        }
        lat.degradation_score = round(_combine(parts), 4)
        lat.quality_detail = {
            **{k: round(v, 4) for k, v in parts.items()},
            "hf_lf_db": round(ratio_db, 2),
            "speech_dbfs": round(level_db, 2),
            "native_sr": float(sr),
        }
