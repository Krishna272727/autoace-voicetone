"""Phase 1 -- voice activity detection.

Writes `speech_ratio` and `max_silence_s`, and caches the speech timeline in
`ctx.cache["vad"]` because Phases 3, 4, 5 and 6 all read it. It is the single
most reused computation in the pipeline, so it is computed once.

**Why a model and not an energy gate.** A loud television crosses any energy
threshold you pick, and this system has to tell "customer is talking" apart from
"customer's TV is talking" -- that is precisely the confusion the brief warns
about. Silero VAD is trained to separate speech from noise specifically, which
an RMS threshold cannot do at any setting. It is also 2 MB, MIT-licensed, ONNX,
and needs no gate acceptance.

Long files are streamed window by window, so a 60-minute call costs the same
memory as a 30-second one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..audio import AudioContext
from ..latent import Latents
from ..models import vad_session

WINDOW = 512                # hop: one probability per 512 samples at 16 kHz
SR = 16_000
STATE_SHAPE = (2, 1, 128)

# The ONNX graph expects the caller to prepend the previous window's last 64
# samples, so each call sees 576 samples. This is not optional and not
# documented in the input shape (which is just [None, None]): feeding a bare
# 512-sample window runs without error and returns near-zero probability for
# every frame, which reads as "no speech anywhere" rather than as a failure.
# Verified empirically -- 512 gives max p=0.12 on a clip that is mostly speech,
# 576 with context gives max p=1.00.
CONTEXT = 64

# Hysteresis: enter speech at the high threshold, leave at the low one. A single
# threshold flickers on and off through every consonant, which would shatter one
# pause into a dozen and make `max_silence_s` meaningless.
ENTER = 0.50
EXIT = 0.35

# A pause shorter than this is inside a word, not between turns.
MIN_SILENCE_S = 0.20
# Speech shorter than this is a click or a door, not a turn.
MIN_SPEECH_S = 0.12
# Speech regions are padded outwards slightly: Silero trims breath and low-energy
# onsets that belong to the speech, and we do not want them counted as noise.
PAD_S = 0.06


@dataclass
class VADTimeline:
    """The shared speech timeline. Times are seconds from the start of the clip."""
    speech: list[tuple[float, float]] = field(default_factory=list)
    silence: list[tuple[float, float]] = field(default_factory=list)
    probs: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    hop_s: float = WINDOW / SR
    duration: float = 0.0

    @property
    def speech_s(self) -> float:
        return float(sum(b - a for a, b in self.speech))

    @property
    def speech_ratio(self) -> float:
        return self.speech_s / self.duration if self.duration > 0 else 0.0

    @property
    def max_silence_s(self) -> float:
        return max((b - a for a, b in self.silence), default=0.0)

    def mask(self, sr: int, n: int) -> np.ndarray:
        """Boolean sample mask at an arbitrary rate, for slicing audio."""
        m = np.zeros(n, dtype=bool)
        for a, b in self.speech:
            i, j = int(a * sr), min(n, int(np.ceil(b * sr)))
            if j > i:
                m[i:j] = True
        return m


def _probs(audio: np.ndarray, session) -> np.ndarray:
    """Per-window speech probability. Streams so memory stays flat."""
    n_win = len(audio) // WINDOW
    if n_win == 0:
        return np.zeros(0, dtype=np.float32)
    state = np.zeros(STATE_SHAPE, dtype=np.float32)
    sr = np.array(SR, dtype=np.int64)
    out = np.empty(n_win, dtype=np.float32)
    ctx_buf = np.zeros(CONTEXT, dtype=np.float32)   # zeros for the first window
    frame = np.empty((1, CONTEXT + WINDOW), dtype=np.float32)
    for i in range(n_win):
        win = audio[i * WINDOW:(i + 1) * WINDOW]
        frame[0, :CONTEXT] = ctx_buf
        frame[0, CONTEXT:] = win
        prob, state = session.run(None, {"input": frame, "state": state, "sr": sr})
        out[i] = float(prob[0, 0])
        ctx_buf = win[-CONTEXT:]
    return out


def _regions(probs: np.ndarray, hop: float, duration: float) -> list[tuple[float, float]]:
    """Hysteresis thresholding, then merge and drop runs that are too short."""
    speech: list[tuple[float, float]] = []
    inside = False
    start = 0.0
    for i, p in enumerate(probs):
        t = i * hop
        if not inside and p >= ENTER:
            inside, start = True, t
        elif inside and p < EXIT:
            speech.append((start, t + hop))
            inside = False
    if inside:
        speech.append((start, duration))

    # Pad, clamp, then merge anything that now touches or is too close together.
    padded = [(max(0.0, a - PAD_S), min(duration, b + PAD_S)) for a, b in speech]
    merged: list[list[float]] = []
    for a, b in padded:
        if merged and a - merged[-1][1] < MIN_SILENCE_S:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= MIN_SPEECH_S]


def _gaps(speech: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """Non-speech runs, including the head and tail of the clip. Leading and
    trailing dead air counts: a call that opens with 20 s of ringback has a long
    silence whether or not anyone eventually speaks."""
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for a, b in speech:
        if a - cursor > 1e-6:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if duration - cursor > 1e-6:
        gaps.append((cursor, duration))
    return gaps


def timeline(ctx: AudioContext) -> VADTimeline:
    """Compute (or return the cached) speech timeline for this file."""
    cached = ctx.cache.get("vad")
    if isinstance(cached, VADTimeline):
        return cached

    audio = np.asarray(ctx.speech, dtype=np.float32).ravel()
    duration = ctx.duration or (len(audio) / SR)
    tl = VADTimeline(duration=float(duration))

    if len(audio) < WINDOW:
        # Shorter than a single VAD window (a sub-32 ms clip). No speech can be
        # established, so the whole clip reads as silence rather than crashing.
        tl.silence = [(0.0, tl.duration)] if tl.duration > 0 else []
        ctx.cache["vad"] = tl
        return tl

    tl.probs = _probs(audio, vad_session())
    tl.speech = _regions(tl.probs, WINDOW / SR, tl.duration)
    tl.silence = _gaps(tl.speech, tl.duration)
    ctx.cache["vad"] = tl
    return tl


class VADPredictor:
    name = "vad"

    def __call__(self, ctx: AudioContext, lat: Latents) -> None:
        tl = timeline(ctx)
        lat.speech_ratio = round(tl.speech_ratio, 5)
        lat.max_silence_s = round(tl.max_silence_s, 3)
        if not tl.speech:
            # Voicemail beep, hold music, dead air, DTMF: valid input with no
            # speech. Say so in a note so the dashboard can explain the low
            # confidence rather than looking broken.
            lat.notes.append("vad: no speech detected in this clip")
