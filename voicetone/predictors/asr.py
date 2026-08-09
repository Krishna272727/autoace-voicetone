"""Shared transcription with timestamps.

Run once per file, cached on `ctx.cache["asr"]`, and used by two consumers:

  * `roles.py`  -- scripted-agent language is a strong role signal, and the
                   acoustic signals alone were measured putting the bot in the
                   customer cluster (see below);
  * `emotion.py`-- lexical valence, computed from the *customer's* lines only.

Transcription is the most expensive stage in the pipeline (~0.4 RTF for
`small.en` int8 on one core), so it is requested conditionally, not always.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import numpy as np

from ..audio import AudioContext

log = logging.getLogger("autoace.asr")

SR = 16_000
MAX_S = float(os.getenv("VOICETONE_ASR_MAX_S", "300"))


@dataclass
class Utterance:
    start: float
    end: float
    text: str

    @property
    def mid(self) -> float:
        return 0.5 * (self.start + self.end)


# Phrases a scripted agent says and a customer essentially never does. These
# are role markers, not sentiment: "I'm so sorry to hear that" is warm, scripted
# and spoken by the bot, which is precisely why attributing it to the customer
# corrupts the emotion estimate.
AGENT_PATTERNS = [
    r"\bi'?m calling (from|about|on behalf)", r"\bthank you for calling\b",
    r"\bhow (can|may) i help\b", r"\bi can help (you )?with (that|this)\b",
    r"\bis (now|this) a good time\b", r"\blet me transfer\b",
    r"\bplease hold\b", r"\bone moment\b", r"\bbear with me\b",
    r"\bi'?m (an? )?(virtual |ai |automated )?(assistant|agent)\b",
    r"\bthis call (may|is) be(ing)? recorded\b",
    r"\bwhat (type|kind) of (service|appointment|issue)\b",
    r"\bis there anything else\b", r"\bhave a (great|good|nice) day\b",
    r"\bi'?m (sorry|so sorry) to hear\b", r"\bi understand your frustration\b",
    r"\bcan i (get|have) your (name|number|account)\b",
    r"\bi'?m \w+ from\b", r"\bspeaking with\b", r"\bhow can i assist\b",
    r"\byou'?re (very )?welcome\b", r"\bmay i ask\b",
]
_AGENT_RE = [re.compile(p, re.I) for p in AGENT_PATTERNS]

# The customer side: requests, complaints, first-person needs.
CUSTOMER_PATTERNS = [
    r"\bi (want|need|would like)\b", r"\bmy (car|vehicle|account|phone|order)\b",
    r"\bcan you\b", r"\bare you a (real )?person\b", r"\bi'?ve been waiting\b",
    r"\bthis is ridiculous\b", r"\bi'?m (not )?happy\b",
    r"\bwhy (is|are|do|did)\b", r"\bi (called|phoned) (about|because)\b",
]
_CUSTOMER_RE = [re.compile(p, re.I) for p in CUSTOMER_PATTERNS]


def agent_score(text: str) -> float:
    """Net evidence that this text is the scripted agent. Positive = agent."""
    if not text:
        return 0.0
    a = sum(1 for r in _AGENT_RE if r.search(text))
    c = sum(1 for r in _CUSTOMER_RE if r.search(text))
    return float(a - c)


def transcribe(ctx: AudioContext) -> list[Utterance]:
    """Transcribe the whole file with timestamps. Cached; never raises."""
    cached = ctx.cache.get("asr")
    if isinstance(cached, list):
        return cached

    out: list[Utterance] = []
    ctx.cache["asr"] = out            # cache failures too, so we retry once only
    x = np.asarray(ctx.speech, dtype=np.float32).ravel()
    if x.size < SR // 2:
        return out
    try:
        from ..models import asr_model
        model = asr_model()
        segments, _ = model.transcribe(
            x[:int(MAX_S * SR)],
            beam_size=1,              # greedy: deterministic and ~2x faster
            temperature=0.0,          # no sampling fallback -> reproducible
            vad_filter=False,         # our own VAD already gates this
            condition_on_previous_text=False,
        )
        for s in segments:
            txt = (s.text or "").strip()
            if txt:
                out.append(Utterance(float(s.start), float(s.end), txt))
    except Exception as exc:                       # noqa: BLE001
        log.info("ASR unavailable: %s", exc)
    return out


def text_for(utts: list[Utterance], regions: list[tuple[float, float]]) -> str:
    """Join the utterances whose midpoint falls inside one speaker's regions."""
    if not regions:
        return " ".join(u.text for u in utts)
    keep = []
    for u in utts:
        if any(a <= u.mid <= b for a, b in regions):
            keep.append(u.text)
    return " ".join(keep).strip()
