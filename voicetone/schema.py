"""Output schema + deterministic consistency layer.

This is the contract. Every predictor writes into a CallAnalysis; nothing
leaves the pipeline without passing through the validators below.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class EmotionalTone(str, Enum):
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"
    UPSET = "upset"
    DISTRESSED = "distressed"


class Intensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NoiseSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AudioQuality(str, Enum):
    CLEAR = "clear"
    SLIGHTLY_IMPAIRED = "slightly_impaired"
    SEVERELY_IMPAIRED = "severely_impaired"


# Ordinal rankings, used by the scorer for adjacent-class accuracy.
TONE_ORDER = ["satisfied", "neutral", "frustrated", "upset", "distressed"]
INTENSITY_ORDER = ["low", "medium", "high"]
SEVERITY_ORDER = ["none", "low", "medium", "high"]
QUALITY_ORDER = ["clear", "slightly_impaired", "severely_impaired"]


class CallAnalysis(BaseModel):
    emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL
    emotional_intensity: Intensity = Intensity.LOW
    background_noise_present: bool = False
    background_noise_type: str = ""
    background_noise_severity: NoiseSeverity = NoiseSeverity.NONE
    audio_quality: AudioQuality = AudioQuality.CLEAR
    speaker_overlap_present: bool = False
    long_silence_present: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _enforce_consistency(self) -> "CallAnalysis":
        # Noise fields must agree with each other.
        if not self.background_noise_present:
            self.background_noise_type = ""
            self.background_noise_severity = NoiseSeverity.NONE
        else:
            if self.background_noise_severity == NoiseSeverity.NONE:
                self.background_noise_severity = NoiseSeverity.LOW
            if not self.background_noise_type.strip():
                self.background_noise_type = "unspecified"

        # Neutral tone cannot be "high" intensity - by definition there is no
        # strong emotion to be intense about.
        if self.emotional_tone == EmotionalTone.NEUTRAL:
            if self.emotional_intensity == Intensity.HIGH:
                self.emotional_intensity = Intensity.MEDIUM

        # Distressed is an escalated state; it is never "low" intensity.
        if self.emotional_tone == EmotionalTone.DISTRESSED:
            if self.emotional_intensity == Intensity.LOW:
                self.emotional_intensity = Intensity.MEDIUM

        self.background_noise_type = self.background_noise_type.strip().lower()
        return self

    def to_flat(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["confidence"] = round(float(d["confidence"]), 3)
        return d


# NOTE: audio_quality is deliberately NOT coupled to the noise fields.
# The provided labels include a call with "sharp static" noise that is still
# labelled audio_quality=clear. Noise and quality are independent heads.
FIELDS = list(CallAnalysis.model_fields.keys())
