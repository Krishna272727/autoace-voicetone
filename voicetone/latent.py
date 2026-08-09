"""Stage 1 output: continuous latent estimates.

Predictors write here, NOT into the schema. Discrete outputs are derived from
these in derive.py, so fields that share a parent cannot contradict each other.

Every field is Optional. None means "not estimated" (predictor absent or
failed), which derive.py treats as unknown and reflects in confidence --
distinct from a confident estimate of zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Latents:
    # --- speech structure ------------------------------------------------
    speech_ratio: float | None = None        # fraction of clip containing speech
    max_silence_s: float | None = None       # longest contiguous non-speech run
    overlap_ratio: float | None = None       # fraction with >=2 active speakers
    n_speakers: int | None = None

    # --- background noise (single parent for present + severity) ---------
    noise_level_db: float | None = None      # noise floor relative to speech
    noise_class_dist: dict[str, float] = field(default_factory=dict)

    # --- technical quality (MUST NOT read any noise_* field) -------------
    degradation_score: float | None = None   # 0.0 pristine .. 1.0 unusable
    quality_detail: dict[str, float] = field(default_factory=dict)

    # --- speaker roles ---------------------------------------------------
    role_confidence: float | None = None     # certainty that customer_mask is right
    customer_speech_s: float | None = None

    # --- emotion (single parent for tone + intensity) --------------------
    arousal: float | None = None             # -1 calm .. +1 activated
    valence: float | None = None             # -1 negative .. +1 positive
    valence_source: str = "none"             # none | acoustic | lexical | fused

    notes: list[str] = field(default_factory=list)

    # Latents filled by a fallback rather than measured. A stubbed value is a
    # real float, so `missing()` cannot see it -- but it carries no evidence,
    # so it must cost the same confidence as an absent one. Without this the
    # baseline would report high confidence in numbers it invented.
    stubbed: list[str] = field(default_factory=list)

    CORE = ("max_silence_s", "overlap_ratio", "noise_level_db",
            "degradation_score", "arousal", "valence")

    def missing(self) -> list[str]:
        """Core latents no predictor supplied. Drives the confidence penalty."""
        return [k for k in self.CORE if getattr(self, k) is None]

    def unevidenced(self) -> list[str]:
        """Core latents that are either absent or stubbed. This, not
        `missing()`, is what confidence should be penalised on."""
        stub = set(self.stubbed)
        return [k for k in self.CORE
                if getattr(self, k) is None or k in stub]

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
