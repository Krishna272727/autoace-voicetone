"""Phase 0 baseline: the honest do-nothing predictor.

It writes a neutral, zero-information estimate for every core latent. That is
deliberately different from writing nothing: writing nothing would leave the
latents at None, which derive.py treats as unknown and answers with schema
defaults plus a large confidence penalty. Writing explicit zeros exercises the
real derivation path end to end, so the harness is measuring the same code the
finished system will use.

It also fixes the reference number every later phase is compared against:
45.8% mean field accuracy on the three provided calls. Anything that does not
beat that is not earning its keep.

Note what is NOT here. An earlier revision hardcoded intensity="medium",
which happened to match 2 of 3 labels and flattered the score to 54.2%.
Deriving intensity honestly from arousal=0 scores 0% on that field and drops
the total to 45.8%. That was an improvement -- a lucky constant was removed
(BUILD_SPEC 3.4).
"""
from __future__ import annotations

from ..audio import AudioContext
from ..latent import Latents

# Comfortably below noise.present_db (-42), i.e. "no audible background".
_QUIET_DB = -60.0


class StubPredictor:
    name = "stub"

    def __call__(self, ctx: AudioContext, lat: Latents) -> None:
        # Only fill what nobody else has estimated, so the stub can be left in
        # the chain as a floor under the real predictors without overwriting
        # them. Registration order therefore does not matter.
        defaults = {
            "speech_ratio": 1.0,
            "max_silence_s": 0.0,
            "overlap_ratio": 0.0,
            "noise_level_db": _QUIET_DB,
            "degradation_score": 0.0,
            "arousal": 0.0,
            "valence": 0.0,
        }
        filled = []
        for key, val in defaults.items():
            if getattr(lat, key) is None:
                setattr(lat, key, val)
                filled.append(key)
        # Record the fakes: these are floats, so missing() cannot see them, and
        # confidence would otherwise be charged nothing for a pure guess.
        lat.stubbed.extend(k for k in filled if k not in lat.stubbed)
        if filled:
            lat.notes.append("stub: neutral defaults for " + ", ".join(filled))
