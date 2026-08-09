"""Stage 2: derive the discrete schema from continuous latents.

ALL thresholds live here, in one dataclass. With only three labelled calls we
cannot train a classifier, but we can fit ~10 thresholds -- so this file is
the actual "model", and calibrating it is calibrating the system.

Design rule: fields that share a latent parent are derived from that ONE
value, so they are mutually consistent by construction rather than by
post-hoc repair.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path

from .latent import Latents
from .schema import CallAnalysis

# config/thresholds.yaml key -> Thresholds attribute.
_YAML_MAP: dict[tuple[str, str], str] = {
    ("noise", "present_db"): "noise_present_db",
    ("noise", "low_db"): "noise_low_db",
    ("noise", "medium_db"): "noise_medium_db",
    ("noise", "high_db"): "noise_high_db",
    ("noise", "type_min_score"): "noise_type_min_score",
    ("quality", "slight"): "quality_slight",
    ("quality", "severe"): "quality_severe",
    ("silence", "long_silence_s"): "long_silence_s",
    ("overlap", "ratio_min"): "overlap_ratio_min",
    ("emotion", "arousal_low"): "arousal_low",
    ("emotion", "arousal_high"): "arousal_high",
    ("emotion", "valence_neutral_band"): "valence_neutral_band",
    ("emotion", "valence_upset"): "valence_upset",
    ("emotion", "distress_arousal"): "distress_arousal",
    ("emotion", "distress_valence"): "distress_valence",
    ("confidence", "base"): "conf_base",
    ("confidence", "missing_penalty"): "conf_missing_penalty",
    ("confidence", "role_weight"): "conf_role_weight",
    ("confidence", "floor"): "conf_floor",
}

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"


@dataclass
class Thresholds:
    # --- background noise: both outputs read noise_level_db --------------
    # Values are noise level in dB relative to speech level (less negative =
    # louder noise). Anchors from the provided calls: call_001 (no noise)
    # measured ~-49 dB; calls 002/003 (medium noise) measured ~-33 dB.
    noise_present_db: float = -42.0
    noise_low_db: float = -42.0
    noise_medium_db: float = -36.0
    noise_high_db: float = -22.0
    noise_type_min_score: float = 0.15      # tagger confidence floor

    # --- audio quality: reads degradation_score ONLY ---------------------
    quality_slight: float = 0.25
    quality_severe: float = 0.55

    # --- silence ---------------------------------------------------------
    # call_003 has a ~7.3s low-energy run but is labelled false, so the
    # threshold sits above that. Conversational pauses are not dead air.
    long_silence_s: float = 10.0

    # --- overlap ---------------------------------------------------------
    overlap_ratio_min: float = 0.02         # ~2% of speech time

    # --- emotion: tone and intensity both read (valence, arousal) --------
    arousal_low: float = 0.20
    arousal_high: float = 0.55
    valence_neutral_band: float = 0.20      # |valence| below this = neutral
    valence_upset: float = -0.45
    distress_arousal: float = 0.80
    distress_valence: float = -0.60

    # --- confidence ------------------------------------------------------
    conf_base: float = 0.85
    conf_missing_penalty: float = 0.09      # per missing core latent
    conf_role_weight: float = 0.20
    conf_floor: float = 0.10


    def to_yaml_dict(self) -> dict:
        """Inverse of load_thresholds -- used by scripts/tune_thresholds.py."""
        out: dict[str, dict[str, float]] = {}
        for (section, key), attr in _YAML_MAP.items():
            out.setdefault(section, {})[key] = getattr(self, attr)
        return out


@functools.lru_cache(maxsize=8)
def load_thresholds(path: str | Path | None = None) -> Thresholds:
    """Load thresholds from YAML. Missing file or key falls back to the
    dataclass default, so the system runs with no config present at all."""
    path = Path(path) if path else DEFAULT_CONFIG
    t = Thresholds()
    try:
        import yaml
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception:                              # noqa: BLE001
        return t                                   # defaults are always valid
    known = {f.name for f in dc_fields(Thresholds)}
    kwargs: dict[str, float] = {}
    for (section, key), attr in _YAML_MAP.items():
        val = (raw.get(section) or {}).get(key)
        if val is not None and attr in known:
            try:
                kwargs[attr] = float(val)
            except (TypeError, ValueError):
                pass
    return Thresholds(**kwargs)


def _bin(value: float, edges: list[float], labels: list[str]) -> str:
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def _margin(value: float, edges: list[float], scale: float) -> float:
    """Distance to the nearest decision boundary, normalised to 0..1.
    Small margin = the call was close = lower confidence."""
    if not edges:
        return 1.0
    d = min(abs(value - e) for e in edges)
    return max(0.0, min(1.0, d / scale))


def derive(lat: Latents, t: Thresholds | None = None) -> CallAnalysis:
    t = t or load_thresholds()
    out: dict = {}
    margins: list[float] = []

    # ---------------- background noise (shared parent) -------------------
    if lat.noise_level_db is not None:
        lvl = lat.noise_level_db
        present = lvl >= t.noise_present_db
        out["background_noise_present"] = present
        if present:
            out["background_noise_severity"] = _bin(
                lvl, [t.noise_medium_db, t.noise_high_db],
                ["low", "medium", "high"])
            # Never emit "unspecified" while noise is present: it scores zero
            # against any real label, whereas a wrong guess from the right
            # family earns partial credit (BUILD_SPEC 5, Phase 3). So the
            # argmax is emitted regardless of its score; a score below
            # type_min_score costs confidence instead of the answer.
            best, score = "", 0.0
            if lat.noise_class_dist:
                # deterministic tie-break: highest score, then alphabetical
                best, score = max(lat.noise_class_dist.items(),
                                  key=lambda kv: (kv[1], [-ord(c) for c in kv[0]]))
            out["background_noise_type"] = best or "static"
            if score < t.noise_type_min_score:
                margins.append(0.0)
        else:
            out["background_noise_severity"] = "none"
            out["background_noise_type"] = ""
        margins.append(_margin(lvl, [t.noise_present_db], 6.0))

    # ---------------- audio quality (independent by design) --------------
    # Deliberately does not read noise_level_db. The provided labels contain
    # "sharp static" with audio_quality=clear; coupling these would fail it.
    if lat.degradation_score is not None:
        d = lat.degradation_score
        out["audio_quality"] = _bin(d, [t.quality_slight, t.quality_severe],
                                    ["clear", "slightly_impaired",
                                     "severely_impaired"])
        margins.append(_margin(d, [t.quality_slight, t.quality_severe], 0.15))

    # ---------------- silence & overlap ----------------------------------
    if lat.max_silence_s is not None:
        out["long_silence_present"] = lat.max_silence_s >= t.long_silence_s
        margins.append(_margin(lat.max_silence_s, [t.long_silence_s], 4.0))

    if lat.overlap_ratio is not None:
        out["speaker_overlap_present"] = lat.overlap_ratio >= t.overlap_ratio_min
        margins.append(_margin(lat.overlap_ratio, [t.overlap_ratio_min], 0.03))

    # ---------------- emotion (shared parent) ----------------------------
    if lat.arousal is not None and lat.valence is not None:
        a, v = lat.arousal, lat.valence
        if a >= t.distress_arousal and v <= t.distress_valence:
            tone = "distressed"
        elif abs(v) < t.valence_neutral_band and a < t.arousal_high:
            tone = "neutral"
        elif v > 0:
            tone = "satisfied"
        elif v <= t.valence_upset or a >= t.arousal_high:
            tone = "upset"
        else:
            tone = "frustrated"
        out["emotional_tone"] = tone
        out["emotional_intensity"] = _bin(a, [t.arousal_low, t.arousal_high],
                                          ["low", "medium", "high"])
        margins.append(_margin(v, [0.0, t.valence_upset], 0.3))
        margins.append(_margin(a, [t.arousal_low, t.arousal_high], 0.2))

    # ---------------- confidence -----------------------------------------
    conf = t.conf_base
    # Penalise stubbed latents exactly like absent ones -- a fallback constant
    # is a real float but carries no evidence.
    conf -= t.conf_missing_penalty * len(lat.unevidenced())
    if margins:
        conf *= (0.65 + 0.35 * (sum(margins) / len(margins)))
    if lat.role_confidence is not None:
        conf *= (1.0 - t.conf_role_weight * (1.0 - lat.role_confidence))
    out["confidence"] = max(t.conf_floor, min(1.0, conf))

    return CallAnalysis(**out)
