"""Per-field scorer against a labels.csv manifest.

Scores each output field independently, because the spec says emotional tone,
background-noise detection and technical audio quality are graded separately.

Reports exact accuracy plus, for ordinal fields, adjacent accuracy (off by one
class). Tone boundaries such as frustrated/upset are genuinely fuzzy, so the
gap between exact and adjacent tells you whether errors are near-misses or
real confusions.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .noise_vocab import match as noise_match
from .noise_vocab import normalize as noise_normalize
from .schema import (FIELDS, INTENSITY_ORDER, QUALITY_ORDER, SEVERITY_ORDER,
                     TONE_ORDER)

ORDINAL = {
    "emotional_tone": TONE_ORDER,
    "emotional_intensity": INTENSITY_ORDER,
    "background_noise_severity": SEVERITY_ORDER,
    "audio_quality": QUALITY_ORDER,
}
BOOLEAN = ["background_noise_present", "speaker_overlap_present",
           "long_silence_present"]
FREETEXT = ["background_noise_type"]
SKIP = ["confidence"]  # scored via calibration, not accuracy


def load_labels(csv_path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("name") or "").strip()
            raw = (row.get("result_json") or "").strip()
            if not name or not raw:
                continue
            try:
                out[name] = json.loads(raw)
            except json.JSONDecodeError:
                continue
    return out


@dataclass
class FieldScore:
    field: str
    n: int = 0
    exact: int = 0
    adjacent: int = 0
    partial: float = 0.0
    confusion: Counter = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.confusion is None:
            self.confusion = Counter()

    @property
    def acc(self) -> float:
        return self.exact / self.n if self.n else 0.0

    @property
    def adj_acc(self) -> float:
        return self.adjacent / self.n if self.n else 0.0

    @property
    def partial_acc(self) -> float:
        return self.partial / self.n if self.n else 0.0


def _macro_f1(pairs: list[tuple[Any, Any]]) -> float:
    """Macro F1 over classes present in truth or prediction."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    classes = set()
    for t, p in pairs:
        classes.add(t); classes.add(p)
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    if not classes:
        return 0.0
    total = 0.0
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        total += 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return total / len(classes)


def score(predictions: dict[str, dict[str, Any]],
          labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, FieldScore] = {f: FieldScore(f) for f in FIELDS
                                     if f not in SKIP}
    pairs: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
    matched = sorted(set(predictions) & set(labels))

    for name in matched:
        pred, truth = predictions[name], labels[name]
        for f in scores:
            if f not in truth:
                continue
            t, p = truth[f], pred.get(f)
            if f in FREETEXT:
                # shared vocabulary layer: "TV" == "television",
                # "sharp static" == "static". Partial credit for near-synonyms.
                sim = noise_match(str(t or ""), str(p or ""))
                hit = sim >= 1.0
                t, p = noise_normalize(str(t or "")), noise_normalize(str(p or ""))
                s_ = scores[f]
                s_.partial += sim
            else:
                hit = t == p
            s = scores[f]
            s.n += 1
            s.exact += int(hit)
            pairs[f].append((t, p))
            if f in ORDINAL:
                order = ORDINAL[f]
                try:
                    s.adjacent += int(abs(order.index(t) - order.index(p)) <= 1)
                except ValueError:
                    pass
            elif hit:
                s.adjacent += 1
            if not hit:
                s.confusion[(t, p)] += 1

    report = {
        "n_scored": len(matched),
        "missing_from_predictions": sorted(set(labels) - set(predictions)),
        "unlabelled_predictions": sorted(set(predictions) - set(labels)),
        "fields": {},
    }
    for f, s in scores.items():
        report["fields"][f] = {
            "n": s.n,
            "partial_credit": round(s.partial_acc, 4) if f in FREETEXT else None,
            "accuracy": round(s.acc, 4),
            "adjacent_accuracy": round(s.adj_acc, 4),
            "macro_f1": round(_macro_f1(pairs[f]), 4) if pairs[f] else 0.0,
            "errors": [{"truth": t, "pred": p, "count": c}
                       for (t, p), c in s.confusion.most_common()],
        }
    graded = [v["accuracy"] for v in report["fields"].values() if v["n"]]
    report["mean_field_accuracy"] = round(sum(graded) / len(graded), 4) if graded else 0.0
    return report


def format_report(rep: dict[str, Any]) -> str:
    lines = [f"scored {rep['n_scored']} file(s)   "
             f"mean field accuracy = {rep['mean_field_accuracy']:.1%}", ""]
    lines.append(f"{'field':<28}{'n':>3}{'acc':>9}{'adj':>8}{'macroF1':>9}")
    lines.append("-" * 57)
    for f, v in rep["fields"].items():
        if not v["n"]:
            continue
        lines.append(f"{f:<28}{v['n']:>3}{v['accuracy']:>9.1%}"
                     f"{v['adjacent_accuracy']:>8.1%}{v['macro_f1']:>9.3f}")
    errs = [(f, e) for f, v in rep["fields"].items() for e in v["errors"]]
    if errs:
        lines += ["", "errors:"]
        for f, e in errs:
            lines.append(f"  {f}: expected {e['truth']!r} -> got {e['pred']!r}"
                         + (f" (x{e['count']})" if e["count"] > 1 else ""))
    if rep["missing_from_predictions"]:
        lines.append(f"\nin labels but not predicted: {rep['missing_from_predictions']}")
    return "\n".join(lines)
