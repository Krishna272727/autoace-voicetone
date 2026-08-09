"""Fit thresholds on the synthetic corpus with grouped cross-validation
(BUILD_SPEC.md 10).

    python scripts/tune_thresholds.py --corpus synthetic --write

Protocol, in order:

1. Load the **synthetic** corpus, never the three real calls.
2. Cache each file's latents once, so the search is over `derive()` only and
   costs seconds rather than re-running the models per candidate.
3. Grid-search each threshold group against the field it controls, maximising
   **macro F1** -- not accuracy, because the classes are heavily imbalanced and
   accuracy would happily discard the rare ones.
4. **Grouped** CV: fold assignment is by `speech_source` + `noise_source` from
   `manifest.csv`, so the same voice and the same noise recording never appear
   in both the fitting and the scoring half. Without this, a threshold that
   memorises one noise clip's spectrum scores well and generalises to nothing.
5. Re-check the three real calls *afterwards*, as a sanity check only.

Why this exists: with thresholds anchored to the three provided files, the
synthetic corpus scored `none` -> `medium` on 13 of 40 clips. Two anchor points
cannot define four severity bands, and the samples never contain `low` or
`high` at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicetone import Latents, Thresholds, analyze_file, derive  # noqa: E402
from voicetone.score import load_labels  # noqa: E402
from voicetone.stack import build_predictors  # noqa: E402

# Which thresholds to search, and over what range, for each field.
SEARCH = {
    "background_noise_severity": {
        "fields": ["background_noise_present", "background_noise_severity"],
        "params": {
            "noise_present_db": [-70 + i for i in range(0, 55, 2)],
            "noise_medium_db": [-60 + i for i in range(0, 50, 2)],
            "noise_high_db": [-40 + i for i in range(0, 35, 2)],
        },
    },
    "audio_quality": {
        "fields": ["audio_quality"],
        "params": {
            "quality_slight": [i / 40 for i in range(2, 26)],
            "quality_severe": [i / 40 for i in range(10, 38)],
        },
    },
    "long_silence_present": {
        "fields": ["long_silence_present"],
        "params": {"long_silence_s": [4 + i for i in range(0, 22)]},
    },
}


def macro_f1(pairs: list[tuple]) -> float:
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    classes = set()
    for t, p in pairs:
        classes |= {t, p}
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    if not classes:
        return 0.0
    total = 0.0
    for c in classes:
        pr = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        total += 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
    return total / len(classes)


def score_with(cache: list[tuple[Latents, dict, str]], t: Thresholds,
               fields: list[str], folds: set[str] | None = None) -> float:
    pairs: dict[str, list[tuple]] = defaultdict(list)
    for lat, truth, group in cache:
        if folds is not None and group not in folds:
            continue
        out = derive(lat, t).to_flat()
        for f in fields:
            if f in truth:
                pairs[f].append((truth[f], out.get(f)))
    if not pairs:
        return 0.0
    return sum(macro_f1(v) for v in pairs.values()) / len(pairs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="synthetic")
    ap.add_argument("--stack", default="vad,quality,noise")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--write", action="store_true",
                    help="write the tuned values back to config/thresholds.yaml")
    a = ap.parse_args()

    corpus = Path(a.corpus)
    labels = load_labels(corpus / "labels.csv")
    if not labels:
        print(f"no labels in {corpus}/labels.csv -- run make_synthetic.py first",
              file=sys.stderr)
        return 1

    # Group key: speech source + noise source. Leakage prevention.
    groups: dict[str, str] = {}
    man = corpus / "manifest.csv"
    if man.exists():
        import csv
        with open(man, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                groups[row["name"]] = f"{row['speech_source']}|{row['noise_source']}"

    print(f"extracting latents for {len(labels)} clips (once)...")
    preds = list(build_predictors(a.stack))
    cache: list[tuple[Latents, dict, str]] = []
    for name, truth in labels.items():
        p = corpus / name
        if not p.exists():
            continue
        r = analyze_file(p, preds)
        if r.status != "ok":
            continue
        lat = Latents(**{k: v for k, v in r.latents.items()
                         if k in Latents.__dataclass_fields__})
        cache.append((lat, truth, groups.get(name, name)))
    print(f"  {len(cache)} usable clips, "
          f"{len(set(g for _, _, g in cache))} distinct groups")

    all_groups = sorted({g for _, _, g in cache})
    folds = [set(all_groups[i::a.folds]) for i in range(a.folds)]

    base = Thresholds()
    tuned = replace(base)
    report = {}

    for name, spec in SEARCH.items():
        fields = spec["fields"]
        before = score_with(cache, base, fields)

        # Coordinate descent over this group's parameters, scored by grouped CV.
        current = replace(tuned)
        for _ in range(2):                          # two passes is enough here
            for param, values in spec["params"].items():
                best_v, best_s = getattr(current, param), -1.0
                for v in values:
                    cand = replace(current, **{param: v})
                    # Held-out score: fit on the other folds, score on this one.
                    s = sum(score_with(cache, cand, fields, folds=f)
                            for f in folds) / len(folds)
                    if s > best_s:
                        best_v, best_s = v, s
                current = replace(current, **{param: best_v})

        after = score_with(cache, current, fields)
        tuned = current
        report[name] = {"macro_f1_before": round(before, 4),
                        "macro_f1_after": round(after, 4),
                        "params": {p: getattr(tuned, p) for p in spec["params"]}}
        print(f"\n{name}")
        print(f"  macro F1  {before:.3f} -> {after:.3f}")
        for p in spec["params"]:
            print(f"    {p:<20} {getattr(base, p):>8} -> {getattr(tuned, p):>8}")

    # --- sanity check on the three real calls, AFTER tuning ---------------
    real = Path("samples")
    if (real / "labels.csv").exists():
        print("\nsanity check on the three provided calls (not tuned on):")
        rl = load_labels(real / "labels.csv")
        agree = total = 0
        for name, truth in rl.items():
            p = real / name
            if not p.exists():
                continue
            r = analyze_file(p, preds)
            if r.status != "ok":
                continue
            lat = Latents(**{k: v for k, v in r.latents.items()
                             if k in Latents.__dataclass_fields__})
            out = derive(lat, tuned).to_flat()
            for f in ("background_noise_present", "background_noise_severity",
                      "audio_quality", "long_silence_present"):
                if f in truth:
                    total += 1
                    agree += int(out.get(f) == truth[f])
        print(f"  {agree}/{total} tuned fields still match "
              f"({agree / total:.1%})" if total else "  no overlap")

    print("\n" + json.dumps(report, indent=2))

    if a.write:
        import yaml
        cfg = Path("config/thresholds.yaml")
        raw = yaml.safe_load(cfg.read_text()) or {}
        data = tuned.to_yaml_dict()
        for section, vals in data.items():
            raw.setdefault(section, {}).update(vals)
        header = (f"# Tuned by scripts/tune_thresholds.py on the synthetic\n"
                  f"# corpus with {a.folds}-fold grouped CV "
                  f"(groups = speech source + noise source).\n"
                  f"# Clips: {len(cache)}.  Scores: "
                  f"{json.dumps({k: v['macro_f1_after'] for k, v in report.items()})}\n"
                  f"# The three provided calls were NOT used for fitting.\n\n")
        cfg.write_text(header + yaml.safe_dump(raw, sort_keys=False))
        print(f"\nwrote {cfg}")
    else:
        print("\n(dry run -- pass --write to update config/thresholds.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
