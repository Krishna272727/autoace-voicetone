"""Batch runner.

    python cli.py samples/ --labels samples/labels.csv --out results

`--stack` selects which predictors run, which is how the ablations in
EXPERIMENTS.md are produced:

    python cli.py samples/ --labels samples/labels.csv --stack none
    python cli.py samples/ --labels samples/labels.csv --stack vad,quality,noise
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from voicetone import FileResult, analyze_file
from voicetone.audio import SUPPORTED
from voicetone.score import format_report, load_labels, score
from voicetone.stack import build_predictors, stack_names


def main() -> int:
    ap = argparse.ArgumentParser(description="AutoAce voice tone batch runner")
    ap.add_argument("folder", help="folder of audio files (searched recursively)")
    ap.add_argument("--labels", default=None, help="labels.csv manifest")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--stack", default=None,
                    help='"all", "none", or e.g. "vad,quality,noise"')
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel files; 1 keeps per-stage timings honest")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    folder = Path(a.folder)
    if not folder.is_dir():
        print(f"not a folder: {folder}", file=sys.stderr)
        return 1
    files = sorted(p for p in folder.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"no supported audio in {folder}", file=sys.stderr)
        return 1

    # Manifest validation, both directions, before any processing starts.
    labels = {}
    if a.labels:
        labels = load_labels(a.labels)
        names = {p.name for p in files}
        for miss in sorted(set(labels) - names):
            print(f"WARN manifest row has no audio file: {miss}")
        for extra in sorted(names - set(labels)):
            print(f"WARN audio file has no manifest row: {extra}")

    predictors = list(build_predictors(a.stack))
    print(f"stack: {', '.join(stack_names(a.stack))}")

    # Models load on first use; warm them once so the first file is not charged
    # for the whole download and the RTF numbers mean something.
    t_warm = time.perf_counter()
    _warm(predictors, files[0])
    print(f"warm-up (model load, excluded from RTF): "
          f"{time.perf_counter() - t_warm:.2f}s\n")

    counter = {"n": 0}

    def run_one(path: Path) -> FileResult:
        r = analyze_file(path, predictors)
        counter["n"] += 1
        tag = "ok " if r.status == "ok" else "FAIL"
        rtf = f"RTF={r.rtf:.3f}" if r.rtf else ""
        print(f"[{counter['n']}/{len(files)}] {tag} {r.name:<26} "
              f"{r.elapsed_s:6.2f}s {rtf}")
        for w in r.warnings:
            print(f"         ! {w}")
        if r.error:
            print(f"         ! {r.error}")
        return r

    t0 = time.perf_counter()
    if a.workers > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            results = list(pool.map(run_one, files))
    else:
        results = [run_one(p) for p in files]
    wall = time.perf_counter() - t0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    preds = {r.name: r.result for r in results if r.status == "ok" and r.result}

    with open(out / "results.json", "w", encoding="utf-8") as fh:
        json.dump([{"name": r.name, "status": r.status, "error": r.error,
                    "elapsed_s": round(r.elapsed_s, 3),
                    "audio_s": round(r.audio_s, 2),
                    "result": r.result, "latents": r.latents,
                    "stage_times": {k: round(v, 4) for k, v in r.stage_times.items()},
                    "warnings": r.warnings}
                   for r in results], fh, indent=2, ensure_ascii=False)
    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "status", "result_json", "error"])
        for r in results:
            w.writerow([r.name, r.status,
                        json.dumps(r.result, ensure_ascii=False) if r.result else "",
                        r.error or ""])

    audio_s = sum(r.audio_s for r in results)
    ok = sum(r.status == "ok" for r in results)
    print(f"\n{ok}/{len(results)} ok | audio {audio_s / 60:.2f} min | "
          f"wall {wall:.2f}s | aggregate RTF "
          f"{wall / audio_s:.4f}" if audio_s else "")
    _stage_table(results, audio_s)
    print(f"wrote {out / 'results.json'} and {out / 'results.csv'}")

    if labels:
        print("\n" + format_report(score(preds, labels)))
    return 0


def _warm(predictors, sample: Path) -> None:
    """Touch every model before timing starts."""
    from voicetone.models import warm
    needed = {getattr(p, "name", "") for p in predictors}
    which = []
    if "vad" in needed:
        which.append("vad")
    if "noise" in needed:
        which.append("tagger")
    if "emotion" in needed:
        which += ["ser", "text", "asr"]
    if not which:
        return
    for key, state in warm(tuple(which)).items():
        if state != "ok":
            print(f"WARN model {key}: {state}")


def _stage_table(results: list[FileResult], audio_s: float) -> None:
    """Per-stage RTF. The memo needs this table and guesses are not acceptable."""
    if not audio_s:
        return
    totals: dict[str, float] = {}
    for r in results:
        for k, v in r.stage_times.items():
            totals[k] = totals.get(k, 0.0) + v
    if not totals:
        return
    print(f"\n{'stage':<14}{'total s':>9}{'RTF':>9}")
    print("-" * 32)
    for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"{k:<14}{v:>9.2f}{v / audio_s:>9.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
