"""Pre-fetch every checkpoint.

Run at Docker build time so the image has no network dependency at runtime and
a cold start cannot time out (BUILD_SPEC.md 4.4).

    python scripts/download_models.py [--check]

`--check` reports what is cached without downloading, and exits non-zero if
anything required is missing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicetone import models  # noqa: E402

REQUIRED = ["vad", "tagger", "ser", "speaker", "text"]
OPTIONAL = ["asr"]          # large; the system degrades without it


def prune_duplicate_weights(dry_run: bool = False) -> int:
    """Delete `pytorch_model.bin` where a `model.safetensors` already exists.

    Several repos publish both formats and transformers ends up with both in
    the cache: measured at 478 + 386 + 361 MB of exact duplicates across the
    sentiment, speaker and SER checkpoints. Nothing loads the `.bin` once the
    safetensors file is present, so it is 1.2 GB of image for no behaviour.

    The match is per SNAPSHOT, and that is the whole correctness argument.

    An earlier version matched per repo: if any revision held a safetensors
    file, the `.bin` was pruned from every revision. That is wrong, and it was
    wrong in production. The two formats usually sit in different revisions --
    several of these repos publish only `.bin` on `main`, and transformers
    follows the safetensors-conversion bot's pull request to find the other.
    Pruning per repo therefore deletes the only weights `main` has, leaving a
    cache that cannot resolve `main` offline at all.

    Transformers does not fail when that happens. It goes to the network,
    re-resolves the PR revision, and loads successfully -- so the container
    silently acquires a runtime dependency on huggingface.co, and on an
    unmerged pull request in someone else's repo. Measured on the deployed
    service: ~10 s of the 60 s warm-up, and a set of outbound requests that
    LICENCES.md states do not happen.

    Matching per snapshot prunes less (only where both formats are genuinely
    duplicated in one revision) and can never remove a revision's last weights.
    The build verifies this rather than assuming it -- see the offline load in
    the Dockerfile, which fails the build if any checkpoint needs the network.

    Deletes the blob, not just the symlink -- the cache stores content in
    `blobs/` and snapshots are links into it, so unlinking the visible name
    frees nothing.
    """
    import os

    hf = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    freed = 0
    for repo in sorted((hf / "hub").glob("models--*")):
        for s in sorted(repo.glob("snapshots/*")):
            bin_ = s / "pytorch_model.bin"
            if not bin_.exists():
                continue
            if not (s / "model.safetensors").exists():
                continue          # this revision's only weights; load-bearing
            blob = bin_.resolve()
            size = blob.stat().st_size if blob.is_file() else 0
            freed += size
            print(f"  {'would prune' if dry_run else 'pruned'} "
                  f"{size / 2**20:7.0f} MB  {repo.name}")
            if not dry_run:
                blob.unlink(missing_ok=True)
                bin_.unlink(missing_ok=True)
    print(f"  {'reclaimable' if dry_run else 'reclaimed'}: {freed / 2**20:.0f} MB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report cache state without downloading")
    ap.add_argument("--skip-asr", action="store_true",
                    help="skip the Whisper download (~500 MB)")
    ap.add_argument("--prune", action="store_true",
                    help="delete .bin weights superseded by .safetensors")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --prune, report what would go without deleting")
    a = ap.parse_args()

    if a.prune:
        return prune_duplicate_weights(dry_run=a.dry_run)

    wanted = list(REQUIRED) + ([] if a.skip_asr else OPTIONAL)
    loaders = {
        "vad": models.vad_session,
        "tagger": models.audioset_tagger,
        "ser": models.ser_model,
        "speaker": models.speaker_embedder,
        "text": models.text_sentiment,
        "asr": models.asr_model,
    }

    if a.check:
        import os
        hf = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
        print(f"HF cache: {hf}  exists={hf.exists()}")
        print(f"VAD file: {models.MODELS_DIR / models.SILERO_VAD_FILE}  "
              f"exists={(models.MODELS_DIR / models.SILERO_VAD_FILE).exists()}")
        return 0

    failures = []
    for key in wanted:
        t = time.perf_counter()
        try:
            loaders[key]()
            print(f"  ok       {key:<8} {time.perf_counter() - t:6.1f}s")
        except Exception as exc:                   # noqa: BLE001
            print(f"  FAILED   {key:<8} {type(exc).__name__}: {exc}")
            if key in REQUIRED:
                failures.append(key)

    if failures:
        print(f"\n{len(failures)} required model(s) unavailable: "
              f"{', '.join(failures)}")
        return 1
    print("\nall models cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
