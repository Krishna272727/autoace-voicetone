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

    The two formats usually sit in DIFFERENT revisions of the same repo, not in
    one snapshot directory: several of these repos publish only `.bin` on
    `main`, and transformers then follows the safetensors-conversion bot's pull
    request. So the match has to be per repo, not per snapshot.

    Deletes the blob, not just the symlink -- the cache stores content in
    `blobs/` and snapshots are links into it, so unlinking the visible name
    frees nothing.

    Off by default, and worth knowing why: this assumes transformers will keep
    choosing the safetensors revision. If a repo later removes it, the pruned
    cache can no longer load offline and the image must be rebuilt.
    """
    import os

    hf = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    freed = 0
    for repo in sorted((hf / "hub").glob("models--*")):
        snaps = list(repo.glob("snapshots/*"))
        if not any((s / "model.safetensors").exists() for s in snaps):
            continue                      # only .bin available; it is load-bearing
        for s in snaps:
            bin_ = s / "pytorch_model.bin"
            if not bin_.exists():
                continue
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
