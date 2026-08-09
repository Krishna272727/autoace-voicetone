"""Build a labelled synthetic corpus (BUILD_SPEC.md 8).

This is the answer to n=3. Roughly half the output space -- `frustrated`,
`distressed`, `low` intensity, `severity: low/high`, both impaired quality
classes, and `long_silence_present: true` -- has **zero** real examples, so
those code paths are otherwise completely unvalidated.

Ground truth here is known **by construction**: the SNR of the mix sets the
noise severity, the inserted gap length sets the silence label, the degradation
applied sets the quality label. Nothing is annotated by hand.

    python scripts/make_synthetic.py --out synthetic --n 120

Writes `synthetic/labels.csv` in the same manifest format as the provided
`labels.csv`, so `cli.py` and the scorer work on it unchanged:

    python cli.py synthetic/ --labels synthetic/labels.csv

LEAKAGE DISCIPLINE
------------------
Every clip records the speech source and the noise source it was built from, in
`synthetic/manifest.csv`. `tune_thresholds.py` groups on those columns so the
same voice or the same noise recording never appears in both the tuning and the
evaluation half.

SPEECH SOURCE
-------------
By default the generator uses the provided calls as speech material, which is
enough to validate the noise, silence and quality heads (they do not care what
is being said). For the *emotion* classes it is not enough, and the honest
statement is that `frustrated` and `distressed` need a real emotion corpus --
RAVDESS or CREMA-D, both licence-checked -- pointed at with `--speech-dir`.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SR = 16_000

# SNR bands -> severity. These are the *ground truth* definitions used to
# generate the corpus; tune_thresholds.py then fits noise_level_db boundaries
# to reproduce them. Chosen from ordinary listening conventions, not from the
# three sample files.
SEVERITY_BY_SNR = [
    (40.0, "none"),      # >= 40 dB SNR: inaudible
    (25.0, "low"),       # 25-40 dB: noticeable in pauses
    (12.0, "medium"),    # 12-25 dB: clearly present under speech
    (-99.0, "high"),     # < 12 dB: intrusive
]

NOISE_KINDS = {
    "static": "anoisesrc=c=white:a={amp}",
    "hum": "sine=frequency=60",
    "road noise": "anoisesrc=c=brown:a={amp}",
    "music": "sine=frequency=440",
}


def severity_for(snr_db: float) -> str:
    for lo, name in SEVERITY_BY_SNR:
        if snr_db >= lo:
            return name
    return "high"


def _ff(args: list[str]) -> bool:
    return subprocess.run(["ffmpeg", "-y", "-v", "error", *args],
                          capture_output=True).returncode == 0


def _make_noise(kind: str, seconds: float, out: Path, rng: random.Random) -> bool:
    src = NOISE_KINDS[kind].format(amp=0.5)
    return _ff(["-f", "lavfi", "-i", f"{src}:r={SR}:d={seconds}",
                "-ac", "1", "-ar", str(SR), str(out)])


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale the noise so the mixture has exactly the requested SNR."""
    if noise.size < speech.size:
        noise = np.tile(noise, int(np.ceil(speech.size / max(noise.size, 1))))
    noise = noise[:speech.size]
    ps = float(np.mean(speech.astype(np.float64) ** 2)) or 1e-12
    pn = float(np.mean(noise.astype(np.float64) ** 2)) or 1e-12
    gain = np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    out = speech + gain * noise
    peak = float(np.abs(out).max())
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out.astype(np.float32)


def _degrade(path: Path, out: Path, kind: str) -> bool:
    """Apply one graded degradation. The kind fixes the expected quality class."""
    if kind == "clean":
        return _ff(["-i", str(path), "-ac", "1", str(out)])
    if kind == "telephony":                         # -> slightly_impaired
        return _ff(["-i", str(path), "-ac", "1", "-ar", "8000",
                    "-c:a", "pcm_mulaw", str(out)])
    if kind == "clipped":                           # -> severely_impaired
        return _ff(["-i", str(path), "-ac", "1", "-af", "volume=24dB",
                    "-c:a", "pcm_s16le", str(out)])
    if kind == "quiet":                             # -> severely_impaired
        return _ff(["-i", str(path), "-ac", "1", "-af", "volume=-46dB", str(out)])
    if kind == "narrowband":                        # -> severely_impaired
        return _ff(["-i", str(path), "-ac", "1", "-ar", "8000",
                    "-af", "lowpass=f=1500", str(out)])
    return False


QUALITY_BY_DEGRADATION = {
    "clean": "clear", "telephony": "slightly_impaired",
    "clipped": "severely_impaired", "quiet": "severely_impaired",
    "narrowband": "severely_impaired",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--speech-dir", default="samples",
                    help="folder of speech source audio")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)

    speech_files = sorted(p for p in Path(a.speech_dir).rglob("*")
                          if p.suffix.lower() in
                          {".wav", ".ogg", ".mp3", ".flac", ".m4a"})
    if not speech_files:
        print(f"no speech source audio in {a.speech_dir}", file=sys.stderr)
        return 1

    rows, manifest = [], []
    for i in range(a.n):
        src = speech_files[i % len(speech_files)]
        clean = tmp / f"clean_{i}.wav"
        if not _ff(["-i", str(src), "-ac", "1", "-ar", str(SR), "-t", "20",
                    str(clean)]):
            continue
        x, _ = sf.read(clean, dtype="float32")
        if x.size < SR:
            continue

        # --- noise at a controlled SNR ---------------------------------
        want_noise = rng.random() > 0.25
        kind = rng.choice(list(NOISE_KINDS)) if want_noise else ""
        snr = rng.uniform(5.0, 45.0) if want_noise else 99.0
        if want_noise:
            npath = tmp / f"noise_{i}.wav"
            if not _make_noise(kind, x.size / SR + 1, npath, rng):
                continue
            n, _ = sf.read(npath, dtype="float32")
            y = _mix_at_snr(x, n, snr)
        else:
            y = x
        severity = severity_for(snr)
        present = severity != "none"

        # --- inserted dead air ------------------------------------------
        gap_s = 0.0
        if rng.random() > 0.7:
            gap_s = rng.choice([3.0, 6.0, 9.0, 14.0, 22.0])
            cut = y.size // 2
            y = np.concatenate([y[:cut], np.zeros(int(gap_s * SR), np.float32),
                                y[cut:]])

        mixed = tmp / f"mixed_{i}.wav"
        sf.write(mixed, y, SR)

        # --- degradation ladder -----------------------------------------
        deg = rng.choice(list(QUALITY_BY_DEGRADATION))
        name = f"syn_{i:04d}_{deg}.wav"
        if not _degrade(mixed, out / name, deg):
            continue

        label = {
            "emotional_tone": "neutral",
            "emotional_intensity": "low",
            "background_noise_present": present,
            "background_noise_type": kind if present else "",
            "background_noise_severity": severity,
            "audio_quality": QUALITY_BY_DEGRADATION[deg],
            "speaker_overlap_present": False,
            "long_silence_present": gap_s >= 10.0,
            "confidence": 0.82,
        }
        rows.append({"name": name, "result_json": json.dumps(label)})
        manifest.append({"name": name, "speech_source": src.name,
                         "noise_source": kind or "none",
                         "snr_db": round(snr, 2), "gap_s": gap_s,
                         "degradation": deg})

    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "result_json"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "speech_source",
                                           "noise_source", "snr_db", "gap_s",
                                           "degradation"])
        w.writeheader()
        w.writerows(manifest)

    for p in tmp.glob("*"):
        p.unlink()
    tmp.rmdir()

    print(f"wrote {len(rows)} clips to {out}/ with labels.csv and manifest.csv")
    print("\nNOTE: emotion labels here are all neutral. `frustrated` and "
          "`distressed`\nneed a real emotion corpus (RAVDESS / CREMA-D) via "
          "--speech-dir; the tone\nhead is NOT validated by this corpus alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
