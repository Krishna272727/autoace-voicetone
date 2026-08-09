# AutoAce — Voice Tone & Background Noise

Analyses call recordings and emits nine structured fields per clip: emotional
tone and intensity, background-noise presence, type and severity, technical
audio quality, speaker overlap, long silence, and a calibrated confidence.

All inference runs on infrastructure you control -- no third-party AI API is
called and no audio is sent to one. Run it locally and nothing leaves your
machine at all; the hosted deployment is described in DEPLOY.md.


> **A note on the `BUILD_SPEC` references.** Comments throughout the code cite
> `BUILD_SPEC §N` — an internal build specification written before
> implementation, which fixed the architecture, the phase order, the
> generalisation rules and the acceptance criteria. It is not included here.
> The decisions it drove are all restated in `MEMO.md` and in the module
> docstrings, so nothing is missing; the citations are provenance, not
> dependencies.

---

## Quick start

```bash
# 1. ffmpeg must be on PATH — check first, everything decodes through it
ffmpeg -version && ffprobe -version
#    macOS: brew install ffmpeg     Ubuntu: sudo apt-get install -y ffmpeg

# 2. Python 3.11 (3.13+ has patchy audio wheels; 3.12 works)
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Pre-fetch the models (optional; they download on first use otherwise)
python scripts/download_models.py

# 4. Run it
python cli.py samples/ --labels samples/labels.csv --out results
```

That prints per-file results, a per-stage timing table and, when labels are
supplied, the scorer report. Results are written to `results/results.json` and
`results/results.csv`.

### The dashboard

```bash
export DASHBOARD_USER=you DASHBOARD_PASS=something-real
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000, log in, upload a ZIP (or several files), watch the
progress bar, then download CSV or JSON. Include a `labels.csv` in the upload
and the scorer report appears inline.

### Docker

```bash
docker build -t autoace .
docker run -p 8000:8000 -e DASHBOARD_USER=you -e DASHBOARD_PASS=secret autoace
```

Models are baked into the image, so there is no runtime download and no cold
start. Health check on `/healthz`.

---

## Output

```json
{
  "emotional_tone": "frustrated",
  "emotional_intensity": "medium",
  "background_noise_present": true,
  "background_noise_type": "office chatter",
  "background_noise_severity": "low",
  "audio_quality": "clear",
  "speaker_overlap_present": false,
  "long_silence_present": false,
  "confidence": 0.82
}
```

| Field | Values |
|---|---|
| `emotional_tone` | `neutral` `satisfied` `frustrated` `upset` `distressed` |
| `emotional_intensity` | `low` `medium` `high` |
| `background_noise_present` | bool |
| `background_noise_type` | closed vocabulary; `""` when no noise |
| `background_noise_severity` | `none` `low` `medium` `high` |
| `audio_quality` | `clear` `slightly_impaired` `severely_impaired` |
| `speaker_overlap_present` | bool |
| `long_silence_present` | bool |
| `confidence` | 0.0–1.0, calibrated from distance-to-threshold |

---

## How it works

Two stages, kept strictly apart.

```
STAGE 1 — measure physical quantities            STAGE 2 — derive the schema

  decode ─┬─ VAD ──────► max_silence_s  ─────────► long_silence_present
          │         └──► speech_ratio
          ├─ DSP ──────► degradation_score ──────► audio_quality
          ├─ min-stats ► noise_level_db ─────┬───► background_noise_present
          │                                  └───► background_noise_severity
          ├─ AudioSet ─► noise_class_dist ───────► background_noise_type
          ├─ pyannote ─► overlap_ratio ──────────► speaker_overlap_present
          └─ roles ────► customer_mask
                   └──► SER + ASR ► arousal ──┬──► emotional_tone
                                     valence ─┴──► emotional_intensity
                                  all margins ───► confidence
```

**Predictors write latents, never schema fields.** Every threshold lives in
`config/thresholds.yaml` and is applied in exactly one place, `derive.py`.

Two consequences that matter:

- **Coupled fields cannot contradict each other.** `present` and `severity` are
  two readings of one continuous number, so "no noise, severity high" is not
  representable. Same for tone and intensity, which both read `(valence,
  arousal)`. Consistency is structural rather than patched afterwards.
- **`audio_quality` never reads a noise variable.** A television in the
  customer's room does not make the *line* bad. A property test sweeps noise
  across its full range with degradation held fixed and asserts the quality
  answer never moves.

### Layout

```
voicetone/
  schema.py        output contract + consistency rules
  latent.py        stage-1 estimates
  derive.py        all thresholds, all coupling
  audio.py         decode, stereo detection
  pipeline.py      registry, two levels of error isolation
  score.py         per-field metrics, adjacent accuracy, macro F1
  noise_vocab.py   closed vocabulary + lenient matching
  models.py        every checkpoint, loaded once per process
  stack.py         predictor assembly
  predictors/      vad, quality, noise, overlap, diarize, asr, roles, emotion
app/               FastAPI dashboard
scripts/           download_models, make_synthetic, tune_thresholds
tests/             property, robustness, determinism, unit
```

### Choosing the stack

`--stack` selects which predictors run; this is how the ablations are produced.

```bash
python cli.py samples/ --labels samples/labels.csv --stack none            # baseline
python cli.py samples/ --labels samples/labels.csv --stack vad,quality,noise
python cli.py samples/ --labels samples/labels.csv                         # everything
```

The DSP-only stack (`vad,quality,noise`) needs no torch models beyond a 2 MB
VAD and runs at **0.09 RTF**. The full stack adds diarization, SER and ASR and
runs at **0.76 RTF**.

---

## Testing

```bash
pytest                       # 76 tests
pytest tests/test_robustness.py   # the malformed-input matrix
```

- **Property tests** — 20 000 random draws over the derivation surface,
  including the regions no sample file reaches. Asserts the noise fields agree,
  neutral is never high-intensity, distressed is never low-intensity,
  confidence stays in range, every enum class is reachable, and quality is
  invariant to noise.
- **Robustness** — zero-byte files, truncated headers, `.txt` renamed to
  `.wav`, video containers, 5.1 surround, true dual-channel, 8 kHz μ-law, pure
  silence, white noise, DTMF, hold music, square waves, half-second clips, and
  filenames with spaces, Cyrillic and emoji. Every case either produces valid
  schema output or a clean `failed` status with a reason. Nothing raises.
- **Determinism** — the same file twice, byte-identical, including the latents
  underneath. Seeds are pinned and ASR is greedy at temperature 0.

---

## Validation beyond the three calls

The provided calls cover roughly half the output space: `frustrated`,
`distressed`, `low` intensity, `low`/`high` severity, both impaired quality
classes and `long_silence_present: true` have **zero** real examples.

```bash
python scripts/make_synthetic.py --out synthetic --n 120
python cli.py synthetic/ --labels synthetic/labels.csv --stack vad,quality,noise
python scripts/tune_thresholds.py --corpus synthetic          # dry run
python scripts/tune_thresholds.py --corpus synthetic --write  # apply
```

Ground truth is known by construction — the mix SNR sets severity, the inserted
gap length sets the silence label, the applied degradation sets the quality
class. The tuner uses 4-fold **grouped** cross-validation, keyed on speech
source and noise source, so the same voice or noise recording never appears in
both halves.

See `EXPERIMENTS.md` for the run-by-run log and `MEMO.md` for findings, cost,
limitations and next steps.

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DASHBOARD_USER` / `DASHBOARD_PASS` | `autoace` / `changeme` | dashboard login; the UI warns while defaults are in use |
| `VOICETONE_STACK` | `all` | `all`, `none`, or e.g. `vad,quality,noise` |
| `BATCH_WORKERS` | CPU count, max 8 | parallel files per batch |
| `MAX_UPLOAD_MB` | 512 | rejected before the body is read into memory |
| `MAX_UNCOMPRESSED_MB` | 2048 | ZIP expansion limit |
| `AUDIO_RETENTION_MIN` | 60 | how long uploaded audio is kept so it can be played back in the results table; `0` deletes it the moment the batch ends and disables playback |
| `COOKIE_SECURE` | off | set to `1` behind TLS |
| `SESSION_SECRET` | derived | cookie-signing key; defaults to a hash of the password, so changing the password invalidates sessions |
| `HF_TOKEN` | unset | enables pyannote overlap detection |
| `VOICETONE_ASR` | `small.en` | Whisper size |
| `TORCH_THREADS` | 1 | per-worker torch threads |

## Data handling

Uploaded audio is retained for `AUDIO_RETENTION_MIN` minutes (default 60) so a
result can be checked by ear in the results table, then deleted automatically —
on expiry, when the job is evicted, and on process shutdown. Set
`AUDIO_RETENTION_MIN=0` to delete it the moment the batch ends and switch
playback off; that is the strictest setting and the right one if audio must not
outlive processing.

Results live in process memory and do not survive a restart. Nothing is sent to
a third party — every model runs in-process. Paid-API disclosure: **none**.

### Verifying a prediction by ear

Each successful row in the results table has a play button. It streams the clip
back from this host only, behind the same session check as everything else, and
stops working once the retention window closes.
