# Licence audit

Every model that ships in the production path, verified before adoption
(BUILD_SPEC.md 14). **No CC-BY-NC checkpoint is used anywhere** — a
non-commercial licence is disqualifying for a production claim.

| Component | Purpose | Licence | Commercial use | Gated |
|---|---|---|---|---|
| ffmpeg / ffprobe | decode, resample, container handling | LGPL-2.1+ (GPL if built with `--enable-gpl`) | yes | no |
| Silero VAD (ONNX) | speech/non-speech timeline | MIT | yes | no |
| `MIT/ast-finetuned-audioset-10-10-0.4593` | AudioSet tagging → noise type | BSD-3-Clause | yes | no |
| `microsoft/wavlm-base-plus-sv` | speaker embeddings → diarization, roles | MIT | yes | no |
| `superb/wav2vec2-base-superb-er` | speech emotion → acoustic valence/arousal | Apache-2.0 | yes | no |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | lexical valence | MIT | yes | no |
| `faster-whisper` / Whisper `small.en` | transcription (CTranslate2) | MIT / MIT weights | yes | no |
| numpy, scipy, librosa, scikit-learn | DSP and clustering | BSD-3-Clause | yes | no |
| FastAPI, uvicorn, Jinja2, pydantic | dashboard | MIT / BSD | yes | no |
| Quicksand (webfont, bundled) | dashboard typography | SIL OFL 1.1 | yes | no |

### On bundling the font

Quicksand ships **inside the repository** (`app/static/fonts/`, ~55 KB for the
variable 300-700 face) rather than being pulled from Google Fonts at runtime.
Two reasons: the container has no outbound network access, so a CDN webfont
would silently fail and drop every page back to a system face at a different
metric; and a request to fonts.gstatic.com on each page load is a third-party
call this system otherwise does not make. OFL-1.1 permits redistribution; the
licence text is bundled alongside the font as `Quicksand-OFL.txt`, which is
what the licence requires.

## Rejected, and why

**`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` — CC-BY-NC-SA-4.0.**
This was the obvious first choice: it predicts arousal, valence and dominance
*directly and continuously*, which is exactly the shape this architecture wants,
and it would have removed the whole four-class-to-circumplex mapping step. It is
non-commercial, so it is disqualifying. The system instead composes valence and
arousal from an Apache-2.0 four-class SER checkpoint plus an MIT text-sentiment
model, which is more machinery for a weaker signal — a real accuracy cost paid
to keep the licence position clean.

**`torchaudio` SQUIM (`SQUIM_OBJECTIVE`) — CC-BY-4.0, licence was fine.**
Rejected on measurements, not licensing: unstable under excerpt length (STOI
0.391 → 0.982 on the same clean file at 3 s vs 10 s), expensive and superlinear
(0.13 → 0.20 RTF), and it responds to background noise, which would have
rebuilt the noise/quality coupling the brief forbids. See
`voicetone/predictors/quality.py`.

**`pyannote/*` — MIT weights, but gated, and measured to not help.** Using them
requires a Hugging Face account, a token, and accepting terms in a browser on
**five** model pages: pyannote 4.x deleted the `OverlappedSpeechDetection`
pipeline BUILD_SPEC names, and redirects `speaker-diarization-3.1` to a further
gated repo, `speaker-diarization-community-1`. The licence permits commercial
use; the *access* requirement makes a human action a build dependency.

Access was obtained and the pipeline run. It scored 1 of 3 on the labelled
calls — identical to abstaining — at RTF 1.08-1.63 against 0.756 for the whole
rest of the pipeline. See MEMO.md 8. It is therefore opt-in behind
`VOICETONE_OVERLAP=pyannote` and `pyannote.audio` is left out of the default
install.

One further note, on data handling rather than licensing: **pyannote.audio 4.x
ships OpenTelemetry exporters and reports every `pipeline_apply`.** No audio is
in the payload, but this system's claim is that nothing leaves the host, so
`overlap.py` calls `telemetry.set_telemetry_metrics(False)` before any model
loads.

## Speech corpora

No speech corpus is redistributed with this repository. The three provided calls
are the client's own data. `scripts/make_synthetic.py` can build its corpus from
RAVDESS or CREMA-D if pointed at them with `--speech-dir`; **both are
research-only for some uses and must be licence-checked before any commercial
deployment**, which is why neither is vendored here and the default generator
uses the client's own audio instead.

## Data handling

All inference is local. No audio, transcript or derived feature is sent to any
third-party API, so the disclosure of paid-API usage is: **none**. Uploaded
audio is deleted as soon as a batch completes; results live in process memory
only and do not survive a restart.
