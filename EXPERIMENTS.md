# EXPERIMENTS

Running log. Every change gets a row with the scorer output and a note on what
changed. A score that drops for a correct reason is recorded as such and not
chased back up (BUILD_SPEC.md 3.4).

```bash
.venv/bin/python cli.py samples/ --labels samples/labels.csv --out results
```

`samples/` holds AutoAce's three production calls and is deliberately not in
this repository; the command is recorded so the runs below can be reproduced
against the same input.

**The three sample calls are sanity anchors, not targets.** n=3, one annotator,
and roughly half the output space has no example at all. A system scoring 100%
here would be a system fitted to three files.

---

## Summary

| # | Phase | Change | Mean field acc | Agg. RTF |
|---|---|---|---|---|
| 1 | 0 | stub with `intensity` hardcoded `"medium"` | 54.2% | 0.028 |
| 2 | 0 | intensity derived honestly from `arousal=0` | **45.8%** | 0.028 |
| 3 | 0 | repo restructured, thresholds to YAML | 45.8% | 0.018 |
| 4 | 1 | Silero VAD → `max_silence_s`, `speech_ratio` | 45.8% | 0.024 |
| 5 | 2 | DSP quality → `degradation_score` | 45.8% | 0.026 |
| 6 | 3 | min-statistics noise + AudioSet tagger | 62.5% | 0.092 |
| 7 | 3 | call-flow classes ignored, closed vocabulary enforced | 66.7% | 0.102 |
| 8 | 4 | overlap: four detectors measured, all failed → abstain | 66.7% | 0.102 |
| 9 | 5+6 | roles + prosody/SER/ASR emotion | 79.2% | 0.753 |
| 10 | 6 | 3-class sentiment replaces binary SST-2 | **83.3%** | 0.756 |

Final: **83.3% mean field accuracy**, up from a 45.8% baseline.

---

## 2 — why the score fell from 54.2% to 45.8%

The Phase 0 stub hardcoded `intensity="medium"`, which coincidentally matched 2
of 3 labels. Deriving it honestly from `arousal=0` scores 0% on that field.
**This was an improvement**: a lucky constant was removed. The same reasoning
governs the overlap decision in run 8.

## 4 — Phase 1, VAD

Silero VAD (ONNX, 2 MB, MIT). One non-obvious detail cost an hour: the ONNX
graph expects the caller to **prepend the previous window's last 64 samples**,
so each call sees 576 samples, not 512. Feeding a bare 512-sample window runs
without error and returns near-zero probability for every frame — it looks like
"this file contains no speech" rather than like a bug. Measured: max p=0.12 on a
mostly-speech clip with 512, max p=1.00 with 576.

Independent confirmation the timeline is right: **call_003's longest pause
measures 7.30 s**, against the 7.3 s quoted in BUILD_SPEC 3.3 from separate
inspection.

| file | duration | speech_ratio | max_silence |
|---|---|---|---|
| call_001 | 30.9 s | 0.469 | 3.69 s |
| call_002 | 35.0 s | 0.703 | 3.49 s |
| call_003 | 171.9 s | 0.735 | **7.30 s** |

Acceptance: all three `false`; a 15 s inserted gap → `true` (measured 15.11 s);
pure silence → `true`, no crash; 2 s and 0.5 s clips → no crash. White noise
gives `speech_ratio = 0`, which is exactly the discrimination an energy gate
cannot make.

## 5 — Phase 2, quality: three sub-scores deleted, SQUIM rejected

Seven sub-scores were specified. Measured against a synthetic ladder (clipping,
band-limiting, level errors, μ-law, low-bitrate Opus, ffmpeg reverb), only four
survived:

| dropped | why |
|---|---|
| spectral roughness (distortion) | ran **backwards** — 0.83 clean vs 0.53 band-limited |
| in-band spectral flatness (robotic) | range 0.31–0.49 across the whole ladder, clean files mid-range |
| envelope-decay slope (reverb) | reverberated copies decayed *faster* than dry; a −46 dB file read 829 dB/s |

**SQUIM was evaluated and rejected** despite an acceptable CC-BY-4.0 licence:

- unstable under excerpt length — STOI 0.391 / 0.947 / 0.982 and PESQ 1.48 /
  1.82 / 2.27 for 3 s / 5 s / 10 s of the *same clean file*;
- 0.132 → 0.198 RTF, superlinear, several times the rest of the DSP stack;
- SI-SDR −16.5 dB on the clean-but-noisy call_002, i.e. it responds to
  background noise and would rebuild the forbidden coupling.

Final ladder (all three samples `clear`, monotonic in every fault):

| input | degradation | verdict |
|---|---|---|
| call_001/2/3 | 0.000 | clear |
| clip drive +10 dB | 0.283 | slightly_impaired |
| clip drive +20 dB | 0.783 | severely_impaired |
| clip drive +34 dB | 1.000 | severely_impaired |
| 8 kHz μ-law | 0.299 | slightly_impaired |
| level −30 dB | 0.390 | slightly_impaired |
| level −46 dB | 0.815 | severely_impaired |
| packet loss 3/s | 0.375 | slightly_impaired |
| packet loss 12/s | 0.900 | severely_impaired |

Two false positives were found and fixed here, both by unit tests:
**Opus intersample overshoot** (the calls decode to |x| up to 1.41) read as
clipping until clipping was made peak-relative *and* flatness-checked — a pure
sine scored 1.0 until the flat-top requirement was added. **Uniform
attenuation** read as continuous packet loss until the dropout floor was made
relative to the speech level.

## 6, 7 — Phase 3, noise

The specified estimator — energy in VAD-negative regions over energy in
VAD-positive regions — **does not work on this material**. Across a grid of 80
percentile pairs the best separation between the no-noise call and the two noisy
ones was **5.3 dB**, and nothing reproduced the −49 / −33 dB anchors in
BUILD_SPEC 3.3. Two causes: the TTS agent's pauses are digitally silent, so
pooling all non-speech frames measures the bot; and steady static persists
*under* speech where a gap-only estimator never looks.

Replaced with **minimum statistics** (per-band sliding minimum, the standard
noise-suppressor approach), which tracks a floor through continuous speech:

| file | gap-based | min-statistics | label |
|---|---|---|---|
| call_001 | −54.2 dB | **−57.9 dB** | no noise |
| call_003 | −48.9 dB | **−47.7 dB** | medium |
| call_002 | −43.6 dB | **−32.8 dB** | medium |

Separation 5.3 → **10.1 dB**, and call_002 lands on the −33 dB anchor. Note the
15 dB spread *within* the single `medium` label — two anchors cannot define four
bands, which is why `low` and `high` remain provisional.

Run 7 added two fixes worth 4.2 points: AudioSet fires **call-flow classes** on
the medium itself ("Dial tone" 0.25, "Sidetone" 0.22 outscored the actual
background sound and took the argmax), now ignored; and emission is constrained
to the closed vocabulary, after "sine wave" escaped as free text.

The DSP signature is calibrated against **synthesised reference noises**, not
the two noisy calls — fitting it to n=2 would be the exact overfitting the spec
warns about, and the two calls do not separate on those features anyway (floor
flatness 0.31 and 0.40, against 0.50 for the file labelled *no noise*).

## 8 — Phase 4, overlap: four detectors, none worked

Measured against constructed ground truth (a single-speaker excerpt and a
deliberately summed two-speaker mix):

| approach | result |
|---|---|
| two-comb harmonic residual | single-speaker 11.4% of frames >0.85, mix 11.1% — identical |
| embedding mixture detection | mix 0.07, no-overlap call 0.14 — **backwards** |
| embedding max-similarity | mix p50 0.92, no-overlap call 0.90 — no separation |
| no-gap speaker changes | no-overlap call 8.26/min, overlapped call 3.80/min — backwards |

So the predictor **abstains**: it writes nothing, `derive.py` falls back to the
schema default and charges the confidence penalty, and the note says why.
pyannote OSD activates automatically when `HF_TOKEN` is set.

Emitting a constant `True` would score 66.7% on this field here, since two of
three calls are labelled with overlap. That is the run-2 lucky constant again —
it is not a measurement, it inverts on inbound or single-speaker recordings, and
it would report confidence in a guess. Field accuracy 33.3% is the honest price.

## 9 — Phases 5 and 6

**Role assignment failed first, in the most instructive way.** Acoustic signals
alone (gap noise floor, level variance, first-speaker prior) put the *bot* in
the customer cluster on call_001 — the two clusters' gap floors differed by
0.4 dB, i.e. noise. The transcript handed to sentiment analysis therefore opened
with "Hi, I'm Erica from Toyota of Braintree." Empathetic TTS is emotionally
toned, so this is not a wash — it is a confident wrong signal.

Two changes fixed it:

1. **Diarize on ASR utterance boundaries, not fixed windows.** 1.25 s windows
   straddle turn changes; utterances do not. (Average-linkage clustering was
   also replaced with spherical k-means, which was producing singleton outlier
   clusters — 192/1 and 11/1 — instead of speaker splits.)
2. **Label the clusters lexically.** Clustering supplies the grouping;
   scripted-agent language supplies the *label*. Neither works alone.

Result — call_003, previously inseparable, now splits cleanly at 0.90
confidence:

```
CUSTOMER: "Erica, I want an appointment for my car, please. I need to have an
           ordinary checkup once every four months..."
AGENT:    "Hi, I'm Erica from Lexington Toyota. I can help with that. What type
           of service do you need?..."
```

The Phase 5 acceptance check (customer segments contain the audible noise,
agent segments do not) passes where it is testable: call_002 shows a **4.8 dB**
higher floor in customer gaps. On call_001 the two differ by 0.4 dB — correctly,
since that call has no background noise — and the system reports low confidence
rather than a forced answer.

## 10 — binary sentiment was the last tone error

SST-2 is binary. An ordinary service request — "I want an appointment for my
car, I need a checkup every four months" — has no sentiment at all, but a binary
head must pick a side, and it picked negative: valence −0.62 turned a call
labelled `satisfied` into `upset`. Swapping to a three-class model
(`cardiffnlp/twitter-roberta-base-sentiment-latest`, MIT) with
`valence = P(pos) − P(neg)` lets neutral text sit near zero.

Tone 33.3% → 66.7%, and the remaining error is off-by-one on the ordinal scale
(`satisfied` → `neutral`) rather than a sign error. **Tone adjacent accuracy is
100%.**

### Ablation — acoustic-only vs fused (the required comparison)

| path | tone + intensity exact | detail |
|---|---|---|
| acoustic only (prosody + SER, no ASR) | **1/3** | call_001 `neutral/low`, call_003 `neutral/medium` |
| fused (acoustic + lexical) | **2/3 tone, 3/3 intensity** | valence sign errors resolved |

Acoustic-only valence is near zero on all three calls (+0.08, −0.03, −0.19):
prosody carries arousal, semantics carry valence, exactly as expected. This is
the measurement behind the claim that lexical fusion is doing the work.

### Stack ablation

| stack | mean field acc | agg. RTF |
|---|---|---|
| `none` (baseline) | 45.8% | 0.018 |
| `vad,quality,noise` (DSP only) | 66.7% | **0.088** |
| full | **83.3%** | 0.756 |

The DSP-only stack gets 80% of the accuracy for 12% of the cost, which is the
configuration to reach for on a large batch.

---

## Final scorer output

```
scored 3 file(s)   mean field accuracy = 83.3%

field                         n      acc     adj  macroF1
---------------------------------------------------------
emotional_tone                3    66.7%  100.0%    0.556
emotional_intensity           3   100.0%  100.0%    1.000
background_noise_present      3   100.0%  100.0%    1.000
background_noise_type         3    66.7%   66.7%    0.556
background_noise_severity     3   100.0%  100.0%    1.000
audio_quality                 3   100.0%  100.0%    1.000
speaker_overlap_present       3    33.3%   33.3%    0.250
long_silence_present          3   100.0%  100.0%    1.000

errors:
  emotional_tone: expected 'satisfied' -> got 'neutral'      (adjacent)
  background_noise_type: expected 'static' -> got 'television'
  speaker_overlap_present: expected True -> got False (x2)   (abstained)
```

Confidence: 0.57 / 0.63 / 0.70 — genuinely varying, not the constant 0.82 in
the reference labels.

## Synthetic corpus and tuning

40 clips, ground truth by construction. `long_silence_s` was tuned from 10.0 to
12.0 (macro F1 0.812 → **1.000**): the inserted gaps make any boundary in (9, 14)
perfect, so the midpoint is more robust than 10.0 sitting on a class edge.

**The noise thresholds were deliberately not tuned.** The tuner moved
`present_db` to −26 dB and agreement on the real calls fell to 66.7%. The cause
is a corpus fault, not a detector fault: the synthetic clips are built on top of
the *provided calls*, which already contain a television and line static, so a
clip labelled `none` by construction genuinely has audible noise in it. The
detector is right and the label is wrong. Rebuilding the corpus from clean
speech is the fix and is the first item in the next-steps list.

## Test suite

76 tests. 20 000 property draws, the full BUILD_SPEC 6 robustness matrix,
determinism, and unit tests per predictor. Two real bugs were found by tests
rather than by inspection (the sine-as-clipping false positive, and the MP4
decode failure caused by probing channels with soundfile instead of ffmpeg).

## pyannote overlap: obtained, run, rejected

Previously recorded as "gated, therefore abstain". Access was granted, so the
question became answerable rather than hypothetical.

Three obstacles before any number came out, each worth recording because the
BUILD_SPEC guidance predates them:

1. `pyannote.audio` 4.x **deleted** the `OverlappedSpeechDetection` pipeline
   that BUILD_SPEC 5 names as primary. The repo still exists; the class does
   not.
2. `Pipeline.from_pretrained(..., use_auth_token=)` was renamed to `token=`.
   The existing code passed the old keyword, so the pyannote path had never
   executed against a 4.x install — it raised TypeError and fell into the
   abstention branch, indistinguishable from "no token set".
3. `speaker-diarization-3.1` transparently pulls `speaker-diarization-community-1`,
   a separately gated repo. Accepting four model pages was not enough; it needs
   five.

A fourth, once running: pyannote's own reader trusts container duration
metadata, which disagrees with the decoded sample count on these Opus files
(`requested chunk [21.0 --> 31.0] resulted in 477105 samples instead of the
expected 480000`). Fixed by passing the already-decoded waveform tensor instead
of a path, which also reuses our ffmpeg decode rather than doing a second one.

### Result

    file          speakers  speech    overlap   ratio    label      RTF
    call_001.ogg      4      13.4s     0.35s    0.0264   no          1.08
    call_002.ogg      4      25.3s     0.88s    0.0347   overlap     1.63
    call_003.ogg      2     119.0s     2.26s    0.0190   overlap     1.34

**1/3, unchanged from abstention.** And unfixable by threshold: call_001
(labelled *no overlap*) ranks above call_003 (labelled *overlap*), so the
ordering is inverted rather than mis-cut. Four speakers found in two 2-party
calls; that over-segmentation is the likely source of the spurious overlap.

A cheaper direct OSD head on `pyannote/segmentation-3.0` was also spiked at
RTF 0.012 and produced the same undifferentiated band (0.0207-0.0265 over all
three files), which corroborates the finding rather than contradicting it.

### Decision

Opt-in behind `VOICETONE_OVERLAP=pyannote`, not enabled by `HF_TOKEN` alone —
tripling runtime should be a decision, not a side effect of setting a token for
some other model. Abstention remains the default.

**The blocker is data, not detectors.** The synthetic corpus has zero overlap
examples, so five candidates have now been judged on three files. Until
BUILD_SPEC 8.6 (spliced turns with deliberate barge-in) exists, `ratio_min` is
unfittable and this comparison stays an anecdote.

Also switched off pyannote 4.x's OpenTelemetry reporting, which fires on every
`pipeline_apply`. No audio in the payload, but it contradicts the local-only
claim and would block on a container with no egress.

## Test suite

129 tests after the pyannote work (76 before the dashboard suite grew).
