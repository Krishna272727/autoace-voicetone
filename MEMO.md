# AutoAce — Voice Tone & Background Noise
### Technical memo

---

## 1. Problem and constraints

Emit nine structured fields per call recording: emotional tone and intensity,
background-noise presence, type and severity, technical audio quality, speaker
overlap, long silence, and a confidence. Graded on a hidden test set that will
not resemble the three provided files, at a cost well under $0.003 per
audio-minute, with a deployed dashboard.

Two constraints from the brief drive the architecture rather than being handled
by convention:

- do not infer frustration or distress from loudness alone;
- do not infer background noise from poor audio quality alone.

Both are enforced structurally, in §3.

## 2. Key findings from the data

**The channels are fake stereo.** L/R correlation is 1.0000 on all three files;
max |L−R| ≈ 0.003, which is Opus noise. These are mono downmixes in a stereo
container, so channel selection cannot separate the speakers here. The hidden
set may contain genuine dual-channel recordings where it can, so the system
branches on measured correlation at runtime and warns when it downmixes.

**Roughly half the output space has no example.** `frustrated`, `distressed`,
`low` intensity, `low` and `high` severity, both impaired quality classes, and
`long_silence_present: true` appear zero times in the labels. Any code path
producing them is unvalidated by the real data, which is why there is a
synthetic corpus and why property tests assert every enum class is reachable.

**Noise and quality are decoupled in the labels.** One call is
`background_noise_type: "sharp static"` *and* `audio_quality: "clear"`. A system
that reads impairment off noise fails that case.

**Confidence is a constant 0.82** in all three reference labels — a placeholder,
not a signal. It is not fitted to. The system emits genuinely varying values
(0.57 / 0.63 / 0.70 on the samples) and says so here.

**The quoted noise anchors did not reproduce.** BUILD_SPEC 3.3 gives −49 dB and
−33 dB for the clean and noisy calls under a gap-based estimator. Across a grid
of 80 percentile pairs, no combination reproduced those numbers and the best
separation was 5.3 dB. This drove a change of estimator (§4).

## 3. Architecture: estimate latents, then derive

Two stages, kept strictly apart.

**Stage 1** predictors write continuous physical estimates into a `Latents`
object. They never write a schema field.
**Stage 2** — `derive.py`, the only place any threshold is applied — turns those
latents into the nine outputs.

The naive alternative gates: predict `present`, and only if true predict
`severity`. Cascades multiply error (90% × 80% = 72% joint) and a wrong
`present=false` locks `severity=none` with no recovery. Instead **one continuous
parent feeds both discrete children**: `present` and `severity` are two readings
of `noise_level_db`, so they cannot contradict each other. Tone and intensity
likewise both read `(valence, arousal)`.

Three consequences:

- consistency is structural, not repaired afterwards;
- confidence falls out for free — distance from a threshold *is* certainty;
- with three labelled calls you cannot train a classifier, but you can fit ~12
  thresholds. This is the only architecture that is actually calibratable with
  the data available.

`audio_quality` never reads a noise variable. A property test sweeps
`noise_level_db` from −70 to −8 dB with `degradation_score` held fixed and
asserts the quality answer never moves.

## 4. Approaches compared

**Noise level: gap-based vs minimum statistics.** The specified gap-based
estimator gave 5.3 dB of separation. Minimum statistics — per-band sliding
minimum, the standard noise-suppressor technique — gave **10.1 dB** and put the
TV call on the −33 dB anchor. The reason is mechanical: the TTS agent's pauses
are digitally silent, so pooling non-speech frames measures the bot, and steady
static persists *under* speech where a gap-only estimator never looks.

**Quality: hand DSP vs SQUIM.** SQUIM (reference-free STOI/PESQ/SI-SDR) was
evaluated as the brief suggests and rejected on measurements, not licensing: on
one clean file, STOI came back 0.391 / 0.947 / 0.982 for 3 s / 5 s / 10 s
excerpts; cost was 0.13–0.20 RTF and superlinear; and SI-SDR read −16.5 dB on a
clean-but-noisy call, meaning it would have rebuilt the forbidden coupling. Four
hand DSP sub-scores survived validation; three were deleted for measuring
nothing (details in `EXPERIMENTS.md`).

**Emotion: acoustic-only vs acoustic+lexical fusion.** This is the headline
comparison.

| path | tone + intensity exact |
|---|---|
| acoustic only (prosody + SER) | 1/3 |
| fused (acoustic + lexical) | 2/3 tone, 3/3 intensity |

Acoustic-only valence sits near zero on all three calls (+0.08, −0.03, −0.19):
prosody carries arousal, semantics carry valence. call_003 is the designed trap
— labelled `satisfied` with the highest speech level of the three — and the
acoustics-only path cannot recover it.

**Sentiment head: binary vs three-class.** SST-2 must assign a side. An ordinary
service request ("I want an appointment for my car") scored −0.62 and turned
`satisfied` into `upset`. A three-class model with `valence = P(pos) − P(neg)`
lets neutral text sit near zero. Tone accuracy 33.3% → 66.7%, worth 4.1 points
of overall accuracy on its own.

**Role assignment: acoustic vs lexical labelling.** Acoustic signals alone put
the bot in the customer cluster (the two clusters' gap floors differed by
0.4 dB — noise). Diarizing on ASR utterance boundaries and labelling the
clusters by scripted-agent language fixed it.

**LLM verification: evaluated and rejected for the production path.** It cannot
hear the audio — it would see only our feature summary and our predictions, so
it cannot catch our errors, only re-reason over the same numbers with more
confidence. It breaks reproducibility, which is 15% of the grade; it adds cost,
latency and data egress, weakening an otherwise airtight privacy story; and it
is slower than the deterministic layer it would be checking. The validation
layer actually needed is the deterministic consistency layer in `schema.py` and
`derive.py`: same guarantees, microseconds, identical every run. An LLM is
useful *offline* — generating validation utterances, clustering noise vocabulary
into a static dictionary — and none of that touches inference.

## 5. Validation

**Results on the three provided calls: 83.3% mean field accuracy**, against a
45.8% majority-class baseline.

| field | accuracy | adjacent | macro F1 |
|---|---|---|---|
| emotional_tone | 66.7% | **100%** | 0.556 |
| emotional_intensity | 100% | 100% | 1.000 |
| background_noise_present | 100% | 100% | 1.000 |
| background_noise_type | 66.7% | 66.7% | 0.556 |
| background_noise_severity | 100% | 100% | 1.000 |
| audio_quality | 100% | 100% | 1.000 |
| speaker_overlap_present | 33.3% | 33.3% | 0.250 |
| long_silence_present | 100% | 100% | 1.000 |

Adjacent accuracy is reported alongside exact match because tone, intensity,
severity and quality are ordinal and their boundaries are genuinely fuzzy. The
single tone error is one class away.

**Read these numbers sceptically.** n=3, one annotator. Two of the 100%s
(`audio_quality`, `long_silence_present`) would also be achieved by a constant,
because all three calls share one value. They are validated on synthetic data
instead, which is the only place those decisions are actually exercised.

**Synthetic corpus.** `scripts/make_synthetic.py` builds a labelled corpus where
ground truth is known by construction: mix SNR sets severity, inserted gap
length sets the silence label, applied degradation sets the quality class.
`scripts/tune_thresholds.py` fits thresholds with 4-fold **grouped**
cross-validation keyed on speech source and noise source, so the same voice or
noise recording never appears in both halves. `long_silence_s` was tuned 10.0 →
12.0, macro F1 0.812 → 1.000.

**The noise thresholds were deliberately left untuned**, and this is worth
stating plainly: the tuner moved `present_db` to −26 dB and agreement on the
real calls fell to 66.7%. The cause is a corpus fault — the synthetic clips are
layered on top of the provided calls, which already contain a television and
line static, so a clip labelled `none` by construction genuinely has audible
noise. The detector is right and the label is wrong. Fitting to it would have
improved a number and broken the system.

**Test suite: 76 tests.** 20 000 property draws over the derivation surface, the
full robustness matrix, a determinism test asserting byte-identical repeat runs,
and unit tests per predictor against analytically-known inputs. Two real bugs
were caught by tests rather than inspection: a pure sine scoring as fully
clipped, and MP4 decode failing because channel probing used soundfile instead
of ffmpeg.

## 6. Cost

All models are open-weight and run locally, so cost is instance time only.

```
instance                  2 vCPU / 4 GB          $0.02 / hour
measured RTF (full)       0.756  single file, single core
measured RTF (DSP only)   0.088  single file, single core
measured RTF, 4 workers   0.431  full stack, 8-core host   <- measured, not extrapolated

cost per audio-minute = $0.02 x (0.431 / 60) = $0.000144
ceiling                                        $0.003
headroom                                       ~21x
```

DSP-only, the same arithmetic gives roughly **$0.000017 per audio-minute**,
~175× headroom.

**Batch scaling is sub-linear and the measured number is used, not the
extrapolated one.** Four workers give 1.75× throughput, not 4×: eight cores are
shared between the batch threads and the models' own inner threads, and the
Python-level work between stages does not parallelise. Quoting 0.756/4 would
have overstated the headroom by 2.3×.

One fix came out of this measurement. All batch threads share a single
`WhisperModel`, and CTranslate2 serialises concurrent `transcribe()` calls
unless `num_workers` is raised — so eight batch workers were queueing on the
slowest stage in the pipeline. Setting `num_workers` to the batch width halved
the ASR stage: **0.360 → 0.182 RTF** on a four-file batch. Replicas share the
weights, so this costs scheduling capacity rather than memory.

Assumptions: batch processing with idle time amortised across a continuously-fed
queue; models resident, loaded once at startup; one process, since a second
worker process would duplicate every checkpoint in memory.

**Paid-API disclosure: none.** No audio, transcript or derived feature leaves
AutoAce-controlled infrastructure. That answers the data-handling requirement
architecturally rather than by promise.

**Retention, stated precisely.** The dashboard plays a clip back beside its
prediction so an evaluator can check the system by ear, which means audio must
outlive the batch. It is held for a bounded, configurable window (default 60
minutes) and purged on expiry, on job eviction and on process shutdown;
`AUDIO_RETENTION_MIN=0` restores delete-on-completion and disables playback.
This is a deliberate loosening of the original posture in exchange for
verifiability, and the window is enforced by test, not just advertised in the
footer.

## 7. Latency

Per-stage, single core, measured on the provided calls:

| stage | RTF | note |
|---|---|---|
| roles (incl. ASR + embeddings) | 0.425 | dominant; ASR is shared with emotion via cache |
| emotion (SER + fusion) | 0.235 | |
| noise (AudioSet tagging) | 0.078 | |
| VAD | 0.006 | |
| quality (DSP) | 0.002 | |
| overlap | 0.000 | abstains without pyannote |
| **full stack** | **0.756** | |
| **DSP-only stack** | **0.088** | 80% of the accuracy for 12% of the cost |

The evaluation is batch, not real-time, so throughput is the headline — but
scaling is sub-linear, and the measured figure is **0.431 effective RTF at four
workers** on an 8-core host (1.75×, not 4×). Models load once at startup (12.7 s
warm-up, excluded from RTF); cold-loading per request would add 10–30 s and
dominate everything.

Per-file latency is the weak spot: a single 35 s call takes ~30 s, which looks
poor in a one-file demo and is irrelevant to a 100-file batch. If per-clip
latency ever matters, `VOICETONE_ASR=tiny.en` or the DSP-only stack are the
levers, at a known accuracy cost.

## 8. Failure modes

- **Speaker overlap is the weakest field, at 33.3%, and pyannote does not fix
  it.** Four DSP fallbacks were built and measured; none separated overlap from
  single-speaker audio (two ran *backwards* on constructed ground truth). Access
  to the gated weights was then obtained and the recommended pipeline actually
  run:

  | file | speech | overlap | ratio | label | RTF |
  |---|---|---|---|---|---|
  | call_001 | 13.4 s | 0.35 s | 0.0264 | no overlap | 1.08 |
  | call_002 | 25.3 s | 0.88 s | 0.0347 | overlap | 1.63 |
  | call_003 | 119.0 s | 2.26 s | 0.0190 | overlap | 1.34 |

  One of three — what abstention already scores. It is not a threshold problem:
  the file labelled *no overlap* scores **higher** than one labelled *overlap*,
  so no cut separates them. The diarizer also reports four speakers in two
  2-party calls, and that over-segmentation is what manufactures the overlap.
  At RTF 1.08–1.63 against 0.756 for the entire rest of the pipeline, it costs
  roughly 3× the runtime for nothing measurable, so it is opt-in behind
  `VOICETONE_OVERLAP=pyannote` rather than on by default.

  The real blocker is validation, not detection: the synthetic corpus contains
  zero overlap examples, so every candidate has been judged on three files.
  BUILD_SPEC 8.6 (spliced agent/customer turns with deliberate barge-in) is
  what would make `overlap.ratio_min` fittable and turn this into a
  measurement.
- **Valence sign errors** remain the top emotion risk. Mitigated by lexical
  fusion, not eliminated; a sarcastic or code-switched utterance will still
  invert. One sample call is Spanish-switching and the transcript degrades.
- **Role assignment** depends on scripted-agent language. A human agent who does
  not read a script, or a non-English call, falls back to the acoustic signals,
  which were measured getting call_001 wrong. `role_confidence` reflects this
  and shrinks the emotion estimate toward neutral when low.
- **Reverberation and codec artifacting are not detected.** Both proxies failed
  validation and were removed rather than left in looking authoritative.
- **`background_noise_type` for line artifacts.** "sharp static" is a
  codec/line condition that AudioSet was not trained on; the DSP signature
  covers it only when genuinely broadband and flat, and stays silent otherwise.
  call_003 is still classified `television`.
- **Unvalidated classes.** `distressed`, `severity: high` and
  `severely_impaired` have no real examples. They are reachable (property tests
  assert it) and exercised synthetically, but their real-world accuracy is
  unknown.
- **Three labels, one annotator.** `frustrated` vs `upset` and `neutral` vs
  `frustrated` are genuinely fuzzy, which is why adjacent accuracy is reported.

## 9. Next steps, in priority order

1. **Rebuild the synthetic corpus from clean speech** (RAVDESS or CREMA-D,
   licence-checked) rather than from the provided calls. This unblocks noise
   threshold tuning, which is currently blocked by the corpus fault in §5, and
   supplies the `frustrated` / `distressed` examples the tone head has never
   seen.
2. **Build overlap ground truth**, then re-open the overlap question. pyannote
   has now been tried and measured (§8) and did not help; what is missing is
   not a better detector but any data to choose one on. Splicing agent and
   customer turns with deliberate barge-in gives ground truth by construction.
3. **Calibration curve.** Bucket synthetic predictions by confidence and verify
   higher buckets are more accurate; apply temperature scaling if not. Confidence
   is currently principled (distance-to-threshold, penalised for unevidenced
   latents) but not empirically calibrated.
4. **A learned fusion** over the latents, once the synthetic corpus is large
   enough to support one without leakage — the threshold layer is deliberately
   simple because n=3 could not support more.
5. **Language detection** to route non-English calls to the acoustic path
   explicitly and lower confidence, rather than degrading silently.
