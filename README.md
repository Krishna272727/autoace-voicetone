# AutoAce — Voice Tone & Background Noise

Upload call recordings, get nine structured fields back per call.

**https://autoace-868989752147.us-central1.run.app**

Drop in individual files or a ZIP; nested folders are searched. Results are
sortable, every clip can be played back in the browser to check a prediction by
ear, and the whole batch downloads as CSV or JSON.

**Start with [`MEMO.md`](MEMO.md)** — the technical memo. Architecture and why
it is shaped that way, the approaches compared and the ones rejected with the
measurements that killed them, validation with per-field accuracy and confusion
matrices, the cost model, the latency table, failure modes, and scale limits.

[`predictions.csv`](predictions.csv) holds the outputs for the three provided
calls, in the supplied schema.

## What comes back

```json
{
  "emotional_tone": "frustrated",       // neutral satisfied frustrated upset distressed
  "emotional_intensity": "medium",      // low medium high
  "background_noise_present": true,
  "background_noise_type": "television",
  "background_noise_severity": "low",   // none low medium high
  "audio_quality": "clear",             // clear slightly_impaired severely_impaired
  "speaker_overlap_present": false,
  "long_silence_present": false,
  "confidence": 0.82
}
```

**83.3% mean field accuracy** on the three provided calls, against a 45.8%
majority-class baseline. All inference runs in the deployment itself; no
third-party AI API is called and none is billed.

## The rest

| | |
|---|---|
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | run-by-run log: what was tried, what it scored, what was rejected |
| [`LICENCES.md`](LICENCES.md) | every model, its licence, and what was rejected on licence grounds |
| [`DEPLOY.md`](DEPLOY.md) | how it is hosted, and why the obvious cheaper options do not work |

The provided call recordings are deliberately not in this repository. Comments
in the code carry the reasoning for anything non-obvious, including the
measurements behind decisions that went the other way.
