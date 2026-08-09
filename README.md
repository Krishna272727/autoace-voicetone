# AutoAce — Voice Tone & Background Noise

Upload call recordings, get nine structured fields back per call.

**https://autoace-868989752147.us-central1.run.app**

Drop in individual files or a ZIP; nested folders are searched. Results are
sortable, every clip can be played back in the browser to check a prediction by
ear, and the whole batch downloads as CSV or JSON.

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


Predictions for the three calls are in `Output format.csv`.

## Reading further

| | |
|---|---|
| `MEMO.md` | architecture, approaches compared, validation, cost, latency, failure modes |
| `EXPERIMENTS.md` | run-by-run log — what was tried, what it scored, what was rejected |
| `LICENCES.md` | every model, its licence, and what was rejected on licence grounds |
| `DEPLOY.md` | how it is hosted, and why the obvious cheaper options do not work |

Comments in the code carry the reasoning for anything non-obvious, including
the measurements behind decisions that went the other way.
