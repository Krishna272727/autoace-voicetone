"""Phase 4 -- speaker overlap.

Writes `overlap_ratio` = overlapped speech time / total speech time.

In a voice-bot call, overlap means **barge-in**: the customer talking over the
TTS. It is common, and two of the three provided calls are labelled with it.

PYANNOTE WAS OBTAINED, RUN, AND MEASURED
-----------------------------------------
BUILD_SPEC 5 names pyannote as the primary. It is gated, so this was originally
written as "unavailable, therefore abstain". Access was later granted and the
recommended pipeline actually run. **It did not beat abstention.**

`pyannote/speaker-diarization-3.1`, overlap = get_overlap() / speech support:

    file          speakers  speech    overlap   ratio    label      RTF
    call_001.ogg      4      13.4s     0.35s    0.0264   no          1.08
    call_002.ogg      4      25.3s     0.88s    0.0347   overlap     1.63
    call_003.ogg      2     119.0s     2.26s    0.0190   overlap     1.34

One of three, which is exactly what abstaining scores. And it is not a
threshold problem: call_001, labelled *no overlap*, scores HIGHER than
call_003, labelled *overlap*. The ordering is inverted, so no cut separates
them. Note also 4 speakers found in two 2-party calls -- the over-segmentation
is what manufactures the spurious overlap.

The cost is the other half of the argument: RTF 1.08-1.63 against 0.756 for the
entire rest of the pipeline single-core. Roughly 3x the runtime, for nothing.

So it is **opt-in via `VOICETONE_OVERLAP=pyannote`**, not merely by having a
token set. Two further notes for anyone enabling it:

  - pyannote 4.x deleted the `OverlappedSpeechDetection` pipeline that
    BUILD_SPEC names, and redirects `speaker-diarization-3.1` to a separate
    gated repo, `speaker-diarization-community-1`. Five model pages must be
    accepted, not one.
  - pyannote 4.x ships OpenTelemetry and reports every pipeline_apply. No audio
    is in the payload, but this system's claim is that nothing leaves the host,
    so `_silence_pyannote_telemetry()` turns it off before any model loads.

A dedicated OSD head built directly on `pyannote/segmentation-3.0` was also
spiked -- far cheaper at RTF 0.012 -- and gave the same undifferentiated band
(0.0207-0.0265 across all three files). Consistent with the finding above.

WHAT THE FALLBACK ACTUALLY DOES, AND WHY IT ABSTAINS
-----------------------------------------------------
**This is the weakest field in the system and the documentation says so.**

Four fallback detectors were built and measured against constructed ground
truth -- a single-speaker excerpt and a deliberately summed two-speaker mix --
plus the three labelled calls. None of them separated overlap from
single-speaker audio:

  1. *Two-comb harmonic analysis.* Estimate the dominant f0, suppress its
     harmonic comb, look for a second comb in the residual at an unrelated
     pitch. Per-frame residual ratios were statistically identical: the
     single-speaker excerpt had 11.4% of frames above 0.85, the deliberately
     mixed file 11.1%. Real speech always contains enough inharmonic energy
     (fricatives, formant noise) to fake a second comb.
  2. *Speaker-embedding mixture detection.* Cluster windows, then look for
     windows sitting between the two centroids. The fully-overlapped mix
     scored 0.07 on that measure against 0.14 for the call labelled *no
     overlap* -- backwards.
  3. *Embedding max-similarity.* Same story: p50 of 0.92 for the mix against
     0.90 for the no-overlap call.
  4. *No-gap speaker changes* (a cluster change with no VAD silence between,
     which is what barge-in physically is). The no-overlap call had the
     *highest* rate at 8.26/min; a genuinely overlapped call had 3.80/min.

Overlapped-speech detection is solved in the field with trained OSD models, and
the honest conclusion is that hand-built DSP does not substitute for one.

So by default this predictor **writes nothing**. `derive.py` then leaves
`speaker_overlap_present` at its schema default and charges the confidence
penalty for an unestimated latent, and the note explains why.

Emitting a constant `True` would in fact score better on the three provided
calls (two of three are labelled with overlap, and barge-in is common in
bot-driven calls). That is precisely the lucky constant BUILD_SPEC 3.4 warns
about: it is not a measurement, it would not generalise to an inbound or
single-speaker recording, and it would report high confidence in a guess.

WHAT WOULD ACTUALLY FIX THIS
-----------------------------
Not another detector. The synthetic corpus contains **zero** overlap examples
(`speaker_overlap_present` is False on all 40 clips), so every candidate above
was judged on three files, which cannot distinguish a working detector from a
lucky one. BUILD_SPEC 8.6 specifies the fix: splice TTS agent turns with human
customer turns including deliberate barge-in, giving ground truth by
construction. With that, `overlap.ratio_min` becomes fittable and the
pyannote-vs-DSP comparison becomes a measurement rather than an anecdote.
"""
from __future__ import annotations

import logging
import os

from ..audio import AudioContext
from ..latent import Latents
from .vad import timeline

log = logging.getLogger("autoace.overlap")

SR = 16_000


def _silence_pyannote_telemetry() -> None:
    """Turn off pyannote 4.x's usage reporting.

    It ships OpenTelemetry exporters and tracks pipeline_init / model_init /
    pipeline_apply. No audio is sent, but this system's whole claim is that
    nothing leaves the host, and an outbound metrics call on every analysed
    file contradicts that claim whatever its payload. Also fails closed: the
    container has no outbound network, so leaving it on would mean a blocked
    request on every call.
    """
    try:
        from pyannote.audio import telemetry
        telemetry.set_telemetry_metrics(False)
    except Exception:                              # noqa: BLE001
        pass                                       # older pyannote, nothing to do


_DIARIZATION_REPO = "pyannote/speaker-diarization-3.1"


def _overlap_ratio_pyannote(ctx: AudioContext) -> float | None:
    """Overlapped speech time / total speech time, via pyannote diarization.

    Opt-in through `VOICETONE_OVERLAP=pyannote`, NOT merely by having a token.
    See the measurements in the module docstring: it costs roughly 3x the whole
    rest of the pipeline and did not beat abstention on the labelled calls, so
    switching it on has to be a decision rather than a side effect of setting
    HF_TOKEN for some other model.
    """
    if os.getenv("VOICETONE_OVERLAP", "").lower() != "pyannote":
        return None
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        return None
    _silence_pyannote_telemetry()
    try:
        import torch
        from pyannote.audio import Pipeline

        # pyannote 4.x renamed use_auth_token -> token; the old keyword raises
        # TypeError, so this path never ran against a 4.x install.
        pipe = Pipeline.from_pretrained(_DIARIZATION_REPO, token=token)
        if pipe is None:                           # gated, terms not accepted
            log.info("pyannote %s not accessible", _DIARIZATION_REPO)
            return None

        # Feed the already-decoded waveform rather than the path. pyannote's
        # own reader trusts the container's duration metadata, which disagrees
        # with the decoded sample count on these Opus files:
        #   "requested chunk [21.0 --> 31.0] resulted in 477105 samples
        #    instead of the expected 480000"
        # Passing a tensor bypasses its reader and reuses our ffmpeg decode.
        out = pipe({"waveform": torch.from_numpy(ctx.speech).float().unsqueeze(0),
                    "sample_rate": ctx.speech_sr})
        # 4.x wraps the Annotation in a DiarizeOutput.
        ann = getattr(out, "speaker_diarization", out)

        # get_overlap() is the timeline where two or more speakers are active.
        # The denominator is the support of all turns, so simultaneous speech
        # counts once rather than twice.
        speech = ann.get_timeline().support().duration()
        if not speech:
            return None
        ratio = float(min(1.0, ann.get_overlap().duration() / speech))
        log.info("overlap %.4f from %s (%d speakers)",
                 ratio, _DIARIZATION_REPO, len(ann.labels()))
        return ratio
    except Exception as exc:                       # noqa: BLE001
        log.info("pyannote diarization unavailable (%s)", exc)
        return None


class OverlapPredictor:
    name = "overlap"

    def __call__(self, ctx: AudioContext, lat: Latents) -> None:
        tl = timeline(ctx)
        if not tl.speech:
            # Nothing is speaking, so nothing can overlap. This one case is a
            # measurement rather than a guess, so it is safe to write.
            lat.overlap_ratio = 0.0
            lat.notes.append("overlap: no speech detected, ratio is 0 by definition")
            return

        ratio = _overlap_ratio_pyannote(ctx)
        if ratio is not None:
            lat.overlap_ratio = round(float(ratio), 5)
            lat.notes.append("overlap: pyannote overlapped-speech-detection")
            return

        # Abstain. See the module docstring: four fallback detectors were built
        # and none separated overlap from single-speaker audio, so no number
        # here would be a measurement. Leaving the latent as None makes the
        # gap visible in `missing()` and costs confidence honestly.
        lat.notes.append(
            "overlap: not estimated -- pyannote OSD needs HF_TOKEN and accepted "
            "terms; no validated DSP fallback exists (see overlap.py)")
