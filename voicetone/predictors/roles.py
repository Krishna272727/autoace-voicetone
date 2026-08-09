"""Phase 5 -- which speaker is the customer.

Writes `customer_speech_s`, `role_confidence` and `n_speakers`, and puts the
customer-only audio in `ctx.cache["customer_audio"]` so Phase 6 runs sentiment
on the right person.

This is the highest-risk stage in the system. A diarizer hands back
`SPEAKER_00` / `SPEAKER_01` with arbitrary labels that mean nothing and are not
stable across files; deciding which cluster is the customer is the actual work,
and getting it wrong contaminates the emotion estimate with the agent's speech.
Empathetic TTS ("I'm so sorry to hear that") is emotionally-toned text, so a
role error does not merely add noise -- it adds a *confident wrong signal*.

WHAT IS NOT USED
----------------
Voice identity. The TTS vendor differs between deployments and the agent may be
a human, so "which cluster sounds synthetic" is not a durable signal.

WHAT IS USED -- the recording path, which survives any vendor change
--------------------------------------------------------------------
1. **Noise floor inside each cluster's own pauses.** The strongest and cheapest
   signal. A TTS agent's inter-word gaps are near-digital-silence; a human on a
   microphone always has room tone, and if there is a television on it is
   audible in their gaps and nowhere else. Measured per cluster, on the gaps
   *within* that cluster's turns.
2. **Level consistency.** Synthesised speech is tightly uniform in level;
   a human moves relative to the microphone.
3. **First speaker.** Outbound bot calls open with the agent's greeting. This
   is a weak prior and is weighted accordingly -- BUILD_SPEC 3.2 warns that
   inbound calls open with the customer.

The votes are combined with weights and the margin becomes `role_confidence`.
When the signals disagree the confidence is low rather than the answer being
forced, and Phase 6 reads that and degrades accordingly.
"""
from __future__ import annotations

import logging

import numpy as np

from ..audio import AudioContext
from ..latent import Latents
from .asr import agent_score, text_for, transcribe
from .diarize import diarize, diarize_segments
from .vad import timeline

log = logging.getLogger("autoace.roles")

SR = 16_000

# Vote weights. The noise-floor signal is the one with a physical mechanism
# behind it; the first-speaker prior is a convention that BUILD_SPEC 3.2
# explicitly says does not always hold.
W_FLOOR = 1.00
W_LEVEL_VAR = 0.45
W_FIRST = 0.30
# Scripted-agent language. Weighted above everything else, because it is the
# only signal here that is about *what the speaker is doing* rather than about
# the recording path, and it is what corrected the measured failure below.
W_LEXICAL = 1.20
# Ask for a transcript only when the acoustic margin is this weak. ASR costs
# ~0.4 RTF; a confident acoustic decision should not pay for it.
LEXICAL_TIEBREAK_BELOW = 0.55

# dB of floor difference that counts as a full-strength vote.
FLOOR_FULL_SCALE = 12.0
LEVEL_VAR_FULL_SCALE = 6.0


def _cluster_gap_floor(x: np.ndarray, wins: list[tuple[float, float]]) -> float:
    """Noise floor (dBFS) in the quiet moments inside one cluster's windows.

    The 10th percentile of short-frame energy: within a talker's own speech,
    that lands on the gaps between words, which is exactly where the bot is
    digitally silent and the human's room is not.
    """
    if not wins:
        return 0.0
    frames = []
    n, h = int(0.020 * SR), int(0.010 * SR)
    for a, b in wins:
        seg = x[int(a * SR):int(b * SR)]
        if seg.size < n:
            continue
        F = np.lib.stride_tricks.sliding_window_view(seg, n)[::h]
        if F.size:
            frames.append(np.sqrt((F.astype(np.float64) ** 2).mean(axis=1)))
    if not frames:
        return 0.0
    rms = np.concatenate(frames)
    return float(20 * np.log10(max(float(np.percentile(rms, 10)), 1e-10)))


def _cluster_level_var(x: np.ndarray, wins: list[tuple[float, float]]) -> float:
    """Standard deviation of per-window speech level, in dB."""
    if len(wins) < 2:
        return 0.0
    lv = []
    for a, b in wins:
        seg = x[int(a * SR):int(b * SR)]
        if seg.size:
            lv.append(20 * np.log10(max(float(np.sqrt((seg.astype(np.float64) ** 2).mean())), 1e-10)))
    return float(np.std(lv)) if len(lv) > 1 else 0.0


class RolePredictor:
    name = "roles"

    def _lexical_roles(self, ctx: AudioContext, lat: Latents,
                       x: np.ndarray, tl) -> bool:
        """Cluster ASR utterances and label the clusters by scripted language.

        Clustering supplies the *grouping* (which utterances share a voice);
        the transcript supplies the *label* (which of those voices is reading a
        script). Neither does the job alone: acoustics cannot tell you what a
        role is, and lexical markers are absent from most individual lines.

        Returns True when it produced an answer.
        """
        utts = transcribe(ctx)
        if len(utts) < 2:
            return False

        d = diarize_segments(ctx, [(u.start, u.end) for u in utts])
        if not d.ok or d.n_speakers < 2:
            return False

        # Map each kept segment back to its utterance text.
        by_span = {(round(u.start, 3), round(u.end, 3)): u for u in utts}
        texts: dict[int, list[str]] = {k: [] for k in range(d.n_speakers)}
        regions: dict[int, list[tuple[float, float]]] = {k: [] for k in range(d.n_speakers)}
        for (a, b), k in zip(d.windows, d.labels):
            regions[int(k)].append((a, b))
            u = by_span.get((round(a, 3), round(b, 3)))
            if u is not None:
                texts[int(k)].append(u.text)

        votes = {k: agent_score(" ".join(t)) for k, t in texts.items()}
        spread = max(votes.values()) - min(votes.values())
        if spread <= 0:
            return False                       # no lexical evidence either way

        # The customer is the cluster reading *less* script.
        customer = min(votes, key=lambda k: votes[k])
        floors = {k: _cluster_gap_floor(x, r) for k, r in regions.items()}

        # Confidence: lexical margin, corroborated by the acoustic floor signal.
        conf = float(np.clip(spread / 4.0, 0.15, 0.9))
        others = [floors[j] for j in floors if j != customer]
        if others and floors[customer] > max(others):
            conf = min(0.95, conf + 0.15)      # room tone agrees with the words
        lat.role_confidence = round(conf, 4)
        lat.n_speakers = d.n_speakers
        lat.customer_speech_s = round(
            float(sum(b - a for a, b in regions[customer])), 2)

        mask = np.zeros(x.size, dtype=bool)
        for a, b in regions[customer]:
            mask[int(a * SR):min(x.size, int(b * SR))] = True
        ctx.cache["customer_audio"] = x[mask] if mask.any() else x
        ctx.cache["customer_regions"] = regions[customer]
        ctx.cache["agent_regions"] = [w for k in regions if k != customer
                                      for w in regions[k]]
        lat.notes.append(
            "roles: utterance-level clustering; customer=cluster%d  "
            "agent_script_score=%s  gap_floor=%s dBFS" % (
                customer, "/".join(f"{votes[k]:.0f}" for k in sorted(votes)),
                "/".join(f"{floors[k]:.0f}" for k in sorted(floors))))
        return True

    def __call__(self, ctx: AudioContext, lat: Latents) -> None:
        tl = timeline(ctx)
        x = np.asarray(ctx.speech, dtype=np.float32).ravel()

        if not tl.speech:
            lat.n_speakers = 0
            lat.role_confidence = 0.0
            lat.customer_speech_s = 0.0
            lat.notes.append("roles: no speech, nothing to attribute")
            return

        # --- preferred path: cluster ASR utterances, label them lexically ---
        if self._lexical_roles(ctx, lat, x, tl):
            return

        d = diarize(ctx)
        if not d.ok or d.n_speakers < 2:
            # One voice: a voicemail, an IVR recording, or a call where the two
            # speakers did not separate. Assume it is the customer -- that is
            # the useful failure direction, since the alternative is discarding
            # the only speech there is -- but say so with low confidence.
            lat.n_speakers = max(1, d.n_speakers)
            lat.role_confidence = 0.35
            lat.customer_speech_s = round(tl.speech_s, 2)
            ctx.cache["customer_audio"] = x
            ctx.cache["customer_regions"] = list(tl.speech)
            lat.notes.append(
                "roles: single speaker cluster; treating all speech as the "
                "customer at low confidence")
            return

        lat.n_speakers = d.n_speakers
        wins = {k: d.cluster_windows(k) for k in range(d.n_speakers)}

        floors = {k: _cluster_gap_floor(x, w) for k, w in wins.items()}
        varis = {k: _cluster_level_var(x, w) for k, w in wins.items()}
        first = min(range(d.n_speakers),
                    key=lambda k: wins[k][0][0] if wins[k] else 1e9)

        # Score each cluster on "how much does this look like the customer".
        scores: dict[int, float] = {}
        for k in range(d.n_speakers):
            others = [j for j in range(d.n_speakers) if j != k]
            # Higher floor in their own pauses -> room tone -> human.
            df = floors[k] - float(np.mean([floors[j] for j in others]))
            s = W_FLOOR * float(np.clip(df / FLOOR_FULL_SCALE, -1.0, 1.0))
            # More level variation -> moving relative to a microphone -> human.
            dv = varis[k] - float(np.mean([varis[j] for j in others]))
            s += W_LEVEL_VAR * float(np.clip(dv / LEVEL_VAR_FULL_SCALE, -1.0, 1.0))
            # Speaking first is weak evidence of being the agent.
            if k == first:
                s -= W_FIRST
            scores[k] = s

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        margin = ranked[0][1] - ranked[1][1]
        total_w = W_FLOOR + W_LEVEL_VAR + W_FIRST
        lexical_note = ""

        # --- lexical tie-break -------------------------------------------
        # The acoustic signals alone were measured getting this wrong. On
        # call_001 the two clusters' gap floors differed by 0.4 dB -- noise --
        # and the bot landed in the "customer" cluster, so the transcript fed
        # to sentiment analysis opened with "Hi, I'm Erica from Toyota of
        # Braintree." Scripted-agent language settles it, and BUILD_SPEC 5
        # names it as a signal to add.
        #
        # ASR is expensive, so it is only requested when the acoustic margin is
        # too small to trust on its own.
        if margin / total_w < LEXICAL_TIEBREAK_BELOW:
            utts = transcribe(ctx)
            if utts:
                agent_votes = {k: agent_score(text_for(utts, wins[k]))
                               for k in range(d.n_speakers)}
                spread = max(agent_votes.values()) - min(agent_votes.values())
                if spread > 0:
                    # Most agent-scripted cluster is the agent; so the customer
                    # is the one with the *lowest* score.
                    for k in scores:
                        scores[k] -= W_LEXICAL * (agent_votes[k] / spread)
                    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
                    margin = ranked[0][1] - ranked[1][1]
                    total_w += W_LEXICAL
                    lexical_note = ("  lexical_agent_score=" +
                                    "/".join(f"{agent_votes[k]:.0f}"
                                             for k in sorted(agent_votes)))

        customer = ranked[0][0]
        lat.role_confidence = round(float(np.clip(margin / total_w, 0.0, 1.0)), 4)

        regions = wins[customer]
        lat.customer_speech_s = round(float(sum(b - a for a, b in regions)), 2)

        mask = np.zeros(x.size, dtype=bool)
        for a, b in regions:
            mask[int(a * SR):min(x.size, int(b * SR))] = True
        ctx.cache["customer_audio"] = x[mask] if mask.any() else x
        ctx.cache["customer_regions"] = regions
        ctx.cache["agent_regions"] = [w for k in range(d.n_speakers)
                                      if k != customer for w in wins[k]]

        lat.notes.append(
            "roles: customer=cluster%d  gap_floor=%s dBFS  level_sd=%s dB  "
            "first_speaker=cluster%d  margin=%.2f" % (
                customer,
                "/".join(f"{floors[k]:.0f}" for k in sorted(floors)),
                "/".join(f"{varis[k]:.1f}" for k in sorted(varis)),
                first, margin) + lexical_note)
