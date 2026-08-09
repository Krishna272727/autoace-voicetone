"""Shared speaker analysis: windowed embeddings, clustering, per-cluster stats.

Phases 4, 5 and 6 all need to know *who is speaking when*, so it is computed
once here and cached on `ctx.cache["diar"]`.

The embedder is `microsoft/wavlm-base-plus-sv` (MIT, ungated). pyannote's
diarization pipeline would be the textbook choice, but its weights are gated
behind a Hugging Face account and manual terms acceptance, which BUILD_SPEC 4.4
explicitly says not to stall on.

Windows are fixed-length and short. An earlier attempt embedded whole VAD
segments and clustered those; on call_002 that produced a similarity matrix
with no clean two-cluster structure, because a single VAD segment routinely
spans a turn change and its embedding is then an average of both speakers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..audio import AudioContext
from .vad import timeline

log = logging.getLogger("autoace.diarize")

SR = 16_000
WIN_S = 1.25                # long enough for a stable embedding
HOP_S = 0.60                # short enough to land inside a single turn
MIN_WIN_S = 0.60
MAX_WINDOWS = 240           # bounds cost on hour-long files

# Below this mean inter-cluster distance, the "two speakers" split is not real
# and the file is treated as single-speaker.
TWO_SPEAKER_MIN_DIST = 0.22
# The smaller cluster must hold at least this share of windows to be a speaker.
MIN_CLUSTER_FRAC = 0.12


@dataclass
class Diarization:
    windows: list[tuple[float, float]] = field(default_factory=list)
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    n_speakers: int = 0
    separation: float = 0.0          # mean cosine distance between centroids
    ok: bool = False

    def cluster_windows(self, k: int) -> list[tuple[float, float]]:
        return [w for w, lb in zip(self.windows, self.labels) if lb == k]

    def speech_s(self, k: int) -> float:
        return float(sum(b - a for a, b in self.cluster_windows(k)))


def _windows(ctx: AudioContext) -> list[tuple[float, float]]:
    """Fixed-length windows lying inside speech regions."""
    out: list[tuple[float, float]] = []
    for a, b in timeline(ctx).speech:
        t = a
        while t + MIN_WIN_S <= b:
            out.append((t, min(t + WIN_S, b)))
            t += HOP_S
    if len(out) > MAX_WINDOWS:
        idx = np.linspace(0, len(out) - 1, MAX_WINDOWS).astype(int)
        out = [out[i] for i in idx]
    return out


def _embed(x: np.ndarray, wins: list[tuple[float, float]]) -> np.ndarray:
    """Speaker embedding per window, one forward pass each.

    Batching was tried and reverted, measured on call_003 (193 windows):

        sequential  27.55 s
        batched     60.18 s      0.46x -- SLOWER
        cosine similarity vs sequential: min 0.912

    Both halves of that are disqualifying. It is slower because these windows
    are 0.6-1.25 s and vary in length, so padding a batch to its longest member
    adds more compute than the per-call overhead it saves -- the arithmetic
    dominates, not the dispatch. And it is wrong because WavLM's x-vector head
    pools statistics over the time axis, which includes the padding: an
    attention mask does not stop the TDNN and stats-pooling layers from seeing
    padded frames, so a padded window gets a different speaker embedding.

    A cosine of 0.912 between "the same window, padded differently" is enough
    to move a window between clusters, which is a role-assignment error, which
    is customer emotion measured on the agent's voice.
    """
    import torch

    from ..models import speaker_embedder
    fe, model = speaker_embedder()
    vecs = []
    for a, b in wins:
        seg = x[int(a * SR):int(b * SR)]
        if seg.size < int(MIN_WIN_S * SR):
            seg = np.pad(seg, (0, int(MIN_WIN_S * SR) - seg.size))
        inp = fe(seg, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            e = model(**inp).embeddings[0].numpy()
        n = np.linalg.norm(e)
        vecs.append(e / n if n > 0 else e)
    return np.asarray(vecs, dtype=np.float32)


def _cluster(E: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Two-way agglomerative clustering on cosine distance.

    Returns (labels, centroids, separation). A weak split is reported by a low
    separation so the caller can fall back to one speaker.
    """
    from sklearn.cluster import KMeans
    if E.shape[0] < 4:
        return (np.zeros(E.shape[0], np.int32),
                E.mean(axis=0, keepdims=True) if E.size else E, 0.0)

    # KMeans on L2-normalised vectors is spherical k-means, i.e. cosine.
    # Average-linkage agglomerative was tried first and produced singleton
    # outlier clusters rather than speaker splits -- 192/1 and 11/1 on files
    # that plainly contain two talkers. KMeans gives balanced, usable splits
    # (18/20, 81/112) on the same embeddings.
    # n_init and random_state are fixed: the determinism test requires
    # byte-identical output across runs.
    labels = KMeans(n_clusters=2, n_init=10,
                    random_state=0).fit_predict(E).astype(np.int32)

    # A split where one side is a handful of windows is an outlier group, not a
    # second speaker.
    counts = np.bincount(labels, minlength=2)
    if counts.min() < max(2, MIN_CLUSTER_FRAC * len(labels)):
        labels = np.zeros(len(labels), np.int32)
    cents = []
    for k in sorted(set(labels.tolist())):
        sel = E[labels == k]
        c = sel.mean(axis=0) if sel.size else np.zeros(E.shape[1], np.float32)
        n = np.linalg.norm(c)
        cents.append(c / n if n > 0 else c)
    C = np.asarray(cents, dtype=np.float32)
    sep = float(1.0 - C[0] @ C[1]) if C.shape[0] > 1 else 0.0
    return labels, C, sep


def diarize_segments(ctx: AudioContext,
                     segments: list[tuple[float, float]]) -> Diarization:
    """Cluster caller-supplied segments instead of fixed windows.

    Used with ASR utterance boundaries, which are turn-aligned. Fixed windows
    are not: at 1.25 s with a 0.60 s hop they straddle turn changes, and the
    resulting clusters were measured mixing both speakers badly enough that the
    "customer" transcript for call_001 came back as the bot's greeting.
    """
    d = Diarization()
    segs = [(a, b) for a, b in segments if b - a >= MIN_WIN_S]
    if len(segs) < 2:
        d.windows = segs
        d.labels = np.zeros(len(segs), np.int32)
        d.n_speakers = 1 if segs else 0
        d.ok = bool(segs)
        return d
    if len(segs) > MAX_WINDOWS:
        idx = np.linspace(0, len(segs) - 1, MAX_WINDOWS).astype(int)
        segs = [segs[i] for i in idx]

    x = np.asarray(ctx.speech, dtype=np.float32).ravel()
    try:
        E = _embed(x, segs)
    except Exception as exc:                       # noqa: BLE001
        log.info("speaker embedder unavailable: %s", exc)
        return d

    labels, cents, sep = _cluster(E)
    d.windows, d.labels, d.embeddings, d.centroids = segs, labels, E, cents
    d.n_speakers = int(len(np.unique(labels)))
    d.separation = sep
    d.ok = True
    return d


def diarize(ctx: AudioContext) -> Diarization:
    cached = ctx.cache.get("diar")
    if isinstance(cached, Diarization):
        return cached

    d = Diarization()
    ctx.cache["diar"] = d                  # cache early: failures cache too
    wins = _windows(ctx)
    if len(wins) < 2:
        d.n_speakers = 1 if wins else 0
        d.windows = wins
        d.labels = np.zeros(len(wins), np.int32)
        d.ok = bool(wins)
        return d

    x = np.asarray(ctx.speech, dtype=np.float32).ravel()
    try:
        E = _embed(x, wins)
    except Exception as exc:                       # noqa: BLE001
        log.info("speaker embedder unavailable: %s", exc)
        return d

    labels, cents, sep = _cluster(E)
    if sep < TWO_SPEAKER_MIN_DIST:
        # The split is not supported by the audio: one voice, or one voice plus
        # noise. Reporting two speakers here would corrupt role assignment.
        labels = np.zeros(len(wins), np.int32)
        c = E.mean(axis=0)
        n = np.linalg.norm(c)
        cents = np.asarray([c / n if n > 0 else c], dtype=np.float32)

    d.windows = wins
    d.labels = labels
    d.embeddings = E
    d.centroids = cents
    d.n_speakers = int(len(np.unique(labels)))
    d.separation = sep
    d.ok = True
    return d
