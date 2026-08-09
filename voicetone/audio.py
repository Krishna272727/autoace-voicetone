"""Audio I/O. Decode once, share everywhere.

Two derivatives are produced from every input:
  - `speech` : 16 kHz mono, for VAD / SER / ASR / tagging
  - `master` : native-rate mono float, for DSP quality + noise analysis

Quality analysis must NOT run on the 16 kHz copy: resampling discards
everything above 8 kHz, which is exactly where hiss, static and codec
artifacts live.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SPEECH_SR = 16_000
STEREO_CORR_THRESHOLD = 0.98  # above this, channels are a duplicated downmix
# Only this much audio is decoded for the channel-correlation probe. Duplicated
# channels are duplicated from the first sample; an hour of them proves nothing
# a minute does not.
STEREO_PROBE_S = 60.0

SUPPORTED = {".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a", ".aac",
             ".wma", ".webm", ".mp4", ".aiff", ".au"}


class AudioLoadError(RuntimeError):
    """Raised when a file cannot be decoded. Caught per-file by the pipeline."""


@dataclass
class AudioContext:
    """Everything downstream predictors need. Built once per file."""
    path: Path
    duration: float
    native_sr: int
    n_channels: int
    true_stereo: bool           # genuine dual-channel telephony recording?
    channel_corr: float
    speech: np.ndarray          # 16 kHz mono
    speech_sr: int
    master: np.ndarray          # native-rate mono
    master_sr: int
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def customer(self) -> np.ndarray:
        """Customer-only 16 kHz audio. Falls back to full mix until Phase 5
        (role assignment) populates the cache."""
        return self.cache.get("customer_audio", self.speech)


def _ffprobe(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise AudioLoadError(f"ffprobe failed: {proc.stderr.strip()[:200]}")
    return json.loads(proc.stdout)


def _decode(path: Path, sr: int | None, out: Path,
            channels: int = 1, limit_s: float | None = None) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-map", "0:a:0",
           "-ac", str(channels), "-c:a", "pcm_f32le"]
    if sr:
        cmd += ["-ar", str(sr)]
    if limit_s:
        cmd += ["-t", str(limit_s)]
    cmd += [str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioLoadError(f"ffmpeg failed: {proc.stderr.strip()[:200]}")


def load(path: str | Path) -> AudioContext:
    path = Path(path)
    if not path.exists():
        raise AudioLoadError(f"file not found: {path}")
    if path.suffix.lower() not in SUPPORTED:
        raise AudioLoadError(f"unsupported extension: {path.suffix}")

    meta = _ffprobe(path)
    streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise AudioLoadError("no audio stream found")
    st = streams[0]
    native_sr = int(st.get("sample_rate", 48_000))
    n_ch = int(st.get("channels", 1))
    duration = float(meta.get("format", {}).get("duration", 0.0) or 0.0)
    if duration <= 0.0:
        raise AudioLoadError("zero-length audio")

    tmp = Path(tempfile.mkdtemp(prefix="autoace_"))
    try:
        m_path, s_path = tmp / "master.wav", tmp / "speech.wav"
        _decode(path, None, m_path)
        _decode(path, SPEECH_SR, s_path)
        master, m_sr = sf.read(str(m_path), dtype="float32")
        speech, s_sr = sf.read(str(s_path), dtype="float32")

        # --- stereo check ---------------------------------------------
        # Never blindly downmix: on a genuine dual-channel recording that
        # would average the two speakers together and destroy free separation.
        #
        # The correlation is measured on an ffmpeg-decoded copy, not by
        # reading the original with soundfile. soundfile only understands
        # libsndfile's own container list, so reading the source directly
        # threw "Format not recognised" on an MP4 with an audio track -- a
        # case BUILD_SPEC 6 requires to succeed. ffmpeg decodes it fine.
        corr, true_stereo = 1.0, False
        if n_ch >= 2:
            c_path = tmp / "channels.wav"
            _decode(path, None, c_path, channels=2, limit_s=STEREO_PROBE_S)
            raw, _ = sf.read(str(c_path), always_2d=True, dtype="float32")
            if raw.shape[1] >= 2:
                L, R = raw[:, 0], raw[:, 1]
                if L.std() > 1e-9 and R.std() > 1e-9:
                    corr = float(np.corrcoef(L, R)[0, 1])
                    if not np.isfinite(corr):
                        corr = 1.0
                true_stereo = corr < STEREO_CORR_THRESHOLD
            del raw
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return AudioContext(
        path=path, duration=duration, native_sr=native_sr, n_channels=n_ch,
        true_stereo=true_stereo, channel_corr=corr,
        speech=np.asarray(speech, dtype=np.float32), speech_sr=int(s_sr),
        master=np.asarray(master, dtype=np.float32), master_sr=int(m_sr),
    )
