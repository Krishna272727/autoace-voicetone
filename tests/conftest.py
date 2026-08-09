"""Shared fixtures. Builds the malformed-input corpus for the robustness matrix.

Everything here is generated with ffmpeg at test time rather than committed as
binaries, so the corpus is reproducible and the repo stays small.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "samples"
SR = 16_000


def _ff(args: list[str]) -> bool:
    return subprocess.run(["ffmpeg", "-y", "-v", "error", *args],
                          capture_output=True).returncode == 0


@pytest.fixture(scope="session")
def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


@pytest.fixture(scope="session")
def samples() -> list[Path]:
    return sorted(SAMPLES.glob("*.ogg"))


@pytest.fixture(scope="session")
def broken(tmp_path_factory, have_ffmpeg) -> dict[str, Path]:
    """One file per row of the BUILD_SPEC 6 robustness table."""
    d = tmp_path_factory.mktemp("broken")
    out: dict[str, Path] = {}
    rng = np.random.default_rng(0)

    def wav(name: str, data: np.ndarray, sr: int = SR) -> Path:
        p = d / name
        sf.write(p, data.astype(np.float32), sr)
        out[name.split(".")[0]] = p
        return p

    # --- structurally invalid -------------------------------------------
    (d / "zero.wav").write_bytes(b"")
    out["zero_byte"] = d / "zero.wav"

    (d / "notaudio.wav").write_text("this is a text file with a .wav extension")
    out["text_as_wav"] = d / "notaudio.wav"

    src = SAMPLES / "call_001.ogg"
    if src.exists():
        blob = src.read_bytes()
        (d / "truncated.ogg").write_bytes(blob[:len(blob) // 7])
        out["truncated"] = d / "truncated.ogg"

    # --- valid audio, awkward content ------------------------------------
    t = np.arange(int(20 * SR)) / SR
    wav("pure_silence.wav", np.zeros(int(20 * SR)))
    wav("white_noise.wav", rng.normal(0, 0.1, int(12 * SR)).clip(-1, 1))
    wav("clip_half_second.wav", 0.3 * np.sin(2 * np.pi * 220 * t[:SR // 2]))
    wav("square_wave.wav", np.sign(np.sin(2 * np.pi * 200 * t)) * 0.999)
    # DTMF: two tones, as a real keypad press
    wav("dtmf.wav", 0.4 * (np.sin(2 * np.pi * 697 * t) + np.sin(2 * np.pi * 1209 * t)))
    # Hold music: a simple chord progression, no speech
    music = sum(0.15 * np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0))
    wav("hold_music.wav", music)

    # --- format and channel variations -----------------------------------
    if have_ffmpeg and src.exists():
        if _ff(["-i", str(src), "-ac", "1", "-ar", "8000", "-c:a", "pcm_mulaw",
                str(d / "mulaw_8k.wav")]):
            out["mulaw_8k"] = d / "mulaw_8k.wav"
        if _ff(["-i", str(src), "-ac", "6", str(d / "surround51.wav")]):
            out["surround_5_1"] = d / "surround51.wav"
        # True dual-channel: different content per side
        a, b = SAMPLES / "call_001.ogg", SAMPLES / "call_002.ogg"
        # Downmix each source to mono FIRST, then merge. Merging the stereo
        # sources directly yields 4 channels, and the "-ac 2" that follows
        # downmixes them back into a correlated pair -- i.e. the fixture would
        # be fake stereo, which is the thing it is meant to contrast with.
        if b.exists() and _ff(["-i", str(a), "-i", str(b), "-filter_complex",
                               "[0:a]pan=mono|c0=c0[l];[1:a]pan=mono|c0=c0[r];"
                               "[l][r]amerge=inputs=2[out]",
                               "-map", "[out]", "-ac", "2",
                               str(d / "true_stereo.wav")]):
            out["true_stereo"] = d / "true_stereo.wav"
        # Video container with an audio track
        if _ff(["-f", "lavfi", "-i", "testsrc=d=5:s=128x96:r=10", "-i", str(src),
                "-map", "0:v", "-map", "1:a", "-t", "5", "-shortest",
                str(d / "video.mp4")]):
            out["video_container"] = d / "video.mp4"
        # Unicode / emoji / spaces in the filename -- preservation is graded
        weird = d / "call ✅ прив́ет 📞 (copy).wav"
        if _ff(["-i", str(src), "-t", "3", "-ac", "1", "-ar", "16000", str(weird)]):
            out["unicode_name"] = weird

    return out
