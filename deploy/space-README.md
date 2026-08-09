---
title: AutoAce Voice Tone
emoji: 🎧
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
short_description: Emotional tone and background noise analysis for call recordings
---

# AutoAce — Voice Tone & Background Noise

Upload call recordings, get nine structured fields per call: emotional tone and
intensity, background noise presence, type and severity, audio quality, speaker
overlap, long silences, and a calibrated confidence.

All inference runs inside this container. No audio is sent to any third-party
API. See `MEMO.md` in the repository for architecture, validation and cost.

**Sign in** with the `DASHBOARD_USER` / `DASHBOARD_PASS` values configured as
Space secrets.
