# Milimo Music v2 — Licensing Matrix

Milimo Music is an open-source, non-commercial AI music generation and production DAW by Mainza Kangombe.

This document summarizes the licensing terms for the Milimo Music platform code, bundled third-party libraries, and supported model weights.

---

## 1. Core Platform

| Component | Repository / Origin | Code License | Weight License | Intended Use |
|---|---|---|---|---|
| **Milimo Music Platform** | `mainza-ai/milimomusic` | Apache-2.0 | N/A | Open-source, non-commercial |
| **Heartlib Engine** | `mainza-ai/milimomusic/heartlib` | Apache-2.0 | Apache-2.0 | Local generation & training |
| **MuScriptor Integration** | `muscriptor/muscriptor` (Kyutai × Mirelo) | MIT | CC BY-NC 4.0 | Music transcription & notation |

---

## 2. Supported Generation & Transcription Models

| Model | Architecture / Provider | Code License | Weight License | Notes |
|---|---|---|---|---|
| **MiniMax Music 3** | `mlx-community/MiniMax-Music3-bf16` / `MiniMaxAI/MiniMax-Music3` | Apache-2.0 | MiniMax Open Weights | Up to 5-minute full song generation with Structured Captions |
| **HeartMuLa-3B** | HeartMuLa Autoregressive LM + HeartCodec (12.5Hz) | Apache-2.0 | Apache-2.0 | Single-GPU / MPS local generation |
| **MuScriptor MT3** | Multi-instrument Automatic Music Transcription | MIT | CC BY-NC 4.0 | Non-commercial only; user responsible for uploaded audio rights |

---

## 3. Audio Processing & Production Tools

| Tool | Capability | License | Maintenance Signal |
|---|---|---|---|
| **BS-Roformer / MelBand-Roformer** | Fast 4-stem separation (Vocals, Drums, Bass, Other) | MIT / CC BY-NC | MSST unified separation framework |
| **Matchering** | Reference mastering (RMS, frequency, stereo width) | GPL-3.0 | Active DSP mastering library |
| **WhisperX** | Forced alignment for lyric-sync & karaoke (.lrc / .srt) | BSD-2-Clause | Active alignment toolkit |
| **RVC v2 / Applio** | Retrieval-based Singing Voice Conversion (SVC) | MIT | Applio maintained fork ecosystem |
| **OpenSheetMusicDisplay** | MusicXML rendering in web browser | MIT | Active open-source notation engine |

---

## 4. User Content & Rights Notice

- When transcribing uploaded audio with MuScriptor, users must hold or have secured necessary rights to the source recording.
- When training custom Voice Identities in the Voice Training Studio, users must explicitly certify that they own or have obtained lawful consent for the target voice.
