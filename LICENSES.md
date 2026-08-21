# Milimo Music — Licensing Matrix

The **Milimo Music platform codebase** is open-source and licensed under the **[Apache License 2.0](LICENSE)** by **[Mainza Kangombe](https://www.linkedin.com/in/mainza-kangombe-6214295)**.

This document summarizes the licensing terms for the Milimo Music application code, bundled third-party libraries, and upstream neural model weights.

---

## 1. Core Platform

| Component | Repository / Origin | Code License | Weight / Model License | Permitted Use |
|---|---|---|---|---|
| **Milimo Music Platform** | `mainza-ai/milimomusic` | **Apache-2.0** | N/A | Permissive open-source (commercial & non-commercial application code use) |
| **Heartlib Engine** | `mainza-ai/milimomusic/heartlib` | **Apache-2.0** | Apache-2.0 | Local generation & training |
| **MuScriptor Integration** | `muscriptor/muscriptor` (Kyutai × Mirelo) | **MIT** | **CC BY-NC 4.0** | Code is MIT; pre-trained neural weights are Non-Commercial only |

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

---

## 5. Bundled Prompt Assets

| Asset | Repository / Origin | License | Notes |
|---|---|---|---|
| **Music caption library** | `skills/music-caption-rewriter` in `MiniMax-AI/MiniMax-Music3` (genre router + 18 family indexes + ~1,000 caption templates), vendored at `backend/data/caption-library/` | **No LICENSE file in the upstream repo** — treat as MiniMax Open Weights / non-commercial project use per MiniMax Music 3 posture | Used to few-shot the in-app caption rewriter (`POST /generate/rewrite_caption`); synthetic captions are newly generated, not copies of templates |
| **Official MiniMax prompting guide** | `multimodalart/minimax-music3-prompting-guide` (HF Space) | Static guide, no license file | Basis for the three-heading caption contract and lyric tag-on-own-line rule; not vendored (referenced only) |
