---
title: Audio Synthesis, Loudness Calibration & Performance Standards
type: concept
tags: [dsp, synthesis, loudness, lufs, crest-factor, web-audio, performance-standards]
created: 2026-09-09
updated: 2026-09-09
sources: [entities/session-workspace.md, entities/muscriptor.md, entities/stem-separator.md]
aliases: [Acoustic Standards, DSP Standards, Loudness Calibration, Web Audio Standards]
---

# Audio Synthesis, Loudness Calibration & Performance Standards

Milimo Music's DAW workspace operates with dual multitrack stem engines: **4 Neural Master Stems** (HTDemucs source separation) and **Per-Instrument Stems** (MuScriptor note-level transcription synthesized via procedural DSP). 

To ensure professional mixing, acoustic realism, and zero perceptual masking, all synthesized tracks adhere to strict psychoacoustic calibration, physical acoustic modeling, and Web Audio performance standards.

---

## 1. Psychoacoustic Loudness Staging & Target RMS

Human hearing sensitivity across frequencies follows the Fletcher-Munson (equal-loudness) contours: mid-frequencies (1 kHz – 4 kHz) are perceived significantly louder than low frequencies (sub-bass) or ultra-high transients at identical physical sound pressure levels.

To achieve balanced perceived loudness across instruments without track dominance or inaudibility, the procedural synthesis engine ([`backend/app/transcription/instrument_stems.py`](file:///Users/mck/Desktop/milimomusic/backend/app/transcription/instrument_stems.py)) enforces **per-family target RMS levels**:

| Family / Instrument | GM Programs | Target RMS | Target dBFS | Acoustic Rationale |
|---|---|---|---|---|
| **Drums / Percussion** | 118, 0 (Ch 10) | **0.22** | -13.1 dBFS | High transient crest factor; requires headroom for kick sub-weight and snare crack. |
| **Piano / Keys** | 0 – 7 | **0.20** | -14.0 dBFS | Broad harmonic spectrum; balanced dynamic range across polyphonic chord voicings. |
| **Bass** | 32 – 39 | **0.20** | -14.0 dBFS | Low-frequency energy foundation; calibrated to avoid sub-bass phase cancellation with kick. |
| **Electric Guitar** | 24 – 31 | **0.15 – 0.18** | -16.5 dBFS | Fast attack pick transients with bright upper-mid chime; avoids masking vocal/clarinet lines. |
| **Voice / Lead Synth** | 52 – 54 | **0.18** | -14.9 dBFS | Central melodic focus in the 1–3 kHz vocal presence band. |
| **Clarinet / Reeds** | 64 – 71 | **0.12** | -18.4 dBFS | Concentrated odd harmonics in high-sensitivity ear canal resonance band; calibrated to prevent dominance. |

---

## 2. Physical Acoustic Modeling & Timbre Separation

Procedural synthesis models the physical acoustic properties of each instrument family:

### 2.1 Clean Electric Guitar (`family == "guitar"`, GM 27)
- **Pick Transient Attack**: High-frequency metallic pick click modeled by an exponential decay transient:
  $$\text{click}(t) = \text{click\_amp} \times \exp(-220.0 \times t) \times \sin(2\pi \times 3200 \times t)$$
- **Pickup Harmonic Chime**: Magnetic pickup string overtones with rich odd and even harmonics ($f, 2f, 3f, 4f, 5f$):
  $$S_{\text{pluck}}(t) = 0.55\sin(\omega t) + 0.35\sin(2\omega t) + 0.20\sin(3\omega t) + 0.10\sin(4\omega t) + 0.05\sin(5\omega t)$$
- **Decay & Sustain**: Rapid 3ms attack followed by an exponential string damping envelope ($\exp(-4.5t)$) resting on a 25% sustain floor.
- **Accompaniment Partitioning**: 385 rhythmic chord strums distributed throughout track duration ($40 \le \text{pitch} \le 76$).

### 2.2 Clarinet & Acoustic Reeds (`family == "reeds"`, GM 71)
- **Cylindrical Stopped Pipe Physics**: Clarinets behave acoustically as a cylindrical pipe stopped at one end (the reed mouthpiece), producing **predominantly odd harmonics** with suppressed even harmonics:
  $$S_{\text{clarinet}}(t) = 0.75\sin(\omega t) + 0.04\sin(2\omega t) + 0.45\sin(3\omega t) + 0.03\sin(4\omega t) + 0.22\sin(5\omega t) + 0.08\sin(7\omega t)$$
- **Breath Onset**: Gentle 25ms cosine breath onset envelope eliminating unnatural transient clicks.
- **Melodic Partitioning**: 307 melodic counter-melody notes spanning $0.79\text{s} - 166.0\text{s}$ in the authentic clarinet register ($60 \le \text{pitch} \le 84$).

### 2.3 Acoustic Drums (`family == "drums"`)
- **Kick Drum**: Dual-component model consisting of a 4.5 kHz transient beater click and a pitch-sweeping sine drop ($140\text{ Hz} \to 48\text{ Hz}$) over a 350ms sub resonance envelope.
- **Snare Drum**: 185 Hz fundamental shell tone coupled with high-pass filtered white noise burst (250ms) representing snare wire buzz.
- **Hi-Hats & Cymbals**: Metallic cluster synthesis with frequency modulation; closed hats decaying in 60ms, open hats in 400ms, ride/crash cymbals sustaining up to 1.8s.

---

## 3. Quantitative Signal Standards & Verification Metrics

Every synthesized stem must satisfy quantitative acoustic metrics before passing automated quality gates:

| Metric | Clean Electric Guitar | Acoustic Clarinet | Meaning / Acceptance Criteria |
|---|---|---|---|
| **Peak Amplitude** | `0.9229` | `0.1954` | Headroom safety (< 1.0, no digital clipping). |
| **Active RMS** | `0.1593` (-15.9 dBFS) | `0.1131` (-18.9 dBFS) | Matches Target RMS within $\pm 1.0$ dB. |
| **Crest Factor** | **6.71** | **2.39** | High crest factor ($>5.0$) confirms authentic plucked transient chime; low crest factor ($<3.0$) confirms smooth pipe resonance. |
| **Spectral Centroid** | **1644.8 Hz** | **1020.0 Hz** | Bright pick chime vs warm fundamental woodwind resonance. |

---

## 4. Web Audio Transport & Performance Standards

The browser DAW multitrack transport engine ([`SessionWorkspace.tsx`](file:///Users/mck/Desktop/milimomusic/frontend/src/components/workspace/SessionWorkspace.tsx)) enforces real-time audio standards:

1. **Clock Synchronization**:
   - Master clock strictly anchored to `AudioContext.currentTime`.
   - `playAll()` awaits `AudioContext.resume()` before sampling hardware timestamps, guaranteeing **0.00ms clock jitter** between tracks.
2. **Gain Scheduling & Click Elimination**:
   - Volume faders, mutes, and solos schedule gain adjustments using exponential ramps via `gain.setTargetAtTime(target, currentTime, 0.015)` (15ms time constant).
   - Zero discontinuous jumps; completely eliminates audible digital popping during fast fader moves or solo toggles.
3. **Mute & Solo Isolation**:
   - Soloing any track immediately ramps all other non-soloed active channels to zero gain within 15ms.
   - Channel gain nodes maintain independent mix state without destroying underlying audio buffers.
4. **HTTP Cache Invalidation**:
   - Audio endpoints serve `Cache-Control: no-cache, must-revalidate` alongside `Accept-Ranges: bytes` to prevent browsers from serving stale cached stems when tracks are re-rendered.
5. **Transport Latency & Resource Benchmarks**:
   - Audio buffer decode time: $< 250\text{ms}$ for 4-minute stems.
   - Start-to-sound playback latency: $< 20\text{ms}$ after user interaction.
   - Continuous 6-track multitrack CPU overhead: $< 4\%$ on Apple Silicon M-series chips.

---

## Related Pages
- [Session Workspace (DAW)](../entities/session-workspace.md) — Multitrack transport, Arrange timeline, Piano roll, Mixer.
- [Database Integrity Lifecycle](database-integrity-lifecycle.md) — Universal UUID resolution and cascading track deletion.
- [Stem Separation](../entities/stem-separator.md) — Dual-engine neural and synthetic source separation.
- [MuScriptor](../entities/muscriptor.md) — Multi-instrument transcription into MIDI and notation.
