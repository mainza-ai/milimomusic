---
title: Neural Singing Voice Conversion (Neural SVC)
type: entity
created: 2026-09-15
updated: 2026-09-15
sources: [sources/v2-refactor-plan.md, sources/readme.md]
tags: [voice, svc, neural, timbre, audio, pitch, formant]
aliases: [Neural SVC, NeuralSVCService, Singing Voice Conversion Engine]
---

# Neural Singing Voice Conversion (Neural SVC)

The **Neural SVC Engine** (`backend/app/services/voice/neural_svc.py`) provides pitch-adaptive vocal timbre transfer, formant morphing, and zero-shot singer identity conversion.

## Background & Evolution

Earlier prototypes in Milimo Music approximated voice conversion via biquad peaking EQ filters. The `NeuralSVCService` replaces that approximation with a true neural acoustic synthesis model:
1. **Pitch Transposition & Phase-Locked Pitch Shifting**: Calculates semi-tone transposition ratios between source vocals and the target voice profile's natural pitch range using Phase Vocoder processing.
2. **Formant Spectral Envelope Warping**: Extracts and morphs the spectral tilt and envelope (via LPC / spectral cepstrum) to impart the target vocal tract resonance and acoustic brightness without distorting pitch stability.
3. **Cross-fade Dry/Wet Blending**: Seamlessly balances converted target character against the performer's original expressive dynamic range.

## Core Interface

```python
class NeuralSVCService:
    @classmethod
    def convert_voice(
        cls,
        source_audio_path: str,
        profile_sample_path: str,
        out_path: str,
        pitch_shift_semitones: float = 0.0,
        formant_shift: float = 0.0,
        dry_wet: float = 1.0,
        f0_method: str = "crepe",
    ) -> Dict[str, Any]:
        """Convert singing timbre of source vocal stem to match target voice profile."""
        ...
```

## Integration with Voice Service

In [VoiceService](voice-service.md) (`backend/app/services/voice_service.py`), when `convert_vocals()` is invoked:
- Verifies artist profile consent and ownership.
- Resolves the target profile reference WAV and source isolated vocal stem.
- Runs `NeuralSVCService.convert_voice` to synthesize the converted vocal.
- Emits real-time SSE progress events (`voice_conversion_progress`).
- Generates `/audio/converted_vocals/<id>.wav` mounted for DAW auditioning.

## Related Pages

- [Voice Service](voice-service.md)
- [Singing Voice Conversion Concept](../concepts/singing-voice-conversion.md)
- [Hardware Coordinator](hardware-coordinator.md)
- [Session Workspace](session-workspace.md)
