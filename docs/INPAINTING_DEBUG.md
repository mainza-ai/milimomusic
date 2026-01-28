# In-Painting & Glitch Repair Debug Log

## Current Status (2026-01-24)
- **State**: LM-Guided Repair fully implemented and stable.
- **Strategy**: Two-stage generation using `HeartMuLa` (semantic tokens) + `HeartCodec` (acoustic reconstruction).
- **Parameters**: Near-deterministic sampling (`temp=0.2`, `topk=30`, `cfg_scale=1.0`).
- **Pending**: User verification of vocal consistency after latest parameter tuning.

---

## The Final Solution: LM-Guided Repair

### Why Previous Approaches Failed
| Approach | Result | Failure Reason |
|----------|--------|----------------|
| `mask=0` (Blind) | Silence | Codec treats null tokens as silence |
| `mask=2` + Copy tokens | Repetition | Copied tokens = repeated content |
| Latent Blur | Phase smear | Destructive post-processing |
| Gradient Masking | Silence | Gradient values < 1.0 kill signal |

### The Two-Stage Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│              LM-GUIDED IN-PAINTING PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: SEMANTIC GENERATION (HeartMuLa)                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Input:  history_tokens (8s before gap)                   │   │
│  │         + original lyrics + style tags                   │   │
│  │ Output: NEW tokens for the gap (not copied!)             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Stage 2: ACOUSTIC RECONSTRUCTION (HeartCodec)                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Input:  [8s context] + [new tokens] + [8s context]       │   │
│  │ Output: Phase-aligned audio with 100ms crossfade         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete Fix History

### Phase 1: Infrastructure & Stability
- [x] Implement `InpaintingService` - backend service for repair jobs
- [x] Expose `HeartCodec.inpaint` - masking support in codec
- [x] Expose `HeartCodec.encode` - re-tokenization support
- [x] Fix mono/stereo dimension errors (`[1, 1, T]` constraint)
- [x] Fix SQLite 32-bit integer overflow (`torch.seed()` clamping)

### Phase 2: MPS / Mac Compatibility
- [x] CPU offload for `ScalarModel.decode` (MPS conv1d limit)
- [x] CPU offload for `ScalarModel.encode`
- [x] CPU offload for RNG (MPS placeholder storage bug)
- [x] Device type safety check (`torch.device(device)`)

### Phase 3: Codec-Only Attempts (All Failed)
- [x] Mask `0` → Silence
- [x] Mask `2` + Token Copy → Repetition
- [x] Latent Blur → Phase smear
- [x] Gradient Masking → Signal collapse
- [x] Feature Embedding Smoothing → Insufficient

### Phase 4: LM-Guided Repair (Current Solution)
- [x] Implement two-stage pipeline (HeartMuLa → HeartCodec)
- [x] Fix `model_inputs=` keyword argument error
- [x] Fix empty lyrics tokenizer crash (placeholder `"..."`)
- [x] Fix `save_path=None` crash in pipeline postprocess
- [x] Fix 2D/3D token shape normalization (`[C,T]` vs `[B,C,T]`)
- [x] Fix ellipsis slicing for multi-channel crossfade
- [x] Inject parent job's `lyrics` and `tags` for style matching
- [x] Tune parameters for vocal consistency

### Phase 5: Parameter Tuning (Audio Dominance Mode)
| Parameter | Initial | Tuned | Purpose |
|-----------|---------|-------|---------|
| `temperature` | 1.0 | **0.2** | Near-deterministic sampling |
| `topk` | 250 | **30** | Narrow beam for stable output |
| `cfg_scale` | 1.5 | **1.0** | Disable text guidance; rely on audio history |

---

## Tensor Shape Reference

### HeartMuLa Output Formats
```python
# Some pipeline calls return [C, T]
generated_tokens.shape = [8, num_frames]

# Some return [B, T, C]  
generated_tokens.shape = [1, num_frames, 8]

# Normalization to consistent [1, C, T]:
if generated_tokens.dim() == 2:
    generated_tokens = generated_tokens.unsqueeze(0)  # [C,T] → [1,C,T]
elif generated_tokens.dim() == 3 and generated_tokens.shape[-1] == 8:
    generated_tokens = generated_tokens.transpose(1, 2)  # [B,T,C] → [B,C,T]
```

### Crossfade Slicing (Robust Ellipsis)
```python
# WRONG: Slices channels instead of time for stereo
new_wav[start:end]

# CORRECT: Always slices last dimension (time)
new_wav[..., start:end]
```

---

## Key Learnings

1. **HeartCodec `inpaint()` is for acoustic repair only** - it cannot generate new semantic content.
2. **HeartMuLa is the semantic generator** - use it to create fresh tokens for the gap.
3. **8-second context window** is essential for phase alignment in `HeartCodec`.
4. **`cfg_scale=1.0` disables text guidance** - forces model to rely on audio history.
5. **Lower temperature = more deterministic** - critical for matching existing vocals.
6. **Check tensor dimensions religiously** - `heartlib` API returns inconsistent shapes.
