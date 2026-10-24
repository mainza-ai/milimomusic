---
title: Track Extension Troubleshooting
type: concept
created: 2026-10-24
updated: 2026-10-24
sources: []
tags: [extension, troubleshooting, continuation, tempo, groove, replay, forensics]
aliases: [Extension debugging, continuation issues]
---

# Track Extension Troubleshooting

Symptom-driven guide to diagnosing and fixing track-extension continuity failures.
For the mechanism itself see [Track Extension](track-extension.md).

## Symptom map

| # | Symptom | Most likely root cause | Fix |
|---|---------|------------------------|-----|
| 1 | Tail continues but at the wrong tempo/beat ("off", "not at the exact cut") | Diverged replay: conditioning text differs from parent run (§R1/R2) | Verify with replay check; fix conditioning or use stem outro |
| 2 | Extension restarts the song instead of continuing | Full-timeline flag wrong (`extension_is_full_timeline`) or fallback rendered | Check `extension_mode`; see §R-fallback |
| 3 | Change lands 1–2s *before* the chosen cut point | Crossfade overlap (tail starts at `cut − crossfade`) + beat-grid snap moved the cut | Expected by design when grooves match; suspect when they don't |
| 4 | `replay_verified: false` on a completed extension | Rendered head doesn't match parent audio — do not trust the tail | See forensic recipe below; route to fallback or stem outro |
| 5 | `extension_mode: fresh_segment_fallback` unexpectedly | MLX/snapshot/seed unavailable, or preflight check failed | Check backend log for the reason; preflight detail names it |
| 6 | Tail sings nothing / wrong words (`tail_lyrics_ok: false`) | Remainder slice empty (past last sung line) or tail diverged | Move cut earlier, add continuation lyrics, or accept instrumental tail |
| 7 | Extension job FAILED with `TypeError: multiple values for 'cancel_event'` | Prefill call bug (fixed 2026-10-24) | Update past the fix; state_restored path works after it |
| 8 | DAW bars don't line up with the audio | Stored beat grid is the 120-BPM fallback (`first_downbeat: 0.0`) | Repair grid (auto at extend time); see §R-grid |

## Forensic recipe (proving divergence vs blaming code)

All commands run from the repo root with the backend venv Python
(`/opt/miniconda3/envs/milimomusic/bin/python`).

**1. Locate the seam.** The splice keeps parent disk audio bit-exact up to the
seam, so the first differing sample *is* the seam:

```python
import wave, numpy as np
def load(f):
    w = wave.open('generated_audio/'+f, 'rb')
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1,2)
    return a.astype(np.float32).mean(axis=1) / 32768.0
p, c = load('PARENT.wav'), load('CHILD.wav')
diff = np.nonzero(c[:len(p)] != p)[0]
print('seam at %.4fs' % (diff[0] / 44100 if len(diff) else -1))
```

Expected: seam ≈ `extend_from_sec − crossfade_sec` (1.5 s overlap by design).
A seam far earlier/later means the cut moved (check beat-grid snap) or the
wrong splice path ran (check `extension_is_full_timeline` vs rendered audio).

**2. Compare groove tracker-free.** Beat trackers misread swung/syncopated
material (classic 3:2 error: a 107.7 BPM groove reads as 161.5). Use onset
autocorrelation on both sides instead of trusting one tracker number:

```python
import librosa
def pulse(y, sr=44100):
    o = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    ac = librosa.autocorrelate(o)
    f = librosa.tempo_frequencies(len(ac), hop_length=512, sr=sr)
    m = (f >= 50) & (f <= 220); i = np.nonzero(m)[0]
    return float(f[i[np.argmax(ac[i])]])
```

If parent pulse ÷ tail pulse (allowing 1/2/2×/1.5× metrical harmonics) differs
by more than ~3%, the tail is a different performance, not a continuation.

**3. Confirm determinism before blaming inputs.** Re-run 300 AR frames twice
with identical inputs and compare codes; then compare against the run's saved
`.mm3state.npz` bundle. Equal ⇒ the machinery is faithful and the *inputs*
differ from the parent run. Unequal ⇒ nondeterminism (weights/snapshot/seed).

**4. Check what the parent run actually used.** The stored `lyrics`/`prompt`
are display copies, not model inputs: since 2026-08-20 the pipeline persists
`extract_final_lyrics(eff_lyrics)`, a lossy cleaner that is *not idempotent* on
lyrics with >14-word lines. New tracks also store the exact model strings in
`model_prompt`/`model_lyrics` — prefer those; their absence on old tracks means
reconstruction is guesswork, and the preflight check will say so.

## Root-cause catalog

### R1 — Remainder slice steers the replay (fixed 2026-10-24)

`split_remaining_lyrics` returns transcription-normalized text (rewritten
headers, reflowed lines) that is generally *not* a byte-substring of the stored
lyrics. The old rule appended anything not contained, so every vocal extension
with timed lyrics conditioned the replay on `parent + remainder` — a different
song from frame 0. The remainder now travels as `tail_lyrics` (fallback-only);
replay conditions on verbatim parent lyrics; genuine additions are flagged
`conditioning_steered`.

### R2 — Lossy conditioning persist (structural; fixed going forward)

The pipeline overwrote `job.lyrics` with the *cleaned* model input while
discarding the exact string. Any >14-word line makes the stored text
unreproducible (`extract(extract(x)) != extract(x)` on such tracks). New tracks
persist `model_prompt`/`model_lyrics` verbatim and extend prefers them; old
affected tracks (e.g. CROSS) cannot be replayed from stored data, period.

### R3 — Dead beat grid

MusiCtor beat detection fails on swung/syncopated material and stores the
`{120 BPM, 4/4, downbeat 0.0}` fallback. Cut snapping, crossfade clamping, and
DAW bars built on it are phase-arbitrary. The extend route now repairs the grid
from the audio (librosa + measured downbeat phase) and writes the repair back.

### R4 — Flow-decode edge at the seam

`_crop_waveform` keeps the right edge only on the last chunk, and chunk count
depends on total length — so even a bit-perfect replay decodes slightly
differently exactly at the seam. Small, but it lives in the worst place.

### R5 — Flow step mismatch (parent 30 vs render 29)

Same AR codes decode to slightly different audio. Correlation stays ~0.99, so
the verification threshold tolerates it, but identical steps would remove it.

### R6 — Fractional-beat crossfade

A 1.5 s overlap is 2.8 beats at 112.3 BPM — phase smear even on a perfect tail.
The overlap is now quantized to whole beats of the (repaired) grid.

### R7 — No audio grounding

Unlike MusicGen/Suno/Udio continuation (audio tokens or audio clips as
context), this stack replays from text only: exact or useless, no graceful
degradation. The mlx MiniMax wrapper exposes no audio encoder, so audio-token
prefill is unavailable without new weights.

### R8 — Prediction ≠ verification

`expected_mode` predicts the path, never fidelity. A preflight probe now
renders ~30 s and compares *before* the full run; diverged reconstructions
route to labeled fallback instead of burning 28 min of GPU.

## CROSS case study (2026-10-24)

Parent `5b7dc92b` (109.13 s, true groove ~112.3 BPM swung). Nine extension
attempts all produced tails at unrelated tempi (161.5/92.3/117.5-pulse
readings) starting 1.5 s before the cut. Evidence chain:

1. Seam forensics: head bit-identical to 107.49 s, then new audio — splice OK.
2. Tracker-free pulse: parent 166.7 (triplet pulse of 112.3) vs tails at
   different pulses — different performances, not continuations.
3. Logic proof: remainder slice not contained in stored lyrics ⇒ old rule
   appended it ⇒ steering confirmed on the actual strings.
4. Determinism proof: fresh 300-frame replay == 18:37 run's saved codes
   (cross-process), yet parent-regeneration with reconstructed inputs gives
   corr ≈ −0.008 vs parent disk ⇒ stored inputs ≠ parent model inputs.
5. Persist path: `extract_final_lyrics` non-idempotent on this track
   (2768 → 953 chars) ⇒ lossy persist is the difference.

Resolution: replay impossible from stored data; published a stem-woven outro
(19-beat loop of the parent's own drums/bass/keys + full-mix variants and a
hook loop) with phase-aligned junctions, verified bit-exact (head 1.0,
tail-vs-source 1.0, pulse 166.7). Kept: parent CROSS + `1745d0b1`
(`stem_outro`, verified). Removed: all nine diverged/failed extension jobs and
their orphaned files (API cascade misses backend mirrors, `.mm3state.npz`,
`song_` mirrors — swept manually).

## Research context

- **MusicGen / MusicGen-Melody**: continuation feeds *audio tokens* of the
  prefix back into the transformer; melody conditioning as prefix. Without
  audio context, continuation quality is strictly weaker.
- **AIR ("Arrange, Inpaint, Refine", IJCAI 2024)**: measured that AR
  continuations "stray further" the longer they run (prefix-CLAP vs full-CLAP
  divergence) — our 109 s prefix + tail is the extreme long-prefix regime.
- **VampNet (masked acoustic modeling)**: inpainting/outpainting/continuation
  via bidirectional prompting — robust because context surrounds the gap.
  Our pure left-to-right text replay is the most fragile design in this space.
- **YuE / DiffRhythm-2 / ACE-Step**: long-form coherence comes from structural
  conditioning (sections with timing), semi-autoregressive blocks, and
  lyric-timing controls — not from replaying a full prefix.
- **Suno / Udio**: extend is *audio-conditioned* (clip + `continueAt`), built
  in ~30–60 s sections, stitched with "Get Whole Song". V6 even continues
  older models' audio — possible only because conditioning is the audio
  itself. Suno documents that some seams never fully disappear.
- Practical seam craft (DJ/studio practice): whole-beat overlaps, phase-aligned
  junctions, optional micro time-stretch (±1% inaudible), short equal-power
  crossfades on transients — all applied in the stem outro builds here.
