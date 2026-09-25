---
title: Director Mode v2 Architecture & Musical Timing
type: concept
tags: [director, video, beat-tracking, downbeats, pacing, lipsync, performer-map, timing, ffmpeg, fidelity-repair, vocal-bypass]
created: 2026-09-16
updated: 2026-09-24
sources: [sources/maestro-creative-studio.md]
aliases: [DirectorV2, BeatAwareDirecting, MusicalShotPlanner]
---

# Director Mode v2 Architecture & Musical Timing

**Director Mode v2** is Milimo Music's production-grade neural directing engine for AI music videos, inspired by the battle-tested architecture in [Maestro](../sources/maestro-creative-studio.md).

Unlike naive video segmenters that divide an audio track into arbitrary 5-second or 10-second chunks, Director Mode v2 treats music as a structured multi-signal timeline: beats, downbeats (bars), structural sections, energy curves, active vocal activity, and transient percussion entrances.

---

## 1. Multi-Signal Audio Analysis Pipeline

The analysis pipeline processes the master audio track and isolated vocal stem to construct an annotated musical context:

```
+-----------------------------------------------------------------------------------+
|                            Master Audio + Vocal Stem                              |
+-----------------------------------------+-----------------------------------------+
                                          |
     +-------------------+----------------+-------------------+-------------------+
     |                   |                                    |                   |
     v                   v                                    v                   v
[Beat & Downbeat] [Section & Energy]                  [Vocal Diarization]  [Percussion Cues]
 Librosa Onsets    Whisper + RMS Energy                Active Intervals     Transient Energy
 Tempo & Bars      Intro/Verse/Chorus/Bridge/Outro     Speaker Allocation   Drum Entrances
```

### 1.1 Beat and Downbeat Tracking
- **Beats**: Extracted via spectral flux onset envelopes using Librosa beat tracking.
- **Downbeats (Measure Starts)**: Detected using rhythmic bar-periodicity estimation. In $4/4$ time, every fourth beat constitutes a downbeat, receiving elevated structural weight.

### 1.2 Section Segmentation and Energy Scoring
- Combines lyric transcription timestamps with Root-Mean-Square (RMS) audio energy curves.
- Detects transitions between structural song sections: `[Intro]`, `[Verse]`, `[Chorus]`, `[Bridge]`, `[Solo]`, `[Outro]`.
- Each section is assigned an energy coefficient $E \in [0.0, 1.0]$, representing dynamic intensity.

### 1.3 Vocal Activity & Performer Diarization
- Maps active vocal spans from the isolated vocal stem (`vocals.wav`).
- When multiple vocalists or characters are defined, diarization assigns speaker identities to each lyric line, allowing the director to focus the camera on the active singer.

### 1.4 Percussion Entrances
- Analyzes transient high-frequency onset energy on the CPU.
- Flags sudden rhythm entrances (e.g. drums entering after a quiet acoustic intro), passing timestamps to the director to guide camera focus onto the drummer.

---

## 2. Scored Accent Boundary Snapping & Editorial Pacing

### 2.1 The Accent Scoring Function
To ensure cuts feel musically organic, potential cut points are evaluated using a hierarchical accent scoring function:

$$\text{Score}(t) = \text{Beat}(t) \cdot 0.5 + \text{Downbeat}(t) \cdot 1.8 + \text{LyricBoundary}(t) \cdot 2.5 + \text{Anchor}(t)$$

Where:
- $\text{Beat}(t)$: Regular rhythmic pulse ($+0.5$ weight).
- $\text{Downbeat}(t)$: Bar/measure boundary ($+1.8$ weight).
- $\text{LyricBoundary}(t)$: Natural pause between sung phrases or sentences ($+2.5$ weight).
- $\text{Anchor}(t)$: Mandatory musical boundaries (Section changes, performer switch points, drum entries).

### 2.2 Cut Speed / Pacing Bias Slider
The user-selectable **Cut Speed** parameter ($\text{bias} \in \{-2, -1, 0, 1, 2\}$) controls how aggressively these musical cues divide the song:

| Cut Speed | Behavior | Ideal Clip Length Formula | Editorial Result |
|---|---|---|---|
| **-2** | Ultra-Long Takes | $L_{\text{target}} = \text{Cap} \times 1.0$ | Uses absolute fewest clips that fit the model ceiling. Snaps cuts near major section changes. |
| **-1** | Extended Cinematic | $L_{\text{target}} = \text{Cap} \times 0.92$ | Favors longer continuous shots while honoring major harmonic transitions. |
| **0** | Balanced Musical | $L_{\text{target}} = \text{Cap} \times (0.92 - 0.20 \cdot E)$ | Natural musical layout; faster cuts during high-energy choruses, longer takes in verses. |
| **+1** | Dynamic Montage | $L_{\text{target}} = \text{Cap} \times (0.80 - 0.20 \cdot E)$ | Rhythmic cuts dividing sections on bar boundaries. |
| **+2** | Rapid Beat-Cuts | $L_{\text{target}} = \text{Cap} \times (0.68 - 0.20 \cdot E)$ | Rapid cutting landing on individual downbeats and lyric syllables. |

---

## 3. Discrete Frame Lattice Snapping & Output Trimming

Frontier video diffusion models (Wan 2.1, MiniMax H3, LTX-Video) operate on discrete temporal frame increments:

$$F_{\text{valid}} = F_{\text{min}} + k \cdot F_{\text{step}} \quad (k \in \mathbb{N}_0)$$

- **Wan 2.1**: Multiples of 4 frames at 16/24 fps.
- **MiniMax H3**: $1 + 8k$ frames at 24 fps (e.g. 81 frames $\approx 3.37\text{s}$, 161 frames $\approx 6.71\text{s}$, 241 frames $\approx 10.04\text{s}$, 345 frames $\approx 14.38\text{s}$).
- **LTX-Video**: Multiples of 8 or 32 frames at 24/25 fps.

### The Audio-Video Drift Trap
If an engine naively generates a 5.0-second video ($120$ frames) for a $4.82$-second musical bar, stitching multiple clips produces **cumulative audio drift** (over a 3-minute song, the video drifts by up to 10 seconds!).

### The Production Solution (`music_output_trim`)
1. **Lattice Snapping**: The director rounds the requested duration **up** to the nearest valid model frame count $F_{\text{gen}}$.
2. **Native Generation**: The video model generates all $F_{\text{gen}}$ frames without artifact-inducing interpolation.
3. **Lossless Sub-Second Trim**: FFmpeg trims the exact fractional tail:
   $$\text{TrimDuration} = t_{\text{cut}} - t_{\text{start}}$$
   $$\text{FFmpeg: } \texttt{-ss 0 -t \{TrimDuration\} -avoid_negative_ts make_zero}$$
This guarantees that audio and video remain **sample-accurate and drift-free** across the entire 3–5 minute song.

---

## 4. Performer Role Ownership & Solo Cutaway Rules

A common failure mode in AI music videos is the **"Flapping Lips" problem**: during a guitar solo or drum break, the visual model animates the lead singer moving their lips as if singing, or depicts a guitarist singing when they are only playing.

Director Mode v2 enforces strict performer ownership:

```python
def music_performance_direction(
    shot_type: str,
    active_speaker: Optional[str],
    has_vocals: bool,
    percussion_entrance: bool,
    instrumental_solo: Optional[str]
) -> Dict[str, str]:
    """Derive precise performer constraints for the LLM prompt."""
    if not has_vocals:
        if instrumental_solo:
            return {
                "camera_focus": f"Intimate dynamic focus on the {instrumental_solo} performance",
                "performer_action": f"Musician passionately playing the {instrumental_solo}, fingers flying over strings/keys",
                "vocal_constraint": "Mouth strictly closed, no singing, no lip movement, pure instrumental focus"
            }
        elif percussion_entrance:
            return {
                "camera_focus": "Rhythmic cutaway to drummer striking cymbals and snares on the downbeat",
                "performer_action": "Drummer executing energetic percussion fill in time with the rhythm",
                "vocal_constraint": "No vocal performance, lips closed"
            }
        else:
            return {
                "camera_focus": "Atmospheric cinematic B-roll, environmental wide shot",
                "performer_action": "Band members in rhythmic groove, instrumental performance",
                "vocal_constraint": "No vocal performance on screen"
            }
    else:
        return {
            "camera_focus": f"Medium close-up tracking shot centered on lead vocalist ({active_speaker or 'singer'})",
            "performer_action": f"Vocalist emotively singing lyrics with natural jaw articulation and eye contact",
            "vocal_constraint": f"Lip-sync exclusively reserved for {active_speaker or 'singer'}; background members do not lip-sync"
        }
```

### 4.2 Visible-Cast Scoping (v2.4.0)
In Maestro v2.4.0, performance instructions were upgraded from global scene injection to **strict per-shot visible-cast scoping**:
- Narrative, dancing, and scenery shots no longer inherit boilerplate lists of vocalists and instrumentalists.
- Only the performers actually staged and visible within that specific shot receive performance constraints (`mouth_movement: closed` vs active emotive singing).
- Recompiling saved music prompts strips out old injected boilerplate, keeping diffusion context compact and focused.

### 4.3 Music Timeline Vocal Bypass & Dialogue Suppression (v2.4.0)
When creating videos for generated music:
- The supplied song audio is the **sole source of truth** for vocal timing, lyrics, and pauses.
- The prompt enhancer completely skips dialogue writing and word-count gating.
- Prevents the LLM from inventing spoken dialogue, voice-over clauses, or conflicting silence directives that corrupt the music video prompt. Visual staging checks and camera directions remain fully enforced.

### 4.4 Localized Multi-Window Repair & 0–5 Fidelity Retries (v2.2.4–v2.4.0)
- **Fidelity Repair Controls**: Users can configure 0 to 5 repair attempts (default 1) with an optional *"Generate even if fidelity checks fail"* switch that continues with the saved draft rather than stalling the pipeline.
- **Card-Local Repairs**: When a multi-window prompt fails camera or continuity fidelity checks, only the affected event card/window prompt is retried. Valid neighboring windows, overall story schedules, and dialogue turn orders are preserved intact.

### 4.5 Speech Pre-Checks & Nonverbal Sound Isolation (v2.2.3)
- **Pre-Flight Duration Validation**: Compares the duration required for exact dialogue against the window length *before* invoking the LLM, preventing fruitless rewrite loops trying to shorten unchangeable lyrics.
- **Nonverbal Sound Cue Isolation**: Authored sound effects (e.g. guitar slide, drum crash, sub sweep) remain in the audio instructions without falsely requiring them to be depicted in the visual action description.

---

## 5. Implementation Reference Code

### 5.1 `director_music_timing.py`
```python
import math
from typing import Any, Dict, List, Optional

def resolve_clip_limits(model_def: Dict[str, Any], requested_seconds: Optional[float] = None) -> Dict[str, Any]:
    fps = float(model_def.get("fps", 24.0))
    minimum = max(1, int(model_def.get("frames_minimum", 25)))
    step = max(1, int(model_def.get("frames_steps", 4)))
    hard_max = int(model_def.get("frames_maximum", 24 * 15))
    
    if requested_seconds is None:
        effective_frames = hard_max
    else:
        seconds = max(minimum / fps, float(requested_seconds))
        target_frames = int(math.floor(seconds * fps + 1e-6))
        effective_frames = min(hard_max, minimum + ((target_frames - minimum) // step) * step)
        
    return {
        "fps": fps,
        "frames_minimum": minimum,
        "frame_step": step,
        "max_frames": effective_frames,
        "max_seconds": effective_frames / fps,
    }

def plan_capped_music_clips(
    analysis: Dict[str, Any],
    maximum_seconds: float,
    fps: float = 24.0,
    frames_steps: int = 4,
    frames_minimum: int = 25,
    energy_bias: int = 0
) -> List[Dict[str, Any]]:
    duration = float(analysis.get("duration", 0.0))
    if duration <= 0:
        return []

    bpm = max(1.0, float(analysis.get("bpm", 120.0)))
    beat_sec = 60.0 / bpm
    beats = [b["time"] for b in analysis.get("beats", [])] or [i * beat_sec for i in range(int(duration / beat_sec) + 1)]
    downbeats = set(analysis.get("downbeats", []))
    sections = analysis.get("sections", [])
    lyrics = analysis.get("lyrics", [])

    # Score timeline accents
    accents: Dict[float, float] = {round(b, 2): 0.5 for b in beats}
    for db in downbeats:
        accents[round(db, 2)] = max(accents.get(round(db, 2), 0.0), 1.8)
    for line in lyrics:
        accents[round(line["start"], 2)] = max(accents.get(round(line["start"], 2), 0.0), 2.5)
        accents[round(line["end"], 2)] = max(accents.get(round(line["end"], 2), 0.0), 2.5)

    # Establish section anchors
    anchors = [0.0]
    for s in sections:
        t = float(s["start"])
        if t > 2.0 and t < duration - 2.0:
            anchors.append(t)
    anchors.append(duration)
    anchors = sorted(list(set(anchors)))

    planned_cuts = [0.0]
    for start, end in zip(anchors[:-1], anchors[1:]):
        span = end - start
        num_clips = max(1, math.ceil(span / maximum_seconds))
        target_len = span / num_clips

        for i in range(1, num_clips):
            ideal_t = start + i * target_len
            # Search nearby candidates within +/- 1.5 seconds for highest accent score
            candidates = [t for t in accents if abs(t - ideal_t) <= 1.5 and t > planned_cuts[-1] + 1.5]
            best_t = max(candidates, key=lambda t: accents[t] - abs(t - ideal_t) * 0.5) if candidates else ideal_t
            planned_cuts.append(best_t)

    planned_cuts.append(duration)
    planned_cuts = sorted(list(set(planned_cuts)))

    # Convert cuts into discrete clips with model-native duration and output trim
    clips = []
    for idx, (c_start, c_end) in enumerate(zip(planned_cuts[:-1], planned_cuts[1:])):
        clip_dur = c_end - c_start
        needed_frames = int(math.ceil(clip_dur * fps))
        valid_frames = frames_minimum + math.ceil(max(0, needed_frames - frames_minimum) / frames_steps) * frames_steps
        gen_duration = valid_frames / fps
        trim_duration = clip_dur

        clips.append({
            "clip_index": idx + 1,
            "start_time": round(c_start, 3),
            "end_time": round(c_end, 3),
            "clip_duration": round(clip_dur, 3),
            "gen_frames": valid_frames,
            "gen_duration": round(gen_duration, 3),
            "trim_duration": round(trim_duration, 3),
            "trim_needed": gen_duration > trim_duration + 0.01
        })
    return clips
```

---

## 6. Related Pages
- [AI Music Video Studio](../entities/video-studio.md)
- [Maestro Ingest](../sources/maestro-creative-studio.md)
- [Non-Destructive Multitrack Timeline](non-destructive-multitrack-timeline.md)
- [Stem Audio-Reactive Video](stem-audio-reactive-video.md)
- [Karaoke & Lyric Sync](../entities/karaoke-lyricsync.md)
