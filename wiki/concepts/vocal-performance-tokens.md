---
title: Expressive Vocal Performance Tokens & Section Casting
type: concept
created: 2026-09-15
updated: 2026-09-15
sources: [sources/v2-refactor-plan.md, sources/readme.md]
tags: [lyrics, tokens, vocal, performance, casting, parsing]
aliases: [Performance Tokens, Vocal Performance Tokens, Section Casting]
---

# Expressive Vocal Performance Tokens & Section Casting

**Expressive Vocal Performance Tokens** (`backend/app/services/lyrics/performance_tokens.py`) enable fine-grained acoustic delivery modulation and multi-character vocal assignment directly within song lyrics.

## Motivation & Architecture

Standard AI text-to-music models treat lyrics as raw text, making it difficult to control singer nuance, vocal techniques (whispering, falsetto, belting), phrasing pauses, or duet assignments. Inspired by professional vocal directing in platforms like VoiceStudio, Milimo Music introduces a deterministic token schema that parses expressive tags and section-level character casting before generation.

## Token Taxonomy

### 1. Acoustic Performance Modifiers

These inline tags instruct the generative backbone on emotional delivery, breath management, and vocal register:

| Token | Meaning | Acoustic Effect |
|-------|---------|-----------------|
| `[breath]` | Audible breath intake | Inserts an audible natural inhalation before phrase onset |
| `[whisper]` | Intimate whisper | Attenuates fundamental vocal pitch ($F_0$), increasing high-frequency noise ratio |
| `[pause 250ms]` | Metric pause | Quantized rhythmic silence inserted between vocal lines |
| `[falsetto]` | Head voice register | Elevates vocal formant resonance into breathy upper octave |
| `[belt]` | Full chest voice power | Drives high dynamic RMS and harmonically rich brassy vocal punch |
| `[vibrato]` | Frequency oscillation | Adds sustained pitch modulation to elongated vowel endings |

### 2. Multi-Agent Section Casting

Section header tags define structured musical boundaries and explicitly cast virtual singers or duets:

- `[Verse 1 | Lead: Maya]` -> Allocates lead vocal timbre to character profile "Maya".
- `[Chorus | Duet: Maya + Marcus]` -> Orchestrates harmony/counterpoint arrangement between two distinct vocal profiles.
- `[Bridge | Backing: Gospel Choir]` -> Instructs polyphonic backing vocal synthesis.

## Token Parser & Lyric Stripping

The parser (`backend/app/services/lyrics/performance_tokens.py`) processes lyrics into two synchronized views:
1. **Model Conditioning Context**: Translates tokens into structured style tags and prompt conditioning prefixes for MiniMax Music 3 / MuLaCover.
2. **Clean Display Lyrics**: Safely strips technical token annotations for clean subtitle display (.lrc/.srt) and score notation.

## Frontend Composer Toolbar

In `ComposerSidebar.tsx`, a dedicated **Expressive Performance** toolbar allows producers to insert tags with a single click, featuring tooltips explaining the acoustic effect of each token.

## Related Pages

- [Lyrics Conditioning](lyrics-conditioning.md)
- [Prompt Structure](prompt-structure.md)
- [AI Co-Writer](../entities/ai-cowriter.md)
- [Neural SVC](../entities/neural-svc.md)
