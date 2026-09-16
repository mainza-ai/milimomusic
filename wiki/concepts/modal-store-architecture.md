---
title: Global Modal Store & DAW Context Routing
type: concept
created: 2026-09-15
updated: 2026-09-15
sources: [sources/v2-refactor-plan.md, sources/readme.md]
tags: [frontend, zustand, store, modals, daw, routing]
aliases: [Modal Store Architecture, useModalStore]
---

# Global Modal Store & DAW Context Routing

The **Global Modal Store** (`frontend/src/stores/useModalStore.ts`) unifies modal lifecycle management across Milimo Music using Zustand, replacing decentralized multi-mount points with single authoritative mounts.

## Problem: Decentralized Component Over-Mounting

In early frontend iterations, complex modals like `<CoverStudioModal>` or `<VoiceStudioModal>` were duplicated across multiple view components:
- `ComposerSidebar.tsx`
- `TrackRowPlayer.tsx`
- `SongsView.tsx`
- `TrackDetailView.tsx`
- `PianoRoll.tsx`

Each view maintained its own local `useState` for modal visibility and track context. This led to:
1. **Memory Bloat**: Multiple unmounted modal DOM trees and duplicate background polling timers (`modelsApi.checkDependencies`).
2. **State Desynchronization**: Actions triggered in one view failed to propagate context to another.
3. **Z-Index & Backdrop Collisions**: Inconsistent overlay layering and event bubbling leaks.

## Solution: Zustand Global Modal Store

```typescript
interface ModalState {
    // Cover Studio
    isCoverStudioOpen: boolean;
    coverStudioTrack: Job | null;
    coverStudioMode: 'audio' | 'midi';
    openCoverStudio: (track?: Job | null, mode?: 'audio' | 'midi') => void;
    closeCoverStudio: () => void;

    // Engine Switcher
    isEngineSwitcherOpen: boolean;
    openEngineSwitcher: () => void;
    closeEngineSwitcher: () => void;

    // Voice Conversion
    isVoiceConvertOpen: boolean;
    voiceConvertTrack: Job | null;
    openVoiceConvert: (track?: Job | null) => void;
    closeVoiceConvert: () => void;
}
```

### Architecture Invariants

1. **Single Mount at Root**: `<CoverStudioModal>` and `<EngineSwitcherModal>` are mounted exactly once at the root level of `App.tsx`.
2. **Context-Preserving Routing**: Any sub-component or DAW view invokes `useModalStore.getState().openCoverStudio(track, 'audio')`. The modal opens with all technical track metadata (stems, BPM, MIDI paths, prompt) pre-populated.
3. **DAW Clip-Level Integration**: In `ArrangeTimeline.tsx`, clicking an audio clip's context menu ("Remix with MuLaCover") immediately opens Cover Studio with the clip's parent job context intact.

## Related Pages

- [Frontend](../entities/frontend.md)
- [Session Workspace](../entities/session-workspace.md)
- [MuLaCover](../entities/mulacover.md)
- [Hardware Coordinator](../entities/hardware-coordinator.md)
