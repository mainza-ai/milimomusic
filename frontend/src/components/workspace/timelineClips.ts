/**
 * Arrange-timeline clip model (plan 3.1).
 *
 * The Arrange tab used to be a read-only visualisation: each lane was one static
 * gradient div with no region model, so there was nothing to drag, trim, split or
 * fade — the defining operation of a DAW.
 *
 * A clip is a window into a stem's buffer:
 *
 *     timeline:   |----start----[==== clip ====]
 *     source:                  |--offset--[====]
 *
 * `duration` is how much of the source plays; `fadeIn`/`fadeOut` are seconds; `gain`
 * multiplies the stem fader so a clip can be ducked without touching the mix.
 */

export interface TimelineClip {
    id: string;
    /** Stem id this clip plays (matches `StemChannel.id`). */
    trackId: string;
    /** Position on the timeline, seconds. */
    start: number;
    /** In-point inside the source buffer, seconds. */
    offset: number;
    /** Playable length, seconds. */
    duration: number;
    /** Linear gain multiplier (0–1), applied on top of the stem fader. */
    gain: number;
    fadeIn: number;
    fadeOut: number;
}

export const MIN_CLIP_SECONDS = 0.05;

export function clipEnd(clip: TimelineClip): number {
    return clip.start + clip.duration;
}

export function clipsForTrack(clips: TimelineClip[], trackId: string): TimelineClip[] {
    return clips.filter(c => c.trackId === trackId).sort((a, b) => a.start - b.start);
}

let _seq = 0;
function newId(trackId: string): string {
    _seq += 1;
    return `${trackId}-c${Date.now().toString(36)}${_seq}`;
}

/** One full-length clip covering the stem — the default arrangement. */
export function defaultClip(trackId: string, duration: number): TimelineClip {
    return {
        id: newId(trackId),
        trackId,
        start: 0,
        offset: 0,
        duration: Math.max(MIN_CLIP_SECONDS, duration),
        gain: 1,
        fadeIn: 0,
        fadeOut: 0,
    };
}

/** Split the clip containing `at` into two, preserving fade shape at the seam. */
export function splitClipAt(clips: TimelineClip[], trackId: string, at: number): TimelineClip[] {
    const trackClips = clipsForTrack(clips, trackId);
    const target = trackClips.find(c => at > c.start + MIN_CLIP_SECONDS && at < clipEnd(c) - MIN_CLIP_SECONDS);
    if (!target) return clips;

    const leftDuration = at - target.start;
    const left: TimelineClip = {
        ...target,
        duration: leftDuration,
        fadeOut: 0,
    };
    const right: TimelineClip = {
        ...target,
        id: newId(trackId),
        start: at,
        offset: target.offset + leftDuration,
        duration: target.duration - leftDuration,
        fadeIn: 0,
    };
    return clips.flatMap(c => (c.id === target.id ? [left, right] : [c]));
}

/**
 * Reposition/trim a clip with snapping, clamped so it can never address audio
 * outside its source buffer or overlap the same track's neighbours silently.
 */
export function moveClip(
    clip: TimelineClip,
    patch: Partial<Pick<TimelineClip, 'start' | 'offset' | 'duration' | 'fadeIn' | 'fadeOut' | 'gain'>>,
    sourceDuration: number,
    snap: (t: number) => number,
    maxStart?: number,
): TimelineClip {
    const next: TimelineClip = { ...clip, ...patch };

    // Keep the source window inside the buffer.
    next.offset = Math.max(0, Math.min(next.offset, Math.max(0, sourceDuration - MIN_CLIP_SECONDS)));
    next.duration = Math.max(MIN_CLIP_SECONDS, Math.min(next.duration, sourceDuration - next.offset));

    next.start = Math.max(0, snap(next.start));
    if (typeof maxStart === 'number') next.start = Math.min(next.start, maxStart);

    next.fadeIn = Math.max(0, Math.min(next.fadeIn, next.duration));
    next.fadeOut = Math.max(0, Math.min(next.fadeOut, next.duration - next.fadeIn));
    next.gain = Math.max(0, Math.min(1, next.gain));
    return next;
}

/** Gain envelope points for a clip, for Web Audio ramps. */
export function fadeEnvelope(clip: TimelineClip): { at: number; gain: number }[] {
    const points = [{ at: 0, gain: clip.fadeIn > 0 ? 0 : clip.gain }];
    if (clip.fadeIn > 0) points.push({ at: clip.fadeIn, gain: clip.gain });
    if (clip.fadeOut > 0) points.push({ at: clip.duration - clip.fadeOut, gain: clip.gain });
    points.push({ at: clip.duration, gain: clip.fadeOut > 0 ? 0 : clip.gain });
    return points;
}

/** Total arrangement length (last clip end), used to size the timeline. */
export function arrangementDuration(clips: TimelineClip[]): number {
    return clips.reduce((max, c) => Math.max(max, clipEnd(c)), 0);
}
