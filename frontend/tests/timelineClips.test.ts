/**
 * Unit tests for the arrange-timeline clip model (plan 3.1).
 *
 * The Arrange tab had no region model at all, so nothing could be moved, trimmed,
 * split or faded. These pin the pure geometry so the UI handlers stay honest.
 */
import { describe, it, expect } from 'vitest';
import {
    MIN_CLIP_SECONDS,
    arrangementDuration,
    clipEnd,
    clipsForTrack,
    defaultClip,
    fadeEnvelope,
    moveClip,
    splitClipAt,
} from '../src/components/workspace/timelineClips';

const snapNone = (t: number) => t;

describe('defaultClip', () => {
    it('covers the whole stem from its start', () => {
        const clip = defaultClip('stem-vocals', 42);
        expect(clip.start).toBe(0);
        expect(clip.offset).toBe(0);
        expect(clip.duration).toBe(42);
        expect(clip.gain).toBe(1);
        expect(clip.fadeIn).toBe(0);
        expect(clip.fadeOut).toBe(0);
    });
});

describe('splitClipAt', () => {
    it('produces two contiguous clips that address the same audio', () => {
        const clip = defaultClip('stem', 10);
        const [left, right] = splitClipAt([clip], 'stem', 4);

        expect(left.start).toBe(0);
        expect(left.duration).toBeCloseTo(4, 6);
        expect(right.start).toBeCloseTo(4, 6);
        expect(right.duration).toBeCloseTo(6, 6);
        // The source in-point advances with the split.
        expect(right.offset).toBeCloseTo(4, 6);
        expect(clipEnd(left)).toBeCloseTo(right.start, 6);
    });

    it('refuses to split outside the clip', () => {
        const clip = defaultClip('stem', 10);
        expect(splitClipAt([clip], 'stem', -1)).toHaveLength(1);
        expect(splitClipAt([clip], 'stem', 20)).toHaveLength(1);
        // …and refuses a degenerate split within the minimum length.
        expect(splitClipAt([clip], 'stem', MIN_CLIP_SECONDS / 2)).toHaveLength(1);
    });

    it('leaves other tracks untouched', () => {
        const a = defaultClip('a', 5);
        const b = defaultClip('b', 5);
        const result = splitClipAt([a, b], 'a', 2);
        expect(clipsForTrack(result, 'b')).toHaveLength(1);
        expect(clipsForTrack(result, 'a')).toHaveLength(2);
    });
});

describe('moveClip', () => {
    const clip = defaultClip('stem', 10);

    it('keeps the source window inside the buffer when trimming the right edge', () => {
        const next = moveClip(clip, { duration: 999 }, 10, snapNone);
        expect(next.duration).toBeCloseTo(10, 6);
    });

    it('cannot start before zero', () => {
        const next = moveClip(clip, { start: -5 }, 10, snapNone);
        expect(next.start).toBe(0);
    });

    it('clamps offset when trimming the left edge past the buffer start', () => {
        const next = moveClip(clip, { offset: -3, duration: 10 }, 10, snapNone);
        expect(next.offset).toBe(0);
    });

    it('never exceeds the source duration for offset + duration', () => {
        const next = moveClip(clip, { offset: 8, duration: 8 }, 10, snapNone);
        expect(next.offset + next.duration).toBeLessThanOrEqual(10 + 1e-9);
    });

    it('clamps fades to the clip length and gain to 0..1', () => {
        const next = moveClip(clip, { fadeIn: 20, fadeOut: 20, gain: 5 }, 10, snapNone);
        expect(next.fadeIn).toBeLessThanOrEqual(next.duration);
        expect(next.fadeIn + next.fadeOut).toBeLessThanOrEqual(next.duration + 1e-9);
        expect(next.gain).toBe(1);
    });

    it('applies the snap function to start', () => {
        const snapQuarter = (t: number) => Math.round(t * 4) / 4;
        const next = moveClip(clip, { start: 1.9 }, 10, snapQuarter);
        expect(next.start).toBeCloseTo(2, 6);
    });
});

describe('fadeEnvelope', () => {
    it('is flat at the clip gain with no fades', () => {
        const env = fadeEnvelope({ ...defaultClip('s', 5), gain: 0.5 });
        expect(env[0]).toEqual({ at: 0, gain: 0.5 });
        expect(env[env.length - 1]).toEqual({ at: 5, gain: 0.5 });
    });

    it('ramps from silence when a fade-in is set', () => {
        const env = fadeEnvelope({ ...defaultClip('s', 5), fadeIn: 1 });
        expect(env[0]).toEqual({ at: 0, gain: 0 });
        expect(env).toContainEqual({ at: 1, gain: 1 });
    });

    it('ramps to silence when a fade-out is set', () => {
        const env = fadeEnvelope({ ...defaultClip('s', 5), fadeOut: 2 });
        expect(env[env.length - 1]).toEqual({ at: 5, gain: 0 });
        expect(env).toContainEqual({ at: 3, gain: 1 });
    });
});

describe('arrangementDuration', () => {
    it('is the last clip end across tracks', () => {
        const a = { ...defaultClip('a', 5), start: 2 };
        const b = { ...defaultClip('b', 5), start: 10 };
        expect(arrangementDuration([a, b])).toBeCloseTo(15, 6);
    });

    it('is zero for an empty arrangement', () => {
        expect(arrangementDuration([])).toBe(0);
    });
});
