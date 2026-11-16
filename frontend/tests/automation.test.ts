/**
 * Unit tests for the automation curve model (plan 3.16).
 */
import { describe, it, expect } from 'vitest';
import {
    addPoint,
    hasAutomation,
    normaliseCurve,
    removePointNear,
    schedulePoints,
    valueAt,
} from '../src/components/workspace/automation';

describe('normaliseCurve', () => {
    it('sorts points by time', () => {
        const curve = normaliseCurve('volume', [{ t: 5, v: 0.2 }, { t: 1, v: 0.9 }]);
        expect(curve.map(p => p.t)).toEqual([1, 5]);
    });

    it('clamps values into the parameter range', () => {
        const volume = normaliseCurve('volume', [{ t: 0, v: 5 }, { t: 1, v: -2 }]);
        expect(volume[0].v).toBe(1);
        expect(volume[1].v).toBe(0);

        const pan = normaliseCurve('pan', [{ t: 0, v: 9 }, { t: 1, v: -9 }]);
        expect(pan[0].v).toBe(1);
        expect(pan[1].v).toBe(-1);
    });

    it('collapses duplicate times, keeping the last value', () => {
        const curve = normaliseCurve('volume', [{ t: 2, v: 0.1 }, { t: 2, v: 0.7 }]);
        expect(curve).toHaveLength(1);
        expect(curve[0].v).toBe(0.7);
    });

    it('drops negative or non-finite times', () => {
        const curve = normaliseCurve('volume', [
            { t: -1, v: 0.5 }, { t: 1, v: 0.5 }, { t: Number.NaN, v: 0.5 },
        ]);
        expect(curve).toHaveLength(1);
        expect(curve[0].t).toBe(1);
    });
});

describe('valueAt', () => {
    const curve = normaliseCurve('volume', [{ t: 0, v: 0.2 }, { t: 10, v: 1.0 }]);

    it('interpolates linearly between points', () => {
        expect(valueAt('volume', curve, 5)).toBeCloseTo(0.6, 5);
    });

    it('holds the first value before the curve starts', () => {
        const late = normaliseCurve('volume', [{ t: 4, v: 0.3 }, { t: 8, v: 0.9 }]);
        expect(valueAt('volume', late, 0)).toBeCloseTo(0.3, 5);
    });

    it('holds the last value after the curve ends', () => {
        expect(valueAt('volume', curve, 99)).toBeCloseTo(1.0, 5);
    });

    it('is neutral for an untouched parameter', () => {
        expect(valueAt('volume', undefined, 3)).toBe(1);
        expect(valueAt('pan', [], 3)).toBe(0);
    });
});

describe('addPoint / removePointNear', () => {
    it('adds and keeps the curve sorted', () => {
        let curve = addPoint('volume', undefined, 4, 0.4);
        curve = addPoint('volume', curve, 1, 0.9);
        expect(curve.map(p => p.t)).toEqual([1, 4]);
    });

    it('removes a point within tolerance only', () => {
        const curve = normaliseCurve('volume', [{ t: 1, v: 0.5 }, { t: 5, v: 0.5 }]);
        const removed = removePointNear(curve, 1.05, 0.1);
        expect(removed.map(p => p.t)).toEqual([5]);
        const untouched = removePointNear(curve, 3, 0.1);
        expect(untouched).toHaveLength(2);
    });
});

describe('schedulePoints', () => {
    const curve = normaliseCurve('volume', [
        { t: 0, v: 0.1 }, { t: 5, v: 0.5 }, { t: 10, v: 1.0 }, { t: 20, v: 0.2 },
    ]);

    it('starts at the interpolated value when seeking into the curve', () => {
        const points = schedulePoints('volume', curve, 7.5, 15);
        expect(points[0].t).toBe(7.5);
        expect(points[0].v).toBeCloseTo(0.75, 5);
    });

    it('only includes later points inside the window', () => {
        const points = schedulePoints('volume', curve, 7.5, 15);
        expect(points.map(p => p.t)).toEqual([7.5, 10]);
    });

    it('is empty without a curve', () => {
        expect(schedulePoints('volume', undefined, 0, 10)).toEqual([]);
    });
});

describe('hasAutomation', () => {
    it('detects a track with any curve', () => {
        expect(hasAutomation({}, 'stem-1')).toBe(false);
        expect(hasAutomation({ 'stem-1': { volume: [] } }, 'stem-1')).toBe(false);
        expect(hasAutomation({ 'stem-1': { volume: [{ t: 0, v: 1 }] } }, 'stem-1')).toBe(true);
        expect(hasAutomation({ 'stem-1': { pan: [{ t: 0, v: 0 }] } }, 'stem-1')).toBe(true);
    });
});
