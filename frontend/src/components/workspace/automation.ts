/**
 * Automation curves (plan 3.16, decision D5).
 *
 * A DAW that can only set a static fader is missing the operation that makes a
 * mix breathe: a level riding up into a chorus, a pan sweeping across a bridge.
 * Curves are stored per stem and per parameter, sampled to breakpoints:
 *
 *     { volume: [{ t: 0, v: 0.8 }, { t: 12, v: 1.0 }, { t: 24, v: 0.7 }] }
 *
 * `v` is normalised 0..1 for volume (multiplied into the stem fader) and -1..1 for
 * pan. Points are sorted by time; the value between points is linear.
 */

export type AutomationParam = 'volume' | 'pan';

export interface AutomationPoint {
    /** Timeline position in seconds. */
    t: number;
    /** Normalised value: 0..1 for volume, -1..1 for pan. */
    v: number;
}

export type AutomationCurve = AutomationPoint[];

/** Per-stem automation, keyed by parameter. */
export type TrackAutomation = Partial<Record<AutomationParam, AutomationCurve>>;

/** Per-stem-id automation map stored in the session document. */
export type AutomationMap = Record<string, TrackAutomation>;

export const PARAM_RANGE: Record<AutomationParam, { min: number; max: number; neutral: number }> = {
    volume: { min: 0, max: 1, neutral: 1 },
    pan: { min: -1, max: 1, neutral: 0 },
};

export function clampValue(param: AutomationParam, value: number): number {
    const { min, max } = PARAM_RANGE[param];
    return Math.max(min, Math.min(max, value));
}

/** Sort + clamp + de-duplicate points at the same time. */
export function normaliseCurve(param: AutomationParam, points: AutomationPoint[]): AutomationCurve {
    const sorted = [...points]
        .filter(p => Number.isFinite(p.t) && Number.isFinite(p.v) && p.t >= 0)
        .sort((a, b) => a.t - b.t);
    const out: AutomationCurve = [];
    for (const point of sorted) {
        const v = clampValue(param, point.v);
        if (out.length && Math.abs(out[out.length - 1].t - point.t) < 1e-6) {
            out[out.length - 1] = { t: point.t, v };
        } else {
            out.push({ t: point.t, v });
        }
    }
    return out;
}

export function addPoint(
    param: AutomationParam,
    curve: AutomationCurve | undefined,
    t: number,
    v: number,
): AutomationCurve {
    return normaliseCurve(param, [...(curve || []), { t, v }]);
}

export function removePointNear(
    curve: AutomationCurve,
    t: number,
    tolerance: number,
): AutomationCurve {
    return curve.filter(p => Math.abs(p.t - t) > tolerance);
}

/** Linear interpolation; before the first point the first value holds. */
export function valueAt(param: AutomationParam, curve: AutomationCurve | undefined, t: number): number {
    const neutral = PARAM_RANGE[param].neutral;
    if (!curve || curve.length === 0) return neutral;
    if (t <= curve[0].t) return curve[0].v;
    const last = curve[curve.length - 1];
    if (t >= last.t) return last.v;
    for (let i = 0; i < curve.length - 1; i++) {
        const a = curve[i];
        const b = curve[i + 1];
        if (t >= a.t && t <= b.t) {
            const span = b.t - a.t || 1e-6;
            const f = (t - a.t) / span;
            return a.v + (b.v - a.v) * f;
        }
    }
    return last.v;
}

/**
 * Breakpoints inside `[from, to]`, plus the value at `from` so a seek into the
 * middle of a curve starts at the right level.
 */
export function schedulePoints(
    param: AutomationParam,
    curve: AutomationCurve | undefined,
    from: number,
    to: number,
): AutomationPoint[] {
    if (!curve || curve.length === 0) return [];
    const points: AutomationPoint[] = [{ t: from, v: valueAt(param, curve, from) }];
    for (const p of curve) {
        if (p.t > from + 1e-6 && p.t <= to) points.push(p);
    }
    return points;
}

export function hasAutomation(automation: AutomationMap, trackId: string): boolean {
    const track = automation[trackId];
    if (!track) return false;
    return Boolean((track.volume && track.volume.length) || (track.pan && track.pan.length));
}
