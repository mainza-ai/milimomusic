import React, { useEffect, useRef, useState } from 'react';
import { trackApi } from '../../api';

// ── Module-level peaks cache ────────────────────────────────────────────────
// One fetch per (jobId, buckets) per session, shared across every surface
// that renders the same waveform. HTTP cache handles cross-reload reuse.
const peaksCache = new Map<string, number[]>();
const inflight = new Map<string, Promise<number[]>>();

async function loadPeaks(jobId: string, buckets: number): Promise<number[]> {
    const key = `${jobId}:${buckets}`;
    const hit = peaksCache.get(key);
    if (hit) return hit;
    const pending = inflight.get(key);
    if (pending) return pending;
    const p = trackApi.getTrackPeaks(jobId, buckets)
        .then(res => {
            peaksCache.set(key, res.peaks);
            inflight.delete(key);
            return res.peaks;
        })
        .catch(err => {
            inflight.delete(key);
            throw err;
        });
    inflight.set(key, p);
    return p;
}

interface StaticWaveformProps {
    jobId: string;
    /** 0..1 playback progress; null/undefined renders the idle wave only. */
    progressFraction?: number | null;
    /** Seek request as a 0..1 fraction of the track. */
    onSeekFraction?: (fraction: number) => void;
    heightClass?: string;
    buckets?: number;
    className?: string;
}

/**
 * Lightweight library waveform: server-computed amplitude envelope rendered
 * as a single SVG path. Costs a few KB per track instead of a full audio
 * download + WebAudio decode per row, and never instantiates an audio engine.
 */
export const StaticWaveform: React.FC<StaticWaveformProps> = ({
    jobId,
    progressFraction = null,
    onSeekFraction,
    heightClass = 'h-14',
    buckets = 240,
    className
}) => {
    const holderRef = useRef<HTMLDivElement>(null);
    const [peaks, setPeaks] = useState<number[] | null>(() => peaksCache.get(`${jobId}:${buckets}`) || null);
    const [failed, setFailed] = useState(false);

    // Fetch only when the card approaches the viewport.
    useEffect(() => {
        if (peaks || failed) return;
        const el = holderRef.current;
        if (!el) return;

        let cancelled = false;
        const start = () => {
            loadPeaks(jobId, buckets)
                .then(p => { if (!cancelled) setPeaks(p); })
                .catch(() => { if (!cancelled) setFailed(true); });
        };

        if (typeof IntersectionObserver === 'undefined') { start(); return; }
        const io = new IntersectionObserver((entries) => {
            if (entries.some(e => e.isIntersecting)) {
                io.disconnect();
                start();
            }
        }, { rootMargin: '300px' });
        io.observe(el);
        return () => { cancelled = true; io.disconnect(); };
    }, [jobId, buckets, peaks, failed]);

    // Single closed polygon: top edge left→right, bottom edge right→left.
    // viewBox is normalized to bucket count × 100 so preserveAspectRatio="none"
    // stretches it to any card width with ONE paint node.
    const wavePath = peaks
        ? (() => {
            const n = peaks.length;
            const stepX = 1 / Math.max(1, n - 1);
            let d = '';
            for (let i = 0; i < n; i++) {
                const x = i * stepX * 1000;
                const half = Math.max(1.5, peaks[i] * 48);
                d += `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${(50 - half).toFixed(2)}`;
            }
            for (let i = n - 1; i >= 0; i--) {
                const x = i * stepX * 1000;
                const half = Math.max(1.5, peaks[i] * 48);
                d += `L${x.toFixed(2)},${(50 + half).toFixed(2)}`;
            }
            return d + 'Z';
        })()
        : null;

    const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (!onSeekFraction) return;
        const rect = e.currentTarget.getBoundingClientRect();
        onSeekFraction(Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)));
    };

    const clampedProgress = progressFraction !== null
        ? Math.min(100, Math.max(0, progressFraction * 100))
        : null;

    return (
        <div
            ref={holderRef}
            onClick={handleClick}
            title={onSeekFraction ? 'Click waveform to seek' : undefined}
            className={`relative w-full ${heightClass} rounded-xl overflow-hidden bg-black/[0.03] dark:bg-white/[0.03] border border-black/[0.06] dark:border-white/10 ${
                onSeekFraction ? 'cursor-pointer' : ''
            } ${className || ''}`}
        >
            {!peaks && !failed && (
                <div className="absolute inset-0 flex items-center justify-around px-2 opacity-30">
                    {Array.from({ length: 36 }).map((_, i) => (
                        <div key={i} className="w-[3px] rounded-full bg-slate-400 dark:bg-slate-500 animate-pulse" style={{ height: '20%' }} />
                    ))}
                </div>
            )}
            {failed && (
                <div className="absolute inset-0 flex items-center justify-center text-[10px] font-mono text-slate-400">
                    Waveform unavailable
                </div>
            )}
            {wavePath && (
                <>
                    <svg
                        viewBox="0 0 1000 100"
                        preserveAspectRatio="none"
                        className="absolute inset-0 w-full h-full"
                        aria-hidden
                    >
                        <path d={wavePath} fill="rgba(20, 184, 166, 0.35)" />
                    </svg>
                    {clampedProgress !== null && clampedProgress > 0 && (
                        <div
                            className="absolute inset-0 pointer-events-none"
                            style={{ clipPath: `inset(0 ${100 - clampedProgress}% 0 0)` }}
                            aria-hidden
                        >
                            <svg
                                viewBox="0 0 1000 100"
                                preserveAspectRatio="none"
                                className="absolute inset-0 w-full h-full"
                            >
                                <path d={wavePath} fill="#14b8a6" />
                            </svg>
                        </div>
                    )}
                </>
            )}
        </div>
    );
};
