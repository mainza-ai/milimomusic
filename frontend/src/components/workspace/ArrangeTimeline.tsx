import React, { useEffect, useMemo, useRef } from 'react';
import { Layers, ZoomIn, ZoomOut, Sparkles, Mic, FastForward } from 'lucide-react';
import type { Job, NoteEvent } from '../../api';
import type { StemChannel } from './SessionWorkspace';
import {
    type TimelineClip,
    clipsForTrack,
    moveClip,
    splitClipAt,
} from './timelineClips';
import {
    type AutomationParam,
    addPoint,
    removePointNear,
} from './automation';
import { useModalStore } from '../../stores/useModalStore';

// Parse-once helpers: these blobs were re-JSON.parsed on EVERY render
// (~12Hz during playback) on the main thread.
function parseGrid(job: Job): { bpm?: number; beats_per_bar?: number } {
    try {
        return job.beat_grid_json
            ? (typeof job.beat_grid_json === 'string' ? JSON.parse(job.beat_grid_json) : job.beat_grid_json)
            : {};
    } catch { return {}; }
}

function parseNotes(job: Job): NoteEvent[] {
    try {
        return job.notes_json
            ? (typeof job.notes_json === 'string' ? JSON.parse(job.notes_json) : job.notes_json)
            : [];
    } catch { return []; }
}

export interface SongSection {
    id: string;
    name: string;
    start: number;
    end: number;
    color: string;
}

function getSectionColor(name: string): string {
    const lower = name.toLowerCase();
    if (lower.includes('intro')) {
        return 'bg-indigo-500/15 dark:bg-indigo-500/25 text-indigo-950 dark:text-indigo-200 border-indigo-500/30 dark:border-indigo-500/50 hover:bg-indigo-500/25';
    }
    if (lower.includes('verse')) {
        return 'bg-sky-500/15 dark:bg-sky-500/25 text-sky-950 dark:text-sky-200 border-sky-500/30 dark:border-sky-500/50 hover:bg-sky-500/25';
    }
    if (lower.includes('chorus') || lower.includes('hook')) {
        return 'bg-amber-500/20 dark:bg-amber-500/30 text-amber-950 dark:text-amber-200 border-amber-500/40 dark:border-amber-500/60 hover:bg-amber-500/30 font-bold';
    }
    if (lower.includes('bridge')) {
        return 'bg-emerald-500/15 dark:bg-emerald-500/25 text-emerald-950 dark:text-emerald-200 border-emerald-500/30 dark:border-emerald-500/50 hover:bg-emerald-500/25';
    }
    if (lower.includes('outro')) {
        return 'bg-purple-500/15 dark:bg-purple-500/25 text-purple-950 dark:text-purple-200 border-purple-500/30 dark:border-purple-500/50 hover:bg-purple-500/25';
    }
    if (lower.includes('solo') || lower.includes('inst')) {
        return 'bg-rose-500/15 dark:bg-rose-500/25 text-rose-950 dark:text-rose-200 border-rose-500/30 dark:border-rose-500/50 hover:bg-rose-500/25';
    }
    return 'bg-teal-500/15 dark:bg-teal-500/20 text-teal-950 dark:text-teal-200 border-teal-500/30 dark:border-teal-500/40 hover:bg-teal-500/25';
}

function parseSongSections(job: Job, totalDuration: number): SongSection[] {
    const sections: SongSection[] = [];
    try {
        if (job.timed_lyrics_json) {
            const raw = typeof job.timed_lyrics_json === 'string'
                ? JSON.parse(job.timed_lyrics_json)
                : job.timed_lyrics_json;
            if (Array.isArray(raw)) {
                const markerItems = raw.filter((item: any) => item.is_section && item.text);
                for (let i = 0; i < markerItems.length; i++) {
                    const item = markerItems[i];
                    const cleanName = item.text.replace(/^\[+|\]+$/g, '').trim();
                    const start = Math.max(0, Number(item.start) || 0);
                    const nextStart = i < markerItems.length - 1 ? Math.max(start, Number(markerItems[i + 1].start) || 0) : totalDuration;
                    const end = Math.max(start + 0.5, Math.min(totalDuration, nextStart));
                    const color = getSectionColor(cleanName);
                    sections.push({
                        id: `sec-${i}-${cleanName}`,
                        name: cleanName,
                        start,
                        end,
                        color
                    });
                }
            }
        }
    } catch { /* ignore */ }

    if (sections.length === 0 && job.lyrics) {
        const matches = Array.from(job.lyrics.matchAll(/\[(.*?)\]/g));
        if (matches.length > 0) {
            const step = totalDuration / matches.length;
            matches.forEach((m, i) => {
                const cleanName = m[1].trim();
                const start = i * step;
                const end = (i + 1) * step;
                const color = getSectionColor(cleanName);
                sections.push({
                    id: `sec-${i}-${cleanName}`,
                    name: cleanName,
                    start,
                    end,
                    color
                });
            });
        }
    }

    return sections;
}

const TimelineWaveform: React.FC<{ peaks: number[] }> = React.memo(({ peaks }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !peaks.length) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;

        if (width === 0 || height === 0) return;

        if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
            canvas.width = width * dpr;
            canvas.height = height * dpr;
        }

        ctx.save();
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, width, height);

        const barWidth = 2;
        const gap = 1.5;
        const totalBars = Math.max(1, Math.floor(width / (barWidth + gap)));
        const step = peaks.length / totalBars;
        const centerY = height / 2;

        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';

        for (let i = 0; i < totalBars; i++) {
            const peakIdx = Math.min(Math.floor(i * step), peaks.length - 1);
            const val = peaks[peakIdx] || 0;
            const barHeight = Math.max(3, val * (height * 0.85));
            const x = i * (barWidth + gap);
            const y = centerY - barHeight / 2;
            const radius = Math.min(1, barWidth / 2);

            ctx.beginPath();
            ctx.roundRect(x, y, barWidth, barHeight, radius);
            ctx.fill();
        }

        ctx.restore();
    }, [peaks]);

    return (
        <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none opacity-50"
        />
    );
});

const TimelineNotesCanvas: React.FC<{ notes: NoteEvent[]; clipDuration: number }> = React.memo(({ notes, clipDuration }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !notes.length || clipDuration <= 0) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;

        if (width === 0 || height === 0) return;

        if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
            canvas.width = width * dpr;
            canvas.height = height * dpr;
        }

        ctx.save();
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, width, height);

        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        const noteHeight = Math.max(2, Math.min(5, height * 0.1));

        for (let i = 0; i < notes.length; i++) {
            const n = notes[i];
            if (n.start_time > clipDuration) continue;
            const left = (n.start_time / clipDuration) * width;
            const noteDur = n.duration !== undefined ? n.duration : (n.end_time ? n.end_time - n.start_time : 0.5);
            const w = Math.max(1.5, (noteDur / clipDuration) * width);
            const top = (height * 0.12) + ((n.pitch % 12) / 12) * (height * 0.65);

            ctx.fillRect(left, top, w, noteHeight);
        }

        ctx.restore();
    }, [notes, clipDuration]);

    return (
        <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none opacity-60"
        />
    );
});

interface TrackHeaderRowProps {
    track: StemChannel;
    onToggleMute: (id: string) => void;
    onToggleSolo: (id: string) => void;
}

const TrackHeaderRow: React.FC<TrackHeaderRowProps> = React.memo(({
    track,
    onToggleMute,
    onToggleSolo
}) => {
    return (
        <div className="h-20 px-4 border-b border-black/[0.04] dark:border-white/5 flex items-center justify-between bg-white/90 dark:bg-[#151722]/90">
            <div className="min-w-0 pr-2">
                <span className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate block">
                    {track.name}
                </span>
                <span className="text-[10px] font-mono text-slate-400 tabular-nums">
                    Vol: {track.volume}%
                </span>
            </div>

            <div className="flex items-center space-x-1">
                <button
                    onClick={(e) => { e.stopPropagation(); onToggleMute(track.id); }}
                    aria-pressed={track.isMuted}
                    title={`Mute ${track.name}`}
                    aria-label={`Mute ${track.name}`}
                    className={`w-7 h-7 rounded-lg text-[10px] font-bold transition-colors cursor-pointer ${
                        track.isMuted
                            ? 'bg-rose-500 text-white shadow-sm'
                            : 'bg-black/[0.04] dark:bg-white/5 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                    }`}
                >
                    M
                </button>
                <button
                    onClick={(e) => { e.stopPropagation(); onToggleSolo(track.id); }}
                    aria-pressed={track.isSolo}
                    title={`Solo ${track.name}`}
                    aria-label={`Solo ${track.name}`}
                    className={`w-7 h-7 rounded-lg text-[10px] font-bold transition-colors cursor-pointer ${
                        track.isSolo
                            ? 'bg-amber-500 text-slate-950 font-extrabold shadow-sm'
                            : 'bg-black/[0.04] dark:bg-white/5 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                    }`}
                >
                    S
                </button>
            </div>
        </div>
    );
});

interface TrackLaneRowProps {
    job: Job;
    track: StemChannel;
    trackNotes: NoteEvent[];
    peaks?: number[];
    stemDur?: number;
    totalDuration: number;
    formatTime: (seconds: number) => string;
}

const TrackLaneRow: React.FC<TrackLaneRowProps> = React.memo(({
    job,
    track,
    trackNotes,
    peaks,
    stemDur,
    totalDuration,
    formatTime
}) => {
    const effectiveDur = stemDur && stemDur > 0 ? Math.min(stemDur, totalDuration) : totalDuration;
    const widthPct = Math.min(100, (effectiveDur / totalDuration) * 100);
    const isVocal = track.name.toLowerCase().includes('vocal');

    return (
        <div className="h-20 border-b border-black/[0.04] dark:border-white/5 p-2 flex items-center relative group">
            <div
                className={`h-16 rounded-xl bg-gradient-to-r ${track.color} p-2 flex items-center justify-between shadow-sm relative overflow-hidden transition-opacity duration-150 ${
                    track.isMuted ? 'opacity-30' : 'opacity-90'
                }`}
                style={{ width: `${widthPct}%` }}
            >
                {peaks && peaks.length > 0 && (
                    <TimelineWaveform peaks={peaks} />
                )}

                {trackNotes.length > 0 && (
                    <TimelineNotesCanvas
                        notes={trackNotes}
                        clipDuration={effectiveDur}
                    />
                )}

                <span className="text-xs font-bold text-white relative z-10 drop-shadow-sm truncate">
                    {track.name} {trackNotes.length > 0 ? `(${trackNotes.length} notes)` : ''}
                </span>

                <div className="flex items-center space-x-1.5 relative z-10">
                    <button
                        type="button"
                        onClick={(e) => {
                            e.stopPropagation();
                            useModalStore.getState().openCoverStudio(job, 'audio', track.audioUrl);
                        }}
                        title={`Remix ${track.name} with MuLaCover`}
                        className="opacity-0 group-hover:opacity-100 px-2 py-0.5 rounded-lg bg-black/40 hover:bg-black/60 text-white text-[10px] font-bold flex items-center gap-1 backdrop-blur-sm border border-white/20 transition-all cursor-pointer shadow-sm active:scale-95"
                    >
                        <Sparkles size={10} className="text-teal-300" />
                        <span>Remix</span>
                    </button>

                    {isVocal && (
                        <button
                            type="button"
                            onClick={(e) => {
                                e.stopPropagation();
                                useModalStore.getState().openVoiceConvert(job, track.audioUrl);
                            }}
                            title={`Singing Voice Conversion for ${track.name}`}
                            className="opacity-0 group-hover:opacity-100 px-2 py-0.5 rounded-lg bg-black/40 hover:bg-black/60 text-white text-[10px] font-bold flex items-center gap-1 backdrop-blur-sm border border-white/20 transition-all cursor-pointer shadow-sm active:scale-95"
                        >
                            <Mic size={10} className="text-purple-300" />
                            <span>Voice</span>
                        </button>
                    )}

                    <span className="text-[10px] font-mono text-white/80 tabular-nums pr-1">
                        {formatTime(effectiveDur)}
                    </span>
                </div>
            </div>
        </div>
    );
});

interface ClipLaneProps {
    clips: TimelineClip[];
    track: StemChannel;
    totalDuration: number;
    snapSec: number;
    sourceDuration: number;
    peaks?: number[];
    readOnly: boolean;
    onChange: (clips: TimelineClip[]) => void;
}

/**
 * Editable clip row (plan 3.1): drag to move, drag an edge to trim, drag a corner
 * handle to fade, double-click to split at the click position. Positions snap to
 * the beat grid when `snapSec > 0`.
 */
const ClipLaneRow: React.FC<ClipLaneProps> = ({
    clips, track, totalDuration, snapSec, sourceDuration, peaks, readOnly, onChange,
}) => {
    const laneRef = useRef<HTMLDivElement>(null);
    const dragRef = useRef<{
        mode: 'move' | 'trim-left' | 'trim-right' | 'fade-in' | 'fade-out';
        id: string; startX: number; original: TimelineClip;
    } | null>(null);

    const pct = (t: number) => (t / Math.max(0.001, totalDuration)) * 100;

    const beginDrag = (e: React.MouseEvent, clip: TimelineClip, mode: 'move' | 'trim-left' | 'trim-right' | 'fade-in' | 'fade-out') => {
        if (readOnly) return;
        e.stopPropagation();
        e.preventDefault();
        dragRef.current = { mode, id: clip.id, startX: e.clientX, original: { ...clip } };

        const move = (ev: MouseEvent) => {
            const drag = dragRef.current;
            const lane = laneRef.current;
            if (!drag || !lane) return;
            const pxPerSec = lane.getBoundingClientRect().width / Math.max(0.001, totalDuration);
            const deltaSec = (ev.clientX - drag.startX) / pxPerSec;
            const snap = (t: number) => (snapSec > 0 ? Math.round(t / snapSec) * snapSec : t);
            const o = drag.original;

            let patch: Parameters<typeof moveClip>[1] = {};
            if (drag.mode === 'move') {
                patch = { start: snap(o.start + deltaSec) };
            } else if (drag.mode === 'trim-left') {
                const newStart = snap(o.start + deltaSec);
                const shift = newStart - o.start;
                // Trimming the left edge moves the source in-point with it.
                patch = {
                    start: newStart,
                    offset: o.offset + shift,
                    duration: o.duration - shift,
                };
            } else if (drag.mode === 'trim-right') {
                patch = { duration: snap(o.duration + deltaSec) };
            } else if (drag.mode === 'fade-in') {
                patch = { fadeIn: Math.max(0, Math.min(o.duration, o.fadeIn + deltaSec)) };
            } else {
                patch = { fadeOut: Math.max(0, Math.min(o.duration, o.fadeOut + deltaSec)) };
            }

            const next = moveClip(o, patch, sourceDuration, snap);
            onChange(clips.map(c => (c.id === drag.id ? next : c)));
        };

        const up = () => {
            dragRef.current = null;
            window.removeEventListener('mousemove', move);
            window.removeEventListener('mouseup', up);
        };
        window.addEventListener('mousemove', move);
        window.addEventListener('mouseup', up);
    };

    const handleDoubleClick = (e: React.MouseEvent) => {
        if (readOnly) return;
        const lane = laneRef.current;
        if (!lane) return;
        const rect = lane.getBoundingClientRect();
        const at = ((e.clientX - rect.left) / rect.width) * totalDuration;
        const snapped = snapSec > 0 ? Math.round(at / snapSec) * snapSec : at;
        onChange(splitClipAt(clips, track.id, snapped));
    };

    const trackClips = clipsForTrack(clips, track.id);

    return (
        <div
            ref={laneRef}
            onDoubleClick={handleDoubleClick}
            className="relative h-20 border-b border-black/[0.04] dark:border-white/5 select-none"
            data-clip-lane={track.id}
        >
            {trackClips.length === 0 && (
                <div className="absolute inset-2 rounded-xl border border-dashed border-slate-300 dark:border-white/10 flex items-center justify-center text-[10px] text-slate-400">
                    No clip — double-click to split the lane's audio here
                </div>
            )}

            {trackClips.map(clip => {
                const left = pct(clip.start);
                const width = Math.max(0.4, pct(clip.duration));
                const fadeInPct = clip.duration > 0 ? (clip.fadeIn / clip.duration) * 100 : 0;
                const fadeOutPct = clip.duration > 0 ? (clip.fadeOut / clip.duration) * 100 : 0;
                return (
                    <div
                        key={clip.id}
                        onMouseDown={(e) => {
                            const target = e.target as HTMLElement;
                            const mode = target.dataset.handle as ClipLaneProps extends never ? never : any;
                            if (mode) beginDrag(e, clip, mode);
                            else beginDrag(e, clip, 'move');
                        }}
                        className={`absolute top-2 bottom-2 rounded-xl overflow-hidden bg-gradient-to-r ${track.color} shadow-sm ${
                            track.isMuted ? 'opacity-30' : 'opacity-95'
                        } ${readOnly ? '' : 'cursor-grab active:cursor-grabbing'}`}
                        style={{ left: `${left}%`, width: `${width}%` }}
                        title={`${track.name} · ${clip.duration.toFixed(2)}s from ${clip.offset.toFixed(2)}s · double-click to split`}
                    >
                        {peaks && peaks.length > 0 && <TimelineWaveform peaks={peaks} />}

                        {/* Fade overlays (visual only; the scheduler applies the ramps) */}
                        {clip.fadeIn > 0 && (
                            <div
                                className="absolute inset-y-0 left-0 bg-black/40"
                                style={{ width: `${fadeInPct}%`, clipPath: 'polygon(0 0, 100% 0, 0 100%)' }}
                            />
                        )}
                        {clip.fadeOut > 0 && (
                            <div
                                className="absolute inset-y-0 right-0 bg-black/40"
                                style={{ width: `${fadeOutPct}%`, clipPath: 'polygon(100% 0, 100% 100%, 0 100%)' }}
                            />
                        )}

                        {!readOnly && (
                            <>
                                <span
                                    data-handle="trim-left"
                                    className="absolute left-0 top-0 bottom-0 w-2 cursor-ew-resize bg-black/20 hover:bg-black/40"
                                />
                                <span
                                    data-handle="trim-right"
                                    className="absolute right-0 top-0 bottom-0 w-2 cursor-ew-resize bg-black/20 hover:bg-black/40"
                                />
                                <span
                                    data-handle="fade-in"
                                    className="absolute left-2 top-1 w-2.5 h-2.5 rounded-full bg-white/80 cursor-ew-resize"
                                    title="Drag to set the fade-in"
                                />
                                <span
                                    data-handle="fade-out"
                                    className="absolute right-2 top-1 w-2.5 h-2.5 rounded-full bg-white/80 cursor-ew-resize"
                                    title="Drag to set the fade-out"
                                />
                            </>
                        )}

                        <span className="absolute left-3 bottom-1 text-[10px] font-bold text-white/90 drop-shadow-sm truncate max-w-[80%]">
                            {track.name}
                        </span>
                    </div>
                );
            })}
        </div>
    );
};

interface AutomationLaneProps {
    trackId: string;
    param: AutomationParam;
    points: { t: number; v: number }[];
    totalDuration: number;
    onChange: (points: { t: number; v: number }[]) => void;
    onParamChange: (param: AutomationParam) => void;
}

/**
 * Volume/pan automation lane (plan 3.16).
 *
 * Click empty space to add a breakpoint, drag a point to move it, double-click a
 * point to delete it. The curve is drawn as an SVG polyline over the same time
 * scale as the clips, so a level ride lines up with what you hear.
 */
const AutomationLaneRow: React.FC<AutomationLaneProps> = ({
    trackId, param, points, totalDuration, onChange, onParamChange,
}) => {
    const laneRef = useRef<HTMLDivElement>(null);
    const H = 48;
    const pct = (t: number) => (t / Math.max(0.001, totalDuration)) * 100;
    const yOf = (v: number) => {
        const min = param === 'volume' ? 0 : -1;
        const max = param === 'volume' ? 1 : 1;
        const norm = (v - min) / Math.max(1e-6, max - min);
        return H - Math.max(0, Math.min(1, norm)) * H;
    };
    const vOf = (clientY: number) => {
        const lane = laneRef.current;
        if (!lane) return param === 'volume' ? 1 : 0;
        const rect = lane.getBoundingClientRect();
        const norm = 1 - Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));
        return param === 'volume' ? norm : norm * 2 - 1;
    };
    const tOf = (clientX: number) => {
        const lane = laneRef.current;
        if (!lane) return 0;
        const rect = lane.getBoundingClientRect();
        return Math.max(0, Math.min(totalDuration, ((clientX - rect.left) / rect.width) * totalDuration));
    };

    const sorted = [...points].sort((a, b) => a.t - b.t);
    const path = sorted.length
        ? sorted.map((p, i) => `${i === 0 ? 'M' : 'L'}${pct(p.t) * 10},${yOf(p.v)}`).join(' ')
        : '';

    return (
        <div className="flex items-stretch border-b border-black/[0.04] dark:border-white/5 bg-black/[0.015] dark:bg-white/[0.01]">
            <div className="w-36 flex-shrink-0 flex flex-col justify-center px-2 gap-1 border-r border-black/[0.06] dark:border-white/10">
                <span className="text-[9px] font-bold uppercase tracking-wide text-slate-500">Automation</span>
                <select
                    value={param}
                    onChange={(e) => onParamChange(e.target.value as AutomationParam)}
                    className="bg-transparent text-[10px] font-mono font-bold text-teal-600 dark:text-teal-400 cursor-pointer focus:outline-none"
                    aria-label={`Automation parameter for ${trackId}`}
                >
                    <option value="volume">Volume</option>
                    <option value="pan">Pan</option>
                </select>
            </div>
            <div
                ref={laneRef}
                className="relative flex-1"
                style={{ height: `${H}px` }}
                onDoubleClick={(e) => {
                    // Double-click empty space adds a breakpoint at the cursor.
                    const target = e.target as HTMLElement;
                    if (target.dataset.automationPoint === 'true') return;
                    onChange(addPoint(param, points, tOf(e.clientX), vOf(e.clientY)));
                }}
                onMouseDown={(e) => {
                    const target = e.target as HTMLElement;
                    const index = target.dataset.automationIndex;
                    if (index === undefined) return;
                    e.stopPropagation();
                    e.preventDefault();
                    const move = (ev: MouseEvent) => {
                        const next = [...sorted];
                        const i = Number(index);
                        if (!next[i]) return;
                        next[i] = { t: tOf(ev.clientX), v: vOf(ev.clientY) };
                        onChange(next);
                    };
                    const up = () => {
                        window.removeEventListener('mousemove', move);
                        window.removeEventListener('mouseup', up);
                    };
                    window.addEventListener('mousemove', move);
                    window.addEventListener('mouseup', up);
                }}
                title="Double-click to add a point · drag a point to move · double-click a point to delete"
            >
                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1000 48" preserveAspectRatio="none" aria-hidden>
                    <line x1="0" y1={yOf(param === 'volume' ? 1 : 0)} x2="1000" y2={yOf(param === 'volume' ? 1 : 0)}
                          stroke="currentColor" className="text-slate-400/30" strokeDasharray="4 4" />
                    {path && <path d={path} fill="none" stroke="#14b8a6" strokeWidth="2" vectorEffect="non-scaling-stroke" />}
                </svg>
                {sorted.map((p, i) => (
                    <button
                        key={`${param}-${i}-${p.t}`}
                        data-automation-point="true"
                        data-automation-index={i}
                        onDoubleClick={(e) => { e.stopPropagation(); onChange(removePointNear(sorted, p.t, 0.05)); }}
                        className="absolute w-2 h-2 -ml-1 -mt-1 rounded-full bg-teal-500 border border-white shadow cursor-ns-resize"
                        style={{ left: `${pct(p.t)}%`, top: `${(yOf(p.v) / H) * 100}%` }}
                        aria-label={`${param} breakpoint at ${p.t.toFixed(2)}s`}
                        title={`${p.t.toFixed(2)}s → ${p.v.toFixed(2)} (double-click to delete)`}
                    />
                ))}
            </div>
        </div>
    );
};

interface ArrangeTimelineProps {
    job: Job;
    stemChannels: StemChannel[];
    currentTime: number;
    duration: number;
    onSeek: (time: number) => void;
    onToggleMute: (id: string) => void;
    onToggleSolo: (id: string) => void;
    /** Real per-stem waveform peaks (normalized 0..1), computed from decoded audio. */
    stemPeaks?: Record<string, number[]>;
    /** Real decoded duration (seconds) per stem id — drives true clip widths. */
    stemDurations?: Record<string, number>;
    onSeparateStems?: () => void;
    isSeparating?: boolean;
    /** Editable arrangement (plan 3.1). Absent = read-only visualisation. */
    clips?: TimelineClip[];
    onClipsChange?: (clips: TimelineClip[]) => void;
    /** Snap step in seconds (0 = free). */
    snapSec?: number;
    /** Per-track automation curves (plan 3.16). */
    automation?: Record<string, Partial<Record<AutomationParam, { t: number; v: number }[]>>>;
    onAutomationChange?: (trackId: string, param: AutomationParam, points: { t: number; v: number }[]) => void;
}

export const ArrangeTimeline: React.FC<ArrangeTimelineProps> = ({
    job,
    stemChannels,
    currentTime,
    duration,
    onSeek,
    onToggleMute,
    onToggleSolo,
    stemPeaks = {},
    stemDurations = {},
    onSeparateStems,
    isSeparating,
    clips,
    onClipsChange,
    snapSec = 0,
    automation = {},
    onAutomationChange,
}) => {
    const [zoom, setZoom] = React.useState(1);
    const totalDuration = duration || 60;
    const progressPercent = Math.min(100, Math.max(0, (currentTime / totalDuration) * 100));

    const handleSeekFromX = (clientX: number, el: HTMLElement) => {
        const rect = el.getBoundingClientRect();
        const clickX = clientX - rect.left;
        onSeek((clickX / rect.width) * totalDuration);
    };

    // Calculate measure markers from the transcription's real beat grid
    // (BPM + beats per bar) so bar markers align with actual note timing.
    // eslint-disable-next-line react-hooks/exhaustive-deps -- parse keyed on the raw JSON string
    const beatGrid = useMemo(() => parseGrid(job), [job.beat_grid_json]);
    const bpm = Number(beatGrid.bpm) > 0 ? Number(beatGrid.bpm) : 120;
    const beatsPerBar = Number(beatGrid.beats_per_bar) > 0 ? Number(beatGrid.beats_per_bar) : 4;
    const barDuration = (60 / bpm) * beatsPerBar;
    const totalBars = Math.max(1, Math.ceil(totalDuration / barDuration));
    const effectiveBars = totalBars / Math.max(0.5, zoom);
    const barStep = effectiveBars > 120 ? 16 : effectiveBars > 60 ? 8 : effectiveBars > 30 ? 4 : effectiveBars > 15 ? 2 : 1;

    // eslint-disable-next-line react-hooks/exhaustive-deps -- parse keyed on the raw JSON string
    const notes = useMemo(() => parseNotes(job), [job.notes_json]);
    const sections = useMemo(() => parseSongSections(job, totalDuration), [job.timed_lyrics_json, job.lyrics, totalDuration]);

    // ── Claim-based note→lane mapping ───────────────────────────────────────
    // Each note is claimed by AT MOST ONE lane (first match wins, walking the
    // lanes top-down). The old heuristic gave every non-bass/drum/vocal note
    // to ALL remaining lanes simultaneously, duplicating content across
    // guitar/piano/strings lanes.
    const notesByLane = useMemo(() => {
        const map: Record<string, NoteEvent[]> = {};
        const claimed = new Set<NoteEvent>();
        const matches = (n: NoteEvent, token: string) =>
            (n.instrument || '').toLowerCase().includes(token);

        // Pass 1: MuScriptor per-instrument parts match their exact instrument.
        stemChannels.forEach(track => {
            if (!track.id.startsWith('part-')) return;
            const target = track.name.toLowerCase().replace(/[^a-z0-9]/g, '');
            map[track.id] = notes.filter(n => {
                if (claimed.has(n)) return false;
                const inst = (n.instrument || '').toLowerCase().replace(/[^a-z0-9]/g, '');
                return inst === target || inst.includes(target) || target.includes(inst);
            });
            map[track.id].forEach(n => claimed.add(n));
        });
        // Pass 2: neural stem groups by keyword.
        const hasExplicitBassNote = notes.some(n => (n.instrument || '').toLowerCase().includes('bass'));
        stemChannels.forEach(track => {
            if (map[track.id]) return;
            const t = track.name.toLowerCase();
            let laneNotes: NoteEvent[] = [];
            if (t.includes('bass')) {
                laneNotes = notes.filter(n => !claimed.has(n) && (matches(n, 'bass') || (!hasExplicitBassNote && n.pitch < 48)));
            } else if (t.includes('drum')) {
                laneNotes = notes.filter(n => !claimed.has(n) && (matches(n, 'drum') || matches(n, 'percussion') || n.channel === 9));
            } else if (t.includes('vocal')) {
                laneNotes = notes.filter(n => !claimed.has(n) && (matches(n, 'vocal') || matches(n, 'voice') || matches(n, 'lead') || matches(n, 'choir')));
            } else if (t.includes('guitar')) {
                laneNotes = notes.filter(n => !claimed.has(n) && (matches(n, 'guitar') || matches(n, 'pluck')));
            } else if (t.includes('piano') || t.includes('key')) {
                laneNotes = notes.filter(n => !claimed.has(n) && (matches(n, 'piano') || matches(n, 'key')));
            } else if (t.includes('string')) {
                laneNotes = notes.filter(n => !claimed.has(n) && (matches(n, 'string') || matches(n, 'violin') || matches(n, 'cello')));
            } else {
                // "Other"/catch-all lane: whatever no explicit lane claimed.
                laneNotes = notes.filter(n => !claimed.has(n));
            }
            map[track.id] = laneNotes;
            laneNotes.forEach(n => claimed.add(n));
        });
        return map;
    }, [stemChannels, notes]);

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    // ── Lanes content memoization ───────────────────────────────────────────
    // Adaptive measure ruler: prevents squished measure numbers at high bar counts
    const rulerBars = useMemo(() => {
        return Array.from({ length: totalBars }, (_, i) => i + 1).map(bar => {
            const leftPct = ((bar - 1) * barDuration / totalDuration) * 100;
            const widthPct = Math.max(0.2, (barDuration / totalDuration) * 100);
            const showLabel = (bar - 1) % barStep === 0;
            return (
                <div
                    key={bar}
                    className={`absolute top-0 bottom-0 border-l ${
                        showLabel ? 'border-slate-500/40 dark:border-slate-600' : 'border-slate-300/40 dark:border-slate-800/30'
                    } pl-1 flex items-center overflow-visible pointer-events-none`}
                    style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                >
                    {showLabel && (
                        <span className="font-bold text-[10px] select-none text-slate-700 dark:text-slate-300 whitespace-nowrap">
                            {bar}
                        </span>
                    )}
                </div>
            );
        });
    }, [totalBars, barDuration, totalDuration, barStep]);

    const lanesContent = useMemo(() => {
        if (stemChannels.length === 0) {
            return (
                <div className="h-40 flex flex-col items-center justify-center p-6 text-center border-b border-black/[0.04] dark:border-white/5 bg-slate-50/50 dark:bg-black/20">
                    <p className="text-sm font-semibold text-slate-800 dark:text-slate-200 mb-1">
                        Stereo Master Track (No Stems Extracted)
                    </p>
                    <p className="text-xs text-slate-500 dark:text-slate-400 max-w-md mb-3">
                        This session is playing the full stereo master mix. Separate this track into neural stems to enable multitrack editing, solo, mute, and per-stem mixing.
                    </p>
                    {onSeparateStems && (
                        <button
                            onClick={(e) => { e.stopPropagation(); onSeparateStems(); }}
                            disabled={isSeparating}
                            className="px-4 py-2 rounded-xl bg-teal-600 hover:bg-teal-500 disabled:opacity-50 text-white text-xs font-semibold shadow-sm transition-colors flex items-center gap-2 cursor-pointer"
                         title="Separate this track into neural stems">
                            {isSeparating ? (
                                <>
                                    <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    <span>Separating Stems with BS-Roformer...</span>
                                </>
                            ) : (
                                <span>Separate into 4 Stems (BS-Roformer)</span>
                            )}
                        </button>
                    )}
                </div>
            );
        }

        const editable = Boolean(clips && onClipsChange);
        return stemChannels.map((track) => (
            <React.Fragment key={track.id}>
            {editable ? (
                <ClipLaneRow
                    key={track.id}
                    clips={clips as TimelineClip[]}
                    track={track}
                    totalDuration={totalDuration}
                    snapSec={snapSec}
                    sourceDuration={stemDurations[track.id] || totalDuration}
                    peaks={stemPeaks[track.id]}
                    readOnly={!onClipsChange}
                    onChange={(next) => onClipsChange?.(next)}
                />
            ) : (
                <TrackLaneRow
                    key={track.id}
                    job={job}
                    track={track}
                    trackNotes={notesByLane[track.id] || []}
                    peaks={stemPeaks[track.id]}
                    stemDur={stemDurations[track.id]}
                    totalDuration={totalDuration}
                    formatTime={formatTime}
                />
            )}
            {onAutomationChange && automation[track.id] && Object.keys(automation[track.id] || {}).length > 0 && (
                <AutomationLaneRow
                    trackId={track.id}
                    param={(Object.keys(automation[track.id] || {})[0] as AutomationParam) || 'volume'}
                    points={automation[track.id]?.volume || []}
                    totalDuration={totalDuration}
                    onChange={(points) => onAutomationChange(
                        track.id,
                        (Object.keys(automation[track.id] || {})[0] as AutomationParam) || 'volume',
                        points,
                    )}
                    onParamChange={(param) => onAutomationChange(track.id, param, [])}
                />
            )}
            </React.Fragment>
        ));
    }, [job, stemChannels, notesByLane, stemPeaks, stemDurations, totalDuration, onSeparateStems,
        isSeparating, clips, onClipsChange, snapSec, zoom, automation, onAutomationChange]);

    return (
        <div className="flex flex-col h-full bg-[#f5f5f7] dark:bg-[#10121a] text-slate-800 dark:text-slate-200 select-none overflow-hidden transition-colors duration-200">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-3 border-b border-black/[0.06] dark:border-white/[0.08] bg-white/95 dark:bg-[#141620]/95 flex-shrink-0">
                <div className="flex items-center space-x-2">
                    <Layers size={16} className="text-teal-600 dark:text-teal-400" />
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-900 dark:text-slate-100">
                        Stem Arrangement & Song Structure
                    </span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20 font-semibold">
                        {job.title || "Active Master"}
                    </span>
                </div>

                <div className="flex items-center space-x-3 text-xs font-mono text-slate-500 dark:text-slate-400">
                    <div className="flex items-center bg-black/[0.04] dark:bg-surface-overlay border border-black/[0.06] dark:border-white/10 rounded-xl p-1 space-x-1">
                        <button
                            onClick={() => setZoom(prev => Math.max(0.5, prev - 0.25))}
                            className="p-1 text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 cursor-pointer"
                            title="Zoom Out"
                            aria-label="Zoom Out"
                        >
                            <ZoomOut size={12} />
                        </button>
                        <button
                            onClick={() => setZoom(1)}
                            className="text-[10px] font-mono px-1.5 tabular-nums hover:text-teal-600 dark:hover:text-teal-400 cursor-pointer"
                            title="Reset Zoom to 100%"
                            aria-label="Reset Zoom to 100%"
                        >
                            {Math.round(zoom * 100)}%
                        </button>
                        <button
                            onClick={() => setZoom(prev => Math.min(3, prev + 0.25))}
                            className="p-1 text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 cursor-pointer"
                            title="Zoom In"
                            aria-label="Zoom In"
                        >
                            <ZoomIn size={12} />
                        </button>
                    </div>
                    <span className="tabular-nums">Playhead: {currentTime.toFixed(1)}s / {totalDuration.toFixed(1)}s</span>
                    <button
                        type="button"
                        onClick={() => useModalStore.getState().openExtendTrack(job)}
                        className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold rounded-lg bg-teal-500/10 hover:bg-teal-500/20 text-teal-700 dark:text-teal-300 border border-teal-500/30 transition-colors cursor-pointer ml-1"
                        title="Extend this track past its current length with AI musical continuity"
                    >
                        <FastForward size={12} />
                        <span>Extend Song</span>
                    </button>
                </div>
            </div>

            {/* Timeline Area — one shared vertical scroller keeps headers and
                lanes aligned while allowing >7 stems to be reached. */}
            <div className="flex-1 overflow-y-auto flex">
                {/* Track Headers (Left Column, sticky during horizontal scroll) */}
                <div className="w-56 bg-white/80 dark:bg-[#12141c] border-r border-black/[0.06] dark:border-white/[0.08] flex flex-col flex-shrink-0 z-30 sticky left-0 shadow-sm">
                    {/* Ruler header */}
                    <div className="h-8 px-4 border-b border-black/[0.06] dark:border-white/5 flex items-center justify-between text-[10px] font-mono font-bold uppercase tracking-wider text-slate-400 bg-white/95 dark:bg-[#141620]/95">
                        <span>Measure Grid</span>
                        <span className="text-[9px] lowercase opacity-70">bars</span>
                    </div>

                    {/* Song Structure header */}
                    {sections.length > 0 && (
                        <div className="h-7 px-4 border-b border-black/[0.06] dark:border-white/5 flex items-center justify-between text-[10px] font-mono font-bold uppercase tracking-wider text-teal-600 dark:text-teal-400 bg-black/[0.02] dark:bg-black/30">
                            <span>Structure</span>
                            <span className="text-[9px] opacity-70">{sections.length} parts</span>
                        </div>
                    )}

                    {stemChannels.length === 0 ? (
                        <div className="h-40 px-4 border-b border-black/[0.04] dark:border-white/5 flex flex-col justify-center bg-white/90 dark:bg-[#151722]/90">
                            <span className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate block">
                                Stereo Master
                            </span>
                            <span className="text-[10px] font-mono text-slate-400">
                                Single Stereo File
                            </span>
                        </div>
                    ) : (
                        stemChannels.map((track) => (
                            <TrackHeaderRow
                                key={track.id}
                                track={track}
                                onToggleMute={onToggleMute}
                                onToggleSolo={onToggleSolo}
                            />
                        ))
                    )}
                </div>

                {/* Horizontal scroll region: ruler + lanes share one coordinate
                    space so bar numbers can NEVER detach from lane positions. */}
                <div className="flex-1 overflow-x-auto overflow-y-hidden">
                    <div className="relative" style={{ minWidth: `${100 * zoom}%` }}>
                        {/* Measure Ruler — clickable to seek, all bars rendered */}
                        <div
                            onClick={(e) => handleSeekFromX(e.clientX, e.currentTarget)}
                            className="h-8 sticky top-0 z-20 bg-black/[0.03] dark:bg-black/40 border-b border-black/[0.08] dark:border-white/5 cursor-pointer text-[10px] font-mono text-slate-600 dark:text-slate-400 select-none"
                            title="Click to move the playhead"
                        >
                            <div className="relative w-full h-full">
                                {rulerBars}
                            </div>
                        </div>

                        {/* Song Structure / Section Marker Track */}
                        {sections.length > 0 && (
                            <div
                                className="h-7 sticky top-8 z-20 bg-black/[0.04] dark:bg-black/60 border-b border-black/[0.08] dark:border-white/5 cursor-pointer text-[10px] font-mono select-none overflow-hidden"
                                title="Song structure sections — click any section to jump"
                            >
                                <div className="relative w-full h-full">
                                    {sections.map(sec => {
                                        const leftPct = (sec.start / totalDuration) * 100;
                                        const widthPct = Math.max(1, ((sec.end - sec.start) / totalDuration) * 100);
                                        const isActive = currentTime >= sec.start && currentTime < sec.end;
                                        return (
                                            <div
                                                key={sec.id}
                                                onClick={(e) => { e.stopPropagation(); onSeek(sec.start); }}
                                                className={`absolute top-0.5 bottom-0.5 rounded-md px-2 flex items-center justify-between border text-[11px] font-extrabold tracking-tight truncate transition-all ${sec.color} ${
                                                    isActive ? 'ring-1 ring-white/90 shadow-sm opacity-100' : 'opacity-90 hover:opacity-100'
                                                }`}
                                                style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                                                title={`${sec.name}: ${formatTime(sec.start)} - ${formatTime(sec.end)} (Click to jump)`}
                                            >
                                                <span className="truncate drop-shadow-sm">{sec.name}</span>
                                                <span className="text-[9px] opacity-80 font-mono hidden md:inline ml-1 font-semibold">
                                                    {formatTime(sec.start)}
                                                </span>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}

                        {/* Tracks Lane & Waveform Blocks — memoized heavy layer */}
                        <div
                            onClick={(e) => handleSeekFromX(e.clientX, e.currentTarget)}
                            className="relative cursor-pointer"
                        >
                            {lanesContent}

                            {/* Interactive Playhead Line */}
                            <div
                                className="absolute top-0 bottom-0 w-0.5 bg-rose-500 z-20 pointer-events-none shadow-lg shadow-rose-500/50 will-change-[left]"
                                style={{ left: `${progressPercent}%` }}
                            >
                                <div className="w-3.5 h-3.5 bg-rose-500 -ml-1.5 -top-1 absolute rounded-full shadow-md border-2 border-white" />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
