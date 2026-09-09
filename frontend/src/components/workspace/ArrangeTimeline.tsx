import React, { useEffect, useMemo, useRef } from 'react';
import { Layers, ZoomIn, ZoomOut } from 'lucide-react';
import type { Job, NoteEvent } from '../../api';
import type { StemChannel } from './SessionWorkspace';

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
                    const lower = cleanName.toLowerCase();
                    let color = 'bg-teal-500/20 text-teal-400 border-teal-500/40 hover:bg-teal-500/30';
                    if (lower.includes('intro')) {
                        color = 'bg-indigo-500/25 text-indigo-300 border-indigo-500/50 hover:bg-indigo-500/35';
                    } else if (lower.includes('verse')) {
                        color = 'bg-sky-500/25 text-sky-300 border-sky-500/50 hover:bg-sky-500/35';
                    } else if (lower.includes('chorus') || lower.includes('hook')) {
                        color = 'bg-amber-500/30 text-amber-300 border-amber-500/60 hover:bg-amber-500/40 font-bold';
                    } else if (lower.includes('bridge')) {
                        color = 'bg-emerald-500/25 text-emerald-300 border-emerald-500/50 hover:bg-emerald-500/35';
                    } else if (lower.includes('outro')) {
                        color = 'bg-purple-500/25 text-purple-300 border-purple-500/50 hover:bg-purple-500/35';
                    }
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
                const lower = cleanName.toLowerCase();
                let color = 'bg-teal-500/20 text-teal-400 border-teal-500/40 hover:bg-teal-500/30';
                if (lower.includes('intro')) color = 'bg-indigo-500/25 text-indigo-300 border-indigo-500/50 hover:bg-indigo-500/35';
                else if (lower.includes('verse')) color = 'bg-sky-500/25 text-sky-300 border-sky-500/50 hover:bg-sky-500/35';
                else if (lower.includes('chorus') || lower.includes('hook')) color = 'bg-amber-500/30 text-amber-300 border-amber-500/60 hover:bg-amber-500/40 font-bold';
                else if (lower.includes('bridge')) color = 'bg-emerald-500/25 text-emerald-300 border-emerald-500/50 hover:bg-emerald-500/35';
                else if (lower.includes('outro')) color = 'bg-purple-500/25 text-purple-300 border-purple-500/50 hover:bg-purple-500/35';
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
    track: StemChannel;
    trackNotes: NoteEvent[];
    peaks?: number[];
    stemDur?: number;
    totalDuration: number;
    formatTime: (seconds: number) => string;
}

const TrackLaneRow: React.FC<TrackLaneRowProps> = React.memo(({
    track,
    trackNotes,
    peaks,
    stemDur,
    totalDuration,
    formatTime
}) => {
    const effectiveDur = stemDur && stemDur > 0 ? Math.min(stemDur, totalDuration) : totalDuration;
    const widthPct = Math.min(100, (effectiveDur / totalDuration) * 100);

    return (
        <div className="h-20 border-b border-black/[0.04] dark:border-white/5 p-2 flex items-center relative">
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
                <span className="text-[10px] font-mono text-white/80 relative z-10 tabular-nums pr-1">
                    {formatTime(effectiveDur)}
                </span>
            </div>
        </div>
    );
});

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
                        showLabel ? 'border-slate-400 dark:border-slate-600' : 'border-slate-200/50 dark:border-slate-800/30'
                    } pl-1 flex items-center overflow-visible pointer-events-none`}
                    style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                >
                    {showLabel && (
                        <span className="font-bold text-[10px] select-none text-slate-500 dark:text-slate-400 whitespace-nowrap">
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
                        >
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

        return stemChannels.map((track) => (
            <TrackLaneRow
                key={track.id}
                track={track}
                trackNotes={notesByLane[track.id] || []}
                peaks={stemPeaks[track.id]}
                stemDur={stemDurations[track.id]}
                totalDuration={totalDuration}
                formatTime={formatTime}
            />
        ));
    }, [stemChannels, notesByLane, stemPeaks, stemDurations, totalDuration, onSeparateStems, isSeparating]);

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
                            className="h-8 sticky top-0 z-20 bg-black/[0.02] dark:bg-black/40 border-b border-black/[0.06] dark:border-white/5 cursor-pointer text-[10px] font-mono text-slate-400 select-none"
                            title="Click to move the playhead"
                        >
                            <div className="relative w-full h-full">
                                {rulerBars}
                            </div>
                        </div>

                        {/* Song Structure / Section Marker Track */}
                        {sections.length > 0 && (
                            <div
                                className="h-7 sticky top-8 z-20 bg-black/[0.03] dark:bg-black/60 border-b border-black/[0.06] dark:border-white/5 cursor-pointer text-[10px] font-mono select-none overflow-hidden"
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
                                                className={`absolute top-0.5 bottom-0.5 rounded-md px-2 flex items-center justify-between border text-[11px] font-bold truncate transition-all ${sec.color} ${
                                                    isActive ? 'ring-1 ring-white/80 shadow-sm opacity-100' : 'opacity-85 hover:opacity-100'
                                                }`}
                                                style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                                                title={`${sec.name}: ${formatTime(sec.start)} - ${formatTime(sec.end)} (Click to jump)`}
                                            >
                                                <span className="truncate">{sec.name}</span>
                                                <span className="text-[9px] opacity-70 font-mono hidden md:inline ml-1">
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
