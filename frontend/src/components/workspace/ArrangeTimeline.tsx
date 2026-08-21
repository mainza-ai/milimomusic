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
    stemDurations = {}
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

    // eslint-disable-next-line react-hooks/exhaustive-deps -- parse keyed on the raw JSON string
    const notes = useMemo(() => parseNotes(job), [job.notes_json]);

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
            map[track.id] = notes.filter(n =>
                !claimed.has(n) && (n.instrument || '').toLowerCase() === track.name.toLowerCase()
            );
            map[track.id].forEach(n => claimed.add(n));
        });
        // Pass 2: neural stem groups by keyword.
        stemChannels.forEach(track => {
            if (map[track.id]) return;
            const t = track.name.toLowerCase();
            let laneNotes: NoteEvent[] = [];
            if (t.includes('bass')) {
                laneNotes = notes.filter(n => !claimed.has(n) && matches(n, 'bass'));
            } else if (t.includes('drum')) {
                laneNotes = notes.filter(n => !claimed.has(n) && (matches(n, 'drum') || matches(n, 'percussion')));
            } else if (t.includes('vocal')) {
                laneNotes = notes.filter(n => !claimed.has(n) && (matches(n, 'vocal') || matches(n, 'lead')));
            } else if (t.includes('guitar')) {
                laneNotes = notes.filter(n => !claimed.has(n) && matches(n, 'guitar'));
            } else if (t.includes('piano') || t.includes('key')) {
                laneNotes = notes.filter(n => !claimed.has(n) && (matches(n, 'piano') || matches(n, 'key')));
            } else if (t.includes('string')) {
                laneNotes = notes.filter(n => !claimed.has(n) && (matches(n, 'string') || matches(n, 'guitar')));
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
    // The clip layer (real waveform bars × lanes + note blocks) is the heavy
    // DOM here. It does NOT depend on time — only the playhead line does — so
    // it's memoized and the ~12Hz playhead ticks stop reconciling it.
    const onSeekRef = useRef(onSeek);
    useEffect(() => { onSeekRef.current = onSeek; });

    const lanesContent = useMemo(() => stemChannels.map((track) => {
        const trackNotes = notesByLane[track.id] || [];
        const peaks = stemPeaks[track.id];
        // TRUE clip width: the stem's real decoded length relative to the
        // timeline — not a forced full-width slab.
        const stemDur = stemDurations[track.id];
        const widthPct = stemDur
            ? Math.min(100, (stemDur / totalDuration) * 100)
            : 100;

        return (
            <div
                key={track.id}
                className="h-20 border-b border-black/[0.04] dark:border-white/5 p-2 flex items-center relative"
            >
                {/* Audio Stem Block with REAL waveform */}
                <div
                    className={`h-16 rounded-xl bg-gradient-to-r ${track.color} p-2 flex items-center justify-between shadow-sm relative overflow-hidden transition-opacity ${
                        track.isMuted ? 'opacity-30' : 'opacity-90'
                    }`}
                    style={{ width: `${widthPct}%` }}
                >
                    {/* Real decoded-amplitude waveform (always drawn when available) */}
                    {peaks && peaks.length > 0 && (
                        <div className="absolute inset-0 flex items-end justify-between opacity-40 pointer-events-none px-1 pb-1">
                            {peaks.map((p, i) => (
                                <div
                                    key={i}
                                    className="w-[2px] bg-white rounded-full self-center"
                                    style={{ height: `${Math.max(8, p * 90)}%` }}
                                />
                            ))}
                        </div>
                    )}

                    {/* Transcribed note blocks overlay — positioned against the
                        CLIP's own duration (the clip is sized to the stem's real
                        length), never against totalDuration. */}
                    {trackNotes.length > 0 && (
                        <div className="absolute inset-0 pointer-events-none p-1">
                            {(() => {
                                const refDur = stemDur || totalDuration;
                                return trackNotes.filter(n => n.start_time <= refDur).map((n, idx) => {
                                    const leftPct = (n.start_time / refDur) * 100;
                                    const noteDur = n.duration !== undefined ? n.duration : (n.end_time ? n.end_time - n.start_time : 0.5);
                                    const wPct = Math.max(0.4, (noteDur / refDur) * 100);
                                    const topPct = 10 + ((n.pitch % 12) / 12) * 60;
                                    return (
                                        <div
                                            key={idx}
                                            className="absolute bg-white/70 rounded-sm shadow-sm"
                                            style={{
                                                left: `${leftPct}%`,
                                                width: `${wPct}%`,
                                                top: `${topPct}%`,
                                                height: '6px'
                                            }}
                                        />
                                    );
                                });
                            })()}
                        </div>
                    )}

                    <span className="text-xs font-bold text-white relative z-10 drop-shadow-sm truncate">
                        {track.name} {trackNotes.length > 0 ? `(${trackNotes.length} notes)` : ''}
                    </span>
                    {/* Real measured duration — no fabricated format claims */}
                    <span className="text-[10px] font-mono text-white/80 relative z-10 tabular-nums pr-1">
                        {stemDur ? formatTime(stemDur) : ''}
                    </span>
                </div>
            </div>
        );
    }), [stemChannels, notesByLane, stemPeaks, stemDurations, totalDuration]);

    return (
        <div className="flex flex-col h-full bg-[#f5f5f7] dark:bg-[#10121a] text-slate-800 dark:text-slate-200 select-none overflow-hidden transition-colors duration-200">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-3 border-b border-black/[0.06] dark:border-white/[0.08] bg-white/70 dark:bg-[#141620]/80 backdrop-blur-xl flex-shrink-0">
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
                            className="p-1 text-slate-500 hover:text-slate-900 dark:hover:text-slate-200"
                            title="Zoom Out"
                            aria-label="Zoom Out"
                        >
                            <ZoomOut size={12} />
                        </button>
                        <span className="text-[10px] font-mono px-1.5 tabular-nums">{Math.round(zoom * 100)}%</span>
                        <button
                            onClick={() => setZoom(prev => Math.min(3, prev + 0.25))}
                            className="p-1 text-slate-500 hover:text-slate-900 dark:hover:text-slate-200"
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
                <div className="w-56 bg-white/80 dark:bg-[#12141c] border-r border-black/[0.06] dark:border-white/[0.08] flex flex-col pt-8 flex-shrink-0 z-30 sticky left-0 shadow-sm">
                    {stemChannels.map((track) => (
                        <div
                            key={track.id}
                            className="h-20 px-4 border-b border-black/[0.04] dark:border-white/5 flex items-center justify-between bg-white/60 dark:bg-[#151722]/80 backdrop-blur-md"
                        >
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
                                    onClick={() => onToggleMute(track.id)}
                                    aria-pressed={track.isMuted}
                                    title={`Mute ${track.name}`}
                                    aria-label={`Mute ${track.name}`}
                                    className={`w-7 h-7 rounded-lg text-[10px] font-bold transition-colors ${
                                        track.isMuted
                                            ? 'bg-rose-500 text-white'
                                            : 'bg-black/[0.04] dark:bg-white/5 text-slate-400 hover:text-slate-700'
                                    }`}
                                >
                                    M
                                </button>
                                <button
                                    onClick={() => onToggleSolo(track.id)}
                                    aria-pressed={track.isSolo}
                                    title={`Solo ${track.name}`}
                                    aria-label={`Solo ${track.name}`}
                                    className={`w-7 h-7 rounded-lg text-[10px] font-bold transition-colors ${
                                        track.isSolo
                                            ? 'bg-amber-500 text-slate-950 font-extrabold'
                                            : 'bg-black/[0.04] dark:bg-white/5 text-slate-400 hover:text-slate-700'
                                    }`}
                                >
                                    S
                                </button>
                            </div>
                        </div>
                    ))}
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
                                {Array.from({ length: totalBars }, (_, i) => i + 1).map(bar => {
                                    const leftPct = ((bar - 1) * barDuration / totalDuration) * 100;
                                    const widthPct = Math.max(0.5, (barDuration / totalDuration) * 100);
                                    return (
                                        <div
                                            key={bar}
                                            className="absolute top-0 bottom-0 border-l border-slate-300 dark:border-slate-700/50 pl-1 flex items-center"
                                            style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                                        >
                                            <span className="font-bold">{bar}</span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Tracks Lane & Waveform Blocks — memoized heavy layer */}
                        <div
                            onClick={(e) => handleSeekFromX(e.clientX, e.currentTarget)}
                            className="relative cursor-pointer"
                        >
                            {lanesContent}

                            {/* Interactive Playhead Line */}
                            <div
                                className="absolute top-0 bottom-0 w-0.5 bg-rose-500 z-20 pointer-events-none transition-all duration-75 shadow-lg shadow-rose-500/50"
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
