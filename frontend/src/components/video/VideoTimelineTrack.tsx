import React, { useState, useMemo } from 'react';
import {
    Film,
    RefreshCw,
    Maximize2,
    Layers,
    Grid,
    Camera,
    Trash2,
    ZoomIn,
    ZoomOut,
    ChevronLeft,
    ChevronRight,
    Type
} from 'lucide-react';
import { api, type VideoClipSegment, type Job } from '../../api';

interface SongSection {
    id: string;
    name: string;
    start: number;
    end: number;
    color: string;
}

function getSectionColor(name: string): string {
    const lower = name.toLowerCase();
    if (lower.includes('intro')) {
        return 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40';
    }
    if (lower.includes('verse')) {
        return 'bg-sky-500/20 text-sky-300 border-sky-500/40';
    }
    if (lower.includes('chorus') || lower.includes('hook')) {
        return 'bg-amber-500/20 text-amber-300 border-amber-500/50 font-bold';
    }
    if (lower.includes('bridge')) {
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
    }
    if (lower.includes('outro')) {
        return 'bg-purple-500/20 text-purple-300 border-purple-500/40';
    }
    return 'bg-teal-500/20 text-teal-300 border-teal-500/40';
}

function parseSections(job?: Job, totalDurationSec: number = 180): SongSection[] {
    if (!job) return [];
    const sections: SongSection[] = [];
    try {
        if (job.timed_lyrics_json) {
            const raw = typeof job.timed_lyrics_json === 'string' ? JSON.parse(job.timed_lyrics_json) : job.timed_lyrics_json;
            if (Array.isArray(raw)) {
                const markers = raw.filter((item: any) => item.is_section && item.text);
                for (let i = 0; i < markers.length; i++) {
                    const clean = markers[i].text.replace(/^\[+|\]+$/g, '').trim();
                    const start = Math.max(0, Number(markers[i].start) || 0);
                    const nextStart = i < markers.length - 1 ? Math.max(start, Number(markers[i + 1].start) || 0) : totalDurationSec;
                    const end = Math.max(start + 0.5, Math.min(totalDurationSec, nextStart));
                    sections.push({
                        id: `sec-${i}`,
                        name: clean,
                        start,
                        end,
                        color: getSectionColor(clean),
                    });
                }
            }
        }
    } catch { /* ignore parse error */ }

    if (sections.length === 0 && job.lyrics) {
        const matches = Array.from(job.lyrics.matchAll(/\[(.*?)\]/g));
        if (matches.length > 0) {
            const step = totalDurationSec / matches.length;
            matches.forEach((m, i) => {
                const clean = m[1].trim();
                sections.push({
                    id: `sec-${i}`,
                    name: clean,
                    start: i * step,
                    end: (i + 1) * step,
                    color: getSectionColor(clean),
                });
            });
        }
    }
    return sections;
}

function getShotIntentBadge(sceneType?: string) {
    switch (sceneType) {
        case 'VOCAL_PERFORMANCE':
            return { label: '🎤 Vocal', cls: 'bg-teal-500/20 text-teal-400 border-teal-500/30' };
        case 'NARRATIVE_STORY':
            return { label: '🎭 Narrative', cls: 'bg-amber-500/20 text-amber-400 border-amber-500/30' };
        case 'METAPHORICAL_VISUAL':
            return { label: '🌌 Metaphor', cls: 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30' };
        case 'INSTRUMENTAL_FOCUS':
            return { label: '🎸 Solo', cls: 'bg-rose-500/20 text-rose-400 border-rose-500/30' };
        case 'ENVIRONMENTAL_BROLL':
        case 'CINEMATIC_BROLL':
        default:
            return { label: '🏙️ B-Roll', cls: 'bg-purple-500/20 text-purple-400 border-purple-500/30' };
    }
}

interface TimedLyricLine {
    id: string;
    text: string;
    start: number;
    end: number;
    words?: Array<{ word: string; start: number; end: number }>;
}

function parseTimedLyrics(job?: Job): TimedLyricLine[] {
    if (!job?.timed_lyrics_json) return [];
    try {
        const raw = typeof job.timed_lyrics_json === 'string'
            ? JSON.parse(job.timed_lyrics_json)
            : job.timed_lyrics_json;
        if (!Array.isArray(raw)) return [];
        return raw
            .filter((item: any) => !item.is_section && item.text && item.text.trim())
            .map((item: any, idx: number) => ({
                id: `line-${idx}`,
                text: item.text.trim(),
                start: Math.max(0, Number(item.start) || 0),
                end: Math.max((Number(item.start) || 0) + 0.3, Number(item.end) || 0),
                words: Array.isArray(item.words) ? item.words : undefined
            }));
    } catch {
        return [];
    }
}

interface VideoTimelineTrackProps {
    clips: VideoClipSegment[];
    keyframes: Record<number, string>;
    aspectRatio?: '16:9' | '9:16' | '1:1' | '21:9';
    activeSong?: Job;
    transitionStyle?: string;
    onRetakeClip: (clipIndex: number) => void;
    onZoomKeyframe: (clipIndex: number, url: string) => void;
    onSeekToTime?: (timeSec: number) => void;
    onClearTimeline?: () => void;
    onReorderClips?: (reordered: VideoClipSegment[]) => void;
    onNudgeLyric?: (lineIndex: number, deltaSec: number) => void;
}

const VideoTimelineTrackComponent: React.FC<VideoTimelineTrackProps> = ({
    clips,
    keyframes,
    aspectRatio = '16:9',
    activeSong,
    transitionStyle = 'beat_cut',
    onRetakeClip,
    onZoomKeyframe,
    onSeekToTime,
    onClearTimeline,
    onReorderClips,
    onNudgeLyric,
}) => {
    const [viewMode, setViewMode] = useState<'timeline' | 'grid'>('timeline');
    const [zoomPxPerSec, setZoomPxPerSec] = useState<number>(36); // Proportional scale: 20 to 80 px/sec

    const aspectClass = useMemo(() => {
        switch (aspectRatio) {
            case '9:16':
                return 'aspect-[9/16]';
            case '1:1':
                return 'aspect-square';
            case '21:9':
                return 'aspect-[21/9]';
            case '16:9':
            default:
                return 'aspect-video';
        }
    }, [aspectRatio]);

    const totalDurationSec = useMemo(() => {
        if (clips.length > 0) {
            const last = clips[clips.length - 1];
            return Math.max(30, last.end_time || 180);
        }
        return activeSong?.duration_ms ? activeSong.duration_ms / 1000 : 180;
    }, [clips, activeSong]);

    const songSections = useMemo(() => {
        return parseSections(activeSong, totalDurationSec);
    }, [activeSong, totalDurationSec]);

    const timedLyrics = useMemo(() => {
        return parseTimedLyrics(activeSong);
    }, [activeSong]);

    const handleShiftClip = (idx: number, direction: 'left' | 'right') => {
        if (!onReorderClips || clips.length < 2) return;
        const targetIdx = direction === 'left' ? idx - 1 : idx + 1;
        if (targetIdx < 0 || targetIdx >= clips.length) return;

        const copy = [...clips];
        const temp = copy[idx];
        copy[idx] = copy[targetIdx];
        copy[targetIdx] = temp;
        onReorderClips(copy);
    };

    if (!clips || clips.length === 0) {
        return null;
    }

    return (
        <section className="space-y-3 p-4 bg-white/70 dark:bg-black/40 backdrop-blur-xl border border-black/[0.06] dark:border-white/10 rounded-2xl shadow-apple-lg">
            {/* Header & Mode / Zoom Switcher */}
            <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-3">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 flex items-center gap-2">
                        <Layers size={14} className="text-teal-500" />
                        <span>Production NLE Multitrack Timeline ({clips.length} Clips)</span>
                    </h3>
                    <span className="text-[10px] font-mono text-slate-400">
                        Total {Math.round(totalDurationSec)}s · Vocals: {clips.filter(c => c.scene_type === 'VOCAL_PERFORMANCE').length} · B-Roll: {clips.filter(c => c.scene_type !== 'VOCAL_PERFORMANCE').length}
                    </span>
                </div>

                <div className="flex items-center gap-2">
                    {/* Zoom Controls for Timeline View */}
                    {viewMode === 'timeline' && (
                        <div className="flex items-center gap-1.5 bg-black/[0.04] dark:bg-white/5 px-2 py-1 rounded-xl border border-black/[0.06] dark:border-white/10 text-xs">
                            <button
                                type="button"
                                onClick={() => setZoomPxPerSec(prev => Math.max(18, prev - 6))}
                                className="p-0.5 rounded text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
                                title="Zoom out timeline"
                            >
                                <ZoomOut size={13} />
                            </button>
                            <span className="text-[10px] font-mono text-slate-400 w-8 text-center">{zoomPxPerSec}px/s</span>
                            <button
                                type="button"
                                onClick={() => setZoomPxPerSec(prev => Math.min(80, prev + 6))}
                                className="p-0.5 rounded text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
                                title="Zoom in timeline"
                            >
                                <ZoomIn size={13} />
                            </button>
                        </div>
                    )}

                    {onClearTimeline && (
                        <button
                            type="button"
                            onClick={() => {
                                if (window.confirm("Are you sure you want to clear the timeline? This will purge all planned scenes and cached keyframe stills for this track.")) {
                                    onClearTimeline();
                                }
                            }}
                            className="px-2.5 py-1 text-xs rounded-xl font-medium text-rose-500 hover:text-rose-600 hover:bg-rose-500/10 border border-rose-500/20 transition-all flex items-center gap-1.5"
                            title="Purge planned scenes, director notes, and keyframes"
                        >
                            <Trash2 size={12} />
                            <span>Clear</span>
                        </button>
                    )}

                    <div className="flex bg-black/[0.04] dark:bg-white/5 p-1 rounded-xl border border-black/[0.06] dark:border-white/10">
                        <button
                            type="button"
                            onClick={() => setViewMode('timeline')}
                            className={`px-2.5 py-1 text-xs rounded-lg font-bold transition-all flex items-center gap-1.5 ${
                                viewMode === 'timeline'
                                    ? 'bg-white dark:bg-white/15 text-teal-600 dark:text-teal-400 shadow-sm'
                                    : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                            }`}
                            title="Proportional linear multitrack timeline"
                        >
                            <Layers size={12} />
                            <span>Timeline</span>
                        </button>
                        <button
                            type="button"
                            onClick={() => setViewMode('grid')}
                            className={`px-2.5 py-1 text-xs rounded-lg font-bold transition-all flex items-center gap-1.5 ${
                                viewMode === 'grid'
                                    ? 'bg-white dark:bg-white/15 text-indigo-600 dark:text-indigo-400 shadow-sm'
                                    : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                            }`}
                            title="Storyboard cards shotboard grid"
                        >
                            <Grid size={12} />
                            <span>Grid</span>
                        </button>
                    </div>
                </div>
            </div>

            {/* Mode A: Horizontal Proportional Multitrack DAW Timeline */}
            {viewMode === 'timeline' && (
                <div className="space-y-2 overflow-x-auto pb-2 pt-1 select-none">
                    {/* Track 1: Song Sections Ruler */}
                    {songSections.length > 0 && (
                        <div className="flex h-5 rounded-lg overflow-hidden border border-black/[0.04] dark:border-white/5 bg-black/[0.02] dark:bg-white/[0.02] min-w-max">
                            {songSections.map((sec) => {
                                const widthPx = Math.max(50, Math.round((sec.end - sec.start) * zoomPxPerSec));
                                return (
                                    <div
                                        key={sec.id}
                                        style={{ width: `${widthPx}px` }}
                                        onClick={() => onSeekToTime?.(sec.start)}
                                        className={`h-full border-r border-black/10 dark:border-white/10 flex items-center px-2 text-[9px] font-mono font-bold truncate cursor-pointer hover:opacity-80 transition-opacity ${sec.color}`}
                                        title={`${sec.name} (${sec.start.toFixed(1)}s - ${sec.end.toFixed(1)}s) - Click to seek`}
                                    >
                                        [{sec.name}]
                                    </div>
                                );
                            })}
                        </div>
                    )}

                    {/* Track 2: Horizontal Video Clip Blocks (Duration Proportional) */}
                    <div className="flex items-stretch gap-2 min-w-max pb-1">
                        {clips.map((clip, idx) => {
                            const kfUrl = keyframes[clip.clip_index];
                            const badge = getShotIntentBadge(clip.scene_type);
                            const energyDots = clip.musical_energy ? '⚡'.repeat(Math.min(5, Math.max(1, clip.musical_energy))) : null;
                            const clipWidthPx = Math.max(170, Math.round(clip.duration * zoomPxPerSec));

                            return (
                                <div
                                    key={clip.clip_index}
                                    style={{ width: `${clipWidthPx}px` }}
                                    onClick={() => onSeekToTime?.(clip.start_time)}
                                    className="flex-shrink-0 p-2 rounded-2xl border transition-all flex flex-col justify-between group relative overflow-hidden cursor-pointer select-none bg-black/[0.02] dark:bg-white/[0.03] border-black/[0.08] dark:border-white/10 hover:border-teal-500/50 hover:shadow-md"
                                    title={`Click to jump playhead to ${clip.time_str} (${clip.duration.toFixed(1)}s)`}
                                >
                                    {/* Keyframe / Poster Image */}
                                    <div className={`relative ${aspectClass} rounded-xl overflow-hidden bg-black/40 border border-black/10 dark:border-white/10 mb-1.5`}>
                                        {kfUrl ? (
                                            <img
                                                src={api.getAudioUrl(kfUrl)}
                                                alt={`Clip ${clip.clip_index}`}
                                                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                                            />
                                        ) : (
                                            <div className="w-full h-full flex flex-col items-center justify-center text-slate-500 text-[10px]">
                                                <Film size={18} className="mb-1 text-slate-600" />
                                                <span>Prompt Ready</span>
                                            </div>
                                        )}

                                        {/* Floating Badge on Thumbnail */}
                                        <div className="absolute top-1 left-1 bg-black/70 backdrop-blur-md px-1.5 py-0.5 rounded text-[8px] font-mono font-bold text-white flex items-center gap-1">
                                            <span>#{clip.clip_index}</span>
                                            {clip.section_label && (
                                                <span className="text-teal-300 font-normal">· {clip.section_label}</span>
                                            )}
                                        </div>

                                        {/* Musical Energy Badge */}
                                        {energyDots && (
                                            <div className="absolute top-1 right-1 bg-black/70 backdrop-blur-md px-1 py-0.5 rounded text-[8px] text-amber-300" title={`Musical Energy: ${clip.musical_energy}/5`}>
                                                {energyDots}
                                            </div>
                                        )}

                                        {/* Zoom Keyframe Lightbox Button */}
                                        {kfUrl && (
                                            <button
                                                type="button"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    onZoomKeyframe(clip.clip_index, kfUrl);
                                                }}
                                                className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-white"
                                                title="View full-resolution keyframe still"
                                            >
                                                <Maximize2 size={16} />
                                            </button>
                                        )}
                                    </div>

                                    {/* Clip Info & Tags */}
                                    <div className="space-y-1 flex-1 flex flex-col justify-between">
                                        <div>
                                            <div className="flex items-center justify-between text-[10px]">
                                                <span className="font-mono font-bold text-slate-700 dark:text-slate-300">
                                                    {clip.time_str} ({clip.duration.toFixed(1)}s)
                                                </span>
                                                <span
                                                    className={`px-1.5 py-0.5 rounded border font-bold uppercase tracking-wider text-[8px] ${badge.cls}`}
                                                >
                                                    {badge.label}
                                                </span>
                                            </div>

                                            {clip.visual_action ? (
                                                <p className="text-[10px] text-slate-800 dark:text-slate-200 font-medium line-clamp-2 mt-0.5">
                                                    <span className="font-bold text-teal-600 dark:text-teal-400">Action:</span> {clip.visual_action}
                                                </p>
                                            ) : (
                                                <p className="text-[10px] text-slate-800 dark:text-slate-200 font-medium line-clamp-2 mt-0.5">
                                                    {clip.prompt}
                                                </p>
                                            )}

                                            {/* Synced Lyrics Snippet */}
                                            {clip.lyrics && (
                                                <p className="text-[9px] italic text-cyan-600 dark:text-cyan-400 truncate mt-0.5">
                                                    "{clip.lyrics}"
                                                </p>
                                            )}
                                        </div>

                                        {/* Footer: Camera, Reorder & Retake */}
                                        <div className="pt-1.5 border-t border-black/[0.04] dark:border-white/5 flex items-center justify-between text-[9px] font-mono text-slate-400">
                                            {/* Shift / Reorder Arrows */}
                                            {onReorderClips && (
                                                <div className="flex items-center gap-0.5" onClick={(e) => e.stopPropagation()}>
                                                    <button
                                                        type="button"
                                                        disabled={idx === 0}
                                                        onClick={() => handleShiftClip(idx, 'left')}
                                                        className="p-0.5 rounded hover:bg-black/10 dark:hover:bg-white/10 disabled:opacity-30 disabled:pointer-events-none"
                                                        title="Shift scene earlier"
                                                    >
                                                        <ChevronLeft size={11} />
                                                    </button>
                                                    <button
                                                        type="button"
                                                        disabled={idx === clips.length - 1}
                                                        onClick={() => handleShiftClip(idx, 'right')}
                                                        className="p-0.5 rounded hover:bg-black/10 dark:hover:bg-white/10 disabled:opacity-30 disabled:pointer-events-none"
                                                        title="Shift scene later"
                                                    >
                                                        <ChevronRight size={11} />
                                                    </button>
                                                </div>
                                            )}

                                            <span className="truncate max-w-[80px] flex items-center gap-1" title={clip.camera}>
                                                <Camera size={9} />
                                                <span>{clip.camera?.split(' ')[0] || 'Cam'}</span>
                                            </span>

                                            {/* Retake Clip Action Button */}
                                            <button
                                                type="button"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    onRetakeClip(clip.clip_index);
                                                }}
                                                className="px-1.5 py-0.5 rounded-md bg-teal-500/10 hover:bg-teal-500/20 text-teal-700 dark:text-teal-300 font-bold flex items-center gap-1 transition-all"
                                                title="Re-prompt and generate a single scene retake"
                                            >
                                                <RefreshCw size={9} />
                                                <span>Retake</span>
                                            </button>
                                        </div>
                                    </div>

                                    {/* Transition Indicator Pill (between scenes) */}
                                    {idx < clips.length - 1 && (
                                        <div className="absolute top-1/2 -right-3 -translate-y-1/2 z-10 pointer-events-none">
                                            <span className="bg-black/80 backdrop-blur-md border border-white/20 text-[7px] text-teal-300 px-1 py-0.5 rounded-full font-mono uppercase tracking-wider shadow">
                                                {transitionStyle === 'crossfade' ? 'fade' : transitionStyle === 'whip_pan' ? 'pan' : 'cut'}
                                            </span>
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>

                    {/* Track 3: Synced Karaoke Lyrics Track */}
                    <div className="pt-1.5 space-y-1">
                        <div className="flex items-center justify-between text-[10px] text-slate-500 px-1">
                            <span className="font-bold flex items-center gap-1.5 text-cyan-600 dark:text-cyan-400">
                                <Type size={12} className="text-cyan-400" />
                                <span>Karaoke Lyrics Track</span>
                                {timedLyrics.length > 0 && (
                                    <span className="text-[9px] font-mono px-1.5 py-0.2 rounded-full bg-cyan-500/10 text-cyan-500 border border-cyan-500/20">
                                        {timedLyrics.length} lines
                                    </span>
                                )}
                            </span>
                            <span className="text-[9px] font-mono text-slate-400">
                                Click line to seek · Nudge [±0.1s] timing
                            </span>
                        </div>

                        {timedLyrics.length > 0 ? (
                            <div
                                style={{ width: `${Math.max(600, Math.round(totalDurationSec * zoomPxPerSec))}px` }}
                                className="relative h-9 rounded-xl border border-cyan-500/20 bg-cyan-500/[0.03] overflow-hidden select-none"
                            >
                                {timedLyrics.map((line, lIdx) => {
                                    const leftPx = Math.round(line.start * zoomPxPerSec);
                                    const widthPx = Math.max(90, Math.round((line.end - line.start) * zoomPxPerSec));

                                    return (
                                        <div
                                            key={line.id}
                                            style={{
                                                position: 'absolute',
                                                left: `${leftPx}px`,
                                                width: `${widthPx}px`
                                            }}
                                            onClick={() => onSeekToTime?.(line.start)}
                                            className="group absolute top-1 bottom-1 rounded-lg border border-cyan-500/30 bg-white/90 dark:bg-black/70 shadow-sm flex items-center justify-between px-2 cursor-pointer hover:border-cyan-400 hover:bg-cyan-500/15 transition-all"
                                            title={`"${line.text}" (${line.start.toFixed(2)}s - ${line.end.toFixed(2)}s) · Click to jump playhead`}
                                        >
                                            <span className="text-[10px] font-semibold text-slate-800 dark:text-slate-200 truncate pr-1">
                                                {line.text}
                                            </span>

                                            {/* Micro Nudge Controls on Hover */}
                                            {onNudgeLyric && (
                                                <div
                                                    className="hidden group-hover:flex items-center gap-0.5 flex-shrink-0"
                                                    onClick={(e) => e.stopPropagation()}
                                                >
                                                    <button
                                                        type="button"
                                                        onClick={() => onNudgeLyric(lIdx, -0.1)}
                                                        className="px-1 py-0.5 rounded text-[8px] font-mono font-bold bg-black/10 dark:bg-white/10 text-slate-400 hover:text-cyan-400 hover:bg-black/20"
                                                        title="Nudge line 0.1s earlier"
                                                    >
                                                        -0.1
                                                    </button>
                                                    <button
                                                        type="button"
                                                        onClick={() => onNudgeLyric(lIdx, 0.1)}
                                                        className="px-1 py-0.5 rounded text-[8px] font-mono font-bold bg-black/10 dark:bg-white/10 text-slate-400 hover:text-cyan-400 hover:bg-black/20"
                                                        title="Nudge line 0.1s later"
                                                    >
                                                        +0.1
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        ) : (
                            <div className="h-7 rounded-xl border border-dashed border-black/10 dark:border-white/10 flex items-center justify-center text-[10px] text-slate-400 font-mono">
                                <span>No timed lyrics found · Click "Acoustically Sync Lyrics ⚡" in Inspector Dock to align</span>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Mode B: Storyboard Shotboard Grid View */}
            {viewMode === 'grid' && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-96 overflow-y-auto pr-1">
                    {clips.map((clip) => {
                        const kfUrl = keyframes[clip.clip_index];
                        const badge = getShotIntentBadge(clip.scene_type);
                        const energyDots = clip.musical_energy ? '⚡'.repeat(Math.min(5, Math.max(1, clip.musical_energy))) : null;
                        return (
                            <div
                                key={clip.clip_index}
                                onClick={() => onSeekToTime?.(clip.start_time)}
                                className="p-3 bg-black/[0.02] dark:bg-white/[0.02] rounded-2xl border border-black/[0.06] dark:border-white/10 flex flex-col justify-between space-y-2 group hover:border-teal-500/40 transition-all cursor-pointer select-none"
                                title={`Click to jump playhead to ${clip.time_str}`}
                            >
                                <div className="flex items-center gap-3">
                                    <div className={`relative w-28 ${aspectClass} rounded-xl overflow-hidden bg-black/40 flex-shrink-0 border border-white/10`}>
                                        {kfUrl ? (
                                            <img
                                                src={api.getAudioUrl(kfUrl)}
                                                alt={`Scene ${clip.clip_index}`}
                                                className="w-full h-full object-cover"
                                            />
                                        ) : (
                                            <div className="w-full h-full flex items-center justify-center text-slate-500 text-[10px]">
                                                <Film size={16} />
                                            </div>
                                        )}
                                        <span className="absolute bottom-1 right-1 text-[8px] bg-black/80 px-1 py-0.5 rounded font-mono font-bold text-teal-400">
                                            #{clip.clip_index}
                                        </span>
                                    </div>

                                    <div className="space-y-1 flex-1 min-w-0">
                                        <div className="flex items-center gap-1.5 flex-wrap">
                                            <span className="text-[10px] font-mono font-bold text-teal-600 dark:text-teal-400 bg-teal-500/10 px-1.5 py-0.5 rounded">
                                                {clip.time_str} ({clip.duration.toFixed(1)}s)
                                            </span>
                                            {clip.section_label && (
                                                <span className="text-[9px] font-mono font-bold text-slate-400 bg-black/[0.04] dark:bg-white/[0.05] px-1.5 py-0.5 rounded">
                                                    [{clip.section_label}]
                                                </span>
                                            )}
                                            <span
                                                className={`text-[9px] font-bold px-1.5 py-0.5 rounded-full border ${badge.cls}`}
                                            >
                                                {badge.label}
                                            </span>
                                            {energyDots && (
                                                <span className="text-[9px] text-amber-300 font-mono ml-auto" title={`Energy: ${clip.musical_energy}/5`}>
                                                    {energyDots}
                                                </span>
                                            )}
                                        </div>
                                        {clip.visual_action ? (
                                            <p className="text-[11px] text-slate-800 dark:text-slate-200 font-medium line-clamp-2">
                                                <span className="font-bold text-teal-600 dark:text-teal-400">Action:</span> {clip.visual_action}
                                            </p>
                                        ) : (
                                            <p className="text-[11px] text-slate-800 dark:text-slate-200 font-medium line-clamp-2">
                                                {clip.prompt}
                                            </p>
                                        )}
                                    </div>
                                </div>

                                {clip.directors_note && (
                                    <div className="px-2 py-0.5 rounded bg-black/[0.03] dark:bg-white/[0.04] text-[9px] text-slate-500 italic truncate" title={clip.directors_note}>
                                        🎬 Note: {clip.directors_note}
                                    </div>
                                )}

                                {clip.lyrics && (
                                    <p className="text-[10px] italic text-cyan-600 dark:text-cyan-400 truncate">
                                        "{clip.lyrics}"
                                    </p>
                                )}

                                <div className="flex items-center justify-between pt-1 border-t border-black/[0.04] dark:border-white/5 text-[10px] font-mono text-slate-400">
                                    <span className="truncate max-w-[140px]" title={clip.camera}>🎥 {clip.camera}</span>
                                    <button
                                        type="button"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onRetakeClip(clip.clip_index);
                                        }}
                                        className="px-2 py-0.5 rounded bg-teal-500/10 hover:bg-teal-500/20 text-teal-600 dark:text-teal-400 font-bold flex items-center gap-1"
                                    >
                                        <RefreshCw size={10} />
                                        <span>Retake</span>
                                    </button>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </section>
    );
};

export const VideoTimelineTrack = React.memo(VideoTimelineTrackComponent);
