import React, { useState, useMemo } from 'react';
import {
    Film,
    RefreshCw,
    Maximize2,
    Layers,
    Grid,
    Camera
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

interface VideoTimelineTrackProps {
    clips: VideoClipSegment[];
    keyframes: Record<number, string>;
    activeSong?: Job;
    onRetakeClip: (clipIndex: number) => void;
    onZoomKeyframe: (clipIndex: number, url: string) => void;
}

export const VideoTimelineTrack: React.FC<VideoTimelineTrackProps> = ({
    clips,
    keyframes,
    activeSong,
    onRetakeClip,
    onZoomKeyframe,
}) => {
    const [viewMode, setViewMode] = useState<'timeline' | 'grid'>('timeline');

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

    if (!clips || clips.length === 0) {
        return null;
    }

    return (
        <section className="space-y-3 p-4 bg-white/70 dark:bg-black/40 backdrop-blur-xl border border-black/[0.06] dark:border-white/10 rounded-2xl shadow-apple-lg">
            {/* Header / View Switcher */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 flex items-center gap-2">
                        <Layers size={14} className="text-teal-500" />
                        <span>Production Multitrack Timeline ({clips.length} Clips)</span>
                    </h3>
                    <span className="text-[10px] font-mono text-slate-400">
                        Total {Math.round(totalDurationSec)}s · Vocals: {clips.filter(c => c.scene_type === 'VOCAL_PERFORMANCE').length} · B-Roll: {clips.filter(c => c.scene_type !== 'VOCAL_PERFORMANCE').length}
                    </span>
                </div>

                <div className="flex bg-black/[0.04] dark:bg-white/5 p-1 rounded-xl border border-black/[0.06] dark:border-white/10">
                    <button
                        type="button"
                        onClick={() => setViewMode('timeline')}
                        className={`px-2.5 py-1 text-xs rounded-lg font-bold transition-all flex items-center gap-1.5 ${
                            viewMode === 'timeline'
                                ? 'bg-white dark:bg-white/15 text-teal-600 dark:text-teal-400 shadow-sm'
                                : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                        }`}
                        title="Linear horizontal multitrack timeline"
                    >
                        <Layers size={12} />
                        <span>Timeline View</span>
                    </button>
                    <button
                        type="button"
                        onClick={() => setViewMode('grid')}
                        className={`px-2.5 py-1 text-xs rounded-lg font-bold transition-all flex items-center gap-1.5 ${
                            viewMode === 'grid'
                                ? 'bg-white dark:bg-white/15 text-indigo-600 dark:text-indigo-400 shadow-sm'
                                : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                        }`}
                        title="Storyboard cards shot grid"
                    >
                        <Grid size={12} />
                        <span>Shotboard Grid</span>
                    </button>
                </div>
            </div>

            {/* Mode A: Horizontal Multitrack DAW Timeline */}
            {viewMode === 'timeline' && (
                <div className="space-y-2 overflow-x-auto pb-2 pt-1 select-none">
                    {/* Track 1: Song Sections Ruler */}
                    {songSections.length > 0 && (
                        <div className="flex h-6 rounded-lg overflow-hidden border border-black/[0.04] dark:border-white/5 bg-black/[0.02] dark:bg-white/[0.02]">
                            {songSections.map((sec) => {
                                const widthPct = Math.max(2, ((sec.end - sec.start) / totalDurationSec) * 100);
                                return (
                                    <div
                                        key={sec.id}
                                        style={{ width: `${widthPct}%` }}
                                        className={`h-full border-r border-black/10 dark:border-white/10 flex items-center px-2 text-[10px] font-mono font-bold truncate ${sec.color}`}
                                        title={`${sec.name} (${sec.start.toFixed(1)}s - ${sec.end.toFixed(1)}s)`}
                                    >
                                        [{sec.name}]
                                    </div>
                                );
                            })}
                        </div>
                    )}

                    {/* Track 2: Horizontal Video Clip Blocks */}
                    <div className="flex gap-2 min-w-max pb-1">
                        {clips.map((clip) => {
                            const kfUrl = keyframes[clip.clip_index];
                            const isVocal = clip.scene_type === 'VOCAL_PERFORMANCE';
                            return (
                                <div
                                    key={clip.clip_index}
                                    className={`w-52 flex-shrink-0 p-2.5 rounded-2xl border transition-all flex flex-col justify-between group relative overflow-hidden ${
                                        isVocal
                                            ? 'bg-teal-500/[0.03] dark:bg-teal-500/[0.05] border-teal-500/20 hover:border-teal-500/40'
                                            : 'bg-purple-500/[0.03] dark:bg-purple-500/[0.05] border-purple-500/20 hover:border-purple-500/40'
                                    }`}
                                >
                                    {/* Keyframe / Poster Image */}
                                    <div className="relative aspect-video rounded-xl overflow-hidden bg-black/40 border border-black/10 dark:border-white/10 mb-2">
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
                                        <div className="absolute top-1 left-1 bg-black/70 backdrop-blur-md px-1.5 py-0.5 rounded text-[9px] font-mono font-bold text-white">
                                            #{clip.clip_index}
                                        </div>

                                        {/* Hover Overlay Action: Zoom Keyframe */}
                                        {kfUrl && (
                                            <button
                                                type="button"
                                                onClick={() => onZoomKeyframe(clip.clip_index, kfUrl)}
                                                className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-white"
                                                title="View full-resolution keyframe still"
                                            >
                                                <Maximize2 size={16} />
                                            </button>
                                        )}
                                    </div>

                                    {/* Clip Info & Tags */}
                                    <div className="space-y-1.5 flex-1 flex flex-col justify-between">
                                        <div>
                                            <div className="flex items-center justify-between text-[10px]">
                                                <span className="font-mono font-bold text-slate-700 dark:text-slate-300">
                                                    {clip.time_str} ({clip.duration.toFixed(1)}s)
                                                </span>
                                                <span
                                                    className={`px-1.5 py-0.2 rounded font-bold uppercase tracking-wider text-[8px] ${
                                                        isVocal
                                                            ? 'bg-teal-500/20 text-teal-400'
                                                            : 'bg-purple-500/20 text-purple-400'
                                                    }`}
                                                >
                                                    {isVocal ? '🎤 Vocal' : '🎥 B-Roll'}
                                                </span>
                                            </div>

                                            <p className="text-[11px] text-slate-800 dark:text-slate-200 font-medium line-clamp-2 mt-1">
                                                {clip.prompt}
                                            </p>
                                        </div>

                                        {/* Lyrics or Camera Tag */}
                                        <div className="pt-1.5 border-t border-black/[0.04] dark:border-white/5 flex items-center justify-between text-[9px] font-mono text-slate-400">
                                            <span className="truncate max-w-[120px] flex items-center gap-1">
                                                <Camera size={10} />
                                                <span>{clip.camera?.split(' ')[0] || 'Camera'}</span>
                                            </span>

                                            {/* Retake Clip Action Button */}
                                            <button
                                                type="button"
                                                onClick={() => onRetakeClip(clip.clip_index)}
                                                className="px-2 py-0.5 rounded-md bg-teal-500/10 hover:bg-teal-500/20 text-teal-700 dark:text-teal-300 font-bold flex items-center gap-1 transition-all"
                                                title="Re-prompt and generate a single scene retake"
                                            >
                                                <RefreshCw size={10} />
                                                <span>Retake</span>
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Mode B: Storyboard 3-Column Grid View */}
            {viewMode === 'grid' && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-96 overflow-y-auto pr-1">
                    {clips.map((clip) => {
                        const kfUrl = keyframes[clip.clip_index];
                        const isVocal = clip.scene_type === 'VOCAL_PERFORMANCE';
                        return (
                            <div
                                key={clip.clip_index}
                                className="p-3 bg-black/[0.02] dark:bg-white/[0.02] rounded-2xl border border-black/[0.06] dark:border-white/10 flex flex-col justify-between space-y-2 group"
                            >
                                <div className="flex items-center gap-3">
                                    <div className="relative w-28 aspect-video rounded-xl overflow-hidden bg-black/40 flex-shrink-0 border border-white/10">
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
                                                {clip.time_str}
                                            </span>
                                            <span
                                                className={`text-[9px] font-bold px-1.5 py-0.5 rounded-full ${
                                                    isVocal
                                                        ? 'bg-teal-500/20 text-teal-300'
                                                        : 'bg-purple-500/20 text-purple-300'
                                                }`}
                                            >
                                                {isVocal ? '🎤 Vocal' : '🎥 B-Roll'}
                                            </span>
                                        </div>
                                        <p className="text-[11px] text-slate-800 dark:text-slate-200 font-medium line-clamp-2">
                                            {clip.prompt}
                                        </p>
                                    </div>
                                </div>

                                {clip.lyrics && (
                                    <p className="text-[10px] italic text-cyan-600 dark:text-cyan-400 truncate">
                                        "{clip.lyrics}"
                                    </p>
                                )}

                                <div className="flex items-center justify-between pt-1 border-t border-black/[0.04] dark:border-white/5 text-[10px] font-mono text-slate-400">
                                    <span>🎥 {clip.camera}</span>
                                    <button
                                        type="button"
                                        onClick={() => onRetakeClip(clip.clip_index)}
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
