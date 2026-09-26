import React from 'react';
import {
    Play,
    Pause,
    Layers,
    Sparkles,
    Video,
    Download,
    Loader2,
    Mic,
    Type,
    Monitor,
    Smartphone,
    Square
} from 'lucide-react';
import type { Job } from '../../api';

export type AspectRatioType = '16:9' | '9:16' | '1:1' | '21:9';

interface VideoTopBarProps {
    completedSongs: Job[];
    selectedSongId: string | null;
    onSelectSong: (id: string) => void;
    activeSong?: Job;
    hasVocals: boolean;
    aspectRatio: AspectRatioType;
    onSelectAspectRatio: (aspect: AspectRatioType) => void;
    resolution?: '720p' | '1080p';
    isPlaying: boolean;
    playingSongId: string | null;
    onTogglePlayAudio: () => void;
    isPlanning: boolean;
    onPlanScenes: () => void;
    isGeneratingKeyframes: boolean;
    onGenerateKeyframes: (force?: boolean) => void;
    hasKeyframes?: boolean;
    isRendering: boolean;
    onRenderVideo: () => void;
    renderedVideoUrl: string | null;
    onDownloadVideo?: () => void;
}

const VideoTopBarComponent: React.FC<VideoTopBarProps> = ({
    completedSongs,
    selectedSongId,
    onSelectSong,
    activeSong,
    hasVocals,
    aspectRatio,
    onSelectAspectRatio,
    resolution = '720p',
    isPlaying,
    playingSongId,
    onTogglePlayAudio,
    isPlanning,
    onPlanScenes,
    isGeneratingKeyframes,
    onGenerateKeyframes,
    hasKeyframes = false,
    isRendering,
    onRenderVideo,
    renderedVideoUrl,
    onDownloadVideo,
}) => {
    const isThisSongPlaying = isPlaying && playingSongId === activeSong?.id;

    return (
        <header className="flex flex-col gap-3 p-3.5 sm:p-4 rounded-2xl bg-white/70 dark:bg-black/40 backdrop-blur-xl border border-black/[0.06] dark:border-white/10 shadow-apple-md">
            {/* TIER 1: Song Source & Audio Telemetry Bar */}
            <div className="flex flex-wrap items-center justify-between gap-2.5 pb-2.5 border-b border-black/[0.05] dark:border-white/[0.06]">
                {/* Left: Track Picker */}
                <div className="flex items-center gap-2 flex-1 min-w-[220px] max-w-md">
                    <select
                        value={selectedSongId || ''}
                        onChange={(e) => onSelectSong(e.target.value)}
                        className="w-full text-xs font-semibold rounded-xl bg-black/[0.04] dark:bg-white/5 border border-black/[0.08] dark:border-white/10 px-3 py-2 text-slate-900 dark:text-slate-100 pr-8 truncate focus:outline-none focus:ring-2 focus:ring-teal-500/40 cursor-pointer shadow-sm"
                    >
                        {completedSongs.map((s) => (
                            <option key={s.id} value={s.id} className="dark:bg-slate-900 text-slate-800 dark:text-slate-200">
                                {s.video_path ? '🎬 ' : '🎵 '}{s.title || s.prompt.slice(0, 32)}
                                {s.video_path ? ' · [Video Ready]' : ''}
                            </option>
                        ))}
                    </select>
                </div>

                {/* Right: Audio Preview & Stem Telemetry Badges */}
                {activeSong && (
                    <div className="flex items-center gap-2 flex-wrap text-[11px]">
                        {/* Audio Play/Pause Button */}
                        <button
                            type="button"
                            onClick={onTogglePlayAudio}
                            className={`px-2.5 py-1.5 rounded-lg font-bold flex items-center gap-1.5 transition-all active:scale-95 shadow-sm ${
                                isThisSongPlaying
                                    ? 'bg-teal-500 text-slate-950 shadow-teal-500/20'
                                    : 'bg-black/[0.05] dark:bg-white/10 text-slate-700 dark:text-slate-200 hover:bg-black/[0.08]'
                            }`}
                            title={isThisSongPlaying ? 'Pause Audio Preview' : 'Play Audio Preview'}
                        >
                            {isThisSongPlaying ? <Pause size={12} /> : <Play size={12} className="ml-0.5" />}
                            <span>{isThisSongPlaying ? 'Pause Audio' : 'Preview'}</span>
                        </button>

                        <span className="font-mono px-2 py-1 rounded-md bg-black/5 dark:bg-white/5 text-slate-600 dark:text-slate-400">
                            ⏱️ {activeSong.duration_ms ? `${Math.round(activeSong.duration_ms / 1000)}s` : 'Unknown'}
                        </span>

                        <span
                            className={`px-2 py-1 rounded-md font-semibold flex items-center gap-1 ${
                                hasVocals
                                    ? 'bg-teal-500/10 text-teal-600 dark:text-teal-400'
                                    : 'bg-amber-500/10 text-amber-600 dark:text-amber-400'
                            }`}
                        >
                            <Mic size={11} />
                            {hasVocals ? 'Vocals Isolated' : 'Full Mix'}
                        </span>

                        <span
                            className={`px-2 py-1 rounded-md font-semibold flex items-center gap-1 ${
                                activeSong.lyrics
                                    ? 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400'
                                    : 'bg-slate-500/10 text-slate-500'
                            }`}
                        >
                            <Type size={11} />
                            {activeSong.lyrics ? 'Lyrics Ready' : 'Instrumental'}
                        </span>
                    </div>
                )}
            </div>

            {/* TIER 2: Master Production Controls & Actions */}
            <div className="flex flex-wrap items-center justify-between gap-3">
                {/* Left: Aspect Ratio & Resolution Specs */}
                <div className="flex items-center gap-2 flex-wrap">
                    {/* Aspect Ratio Switcher */}
                    <div className="flex items-center gap-1 bg-black/[0.03] dark:bg-white/5 p-1 rounded-xl border border-black/[0.06] dark:border-white/10">
                        <button
                            type="button"
                            onClick={() => onSelectAspectRatio('16:9')}
                            className={`px-2.5 py-1 text-xs rounded-lg font-bold flex items-center gap-1 transition-all ${
                                aspectRatio === '16:9'
                                    ? 'bg-white dark:bg-white/15 text-slate-900 dark:text-white shadow-sm'
                                    : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                            }`}
                            title="16:9 Widescreen (YouTube, Desktop)"
                        >
                            <Monitor size={12} />
                            <span>16:9</span>
                        </button>
                        <button
                            type="button"
                            onClick={() => onSelectAspectRatio('9:16')}
                            className={`px-2.5 py-1 text-xs rounded-lg font-bold flex items-center gap-1 transition-all ${
                                aspectRatio === '9:16'
                                    ? 'bg-white dark:bg-white/15 text-slate-900 dark:text-white shadow-sm'
                                    : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                            }`}
                            title="9:16 Vertical (TikTok, Instagram Reels, Shorts)"
                        >
                            <Smartphone size={12} />
                            <span>9:16</span>
                        </button>
                        <button
                            type="button"
                            onClick={() => onSelectAspectRatio('1:1')}
                            className={`px-2.5 py-1 text-xs rounded-lg font-bold flex items-center gap-1 transition-all ${
                                aspectRatio === '1:1'
                                    ? 'bg-white dark:bg-white/15 text-slate-900 dark:text-white shadow-sm'
                                    : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                            }`}
                            title="1:1 Square (Instagram Feed)"
                        >
                            <Square size={12} />
                            <span>1:1</span>
                        </button>
                        <button
                            type="button"
                            onClick={() => onSelectAspectRatio('21:9')}
                            className={`px-2.5 py-1 text-xs rounded-lg font-bold flex items-center gap-1 transition-all ${
                                aspectRatio === '21:9'
                                    ? 'bg-white dark:bg-white/15 text-slate-900 dark:text-white shadow-sm'
                                    : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                            }`}
                            title="21:9 Ultra-Widescreen Cinemascope"
                        >
                            <span>21:9</span>
                        </button>
                    </div>

                    {/* Output Resolution Pill */}
                    <div className="px-2.5 py-1 text-[11px] font-mono font-bold rounded-lg bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20" title="Active Output Resolution">
                        {resolution}
                    </div>
                </div>

                {/* Right: Master Production Actions */}
                <div className="flex items-center gap-2 flex-wrap">
                    {/* Plan Scene Breakdown */}
                    <button
                        type="button"
                        onClick={onPlanScenes}
                        disabled={isPlanning || isRendering || !activeSong}
                        className="px-3.5 py-1.5 bg-black/[0.05] dark:bg-white/10 hover:bg-black/[0.08] dark:hover:bg-white/15 text-slate-800 dark:text-slate-200 font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-all disabled:opacity-50"
                        title="Plan intelligent scene cuts snapped to musical beats & lyrics"
                    >
                        {isPlanning ? <Loader2 size={13} className="animate-spin text-teal-500" /> : <Layers size={13} />}
                        <span>{isPlanning ? 'Planning…' : 'Plan Scenes'}</span>
                    </button>

                    {/* Pre-Render Keyframes */}
                    <button
                        type="button"
                        onClick={() => onGenerateKeyframes(Boolean(hasKeyframes))}
                        disabled={isGeneratingKeyframes || isRendering || !activeSong}
                        className="px-3.5 py-1.5 bg-purple-500/15 hover:bg-purple-500/25 text-purple-700 dark:text-purple-300 font-bold text-xs rounded-xl flex items-center space-x-1.5 border border-purple-500/20 transition-all disabled:opacity-50"
                        title="Pre-render visual keyframe stills for each planned scene before video diffusion"
                    >
                        {isGeneratingKeyframes ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
                        <span>{isGeneratingKeyframes ? 'Keyframes…' : (hasKeyframes ? 'Regenerate Stills' : 'Pre-Render Stills')}</span>
                    </button>

                    {/* Render Video */}
                    <button
                        type="button"
                        onClick={onRenderVideo}
                        disabled={isRendering || !activeSong}
                        className="px-4 py-1.5 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 shadow-md shadow-teal-500/20 active:scale-95 transition-all disabled:opacity-50"
                        title="Execute multi-scene video diffusion with vocal lip-syncing & subtitle burn"
                    >
                        {isRendering ? <Loader2 size={13} className="animate-spin" /> : <Video size={13} />}
                        <span>{isRendering ? 'Rendering Video…' : 'Render Video ⚡'}</span>
                    </button>

                    {/* Download MP4 Video */}
                    {renderedVideoUrl && onDownloadVideo && (
                        <button
                            type="button"
                            onClick={onDownloadVideo}
                            className="px-3.5 py-1.5 bg-black/[0.05] dark:bg-white/10 hover:bg-black/[0.08] dark:hover:bg-white/15 text-slate-800 dark:text-slate-200 font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-all shadow-sm"
                            title="Download Rendered Production Video"
                        >
                            <Download size={13} />
                            <span>Export MP4</span>
                        </button>
                    )}
                </div>
            </div>
        </header>
    );
};

export const VideoTopBar = React.memo(VideoTopBarComponent);
