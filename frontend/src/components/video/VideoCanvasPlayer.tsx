import React, { useRef, useState, useEffect } from 'react';
import {
    Film,
    RefreshCw,
    Trash2,
    Share2,
    Loader2,
    AlertCircle,
    CheckCircle2,
    Layers,
    Video,
    X,
    Square,
    Mic,
    Play
} from 'lucide-react';
import { api, galleryApi, type Job, type VideoTaskStatus } from '../../api';
import { useAudioTime } from '../../context/AudioEngineContext';
import { LyricCanvasOverlay } from './LyricCanvasOverlay';
import type { AspectRatioType } from './VideoTopBar';

interface VideoCanvasPlayerProps {
    activeSong?: Job;
    renderedVideoUrl: string | null;
    aspectRatio: AspectRatioType;
    isRendering: boolean;
    activeTask: VideoTaskStatus | null;
    isDeletingVideo: boolean;
    onDeleteVideo: () => void;
    onRegenerateVideo: () => void;
    onRouteToDirector: () => void;
    isRouting: boolean;
    onPlanScenes?: () => void;
    onRenderVideo?: () => void;
    onCancelRender?: () => void;
    isPlanning?: boolean;
    seekTime?: number | null;
    onDismissTask?: () => void;
    // Fast-Path Lyric Studio & WYSIWYG
    isPlayingAudio?: boolean;
    stylePreset?: string;
    backgroundMode?: string;
    fontFamily?: string;
    fontSizeOverride?: number;
    onSeek?: (timeSec: number) => void;
    onRenderLyricVideo?: () => void;
    isRenderingLyricVideo?: boolean;
}

const VideoCanvasPlayerComponent: React.FC<VideoCanvasPlayerProps> = ({
    activeSong,
    renderedVideoUrl,
    aspectRatio,
    isRendering,
    activeTask,
    isDeletingVideo,
    onDeleteVideo,
    onRegenerateVideo,
    onRouteToDirector,
    isRouting,
    onPlanScenes,
    onRenderVideo,
    onCancelRender,
    isPlanning = false,
    seekTime,
    onDismissTask,
    isPlayingAudio = false,
    stylePreset = 'neon',
    backgroundMode = 'cover_art',
    fontFamily,
    fontSizeOverride,
    onSeek,
    onRenderLyricVideo,
    isRenderingLyricVideo = false,
}) => {
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);

    // Audio Engine synchronization for 60fps lyric sweeps
    const { currentTime: audioCurrentTime } = useAudioTime();
    const [videoCurrentTime, setVideoCurrentTime] = useState(0);
    const [showLyricOverlay, setShowLyricOverlay] = useState(false);

    const effectiveTime = renderedVideoUrl ? videoCurrentTime : audioCurrentTime;

    // split comparison state
    const [splitCompareActive, setSplitCompareActive] = useState(false);
    const [splitRatio, setSplitRatio] = useState(0.5); // 0 to 1
    const [isDraggingSplit, setIsDraggingSplit] = useState(false);
    const [videoError, setVideoError] = useState(false);
    const [isVideoPlaying, setIsVideoPlaying] = useState(false);

    useEffect(() => {
        setVideoError(false);
        if (renderedVideoUrl) {
            setShowLyricOverlay(false);
        }
    }, [renderedVideoUrl]);

    const toggleVideoPlayback = () => {
        if (!videoRef.current) return;
        if (videoRef.current.paused) {
            videoRef.current.play().catch((err) => {
                console.warn('Video playback request error:', err);
            });
        } else {
            videoRef.current.pause();
        }
    };

    // Sync playhead when seeking from timeline with settle guard
    useEffect(() => {
        if (seekTime !== undefined && seekTime !== null && videoRef.current) {
            videoRef.current.currentTime = seekTime;
            if (videoRef.current.paused) {
                videoRef.current.play().catch(() => {});
            }
        }
    }, [seekTime]);

    // Sync video with external isPlayingAudio transport if user toggles DAW play button
    useEffect(() => {
        if (videoRef.current && renderedVideoUrl) {
            if (isPlayingAudio && videoRef.current.paused) {
                videoRef.current.play().catch(() => {});
            } else if (!isPlayingAudio && !videoRef.current.paused) {
                videoRef.current.pause();
            }
        }
    }, [isPlayingAudio, renderedVideoUrl]);

    // Aspect ratio classes
    const aspectClass =
        aspectRatio === '9:16'
            ? 'aspect-[9/16] max-h-[75vh] w-auto mx-auto'
            : aspectRatio === '1:1'
            ? 'aspect-square max-h-[70vh] w-auto mx-auto'
            : aspectRatio === '21:9'
            ? 'aspect-[21/9] w-full'
            : 'aspect-video w-full';

    // Handle split drag
    const handleSplitMouseDown = (e: React.MouseEvent) => {
        e.preventDefault();
        setIsDraggingSplit(true);
    };

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (!isDraggingSplit || !containerRef.current) return;
            const rect = containerRef.current.getBoundingClientRect();
            const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
            setSplitRatio(x / rect.width);
        };

        const handleMouseUp = () => {
            if (isDraggingSplit) setIsDraggingSplit(false);
        };

        if (isDraggingSplit) {
            window.addEventListener('mousemove', handleMouseMove);
            window.addEventListener('mouseup', handleMouseUp);
        }
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isDraggingSplit]);

    return (
        <div className="space-y-3">
            {/* Main Video Viewport Canvas */}
            <div
                ref={containerRef}
                className={`relative ${aspectClass} rounded-2xl bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 border border-white/10 flex flex-col items-center justify-center text-center overflow-hidden shadow-apple-2xl group select-none`}
            >
                {renderedVideoUrl ? (
                    <>
                        {/* Primary Rendered Video Element */}
                        <video
                            ref={videoRef}
                            src={api.getAudioUrl(renderedVideoUrl)}
                            poster={galleryApi.getThumbnailUrl(renderedVideoUrl.split('/').pop() || renderedVideoUrl)}
                            controls
                            playsInline
                            preload="auto"
                            onTimeUpdate={(e) => setVideoCurrentTime(e.currentTarget.currentTime)}
                            onPlay={() => setIsVideoPlaying(true)}
                            onPause={() => setIsVideoPlaying(false)}
                            onError={() => {
                                if (videoRef.current?.error) {
                                    setVideoError(true);
                                }
                            }}
                            onLoadedData={() => setVideoError(false)}
                            onClick={toggleVideoPlayback}
                            className="w-full h-full object-cover rounded-xl cursor-pointer"
                        />

                        {/* Center Play Button HUD when Paused */}
                        {!isVideoPlaying && !videoError && (
                            <div
                                onClick={toggleVideoPlayback}
                                className="absolute inset-0 flex items-center justify-center bg-black/25 hover:bg-black/35 backdrop-blur-[1px] transition-all cursor-pointer z-10"
                            >
                                <button
                                    type="button"
                                    className="w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-teal-500/95 hover:bg-teal-400 text-slate-950 flex items-center justify-center shadow-apple-2xl transform hover:scale-110 active:scale-95 transition-all duration-200 pl-1"
                                    title="Play Video"
                                >
                                    <Play size={36} className="fill-current text-slate-950" />
                                </button>
                            </div>
                        )}

                        {/* Live Subtitle Overlay Preview over rendered video */}
                        {showLyricOverlay && activeSong && (
                            <div className="absolute inset-0 pointer-events-none">
                                <LyricCanvasOverlay
                                    activeSong={activeSong}
                                    currentTime={videoCurrentTime}
                                    isPlaying={Boolean(videoRef.current && !videoRef.current.paused)}
                                    stylePreset={stylePreset}
                                    aspectRatio={aspectRatio}
                                    backgroundMode={backgroundMode}
                                    fontFamily={fontFamily}
                                    fontSizeOverride={fontSizeOverride}
                                    isOverVideo={true}
                                    onSeek={(t) => {
                                        if (videoRef.current) {
                                            videoRef.current.currentTime = t;
                                        }
                                        onSeek?.(t);
                                    }}
                                />
                            </div>
                        )}

                        {/* Playback Error Fallback Overlay */}
                        {videoError && (
                            <div className="absolute inset-0 bg-slate-950/90 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center z-20">
                                <div className="w-12 h-12 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-400 flex items-center justify-center mb-3">
                                    <AlertCircle size={24} />
                                </div>
                                <h4 className="text-sm font-bold text-white mb-1">Video playback unavailable</h4>
                                <p className="text-xs text-slate-400 max-w-sm mb-4">
                                    The video file could not be loaded or was removed. You can regenerate it or clear the video reference.
                                </p>
                                <div className="flex items-center gap-2">
                                    <button
                                        type="button"
                                        onClick={onRegenerateVideo}
                                        disabled={isRendering || isDeletingVideo}
                                        className="px-3 py-1.5 bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-slate-950 font-bold text-xs rounded-lg flex items-center gap-1.5 shadow-md"
                                    >
                                        <RefreshCw size={12} />
                                        <span>Regenerate Video</span>
                                    </button>
                                    <button
                                        type="button"
                                        onClick={onDeleteVideo}
                                        disabled={isDeletingVideo}
                                        className="px-3 py-1.5 bg-white/10 hover:bg-white/20 text-white font-medium text-xs rounded-lg flex items-center gap-1.5 border border-white/10"
                                    >
                                        {isDeletingVideo ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
                                        <span>{isDeletingVideo ? 'Deleting…' : 'Clear / Delete'}</span>
                                    </button>
                                </div>
                            </div>
                        )}

                        {/* Top Floating Telemetry Overlay */}
                        <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                            <div className="flex items-center gap-1.5 bg-black/70 backdrop-blur-md px-2.5 py-1 rounded-lg border border-white/10 text-[10px] font-mono font-bold text-teal-300">
                                <span>🎬 {activeSong?.title || 'Production Music Video'}</span>
                                <span className="text-white/40">·</span>
                                <span className="text-white/80">{aspectRatio}</span>
                            </div>

                            <div className="flex items-center gap-2 pointer-events-auto">
                                <button
                                    type="button"
                                    onClick={() => setShowLyricOverlay(!showLyricOverlay)}
                                    className={`px-2.5 py-1 rounded-lg text-[10px] font-bold backdrop-blur-md border transition-all ${
                                        showLyricOverlay
                                            ? 'bg-teal-500/80 text-slate-950 border-teal-400 font-extrabold shadow-sm'
                                            : 'bg-black/60 text-slate-300 border-white/20 hover:text-white'
                                    }`}
                                    title="Toggle Synchronized Overlay Subtitles over Video"
                                >
                                    {showLyricOverlay ? '🎤 Lyric Overlay: ON' : '🎤 Lyric Overlay: OFF'}
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setSplitCompareActive(!splitCompareActive)}
                                    className={`px-2.5 py-1 rounded-lg text-[10px] font-bold backdrop-blur-md border transition-all ${
                                        splitCompareActive
                                            ? 'bg-indigo-500 text-white border-indigo-400 shadow-md'
                                            : 'bg-black/60 text-slate-300 border-white/20 hover:text-white'
                                    }`}
                                    title="Toggle Split A/B Comparison View"
                                >
                                    {splitCompareActive ? '✕ Close Split' : '↔ Split Compare'}
                                </button>
                            </div>
                        </div>

                        {/* Split A/B Comparison Overlay Curtain */}
                        {splitCompareActive && (
                            <>
                                {/* Draggable vertical divider line */}
                                <div
                                    className="absolute top-0 bottom-0 z-20 cursor-ew-resize flex items-center justify-center"
                                    style={{ left: `${splitRatio * 100}%` }}
                                    onMouseDown={handleSplitMouseDown}
                                >
                                    <div className="w-0.5 h-full bg-indigo-400 shadow-[0_0_10px_rgba(129,140,248,0.8)]" />
                                    <div className="absolute w-6 h-6 rounded-full bg-indigo-500 border-2 border-white flex items-center justify-center text-[10px] text-white font-bold shadow-lg pointer-events-none">
                                        ↔
                                    </div>
                                </div>

                                {/* Floating Split Ratio Controls */}
                                <div className="absolute bottom-16 left-4 right-4 bg-black/80 backdrop-blur-md rounded-xl p-2.5 border border-indigo-500/30 flex items-center gap-3 z-30">
                                    <span className="text-[10px] font-bold text-slate-300">Take A (Original)</span>
                                    <input
                                        type="range"
                                        min={0}
                                        max={1}
                                        step={0.01}
                                        value={splitRatio}
                                        onChange={(e) => setSplitRatio(parseFloat(e.target.value))}
                                        className="flex-1 accent-indigo-500 h-1.5 bg-white/20 rounded cursor-pointer"
                                    />
                                    <span className="text-[10px] font-bold text-indigo-400">
                                        Take B / Retake ({Math.round(splitRatio * 100)}%)
                                    </span>
                                </div>
                            </>
                        )}

                        {/* Bottom Action Bar (Floating on hover) */}
                        <div className="absolute bottom-3 right-3 flex items-center gap-2 z-10">
                            <button
                                type="button"
                                onClick={onRouteToDirector}
                                disabled={isRendering || isDeletingVideo || isRouting}
                                className="px-3 py-1.5 bg-black/60 hover:bg-black/80 text-white font-bold text-[11px] rounded-lg flex items-center gap-1.5 backdrop-blur-md border border-white/20 shadow-md transition-all disabled:opacity-50"
                                title="Route this video to Gallery & Director reference input"
                            >
                                {isRouting ? <Loader2 size={12} className="animate-spin text-teal-400" /> : <Share2 size={12} className="text-teal-400" />}
                                <span>{isRouting ? 'Routing…' : 'To Director'}</span>
                            </button>

                            <button
                                type="button"
                                onClick={onRegenerateVideo}
                                disabled={isRendering || isDeletingVideo}
                                className="px-3 py-1.5 bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-slate-950 font-bold text-[11px] rounded-lg flex items-center gap-1.5 shadow-md transition-all disabled:opacity-50"
                                title="Re-render with the exact stored pipeline configuration"
                            >
                                {isRendering ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                                <span>{isRendering ? 'Rendering…' : 'Regenerate'}</span>
                            </button>

                            <button
                                type="button"
                                onClick={onDeleteVideo}
                                disabled={isRendering || isDeletingVideo}
                                className="px-3 py-1.5 bg-rose-500/90 hover:bg-rose-500 text-white font-bold text-[11px] rounded-lg flex items-center gap-1.5 shadow-md transition-all disabled:opacity-50"
                                title="Delete this rendered video (audio and track stay untouched)"
                            >
                                {isDeletingVideo ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
                                <span>{isDeletingVideo ? 'Deleting…' : 'Delete'}</span>
                            </button>
                        </div>
                    </>
                ) : activeSong ? (
                    /* WYSIWYG Live Lyric Canvas Player & Pre-Render Mode */
                    <div className="relative w-full h-full flex flex-col items-center justify-center">
                        <LyricCanvasOverlay
                            activeSong={activeSong}
                            currentTime={effectiveTime}
                            isPlaying={isPlayingAudio}
                            stylePreset={stylePreset}
                            aspectRatio={aspectRatio}
                            backgroundMode={backgroundMode}
                            fontFamily={fontFamily}
                            fontSizeOverride={fontSizeOverride}
                            isOverVideo={false}
                            onSeek={onSeek}
                        />

                        {/* Live Rendering Modal Card Overlay */}
                        {isRendering && (
                            <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center z-30">
                                <div className="w-16 h-16 rounded-2xl bg-teal-500/10 border border-teal-500/20 text-teal-400 flex items-center justify-center shadow-lg relative mb-4 animate-bounce">
                                    <Film size={32} />
                                </div>
                                <h3 className="text-lg font-extrabold text-white tracking-tight">
                                    Rendering Video…
                                </h3>
                                <p className="text-xs text-slate-300 mt-1 max-w-sm">
                                    {activeTask?.step || 'Composing frames and synthesizing visuals…'}
                                </p>
                                {onCancelRender && (
                                    <div className="pt-4">
                                        <button
                                            type="button"
                                            onClick={onCancelRender}
                                            className="px-4 py-2 bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 text-xs font-bold rounded-xl flex items-center space-x-1.5 shadow-sm transition-all"
                                        >
                                            <Square size={12} className="fill-current" />
                                            <span>Stop Video Generation</span>
                                        </button>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Top Floating Telemetry Overlay */}
                        <div className="absolute top-3 left-3 flex items-center gap-2 z-20 pointer-events-none">
                            <div className="flex items-center gap-1.5 bg-black/70 backdrop-blur-md px-2.5 py-1 rounded-lg border border-white/10 text-[10px] font-mono font-bold text-teal-300">
                                <span>🎵 {activeSong.title || 'Track Preview'}</span>
                                <span className="text-white/40">·</span>
                                <span className="text-white/80">{aspectRatio}</span>
                                <span className="text-white/40">·</span>
                                <span className="text-cyan-400">WYSIWYG Lyric Canvas</span>
                            </div>
                        </div>

                        {/* Floating Quick Action Bar */}
                        {!isRendering && (
                            <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between z-20 pointer-events-auto">
                                <div className="flex items-center gap-2">
                                    {onRenderLyricVideo && (
                                        <button
                                            type="button"
                                            onClick={onRenderLyricVideo}
                                            disabled={isRendering || isRenderingLyricVideo}
                                            className="px-4 py-2 bg-gradient-to-r from-teal-400 to-cyan-400 hover:from-teal-300 hover:to-cyan-300 text-slate-950 font-bold text-xs rounded-xl flex items-center gap-1.5 shadow-lg shadow-teal-500/20 active:scale-95 transition-all disabled:opacity-50"
                                            title="Render fast local lyric video in under 30 seconds"
                                        >
                                            {isRenderingLyricVideo ? <Loader2 size={13} className="animate-spin" /> : <Mic size={13} />}
                                            <span>Create Lyric Video 🎤</span>
                                        </button>
                                    )}
                                </div>

                                <div className="flex items-center gap-2">
                                    {onPlanScenes && (
                                        <button
                                            type="button"
                                            onClick={onPlanScenes}
                                            disabled={isPlanning}
                                            className="px-3.5 py-2 bg-black/60 hover:bg-black/80 text-white font-bold text-xs rounded-xl flex items-center gap-1.5 backdrop-blur-md border border-white/15 transition-all disabled:opacity-50"
                                        >
                                            {isPlanning ? <Loader2 size={13} className="animate-spin text-teal-400" /> : <Layers size={13} />}
                                            <span>Plan Scenes</span>
                                        </button>
                                    )}
                                    {onRenderVideo && (
                                        <button
                                            type="button"
                                            onClick={onRenderVideo}
                                            disabled={isRendering}
                                            className="px-3.5 py-2 bg-white/10 hover:bg-white/20 text-white font-bold text-xs rounded-xl flex items-center gap-1.5 backdrop-blur-md border border-white/15 transition-all disabled:opacity-50"
                                        >
                                            <Video size={13} />
                                            <span>Render Full Video</span>
                                        </button>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                ) : (
                    /* Empty Slate / No Track Selected View */
                    <div className="p-8 flex flex-col items-center justify-center max-w-md space-y-4">
                        <div className="relative">
                            <div className="absolute inset-0 bg-teal-500/20 rounded-full blur-xl animate-pulse" />
                            <div className="w-16 h-16 rounded-2xl bg-teal-500/10 border border-teal-500/20 text-teal-400 flex items-center justify-center shadow-lg relative">
                                <Film size={32} />
                            </div>
                        </div>

                        <div>
                            <h3 className="text-lg font-extrabold text-white tracking-tight">
                                AI Music Video Studio
                            </h3>
                            <p className="text-xs text-slate-400 mt-1 line-clamp-2">
                                Select a completed track from the top bar to preview live lyrics and create a music video.
                            </p>
                        </div>
                    </div>
                )}
            </div>

            {/* Multi-Stage Live Pipeline Rendering / Director Planning HUD */}
            {activeTask && (
                <div
                    className={`p-4 rounded-2xl border space-y-3 transition-all ${
                        activeTask.status === 'error'
                            ? 'bg-rose-500/10 border-rose-500/30'
                            : activeTask.status === 'cancelled'
                            ? 'bg-amber-500/10 border-amber-500/30'
                            : activeTask.status === 'completed'
                            ? 'bg-teal-500/10 border-teal-500/30'
                            : 'bg-black/[0.03] dark:bg-white/5 border-black/[0.06] dark:border-white/10'
                    }`}
                >
                    <div className="flex items-center justify-between text-xs">
                        <span className="font-bold flex items-center gap-2 text-slate-800 dark:text-slate-200">
                            {activeTask.status === 'error' && <AlertCircle size={14} className="text-rose-500" />}
                            {activeTask.status === 'cancelled' && <Square size={12} className="text-amber-500 fill-current" />}
                            {activeTask.status === 'completed' && <CheckCircle2 size={14} className="text-teal-500" />}
                            {activeTask.status === 'processing' && <Loader2 size={14} className="animate-spin text-teal-500" />}
                            <span>
                                {activeTask.id.startsWith('plan_') ? 'AI Visual Director: ' : 'Pipeline Stage: '}
                                <strong className="uppercase font-mono text-teal-600 dark:text-teal-400">
                                    {activeTask.step.replace(/_/g, ' ')}
                                </strong>
                            </span>
                        </span>
                        <div className="flex items-center gap-2">
                            {activeTask.status === 'processing' && onCancelRender && (
                                <button
                                    type="button"
                                    onClick={onCancelRender}
                                    className="px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/20 flex items-center gap-1 transition-colors"
                                    title={activeTask.id.startsWith('plan_') ? "Cancel scene planning" : "Cancel video generation"}
                                >
                                    <Square size={9} className="fill-current" />
                                    <span>{activeTask.id.startsWith('plan_') ? "Stop Planning" : "Stop"}</span>
                                </button>
                            )}
                            <span className="font-mono text-xs font-bold text-slate-700 dark:text-slate-300">
                                {activeTask.progress}%
                            </span>
                            {onDismissTask && (activeTask.status === 'error' || activeTask.status === 'completed' || activeTask.status === 'cancelled') && (
                                <button
                                    onClick={onDismissTask}
                                    title="Dismiss status"
                                    className="p-0.5 rounded-md hover:bg-black/10 dark:hover:bg-white/10 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
                                >
                                    <X size={13} />
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full h-1.5 bg-black/[0.06] dark:bg-white/10 rounded-full overflow-hidden">
                        <div
                            className={`h-full rounded-full transition-all duration-500 ${
                                activeTask.status === 'cancelled'
                                    ? 'bg-amber-500'
                                    : activeTask.id.startsWith('plan_')
                                    ? 'bg-gradient-to-r from-purple-500 via-pink-500 to-teal-400 animate-pulse'
                                    : 'bg-gradient-to-r from-teal-500 to-cyan-400'
                            }`}
                            style={{ width: `${Math.max(3, activeTask.progress)}%` }}
                        />
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-slate-500 font-mono">
                        <span>
                            {activeTask.id.startsWith('plan_')
                                ? (activeTask.total_clips > 0 ? `${activeTask.total_clips} scenes planned` : 'Synchronizing downbeats, lyrics & cinematic prompts')
                                : `Clip ${activeTask.current_clip} of ${activeTask.total_clips}`}
                        </span>
                        {activeTask.error && <span className="text-rose-500 font-semibold">{activeTask.error}</span>}
                    </div>

                    {/* Fallback metadata banner */}
                    {activeTask.fallback_used && (
                        <div className="flex items-center gap-2 p-2 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-[11px]">
                            <AlertCircle size={14} className="shrink-0 text-amber-400" />
                            <span>
                                <strong>{activeTask.id.startsWith('plan_') ? 'Deterministic Fallback Director:' : 'Cinematic Animatic Mode:'}</strong> {activeTask.fallback_reason || (activeTask.id.startsWith('plan_') ? 'LLM service was unreachable; used acoustic downbeat pacing.' : 'Selected neural diffusion engine offline; used synchronized keyframe animatic.')}
                            </span>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export const VideoCanvasPlayer = React.memo(VideoCanvasPlayerComponent);
