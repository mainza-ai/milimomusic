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
    Video
} from 'lucide-react';
import { api, galleryApi, type Job, type VideoTaskStatus } from '../../api';
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
    isPlanning?: boolean;
}

export const VideoCanvasPlayer: React.FC<VideoCanvasPlayerProps> = ({
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
    isPlanning = false,
}) => {
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);

    // Split comparison state
    const [splitCompareActive, setSplitCompareActive] = useState(false);
    const [splitRatio, setSplitRatio] = useState(0.5); // 0 to 1
    const [isDraggingSplit, setIsDraggingSplit] = useState(false);

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
                            className="w-full h-full object-cover rounded-xl"
                        />

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
                ) : (
                    /* Empty Slate / Ready to Render View */
                    <div className="p-8 flex flex-col items-center justify-center max-w-md space-y-4">
                        <div className="relative">
                            <div className="absolute inset-0 bg-teal-500/20 rounded-full blur-xl animate-pulse" />
                            <div className="w-16 h-16 rounded-2xl bg-teal-500/10 border border-teal-500/20 text-teal-400 flex items-center justify-center shadow-lg relative">
                                <Film size={32} className={isRendering ? 'animate-bounce' : ''} />
                            </div>
                        </div>

                        <div>
                            <h3 className="text-lg font-extrabold text-white tracking-tight">
                                {isRendering ? 'Generating Music Video…' : (activeSong?.title || 'AI Music Video Studio')}
                            </h3>
                            <p className="text-xs text-slate-400 mt-1 line-clamp-2">
                                {isRendering
                                    ? `Executing multi-scene generation and stem alignment (Stage: ${activeTask?.step || 'diffusion'})…`
                                    : activeSong
                                    ? `Directing: "${activeSong.prompt}"`
                                    : 'Select a track to start multi-scene video production.'}
                            </p>
                        </div>

                        {!isRendering && activeSong && (
                            <div className="flex items-center gap-2 pt-2">
                                {onPlanScenes && (
                                    <button
                                        type="button"
                                        onClick={onPlanScenes}
                                        disabled={isPlanning}
                                        className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white font-bold text-xs rounded-xl flex items-center space-x-1.5 backdrop-blur-md transition-all disabled:opacity-50"
                                    >
                                        {isPlanning ? <Loader2 size={13} className="animate-spin" /> : <Layers size={13} />}
                                        <span>Plan Scene Breakdown</span>
                                    </button>
                                )}
                                {onRenderVideo && (
                                    <button
                                        type="button"
                                        onClick={onRenderVideo}
                                        className="px-4 py-2 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 shadow-md shadow-teal-500/20 transition-all active:scale-95"
                                    >
                                        <Video size={13} />
                                        <span>Render Production Video</span>
                                    </button>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Multi-Stage Live Pipeline Rendering HUD */}
            {activeTask && (
                <div
                    className={`p-4 rounded-2xl border space-y-3 transition-all ${
                        activeTask.status === 'error'
                            ? 'bg-rose-500/10 border-rose-500/30'
                            : activeTask.status === 'completed'
                            ? 'bg-teal-500/10 border-teal-500/30'
                            : 'bg-black/[0.03] dark:bg-white/5 border-black/[0.06] dark:border-white/10'
                    }`}
                >
                    <div className="flex items-center justify-between text-xs">
                        <span className="font-bold flex items-center gap-2 text-slate-800 dark:text-slate-200">
                            {activeTask.status === 'error' && <AlertCircle size={14} className="text-rose-500" />}
                            {activeTask.status === 'completed' && <CheckCircle2 size={14} className="text-teal-500" />}
                            {activeTask.status === 'processing' && <Loader2 size={14} className="animate-spin text-teal-500" />}
                            <span>
                                Pipeline Stage: <strong className="uppercase font-mono text-teal-600 dark:text-teal-400">{activeTask.step.replace(/_/g, ' ')}</strong>
                            </span>
                        </span>
                        <span className="font-mono text-xs font-bold text-slate-700 dark:text-slate-300">
                            {activeTask.progress}%
                        </span>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full h-1.5 bg-black/[0.06] dark:bg-white/10 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-teal-500 to-cyan-400 rounded-full transition-all duration-500"
                            style={{ width: `${Math.max(3, activeTask.progress)}%` }}
                        />
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-slate-500 font-mono">
                        <span>
                            Clip {activeTask.current_clip} of {activeTask.total_clips}
                        </span>
                        {activeTask.error && <span className="text-rose-500 font-semibold">{activeTask.error}</span>}
                    </div>
                </div>
            )}
        </div>
    );
};
