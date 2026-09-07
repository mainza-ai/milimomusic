import React, { useState, useEffect } from 'react';
import { type Job, videoApi, type StoryboardScene } from '../../api';
import { Play, Film, Wand2, Download, Video, Sparkles, Loader2, CheckCircle2 } from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';
import { AppFooter } from '../ui/AppFooter';

interface MusicVideosViewProps {
    songs: Job[];
    onPlay: (job: Job) => void;
}

export const MusicVideosView: React.FC<MusicVideosViewProps> = ({ songs, onPlay }) => {
    const completedSongs = songs.filter(s => s.status === 'completed' && s.audio_path);
    const [selectedSongId, setSelectedSongId] = useState<string | null>(completedSongs[0]?.id || null);
    const [videoStyle, setVideoStyle] = useState<'neon-cyberpunk' | 'anime-cinematic' | 'retro-vhs' | 'minimal-lyrics'>('neon-cyberpunk');
    const [isGeneratingStory, setIsGeneratingStory] = useState(false);
    const [isRenderingVideo, setIsRenderingVideo] = useState(false);
    const [renderedVideoUrl, setRenderedVideoUrl] = useState<string | null>(null);
    const [storyboardScenes, setStoryboardScenes] = useState<StoryboardScene[]>([
        { time: '0:00 - 0:15', prompt: 'Neon cityscape reflections across rain-soaked streets with cyan backlighting', camera: 'Slow drone zoom forward', lighting: 'Cyan edge luminescence' },
        { time: '0:15 - 0:45', prompt: 'Silhouetted singer at the edge of a cybernetic rooftop under holographic billboard stars', camera: '360 orbit medium shot', lighting: 'Warm amber rim flare' },
        { time: '0:45 - 1:15', prompt: 'Fast highway pursuit through glowing neon tunnels with rhythmic audio particle reactive pulses', camera: 'Low angle speed tracking', lighting: 'Pulsing stroboscopic neon' }
    ]);

    const activeSong = completedSongs.find(s => s.id === selectedSongId);

    useEffect(() => {
        if (activeSong?.video_path) {
            setRenderedVideoUrl(activeSong.video_path);
        } else {
            setRenderedVideoUrl(null);
        }
    }, [activeSong]);

    const handleGenerateStoryboard = async () => {
        if (!activeSong) return;
        try {
            setIsGeneratingStory(true);
            const scenes = await videoApi.generateStoryboard(activeSong.id, videoStyle);
            if (scenes && scenes.length > 0) {
                setStoryboardScenes(scenes);
            }
        } catch (err) {
            console.error('Failed to generate storyboard:', err);
        } finally {
            setIsGeneratingStory(false);
        }
    };

    const handleRenderVideo = async () => {
        if (!activeSong) return;
        try {
            setIsRenderingVideo(true);
            const res = await videoApi.renderVideo(activeSong.id, videoStyle, '720p');
            if (res.video_url) {
                setRenderedVideoUrl(res.video_url);
            }
        } catch (err) {
            console.error('Failed to render music video:', err);
        } finally {
            setIsRenderingVideo(false);
        }
    };

    return (
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6 flex flex-col justify-between min-h-full">
            <div className="space-y-6">
                {/* Header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div>
                        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white flex items-center gap-3">
                            <span className="p-2 rounded-2xl bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border border-cyan-500/20">
                                🎬
                            </span>
                            <span>AI Music Video Studio</span>
                            <span className="px-2.5 py-1 text-[11px] font-mono font-bold rounded-full bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20 flex items-center gap-1">
                                <Sparkles size={12} />
                                Production Ready
                            </span>
                        </h1>
                        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
                            Generate LLM cinematic storyboards and render real audio-reactive music videos with FFmpeg
                        </p>
                    </div>

                    {renderedVideoUrl && (
                        <a
                            href={renderedVideoUrl}
                            download={`${activeSong?.title || 'track'}_music_video.mp4`}
                            className="px-4 py-2 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 shadow-md shadow-teal-500/20 active:scale-95 transition-all self-start sm:self-auto"
                        >
                            <Download size={14} />
                            <span>Download MP4 Video</span>
                        </a>
                    )}
                </div>

                {/* Studio Workspace */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left Panel: Track Picker & Visual Styles */}
                    <div className="space-y-4">
                        <GlassCard className="p-4 space-y-3">
                            <label className="text-xs font-bold uppercase tracking-wider text-slate-400 block">
                                Select Studio Track
                            </label>
                            <select
                                value={selectedSongId || ''}
                                onChange={(e) => setSelectedSongId(e.target.value)}
                                className="w-full apple-input text-xs font-mono"
                            >
                                {completedSongs.map(s => (
                                    <option key={s.id} value={s.id}>
                                        {s.title || s.prompt.slice(0, 30)}
                                    </option>
                                ))}
                            </select>
                        </GlassCard>

                        <GlassCard className="p-4 space-y-3">
                            <label className="text-xs font-bold uppercase tracking-wider text-slate-400 block">
                                Visual Aesthetic Preset
                            </label>
                            <div className="space-y-2">
                                {[
                                    { id: 'neon-cyberpunk', name: 'Cyberpunk Neon', desc: 'Holographic glows, dark alleys, cyan & teal visualizers' },
                                    { id: 'anime-cinematic', name: 'Anime Cinematic', desc: 'Makoto Shinkai twilight gradients & amber flares' },
                                    { id: 'retro-vhs', name: '80s Retro VHS', desc: 'Purple scanlines, warm tape grain, retro synth vibes' },
                                    { id: 'minimal-lyrics', name: 'Minimal Audio Canvas', desc: 'Deep obsidian backdrop with high-precision reactive waves' }
                                ].map(style => (
                                    <button
                                        key={style.id}
                                        onClick={() => setVideoStyle(style.id as any)}
                                        className={`w-full text-left p-2.5 rounded-xl border text-xs transition-all ${
                                            videoStyle === style.id
                                                ? 'bg-teal-500/10 border-teal-500/30 text-teal-700 dark:text-teal-300 font-bold'
                                                : 'bg-black/[0.02] dark:bg-white/[0.02] border-transparent text-slate-600 dark:text-slate-400 hover:bg-black/[0.04] dark:hover:bg-white/5'
                                        }`}
                                    >
                                        <div>{style.name}</div>
                                        <div className="text-[10px] text-slate-400 font-normal">{style.desc}</div>
                                    </button>
                                ))}
                            </div>
                        </GlassCard>
                    </div>

                    {/* Right Panel: Video Preview & Storyboard Prompt Sequence */}
                    <div className="lg:col-span-2 space-y-4">
                        <GlassCard className="p-6 space-y-6">
                            {/* Video Canvas / Player */}
                            <div className="relative aspect-video rounded-2xl bg-gradient-to-br from-slate-900 via-slate-800 to-slate-950 border border-white/10 flex flex-col items-center justify-center p-6 text-center overflow-hidden shadow-apple-lg group">
                                {renderedVideoUrl ? (
                                    <video
                                        src={renderedVideoUrl}
                                        controls
                                        autoPlay
                                        className="w-full h-full object-cover rounded-xl"
                                    />
                                ) : (
                                    <>
                                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(20,184,166,0.15),transparent_70%)] pointer-events-none" />
                                        <Film size={48} className={`text-teal-400 mb-3 ${isRenderingVideo ? 'animate-bounce' : 'animate-pulse'}`} />
                                        <h3 className="text-lg font-bold text-white">
                                            {isRenderingVideo ? 'Rendering Video with FFmpeg…' : (activeSong?.title || "AI Music Video Studio")}
                                        </h3>
                                        <p className="text-xs text-slate-400 max-w-md mt-1">
                                            {isRenderingVideo
                                                ? 'Synthesizing audio-reactive waveform visualizer and high-definition video container…'
                                                : activeSong
                                                ? `Synchronized to: ${activeSong.prompt.slice(0, 60)}...`
                                                : 'Select a track to render storyboard.'}
                                        </p>

                                        <div className="mt-4 flex items-center gap-2 flex-wrap justify-center">
                                            <button
                                                onClick={() => activeSong && onPlay(activeSong)}
                                                className="px-4 py-2 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 shadow-md transition-all active:scale-95"
                                            >
                                                <Play size={13} className="ml-0.5" />
                                                <span>Preview Audio</span>
                                            </button>
                                            <button
                                                onClick={handleGenerateStoryboard}
                                                disabled={isGeneratingStory || isRenderingVideo}
                                                className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white font-bold text-xs rounded-xl flex items-center space-x-1.5 backdrop-blur-md transition-all disabled:opacity-50"
                                            >
                                                {isGeneratingStory ? <Loader2 size={13} className="animate-spin" /> : <Wand2 size={13} />}
                                                <span>{isGeneratingStory ? 'Directing Story…' : 'AI Storyboard'}</span>
                                            </button>
                                            <button
                                                onClick={handleRenderVideo}
                                                disabled={isRenderingVideo || !activeSong}
                                                className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 shadow-md transition-all disabled:opacity-50"
                                            >
                                                {isRenderingVideo ? <Loader2 size={13} className="animate-spin" /> : <Video size={13} />}
                                                <span>{isRenderingVideo ? 'Rendering MP4…' : 'Render MP4 Video'}</span>
                                            </button>
                                        </div>
                                    </>
                                )}
                            </div>

                            {/* Scene Sequence Storyboard */}
                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                                        Cinematic Storyboard Timeline ({videoStyle})
                                    </h4>
                                    <span className="text-[10px] font-mono text-teal-600 dark:text-teal-400 font-bold flex items-center gap-1">
                                        <CheckCircle2 size={12} />
                                        {storyboardScenes.length} Scenes Directed
                                    </span>
                                </div>

                                <div className="space-y-2">
                                    {storyboardScenes.map((scene, idx) => (
                                        <div
                                            key={idx}
                                            className="p-3 bg-black/[0.02] dark:bg-white/[0.02] rounded-xl border border-black/[0.04] dark:border-white/5 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs"
                                        >
                                            <div className="flex items-center space-x-3">
                                                <span className="font-mono text-[11px] font-bold text-teal-600 dark:text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded-md flex-shrink-0">
                                                    {scene.time}
                                                </span>
                                                <span className="text-slate-700 dark:text-slate-300 font-medium">
                                                    {scene.prompt}
                                                </span>
                                            </div>
                                            <span className="text-[10px] font-mono text-slate-400 sm:text-right flex-shrink-0">
                                                🎥 {scene.camera}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </GlassCard>
                    </div>
                </div>
            </div>

            {/* Global Creator Footer */}
            <AppFooter />
        </div>
    );
};

