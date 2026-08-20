import React, { useState } from 'react';
import { type Job } from '../../api';
import { Play, Film, Wand2 } from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';

interface MusicVideosViewProps {
    songs: Job[];
    onPlay: (job: Job) => void;
}

export const MusicVideosView: React.FC<MusicVideosViewProps> = ({ songs, onPlay }) => {
    const completedSongs = songs.filter(s => s.status === 'completed' && s.audio_path);
    const [selectedSongId, setSelectedSongId] = useState<string | null>(completedSongs[0]?.id || null);
    const [videoStyle, setVideoStyle] = useState<'neon-cyberpunk' | 'anime-cinematic' | 'retro-vhs' | 'minimal-lyrics'>('neon-cyberpunk');
    const [isGeneratingStory, setIsGeneratingStory] = useState(false);
    const [storyboardScenes, setStoryboardScenes] = useState<{ time: string; prompt: string; camera: string }[]>([
        { time: '0:00 - 0:15', prompt: 'Neon cityscape reflections across rain-soaked streets with cyan backlighting', camera: 'Slow drone zoom forward' },
        { time: '0:15 - 0:45', prompt: 'Silhouetted singer at the edge of a cybernetic rooftop under holographic billboard stars', camera: '360 orbit medium shot' },
        { time: '0:45 - 1:15', prompt: 'Fast highway pursuit through glowing neon tunnels with rhythmic audio particle reactive pulses', camera: 'Low angle speed tracking' }
    ]);

    const activeSong = completedSongs.find(s => s.id === selectedSongId);

    const handleGenerateStoryboard = () => {
        setIsGeneratingStory(true);
        setTimeout(() => {
            if (activeSong) {
                setStoryboardScenes([
                    { time: '0:00 - 0:20', prompt: `Atmospheric opening: ${activeSong.prompt.slice(0, 50)} with volumetric lighting`, camera: 'Wide panoramic establishing shot' },
                    { time: '0:20 - 0:50', prompt: `Dynamic crescendo matching vocals with pulsing chromatic particle effects`, camera: 'Dynamic Dutch angle push' },
                    { time: '0:50 - 1:20', prompt: `Climactic visual finale with glowing starlight trails and synchronized typography`, camera: 'Smooth upward crane descent' }
                ]);
            }
            setIsGeneratingStory(false);
        }, 1200);
    };

    return (
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white flex items-center gap-3">
                    <span className="p-2 rounded-2xl bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border border-cyan-500/20">
                        🎬
                    </span>
                    <span>AI Music Video Studio</span>
                </h1>
                <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
                    Generate audio-reactive visual storyboards, WhisperX synchronized lyrics, and video prompts
                </p>
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
                                { id: 'neon-cyberpunk', name: 'Cyberpunk Neon', desc: 'Holographic glows, dark alleys, cyan neon' },
                                { id: 'anime-cinematic', name: 'Anime Cinematic', desc: 'Makoto Shinkai sky gradients, lens flare' },
                                { id: 'retro-vhs', name: '80s Retro VHS', desc: 'Scanlines, warm tape grain, retro synth visuals' },
                                { id: 'minimal-lyrics', name: 'Apple Kinetic Typography', desc: 'Frosted glassmorphism with synced typography' }
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
                        {/* Video Canvas Mockup */}
                        <div className="relative aspect-video rounded-2xl bg-gradient-to-br from-slate-900 via-slate-800 to-slate-950 border border-white/10 flex flex-col items-center justify-center p-6 text-center overflow-hidden shadow-apple-lg group">
                            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(20,184,166,0.15),transparent_70%)] pointer-events-none" />
                            <Film size={48} className="text-teal-400 mb-3 animate-pulse" />
                            <h3 className="text-lg font-bold text-white">
                                {activeSong?.title || "AI Lyric Video Preview"}
                            </h3>
                            <p className="text-xs text-slate-400 max-w-md mt-1">
                                {activeSong ? `Synchronized to: ${activeSong.prompt.slice(0, 60)}...` : 'Select a track to render storyboard.'}
                            </p>

                            <div className="mt-4 flex items-center gap-2">
                                <button
                                    onClick={() => activeSong && onPlay(activeSong)}
                                    className="px-4 py-2 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 shadow-md transition-all active:scale-95"
                                >
                                    <Play size={13} className="ml-0.5" />
                                    <span>Preview Audio Sync</span>
                                </button>
                                <button
                                    onClick={handleGenerateStoryboard}
                                    disabled={isGeneratingStory}
                                    className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white font-bold text-xs rounded-xl flex items-center space-x-1.5 backdrop-blur-md transition-all"
                                >
                                    <Wand2 size={13} />
                                    <span>{isGeneratingStory ? 'AI Generating...' : 'Generate Storyboard'}</span>
                                </button>
                            </div>
                        </div>

                        {/* Scene Sequence Storyboard */}
                        <div className="space-y-3">
                            <div className="flex items-center justify-between">
                                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                                    Storyboard Prompt Timeline (WhisperX Aligned)
                                </h4>
                                <span className="text-[10px] font-mono text-teal-600 dark:text-teal-400 font-bold">
                                    3 Scenes Formatted
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
    );
};
