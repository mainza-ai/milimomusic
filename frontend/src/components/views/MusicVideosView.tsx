import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
    type Job,
    videoApi,
    api,
    type StoryboardScene,
    type VideoPlanResult,
    type VideoTaskStatus,
    type VideoPlanParams,
    type VideoRenderParams
} from '../../api';
import {
    Play,
    Pause,
    Film,
    Wand2,
    Download,
    Video,
    Sparkles,
    Loader2,
    CheckCircle2,
    Mic,
    Type,
    Layers,
    AlertCircle,
    RefreshCw,
    Trash2
} from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';
import { toast } from '../../utils/toast';
import { AppFooter } from '../ui/AppFooter';

interface MusicVideosViewProps {
    songs: Job[];
    onPlay: (job: Job) => void;
    initialSelectedSongId?: string | null;
    isPlaying?: boolean;
    playingSongId?: string | null;
    onUpdateSong?: (job: Job) => void;
}

export type VideoModelKey = 'hailuo_h3' | 'hunyuan' | 'cogvideox' | 'wan2.1' | 'audioreactive';

export const MODEL_CONSTRAINTS: Record<VideoModelKey, { label: string; minSec: number; maxSec: number; defaultSec: number; desc: string }> = {
    'hailuo_h3': { label: 'MiniMax Hailuo H3', minSec: 5.0, maxSec: 15.0, defaultSec: 15.0, desc: 'MiniMax Hailuo H3 flagship DiT — up to 15.0s maximum duration per generation' },
    'hunyuan': { label: 'Tencent HunyuanVideo', minSec: 4.0, maxSec: 15.0, defaultSec: 15.0, desc: 'Tencent HunyuanVideo 13B DiT — up to 15.0s extended visual takes' },
    'cogvideox': { label: 'CogVideoX 1.5 Engine', minSec: 3.0, maxSec: 10.0, defaultSec: 10.0, desc: 'THUDM CogVideoX 1.5 — up to 10.0s clip generation (161 frames at 16fps)' },
    'wan2.1': { label: 'Wan 2.1 Engine', minSec: 2.0, maxSec: 5.0, defaultSec: 5.0, desc: 'Alibaba Wan 2.1 — 5.0s standard limit (81 frames at 16fps) with musical cuts' },
    'audioreactive': { label: 'Audio-Reactive Full', minSec: 5.0, maxSec: 120.0, defaultSec: 120.0, desc: 'Continuous full-timeline audio reactive spectrum & waveform visualizer' },
};

export const isValidVideoEngine = (e: string): e is VideoModelKey => e in MODEL_CONSTRAINTS;

/** Window event dispatched when Models & HW activation changes (see ModelsManagerModal). */
export const VIDEO_ACTIVE_MODEL_EVENT = 'milimo:model-activated';

export const MusicVideosView: React.FC<MusicVideosViewProps> = ({
    songs,
    onPlay,
    initialSelectedSongId,
    isPlaying = false,
    playingSongId = null,
    onUpdateSong
}) => {
    const completedSongs = useMemo<Job[]>(() => songs.filter((s: Job) => s.status === 'completed' && Boolean(s.audio_path)), [songs]);
    const [selectedSongId, setSelectedSongId] = useState<string | null>(() => {
        if (initialSelectedSongId) return initialSelectedSongId;
        const saved = localStorage.getItem('milimo_selected_video_song_id');
        if (saved && completedSongs.some(s => s.id === saved)) return saved;
        const withVideo = completedSongs.find(s => !!s.video_path);
        return withVideo?.id || completedSongs[0]?.id || null;
    });

    const handleSelectSong = (id: string) => {
        setSelectedSongId(id);
        localStorage.setItem('milimo_selected_video_song_id', id);
    };

    useEffect(() => {
        if (!selectedSongId && completedSongs.length > 0) {
            const saved = localStorage.getItem('milimo_selected_video_song_id');
            const target = completedSongs.find(s => s.id === saved) || completedSongs.find(s => !!s.video_path) || completedSongs[0];
            if (target) setSelectedSongId(target.id);
        }
    }, [completedSongs, selectedSongId]);

    // Style & Model Engine settings
    const [videoModel, setVideoModel] = useState<VideoModelKey>('hailuo_h3');
    // Engine the user marked active in Models & HW — the page follows it by
    // default and on activation changes; manual override is still allowed.
    const [activeVideoEngine, setActiveVideoEngine] = useState<VideoModelKey | null>(null);
    const [modelRegistry, setModelRegistry] = useState<Record<string, any>>({});
    const [clipDuration, setClipDuration] = useState<number>(() => {
        const saved = localStorage.getItem('milimo_video_clip_len_hailuo_h3');
        return saved ? Math.min(15.0, Math.max(5.0, parseFloat(saved))) : 15.0;
    });

    const selectEngine = useCallback((model: VideoModelKey) => {
        setVideoModel(model);
        const conf = MODEL_CONSTRAINTS[model];
        const saved = localStorage.getItem(`milimo_video_clip_len_${model}`);
        const resolved = saved ? Math.min(conf.maxSec, Math.max(conf.minSec, parseFloat(saved))) : conf.defaultSec;
        setClipDuration(resolved);
    }, []);

    useEffect(() => {
        let alive = true;
        videoApi.getVideoModels().then((reg) => { if (alive) setModelRegistry(reg); }).catch(() => {});

        const applyActiveEngine = () => {
            videoApi.getActiveVideoEngine()
                .then((r) => {
                    if (!alive) return;
                    if (r.engine && isValidVideoEngine(r.engine)) {
                        setActiveVideoEngine(r.engine);
                        // Follow the Models & HW active engine (manual override is
                        // transient until the next activation change).
                        selectEngine(r.engine);
                    }
                })
                .catch(() => {});
        };
        applyActiveEngine();
        window.addEventListener(VIDEO_ACTIVE_MODEL_EVENT, applyActiveEngine);
        return () => {
            alive = false;
            window.removeEventListener(VIDEO_ACTIVE_MODEL_EVENT, applyActiveEngine);
        };
    }, [selectEngine]);

    const [videoStyle, setVideoStyle] = useState<'neon-cyberpunk' | 'anime-cinematic' | 'retro-vhs' | 'minimal-lyrics'>('neon-cyberpunk');
    const [resolution, setResolution] = useState<'720p' | '1080p'>('720p');
    const [isDeletingVideo, setIsDeletingVideo] = useState(false);

    const handleSelectModel = (model: VideoModelKey) => {
        selectEngine(model);
    };

    const handleClipDurationChange = (val: number) => {
        const conf = MODEL_CONSTRAINTS[videoModel];
        const clamped = Math.min(conf.maxSec, Math.max(conf.minSec, val));
        setClipDuration(clamped);
        localStorage.setItem(`milimo_video_clip_len_${videoModel}`, clamped.toString());
    };

    const handleResetDurationToMax = () => {
        const conf = MODEL_CONSTRAINTS[videoModel];
        setClipDuration(conf.maxSec);
        localStorage.setItem(`milimo_video_clip_len_${videoModel}`, conf.maxSec.toString());
    };

    useEffect(() => {
        if (initialSelectedSongId) {
            setSelectedSongId(initialSelectedSongId);
        }
    }, [initialSelectedSongId]);

    // Advanced Lip Sync & Lyric Options
    const [enableLipSync, setEnableLipSync] = useState(true);
    const [burnSubtitles, setBurnSubtitles] = useState(true);
    const [subtitleStyle, setSubtitleStyle] = useState<'neon' | 'cinematic' | 'karaoke'>('neon');

    // Planning & Task Tracking
    const [isPlanning, setIsPlanning] = useState(false);
    const [planResult, setPlanResult] = useState<VideoPlanResult | null>(null);

    const [activeTask, setActiveTask] = useState<VideoTaskStatus | null>(null);
    const [isRendering, setIsRendering] = useState(false);
    const [renderedVideoUrl, setRenderedVideoUrl] = useState<string | null>(null);
    const pollRef = useRef<number | undefined>(undefined);

    // Unmount cleanup to prevent leaking video polling interval
    useEffect(() => {
        return () => {
            if (pollRef.current) {
                window.clearInterval(pollRef.current);
                pollRef.current = undefined;
            }
        };
    }, []);

    // Fallback legacy storyboard scenes
    const [isGeneratingStory, setIsGeneratingStory] = useState(false);
    const [storyboardScenes, setStoryboardScenes] = useState<StoryboardScene[]>([
        { time: '0:00 - 0:15', prompt: 'Neon cityscape reflections across rain-soaked streets with cyan backlighting', camera: 'Slow drone zoom forward', lighting: 'Cyan edge luminescence' },
        { time: '0:15 - 0:45', prompt: 'Silhouetted singer at the edge of a cybernetic rooftop under holographic billboard stars', camera: '360 orbit medium shot', lighting: 'Warm amber rim flare' },
        { time: '0:45 - 1:15', prompt: 'Fast highway pursuit through glowing neon tunnels with rhythmic audio particle reactive pulses', camera: 'Low angle speed tracking', lighting: 'Pulsing stroboscopic neon' }
    ]);

    const activeSong = completedSongs.find(s => s.id === selectedSongId);

    // Check if song has isolated stems
    let hasVocals = false;
    if (activeSong?.stems_json) {
        try {
            const parsed = typeof activeSong.stems_json === 'string' ? JSON.parse(activeSong.stems_json) : activeSong.stems_json;
            if (parsed && (parsed.vocals || parsed.vocals_path)) {
                hasVocals = true;
            }
        } catch { /* ignore parse error */ }
    }

    useEffect(() => {
        if (!selectedSongId) {
            setRenderedVideoUrl(null);
            return;
        }
        if (activeSong?.video_path) {
            setRenderedVideoUrl(activeSong.video_path);
        } else {
            videoApi.getVideo(selectedSongId).then(res => {
                if (res?.has_video && res.video_path) {
                    setRenderedVideoUrl(res.video_path);
                    if (activeSong && onUpdateSong) {
                        onUpdateSong({ ...activeSong, video_path: res.video_path });
                    }
                } else {
                    setRenderedVideoUrl(null);
                }
            }).catch(() => setRenderedVideoUrl(null));
        }
        setPlanResult(null);
        setActiveTask(null);
    }, [selectedSongId, activeSong?.video_path]);

    useEffect(() => {
        return () => {
            if (pollRef.current) window.clearInterval(pollRef.current);
        };
    }, []);

    // Plan Scenes Breakdown
    const handlePlanScenes = async () => {
        if (!activeSong) return;
        try {
            setIsPlanning(true);
            const params: VideoPlanParams = {
                model_name: videoModel,
                max_clip_duration: clipDuration,
                bpm: 120,
                visual_style: videoStyle
            };
            const plan = await videoApi.planVideo(activeSong.id, params);
            setPlanResult(plan);
            toast(`Scene plan created: ${plan.total_clips} scenes ready for production.`, 'success');
        } catch (err: any) {
            console.error('Failed to plan video scenes:', err);
            toast(err?.response?.data?.detail || 'Failed to plan video scenes. Please ensure the track is completed.', 'error');
        } finally {
            setIsPlanning(false);
        }
    };

    // Render Advanced Production Video
    const handleRenderAdvancedVideo = async () => {
        if (!activeSong) return;
        try {
            setIsRendering(true);
            const params: VideoRenderParams = {
                model_name: videoModel,
                visual_style: videoStyle,
                resolution,
                enable_lip_sync: enableLipSync,
                burn_lyrics: burnSubtitles,
                subtitle_style: subtitleStyle,
                max_clip_duration: clipDuration,
                mode: 'production_multiclip'
            };

            const taskInit = await videoApi.renderAdvancedVideo(activeSong.id, params);
            setActiveTask({
                id: taskInit.task_id,
                job_id: activeSong.id,
                status: 'processing',
                step: 'planning',
                progress: 5,
                total_clips: planResult?.total_clips || 1,
                current_clip: 0
            });

            // Poll task status
            if (pollRef.current) window.clearInterval(pollRef.current);
            pollRef.current = window.setInterval(async () => {
                try {
                    const status = await videoApi.getVideoTaskStatus(taskInit.task_id);
                    setActiveTask(status);

                    if (status.status === 'completed') {
                        window.clearInterval(pollRef.current);
                        setIsRendering(false);
                        if (status.video_url) {
                            setRenderedVideoUrl(status.video_url);
                            if (activeSong && onUpdateSong) {
                                onUpdateSong({ ...activeSong, video_path: status.video_url });
                            }
                        }
                    } else if (status.status === 'error') {
                        window.clearInterval(pollRef.current);
                        setIsRendering(false);
                    }
                } catch { /* transient error */ }
            }, 1000);
        } catch (err) {
            console.error('Failed to start video rendering:', err);
            toast('Failed to start video rendering. Please check backend service.', 'error');
            setIsRendering(false);
        }
    };

    // Apply the render config the existing video was generated with so a
    // regeneration is faithful (same engine, style, resolution, lip-sync,
    // subtitles, clip length) instead of whatever the page currently shows.
    const applyStoredVideoConfig = (job: Job) => {
        if (!job?.video_config_json) return;
        try {
            const cfg = JSON.parse(job.video_config_json) as Record<string, any>;
            if (cfg.model_name && isValidVideoEngine(cfg.model_name)) {
                selectEngine(cfg.model_name);
            }
            if (cfg.max_clip_duration) {
                const engine = (cfg.model_name && isValidVideoEngine(cfg.model_name)) ? cfg.model_name : videoModel;
                const conf = MODEL_CONSTRAINTS[engine];
                const clamped = Math.min(conf.maxSec, Math.max(conf.minSec, parseFloat(cfg.max_clip_duration)));
                setClipDuration(clamped);
            }
            if (cfg.visual_style && ['neon-cyberpunk', 'anime-cinematic', 'retro-vhs', 'minimal-lyrics'].includes(cfg.visual_style)) {
                setVideoStyle(cfg.visual_style);
            }
            if (cfg.resolution === '1080p' || cfg.resolution === '720p') setResolution(cfg.resolution);
            if (typeof cfg.enable_lip_sync === 'boolean') setEnableLipSync(cfg.enable_lip_sync);
            if (typeof cfg.burn_lyrics === 'boolean') setBurnSubtitles(cfg.burn_lyrics);
            if (cfg.subtitle_style && ['neon', 'cinematic', 'karaoke'].includes(cfg.subtitle_style)) setSubtitleStyle(cfg.subtitle_style);
        } catch { /* unreadable config — fall back to current page settings */ }
    };

    const handleRegenerateVideo = async () => {
        if (!activeSong) return;
        applyStoredVideoConfig(activeSong);
        await handleRenderAdvancedVideo();
    };

    const handleDeleteVideo = async () => {
        if (!activeSong) return;
        if (!window.confirm(`Delete the rendered video for "${activeSong.title || activeSong.prompt.slice(0, 40)}"? The track and audio stay untouched.`)) return;
        setIsDeletingVideo(true);
        try {
            await videoApi.deleteVideo(activeSong.id);
            setRenderedVideoUrl(null);
            onUpdateSong?.({ ...activeSong, video_path: undefined as any, video_config_json: undefined as any });
            toast('Video deleted.', 'success');
        } catch (e: any) {
            toast(e?.response?.data?.detail || 'Failed to delete video.', 'error');
        } finally {
            setIsDeletingVideo(false);
        }
    };

    // Generate legacy storyboard
    const handleGenerateStoryboard = async () => {
        if (!activeSong) return;
        try {
            setIsGeneratingStory(true);
            const scenes = await videoApi.generateStoryboard(activeSong.id, videoStyle);
            if (scenes && scenes.length > 0) {
                setStoryboardScenes(scenes);
                toast('Storyboard sequence generated with dynamic directing prompts.', 'success');
            }
        } catch (err: any) {
            console.error('Failed to generate storyboard:', err);
            toast(err?.response?.data?.detail || 'Failed to generate storyboard scenes.', 'error');
        } finally {
            setIsGeneratingStory(false);
        }
    };

    return (
        <div className="flex-1 overflow-y-auto p-6 md:p-8 pb-28 sm:pb-32 space-y-6 flex flex-col justify-between min-h-full">
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
                                Production Multi-Scene Pipeline
                            </span>
                        </h1>
                        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
                            Model duration constraint handling (H3 15s, Hunyuan 15s, CogVideoX 10s, Wan 5s), isolated vocal stem lip-syncing & burned subtitles
                        </p>
                    </div>

                    {renderedVideoUrl && (
                        <a
                            href={api.getAudioUrl(renderedVideoUrl)}
                            download={`${activeSong?.title || 'track'}_music_video.mp4`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-4 py-2 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 shadow-md shadow-teal-500/20 active:scale-95 transition-all self-start sm:self-auto"
                        >
                            <Download size={14} />
                            <span>Download MP4 Video</span>
                        </a>
                    )}
                </div>

                {/* Studio Workspace Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left Panel: Controls, Engine Selector & Pipeline Settings */}
                    <div className="space-y-4">
                        {/* Track Picker */}
                        <GlassCard className="p-4 space-y-3">
                            <label className="text-xs font-bold uppercase tracking-wider text-slate-400 block">
                                Select Track for Music Video
                            </label>
                            <select
                                value={selectedSongId || ''}
                                onChange={(e) => handleSelectSong(e.target.value)}
                                className="w-full apple-input text-xs font-mono"
                            >
                                {completedSongs.map(s => (
                                    <option key={s.id} value={s.id}>
                                        {s.video_path ? '🎬 ' : '🎵 '}{s.title || s.prompt.slice(0, 30)}{s.video_path ? ' · [Video Ready]' : ''}
                                    </option>
                                ))}
                            </select>

                            {activeSong && (
                                <div className="flex items-center gap-2 flex-wrap pt-1 text-[11px]">
                                    <span className="font-mono px-2 py-0.5 rounded-md bg-black/5 dark:bg-white/5 text-slate-600 dark:text-slate-300">
                                        ⏱️ {activeSong.duration_ms ? `${Math.round(activeSong.duration_ms / 1000)}s` : 'Unknown'}
                                    </span>
                                    <span className={`px-2 py-0.5 rounded-md font-semibold flex items-center gap-1 ${
                                        hasVocals
                                            ? 'bg-teal-500/10 text-teal-600 dark:text-teal-400'
                                            : 'bg-amber-500/10 text-amber-600 dark:text-amber-400'
                                    }`}>
                                        <Mic size={11} />
                                        {hasVocals ? 'Vocals Isolated' : 'Full Audio'}
                                    </span>
                                    <span className={`px-2 py-0.5 rounded-md font-semibold flex items-center gap-1 ${
                                        activeSong.lyrics
                                            ? 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400'
                                            : 'bg-slate-500/10 text-slate-500'
                                    }`}>
                                        <Type size={11} />
                                        {activeSong.lyrics ? 'Lyrics Ready' : 'Instrumental'}
                                    </span>
                                </div>
                            )}
                        </GlassCard>

                        {/* Model Duration Constraint & Custom Length Setting */}
                        <GlassCard className="p-4 space-y-3">
                            <div className="flex items-center justify-between">
                                <label className="text-xs font-bold uppercase tracking-wider text-slate-400 block">
                                    Video Engine & Clip Duration
                                </label>
                                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-600 dark:text-teal-400 font-bold">
                                    Default = Max Length
                                </span>
                            </div>

                            <div className="space-y-2">
                                {(Object.keys(MODEL_CONSTRAINTS) as Array<keyof typeof MODEL_CONSTRAINTS>).map(key => {
                                    const conf = MODEL_CONSTRAINTS[key];
                                    return (
                                        <button
                                            key={key}
                                            onClick={() => handleSelectModel(key as any)}
                                            className={`w-full text-left p-2.5 rounded-xl border text-xs transition-all ${
                                                videoModel === key
                                                    ? 'bg-teal-500/10 border-teal-500/30 text-teal-700 dark:text-teal-300 font-bold shadow-sm'
                                                    : 'bg-black/[0.02] dark:bg-white/[0.02] border-transparent text-slate-600 dark:text-slate-400 hover:bg-black/[0.04] dark:hover:bg-white/5'
                                            }`}
                                        >
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center gap-1.5">
                                                    <span>{conf.label}</span>
                                                    {activeVideoEngine === key && (
                                                        <span className="text-[9px] font-mono px-1 py-0.5 rounded bg-teal-500/15 text-teal-700 dark:text-teal-300 font-bold border border-teal-500/20">
                                                            ● Active in Models
                                                        </span>
                                                    )}
                                                    {modelRegistry[key]?.local_weights_present && (
                                                        <span className="text-[9px] font-mono px-1 py-0.5 rounded bg-amber-500/10 text-amber-600 dark:text-amber-400 font-bold">
                                                            ⚡ Local Ready
                                                        </span>
                                                    )}
                                                </div>
                                                <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-teal-500/10 text-teal-600 dark:text-teal-400">
                                                    Max: {conf.maxSec}s
                                                </span>
                                            </div>
                                            <div className="text-[10px] text-slate-400 font-normal mt-0.5">{conf.desc}</div>
                                        </button>
                                    );
                                })}
                            </div>

                            {/* Clip Length Adjustment Slider & Stepper */}
                            <div className="pt-2 border-t border-black/[0.06] dark:border-white/5 space-y-2">
                                <div className="flex items-center justify-between text-xs">
                                    <span className="font-semibold text-slate-700 dark:text-slate-300">
                                        Clip Duration Setting
                                    </span>
                                    <div className="flex items-center gap-1.5">
                                        <span className="font-mono font-bold text-teal-600 dark:text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded-md text-xs">
                                            {clipDuration.toFixed(1)}s
                                        </span>
                                        {clipDuration !== MODEL_CONSTRAINTS[videoModel].maxSec && (
                                            <button
                                                onClick={handleResetDurationToMax}
                                                className="text-[10px] text-teal-600 dark:text-teal-400 hover:underline font-mono"
                                                title="Reset to model max length"
                                            >
                                                Reset Max ({MODEL_CONSTRAINTS[videoModel].maxSec}s)
                                            </button>
                                        )}
                                    </div>
                                </div>

                                <div className="flex items-center gap-3">
                                    <span className="text-[10px] font-mono text-slate-400">
                                        {MODEL_CONSTRAINTS[videoModel].minSec}s
                                    </span>
                                    <input
                                        type="range"
                                        min={MODEL_CONSTRAINTS[videoModel].minSec}
                                        max={MODEL_CONSTRAINTS[videoModel].maxSec}
                                        step={0.5}
                                        value={clipDuration}
                                        onChange={(e) => handleClipDurationChange(parseFloat(e.target.value))}
                                        className="flex-1 accent-teal-500 h-1.5 bg-black/[0.06] dark:bg-white/10 rounded-lg cursor-pointer"
                                    />
                                    <span className="text-[10px] font-mono text-slate-400">
                                        {MODEL_CONSTRAINTS[videoModel].maxSec}s
                                    </span>
                                </div>

                                <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400 pt-0.5">
                                    <span>
                                        Pacing: {clipDuration <= 3.5 ? '⚡ Fast cuts' : clipDuration <= 6.0 ? '🎬 Standard cinematic' : '🌊 Extended takes'}
                                    </span>
                                    {activeSong?.duration_ms && (
                                        <span className="font-mono text-teal-600 dark:text-teal-400">
                                            Est. ~{Math.ceil(Math.round(activeSong.duration_ms / 1000) / clipDuration)} scenes
                                        </span>
                                    )}
                                </div>
                            </div>
                        </GlassCard>

                        {/* Lip Syncing & Vocal Stem Alignment */}
                        <GlassCard className="p-4 space-y-3">
                            <div className="flex items-center justify-between">
                                <label className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                                    <Mic size={13} className="text-teal-500" />
                                    <span>Vocal Lip-Syncing</span>
                                </label>
                                <input
                                    type="checkbox"
                                    checked={enableLipSync}
                                    onChange={(e) => setEnableLipSync(e.target.checked)}
                                    className="rounded border-slate-700 text-teal-500 focus:ring-teal-500"
                                />
                            </div>

                            {enableLipSync && (
                                <p className="text-[11px] text-slate-500 dark:text-slate-400 pt-1">
                                    Singing movements are mapped strictly to the isolated vocal stem (<code className="text-teal-400 font-mono">vocals.mp3 / vocals.wav</code>) to prevent mouth distortion from heavy percussion or bass.
                                </p>
                            )}
                        </GlassCard>

                        {/* Synchronized Subtitles & Karaoke Burning */}
                        <GlassCard className="p-4 space-y-3">
                            <div className="flex items-center justify-between">
                                <label className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                                    <Type size={13} className="text-cyan-500" />
                                    <span>Burn Subtitles / Lyrics</span>
                                </label>
                                <input
                                    type="checkbox"
                                    checked={burnSubtitles}
                                    onChange={(e) => setBurnSubtitles(e.target.checked)}
                                    className="rounded border-slate-700 text-cyan-500 focus:ring-cyan-500"
                                />
                            </div>

                            {burnSubtitles && (
                                <div className="flex gap-2 pt-1">
                                    {(['neon', 'cinematic', 'karaoke'] as const).map(style => (
                                        <button
                                            key={style}
                                            onClick={() => setSubtitleStyle(style)}
                                            className={`flex-1 py-1 text-xs rounded-lg border capitalize transition-all ${
                                                subtitleStyle === style
                                                    ? 'bg-cyan-500/10 border-cyan-500/40 text-cyan-600 dark:text-cyan-400 font-bold'
                                                    : 'border-black/5 dark:border-white/5 text-slate-400'
                                            }`}
                                        >
                                            {style}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </GlassCard>

                        {/* Visual Aesthetic Preset & Resolution */}
                        <GlassCard className="p-4 space-y-3">
                            <label className="text-xs font-bold uppercase tracking-wider text-slate-400 block">
                                Visual Aesthetic Preset
                            </label>
                            <div className="grid grid-cols-2 gap-1.5">
                                {[
                                    { id: 'neon-cyberpunk', name: 'Cyberpunk' },
                                    { id: 'anime-cinematic', name: 'Anime' },
                                    { id: 'retro-vhs', name: '80s VHS' },
                                    { id: 'minimal-lyrics', name: 'Minimal' }
                                ].map(style => (
                                    <button
                                        key={style.id}
                                        onClick={() => setVideoStyle(style.id as any)}
                                        className={`py-1.5 px-2 rounded-lg border text-xs text-center transition-all ${
                                            videoStyle === style.id
                                                ? 'bg-teal-500/10 border-teal-500/30 text-teal-700 dark:text-teal-300 font-bold'
                                                : 'bg-black/[0.02] dark:bg-white/[0.02] border-transparent text-slate-600 dark:text-slate-400'
                                        }`}
                                    >
                                        {style.name}
                                    </button>
                                ))}
                            </div>

                            <div className="flex items-center justify-between pt-1">
                                <span className="text-xs text-slate-400 font-bold">Output Quality</span>
                                <div className="flex gap-1.5">
                                    <button
                                        onClick={() => setResolution('720p')}
                                        className={`px-2.5 py-0.5 text-xs rounded-md ${resolution === '720p' ? 'bg-teal-500 text-slate-950 font-bold' : 'bg-black/5 dark:bg-white/5 text-slate-400'}`}
                                    >
                                        720p
                                    </button>
                                    <button
                                        onClick={() => setResolution('1080p')}
                                        className={`px-2.5 py-0.5 text-xs rounded-md ${resolution === '1080p' ? 'bg-teal-500 text-slate-950 font-bold' : 'bg-black/5 dark:bg-white/5 text-slate-400'}`}
                                    >
                                        1080p
                                    </button>
                                </div>
                            </div>
                        </GlassCard>
                    </div>

                    {/* Right Panel: Video Canvas, Live Rendering HUD & Scene Timeline */}
                    <div className="lg:col-span-2 space-y-4">
                        <GlassCard className="p-6 space-y-6">
                            {/* Video Canvas / Player */}
                            <div className="relative aspect-video rounded-2xl bg-gradient-to-br from-slate-900 via-slate-800 to-slate-950 border border-white/10 flex flex-col items-center justify-center p-6 text-center overflow-hidden shadow-apple-lg group">
                                {renderedVideoUrl ? (
                                    <>
                                        <video
                                            src={api.getAudioUrl(renderedVideoUrl)}
                                            controls
                                            autoPlay
                                            className="w-full h-full object-cover rounded-xl"
                                        />
                                        <div className="absolute bottom-3 right-3 flex items-center gap-2">
                                            <button
                                                onClick={handleRegenerateVideo}
                                                disabled={isRendering || isDeletingVideo}
                                                className="px-3 py-1.5 bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-slate-950 font-bold text-[11px] rounded-lg flex items-center gap-1.5 shadow-md transition-all disabled:opacity-50"
                                                title="Re-render with the same configuration this video was generated with"
                                            >
                                                {isRendering ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                                                <span>{isRendering ? 'Rendering…' : 'Regenerate'}</span>
                                            </button>
                                            <button
                                                onClick={handleDeleteVideo}
                                                disabled={isRendering || isDeletingVideo}
                                                className="px-3 py-1.5 bg-rose-500/90 hover:bg-rose-500 text-white font-bold text-[11px] rounded-lg flex items-center gap-1.5 shadow-md transition-all disabled:opacity-50"
                                                title="Delete this rendered video (track and audio stay untouched)"
                                            >
                                                {isDeletingVideo ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
                                                <span>{isDeletingVideo ? 'Deleting…' : 'Delete'}</span>
                                            </button>
                                        </div>
                                    </>
                                ) : (
                                    <>
                                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(20,184,166,0.15),transparent_70%)] pointer-events-none" />
                                        <Film size={48} className={`text-teal-400 mb-3 ${isRendering ? 'animate-bounce' : 'animate-pulse'}`} />
                                        <h3 className="text-lg font-bold text-white">
                                            {isRendering ? 'Rendering Production Music Video…' : (activeSong?.title || "AI Music Video Studio")}
                                        </h3>
                                        <p className="text-xs text-slate-400 max-w-md mt-1">
                                            {isRendering
                                                ? `Executing multi-scene generation and stem alignment pipeline (Step: ${activeTask?.step})…`
                                                : activeSong
                                                ? `Synchronized to: ${activeSong.prompt.slice(0, 60)}...`
                                                : 'Select a track to start music video generation.'}
                                        </p>

                                        <div className="mt-4 flex items-center gap-2 flex-wrap justify-center">
                                            <button
                                                onClick={() => activeSong && onPlay(activeSong)}
                                                className="px-4 py-2 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 shadow-md transition-all active:scale-95"
                                            >
                                                {isPlaying && playingSongId === activeSong?.id ? (
                                                    <>
                                                        <Pause size={13} className="ml-0.5" />
                                                        <span>Pause Audio</span>
                                                    </>
                                                ) : (
                                                    <>
                                                        <Play size={13} className="ml-0.5" />
                                                        <span>Preview Audio</span>
                                                    </>
                                                )}
                                            </button>
                                            <button
                                                onClick={handlePlanScenes}
                                                disabled={isPlanning || isRendering || !activeSong}
                                                className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white font-bold text-xs rounded-xl flex items-center space-x-1.5 backdrop-blur-md transition-all disabled:opacity-50"
                                            >
                                                {isPlanning ? <Loader2 size={13} className="animate-spin" /> : <Layers size={13} />}
                                                <span>{isPlanning ? 'Planning…' : 'Plan Scene Breakdown'}</span>
                                            </button>
                                            <button
                                                onClick={handleRenderAdvancedVideo}
                                                disabled={isRendering || !activeSong}
                                                className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 shadow-md transition-all disabled:opacity-50"
                                            >
                                                {isRendering ? <Loader2 size={13} className="animate-spin" /> : <Video size={13} />}
                                                <span>{isRendering ? 'Rendering Video…' : 'Render Production Video'}</span>
                                            </button>
                                        </div>
                                    </>
                                )}
                            </div>

                            {/* Multi-Stage Live Rendering HUD */}
                            {activeTask && (
                                <div className={`p-4 rounded-2xl border space-y-3 ${
                                    activeTask.status === 'error'
                                        ? 'bg-rose-500/10 border-rose-500/30'
                                        : activeTask.status === 'completed'
                                        ? 'bg-teal-500/10 border-teal-500/30'
                                        : 'bg-black/[0.03] dark:bg-white/5 border-black/[0.06] dark:border-white/10'
                                }`}>
                                    <div className="flex items-center justify-between text-xs">
                                        <span className="font-bold flex items-center gap-2 text-slate-800 dark:text-slate-200">
                                            {activeTask.status === 'error' && <AlertCircle size={14} className="text-rose-500" />}
                                            {activeTask.status === 'completed' && <CheckCircle2 size={14} className="text-teal-500" />}
                                            {activeTask.status === 'processing' && <Loader2 size={14} className="animate-spin text-teal-500" />}
                                            <span>Pipeline Stage: <strong className="uppercase">{activeTask.step.replace(/_/g, ' ')}</strong></span>
                                        </span>
                                        <span className="font-mono text-slate-500">
                                            {activeTask.progress}%
                                        </span>
                                    </div>

                                    {/* Progress Bar */}
                                    <div className="w-full h-1.5 bg-black/[0.06] dark:bg-white/10 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-gradient-to-r from-teal-500 to-cyan-400 rounded-full transition-all duration-500"
                                            style={{ width: `${activeTask.progress}%` }}
                                        />
                                    </div>

                                    <div className="flex items-center justify-between text-[11px] text-slate-500">
                                        <span>Clip {activeTask.current_clip} of {activeTask.total_clips}</span>
                                        {activeTask.error && (
                                            <span className="text-rose-500 font-semibold">{activeTask.error}</span>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Interactive Scene Plan Timeline */}
                            {planResult && (
                                <div className="space-y-3">
                                    <div className="flex items-center justify-between">
                                        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                                            <span>Planned Scene Segments ({planResult.clips?.length || planResult.total_clips} Clips)</span>
                                            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-600 dark:text-teal-400">
                                                Max {planResult.max_clip_duration}s / clip
                                            </span>
                                        </h4>
                                        <span className="text-[10px] text-slate-400 font-mono">
                                            Vocals: {planResult.vocal_clips_count} · B-Roll: {planResult.broll_clips_count}
                                        </span>
                                    </div>

                                    <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
                                        {(planResult.clips || []).map((scene) => (
                                            <div
                                                key={scene.clip_index}
                                                className="p-3 bg-black/[0.02] dark:bg-white/[0.02] rounded-xl border border-black/[0.04] dark:border-white/5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
                                            >
                                                <div className="space-y-1">
                                                    <div className="flex items-center space-x-2">
                                                        <span className="font-mono text-[11px] font-bold text-teal-600 dark:text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded-md flex-shrink-0">
                                                            {scene.time_str} ({scene.duration.toFixed(1)}s)
                                                        </span>
                                                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                                                            scene.scene_type === 'VOCAL_PERFORMANCE'
                                                                ? 'bg-teal-500/20 text-teal-700 dark:text-teal-300'
                                                                : 'bg-purple-500/20 text-purple-700 dark:text-purple-300'
                                                        }`}>
                                                            {scene.scene_type === 'VOCAL_PERFORMANCE' ? '🎤 Vocal Performance (Lip-Sync)' : '🎥 Cinematic B-Roll'}
                                                        </span>
                                                        <span className="text-[10px] font-mono text-slate-400">
                                                            Camera: {scene.camera}
                                                        </span>
                                                    </div>
                                                    <p className="text-slate-700 dark:text-slate-300 font-medium">
                                                        {scene.prompt}
                                                    </p>
                                                    {scene.lyrics && (
                                                        <p className="text-[11px] italic text-cyan-600 dark:text-cyan-400">
                                                            "{scene.lyrics}"
                                                        </p>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Fallback Storyboard Sequence (when no plan is generated yet) */}
                            {!planResult && (
                                <div className="space-y-3">
                                    <div className="flex items-center justify-between">
                                        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                                            Cinematic Storyboard Sequence ({videoStyle})
                                        </h4>
                                        <div className="flex items-center space-x-2">
                                            <button
                                                onClick={handleGenerateStoryboard}
                                                disabled={isGeneratingStory}
                                                className="text-[11px] text-teal-600 dark:text-teal-400 hover:underline flex items-center gap-1"
                                            >
                                                {isGeneratingStory ? <Loader2 size={11} className="animate-spin" /> : <Wand2 size={11} />}
                                                <span>Regenerate Directing Notes</span>
                                            </button>
                                        </div>
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
                            )}
                        </GlassCard>
                    </div>
                </div>
            </div>

            {/* Global Creator Footer */}
            <AppFooter />
        </div>
    );
};
