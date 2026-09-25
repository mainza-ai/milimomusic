import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
    type Job,
    videoApi,
    api,
    galleryApi,
    type VideoPlanResult,
    type VideoTaskStatus,
    type VideoPlanParams,
    type VideoRenderParams
} from '../../api';
import { toast } from '../../utils/toast';
import { AppFooter } from '../ui/AppFooter';

import { VideoTopBar, type AspectRatioType } from '../video/VideoTopBar';
import { VideoCanvasPlayer } from '../video/VideoCanvasPlayer';
import { VideoInspectorDock } from '../video/VideoInspectorDock';
import { VideoTimelineTrack } from '../video/VideoTimelineTrack';
import { ClipRetakeModal } from '../video/ClipRetakeModal';
import { KeyframeZoomModal } from '../video/KeyframeZoomModal';

interface MusicVideosViewProps {
    songs: Job[];
    onPlay: (job: Job) => void;
    initialSelectedSongId?: string | null;
    isPlaying?: boolean;
    playingSongId?: string | null;
    onUpdateSong?: (job: Job) => void;
}

export type VideoModelKey = 'wan_14b' | 'wan_1.3b' | 'ltx_video' | 'cogvideox' | 'hailuo_h3' | 'hunyuan' | 'audioreactive';

export const MODEL_CONSTRAINTS: Record<string, { label: string; minSec: number; maxSec: number; defaultSec: number; desc: string }> = {
    'wan_14b': { label: 'Wan 2.1 14B Flagship', minSec: 2.0, maxSec: 5.0, defaultSec: 5.0, desc: 'Alibaba Wan 2.1 14B DiT with 3D temporal diffusion & keyframe I2V' },
    'wan_1.3b': { label: 'Wan 2.1 1.3B Fast', minSec: 2.0, maxSec: 5.0, defaultSec: 5.0, desc: 'Lightweight text-to-video diffusion for rapid local preview' },
    'ltx_video': { label: 'LTX-Video 0.9B Realtime', minSec: 3.0, maxSec: 10.0, defaultSec: 5.0, desc: 'Lightricks 0.9B real-time DiT (24 fps) for quick scene rendering' },
    'cogvideox': { label: 'CogVideoX 1.5 Engine', minSec: 3.0, maxSec: 10.0, defaultSec: 10.0, desc: 'THUDM CogVideoX 1.5 — 5B 3D causal VAE model' },
    'hailuo_h3': { label: 'MiniMax Hailuo H3', minSec: 5.0, maxSec: 15.0, defaultSec: 15.0, desc: 'MiniMax Hailuo H3 flagship DiT — up to 15.0s maximum duration' },
    'hunyuan': { label: 'Tencent HunyuanVideo', minSec: 4.0, maxSec: 15.0, defaultSec: 15.0, desc: 'Tencent HunyuanVideo 13B DiT — up to 15.0s extended visual takes' },
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

    // Active song instance
    const activeSong = completedSongs.find(s => s.id === selectedSongId);

    // Style & Model Engine settings
    const [videoModel, setVideoModel] = useState<VideoModelKey>('wan_14b');
    const [videoProvider, setVideoProvider] = useState<'local' | 'cloud_fal' | 'cloud_replicate'>('local');
    const [lipSyncEngine, setLipSyncEngine] = useState<'live_portrait' | 'fallback'>('live_portrait');
    const [aspectRatio, setAspectRatio] = useState<AspectRatioType>('16:9');
    const [keyframes, setKeyframes] = useState<Record<number, string>>({});
    const [isGeneratingKeyframes, setIsGeneratingKeyframes] = useState(false);

    // Active model variant tracker
    const [activeVideoEngine, setActiveVideoEngine] = useState<VideoModelKey | null>(null);
    const [modelRegistry, setModelRegistry] = useState<Record<string, any>>({});
    const [clipDuration, setClipDuration] = useState<number>(() => {
        const saved = localStorage.getItem('milimo_video_clip_len_wan_14b') || localStorage.getItem('milimo_video_clip_len_hailuo_h3');
        return saved ? Math.min(5.0, Math.max(2.0, parseFloat(saved))) : 5.0;
    });

    const selectEngine = useCallback((model: VideoModelKey) => {
        const canonical = model === ('wan2.1' as any) ? 'wan_14b' : model;
        setVideoModel(canonical);
        const conf = MODEL_CONSTRAINTS[canonical] || MODEL_CONSTRAINTS['wan_14b'];
        const saved = localStorage.getItem(`milimo_video_clip_len_${canonical}`);
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
    const [transitionStyle, setTransitionStyle] = useState<'beat_cut' | 'crossfade' | 'flash' | 'whip_pan' | 'glitch'>('beat_cut');
    const [isDeletingVideo, setIsDeletingVideo] = useState(false);

    const handleClipDurationChange = (val: number) => {
        const conf = MODEL_CONSTRAINTS[videoModel] || MODEL_CONSTRAINTS['wan_14b'];
        const clamped = Math.min(conf.maxSec, Math.max(conf.minSec, val));
        setClipDuration(clamped);
        localStorage.setItem(`milimo_video_clip_len_${videoModel}`, clamped.toString());
    };

    const handleResetDurationToMax = () => {
        const conf = MODEL_CONSTRAINTS[videoModel] || MODEL_CONSTRAINTS['wan_14b'];
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

    // Director Mode v2 Controls (Maestro v2.4.0)
    const [pacingBias, setPacingBias] = useState<number>(0); // -2 (slow/cinematic) to +2 (rapid montage)
    const [vocalBypass, setVocalBypass] = useState<boolean>(true);
    const [fidelityRetries, setFidelityRetries] = useState<number>(1);
    const [autoContinue, setAutoContinue] = useState<boolean>(false);
    const [visibleCast, setVisibleCast] = useState<string[]>(['Lead Vocalist']);
    const [characterPromptNote, setCharacterPromptNote] = useState<string>('');

    // Planning & Task Tracking
    const [isPlanning, setIsPlanning] = useState(false);
    const [planResult, setPlanResult] = useState<VideoPlanResult | null>(null);

    const [activeTask, setActiveTask] = useState<VideoTaskStatus | null>(null);
    const [isRendering, setIsRendering] = useState(false);
    const [renderedVideoUrl, setRenderedVideoUrl] = useState<string | null>(null);
    const pollRef = useRef<number | undefined>(undefined);

    // Modal states
    const [retakeModalOpen, setRetakeModalOpen] = useState(false);
    const [retakeClipIndex, setRetakeClipIndex] = useState<number | null>(null);
    const [isRetaking, setIsRetaking] = useState(false);
    const [zoomKeyframe, setZoomKeyframe] = useState<{ clipIndex: number; url: string } | null>(null);

    // Unmount cleanup to prevent leaking video polling interval
    useEffect(() => {
        return () => {
            if (pollRef.current) {
                window.clearInterval(pollRef.current);
                pollRef.current = undefined;
            }
        };
    }, []);

    // Check if song has isolated stems
    const hasVocals = useMemo(() => {
        if (!activeSong?.stems_json) return false;
        try {
            const parsed = typeof activeSong.stems_json === 'string' ? JSON.parse(activeSong.stems_json) : activeSong.stems_json;
            return Boolean(parsed && (parsed.vocals || parsed.vocals_path));
        } catch {
            return false;
        }
    }, [activeSong?.stems_json]);

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

    // Plan Scenes Breakdown
    const handlePlanScenes = async () => {
        if (!activeSong) return;
        try {
            setIsPlanning(true);
            const params: VideoPlanParams = {
                model_name: videoModel,
                max_clip_duration: clipDuration,
                bpm: 120,
                visual_style: videoStyle,
                aspect_ratio: aspectRatio,
                provider: videoProvider,
                pacing_bias: pacingBias,
                music_timeline_vocal_bypass: vocalBypass,
                fidelity_retries: fidelityRetries,
                auto_continue: autoContinue,
                visible_cast: visibleCast,
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

    // Pre-Render Scene Keyframes for Storyboard Preview
    const handleGenerateKeyframes = async () => {
        if (!activeSong) return;
        try {
            setIsGeneratingKeyframes(true);
            const res = await videoApi.generateKeyframes(activeSong.id, videoStyle, resolution);
            if (res && res.keyframes) {
                const kfMap: Record<number, string> = {};
                for (const kf of res.keyframes) {
                    if (kf.keyframe_url) {
                        kfMap[kf.clip_index] = kf.keyframe_url;
                    }
                }
                setKeyframes(kfMap);
                toast(`Generated ${res.keyframes.length} scene keyframes for review.`, 'success');
            }
        } catch (err: any) {
            console.error('Failed to generate keyframes:', err);
            toast('Failed to generate scene keyframes.', 'error');
        } finally {
            setIsGeneratingKeyframes(false);
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
                aspect_ratio: aspectRatio,
                provider: videoProvider,
                lip_sync_engine: lipSyncEngine,
                enable_lip_sync: enableLipSync,
                burn_lyrics: burnSubtitles,
                subtitle_style: subtitleStyle,
                transition_style: transitionStyle,
                max_clip_duration: clipDuration,
                mode: 'production_multiclip',
                pacing_bias: pacingBias,
                music_timeline_vocal_bypass: vocalBypass,
                fidelity_retries: fidelityRetries,
                auto_continue: autoContinue,
                visible_cast: visibleCast,
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

    const applyStoredVideoConfig = (job: Job) => {
        if (!job?.video_config_json) return;
        try {
            const cfg = JSON.parse(job.video_config_json) as Record<string, any>;
            if (cfg.model_name && isValidVideoEngine(cfg.model_name)) {
                selectEngine(cfg.model_name);
            }
            if (cfg.max_clip_duration) {
                const engine = (cfg.model_name && isValidVideoEngine(cfg.model_name)) ? cfg.model_name : videoModel;
                const conf = MODEL_CONSTRAINTS[engine] || MODEL_CONSTRAINTS['wan_14b'];
                const clamped = Math.min(conf.maxSec, Math.max(conf.minSec, parseFloat(cfg.max_clip_duration)));
                setClipDuration(clamped);
            }
            if (cfg.visual_style && ['neon-cyberpunk', 'anime-cinematic', 'retro-vhs', 'minimal-lyrics'].includes(cfg.visual_style)) {
                setVideoStyle(cfg.visual_style);
            }
            if (cfg.aspect_ratio && ['16:9', '9:16', '1:1', '21:9'].includes(cfg.aspect_ratio)) {
                setAspectRatio(cfg.aspect_ratio);
            }
            if (cfg.resolution === '1080p' || cfg.resolution === '720p') setResolution(cfg.resolution);
            if (typeof cfg.enable_lip_sync === 'boolean') setEnableLipSync(cfg.enable_lip_sync);
            if (typeof cfg.burn_lyrics === 'boolean') setBurnSubtitles(cfg.burn_lyrics);
            if (cfg.subtitle_style && ['neon', 'cinematic', 'karaoke'].includes(cfg.subtitle_style)) setSubtitleStyle(cfg.subtitle_style);
        } catch { /* fallback to current state */ }
    };

    const handleRegenerateVideo = async () => {
        if (!activeSong) return;
        applyStoredVideoConfig(activeSong);
        await handleRenderAdvancedVideo();
    };

    const [isRouting, setIsRouting] = useState(false);

    const handleRouteToDirector = async () => {
        if (!renderedVideoUrl) return;
        setIsRouting(true);
        try {
            const filename = renderedVideoUrl.split('/').pop() || renderedVideoUrl;
            await galleryApi.routeMedia(renderedVideoUrl, 'references', activeSong?.id);
            toast(`Dispatched ${filename} to Director References!`, 'success');
        } catch (err: any) {
            toast(`Dispatch failed: ${err.message}`, 'error');
        } finally {
            setIsRouting(false);
        }
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

    // Storyboard notes generator
    const [isGeneratingStory, setIsGeneratingStory] = useState(false);
    const handleGenerateStoryboard = async () => {
        if (!activeSong) return;
        try {
            setIsGeneratingStory(true);
            const scenes = await videoApi.generateStoryboard(activeSong.id, videoStyle);
            if (scenes && scenes.length > 0) {
                toast('Storyboard sequence generated with dynamic directing prompts.', 'success');
            }
        } catch (err: any) {
            toast(err?.response?.data?.detail || 'Failed to generate storyboard.', 'error');
        } finally {
            setIsGeneratingStory(false);
        }
    };

    // Retake handlers
    const handleOpenRetakeModal = (clipIndex: number) => {
        setRetakeClipIndex(clipIndex);
        setRetakeModalOpen(true);
    };

    const handleConfirmRetake = async (clipIndex: number, newPrompt: string, camera: string, lighting: string) => {
        if (!activeSong) return;
        setIsRetaking(true);
        try {
            const res = await videoApi.retakeScene(activeSong.id, clipIndex, {
                prompt: newPrompt,
                camera,
                lighting
            });
            if (res.keyframe_url) {
                setKeyframes(prev => ({ ...prev, [clipIndex]: res.keyframe_url! }));
            }
            if (planResult?.clips) {
                const updatedClips = planResult.clips.map(c => {
                    if (c.clip_index === clipIndex) {
                        return { ...c, prompt: newPrompt, camera, lighting };
                    }
                    return c;
                });
                setPlanResult({ ...planResult, clips: updatedClips });
            }
            toast(`Retake generated for Scene #${clipIndex}!`, 'success');
            setRetakeModalOpen(false);
        } catch (err: any) {
            toast(`Failed to generate retake: ${err.message}`, 'error');
        } finally {
            setIsRetaking(false);
        }
    };

    const activeRetakeSegment = useMemo(() => {
        if (retakeClipIndex === null || !planResult?.clips) return undefined;
        return planResult.clips.find(c => c.clip_index === retakeClipIndex);
    }, [retakeClipIndex, planResult?.clips]);

    return (
        <div className="flex-1 overflow-y-auto p-4 md:p-6 pb-28 sm:pb-32 space-y-6 flex flex-col justify-between min-h-full">
            <div className="space-y-6 max-w-[1600px] mx-auto w-full">
                {/* ZONE 1: TOP MASTER STUDIO BAR */}
                <VideoTopBar
                    completedSongs={completedSongs}
                    selectedSongId={selectedSongId}
                    onSelectSong={handleSelectSong}
                    activeSong={activeSong}
                    hasVocals={hasVocals}
                    aspectRatio={aspectRatio}
                    onSelectAspectRatio={setAspectRatio}
                    isPlaying={isPlaying}
                    playingSongId={playingSongId}
                    onTogglePlayAudio={() => activeSong && onPlay(activeSong)}
                    isPlanning={isPlanning}
                    onPlanScenes={handlePlanScenes}
                    isGeneratingKeyframes={isGeneratingKeyframes}
                    onGenerateKeyframes={handleGenerateKeyframes}
                    isRendering={isRendering}
                    onRenderVideo={handleRenderAdvancedVideo}
                    renderedVideoUrl={renderedVideoUrl}
                    onDownloadVideo={() => {
                        if (renderedVideoUrl) {
                            api.downloadUrlAsFile(api.getAudioUrl(renderedVideoUrl), `${activeSong?.title || 'track'}_music_video.mp4`);
                        }
                    }}
                />

                {/* ZONE 2: DUAL WORKSPACE (Center Viewport + Right Inspector Dock) */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                    {/* Center / Left Viewport (7 Cols on desktop) */}
                    <div className="lg:col-span-7 xl:col-span-8 space-y-4">
                        <VideoCanvasPlayer
                            activeSong={activeSong}
                            renderedVideoUrl={renderedVideoUrl}
                            aspectRatio={aspectRatio}
                            isRendering={isRendering}
                            activeTask={activeTask}
                            isDeletingVideo={isDeletingVideo}
                            onDeleteVideo={handleDeleteVideo}
                            onRegenerateVideo={handleRegenerateVideo}
                            onRouteToDirector={handleRouteToDirector}
                            isRouting={isRouting}
                            onPlanScenes={handlePlanScenes}
                            onRenderVideo={handleRenderAdvancedVideo}
                            isPlanning={isPlanning}
                        />
                    </div>

                    {/* Right Tabbed Inspector Dock (5 Cols on desktop) */}
                    <div className="lg:col-span-5 xl:col-span-4 min-h-[500px]">
                        <VideoInspectorDock
                            videoModel={videoModel}
                            onSelectModel={selectEngine}
                            modelConstraints={MODEL_CONSTRAINTS}
                            modelRegistry={modelRegistry}
                            activeVideoEngine={activeVideoEngine}
                            videoProvider={videoProvider}
                            onSelectProvider={setVideoProvider}
                            clipDuration={clipDuration}
                            onChangeClipDuration={handleClipDurationChange}
                            onResetClipDuration={handleResetDurationToMax}
                            resolution={resolution}
                            onChangeResolution={setResolution}
                            videoStyle={videoStyle}
                            onSelectStyle={setVideoStyle}
                            pacingBias={pacingBias}
                            onChangePacingBias={setPacingBias}
                            vocalBypass={vocalBypass}
                            onChangeVocalBypass={setVocalBypass}
                            fidelityRetries={fidelityRetries}
                            onChangeFidelityRetries={setFidelityRetries}
                            autoContinue={autoContinue}
                            onChangeAutoContinue={setAutoContinue}
                            onGenerateStoryboard={handleGenerateStoryboard}
                            isGeneratingStory={isGeneratingStory}
                            enableLipSync={enableLipSync}
                            onChangeEnableLipSync={setEnableLipSync}
                            lipSyncEngine={lipSyncEngine}
                            onChangeLipSyncEngine={setLipSyncEngine}
                            burnSubtitles={burnSubtitles}
                            onChangeBurnSubtitles={setBurnSubtitles}
                            subtitleStyle={subtitleStyle}
                            onChangeSubtitleStyle={setSubtitleStyle}
                            transitionStyle={transitionStyle}
                            onChangeTransitionStyle={setTransitionStyle}
                            visibleCast={visibleCast}
                            onToggleCastMember={(cast) => {
                                if (visibleCast.includes(cast)) {
                                    setVisibleCast(visibleCast.filter(c => c !== cast));
                                } else {
                                    setVisibleCast([...visibleCast, cast]);
                                }
                            }}
                            characterPromptNote={characterPromptNote}
                            onChangeCharacterPromptNote={setCharacterPromptNote}
                        />
                    </div>
                </div>

                {/* ZONE 3: FULL-WIDTH HORIZONTAL MULTITRACK PRODUCTION TIMELINE */}
                {planResult?.clips && planResult.clips.length > 0 && (
                    <VideoTimelineTrack
                        clips={planResult.clips}
                        keyframes={keyframes}
                        activeSong={activeSong}
                        onRetakeClip={handleOpenRetakeModal}
                        onZoomKeyframe={(clipIndex, url) => setZoomKeyframe({ clipIndex, url })}
                    />
                )}
            </div>

            {/* Modal 1: Single-Clip Retake Studio Modal */}
            <ClipRetakeModal
                isOpen={retakeModalOpen}
                onClose={() => setRetakeModalOpen(false)}
                clipIndex={retakeClipIndex}
                clipSegment={activeRetakeSegment}
                onConfirmRetake={handleConfirmRetake}
                isRetaking={isRetaking}
            />

            {/* Modal 2: Full-Resolution Keyframe Still Zoom Lightbox */}
            <KeyframeZoomModal
                isOpen={zoomKeyframe !== null}
                onClose={() => setZoomKeyframe(null)}
                clipIndex={zoomKeyframe?.clipIndex ?? null}
                keyframeUrl={zoomKeyframe?.url ?? null}
                clipSegment={zoomKeyframe ? planResult?.clips?.find(c => c.clip_index === zoomKeyframe.clipIndex) : undefined}
            />

            {/* Global Creator Footer */}
            <AppFooter />
        </div>
    );
};
