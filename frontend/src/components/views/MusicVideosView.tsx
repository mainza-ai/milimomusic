import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
    type Job,
    videoApi,
    api,
    galleryApi,
    type VideoPlanResult,
    type VideoTaskStatus,
    type VideoPlanParams,
    type VideoRenderParams,
    type DirectorTreatment,
    type VideoClipSegment
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

export function normalizeVideoEngine(raw?: string | null): VideoModelKey {
    if (!raw) return 'wan_14b';
    const l = raw.toLowerCase().trim();
    if (l in MODEL_CONSTRAINTS) return l as VideoModelKey;
    if (l.includes('1.3') || l.includes('1_3')) return 'wan_1.3b';
    if (l.includes('wan') || l === 'wan2.1') return 'wan_14b';
    if (l.includes('ltx')) return 'ltx_video';
    if (l.includes('cog')) return 'cogvideox';
    if (l.includes('hailuo') || l.includes('minimax') || l.includes('h3')) return 'hailuo_h3';
    if (l.includes('hunyuan')) return 'hunyuan';
    if (l.includes('reactive')) return 'audioreactive';
    return 'wan_14b';
}

export const isValidVideoEngine = (e: string): e is VideoModelKey => e in MODEL_CONSTRAINTS || e === 'wan2.1';

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
    const [videoModel, setVideoModel] = useState<VideoModelKey>(() => {
        const saved = localStorage.getItem('milimo_active_video_engine');
        return saved ? normalizeVideoEngine(saved) : 'wan_14b';
    });
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

    const selectEngine = useCallback((model: VideoModelKey, persistToBackend = false) => {
        const canonical = normalizeVideoEngine(model);
        setVideoModel(canonical);
        const conf = MODEL_CONSTRAINTS[canonical] || MODEL_CONSTRAINTS['wan_14b'];
        const saved = localStorage.getItem(`milimo_video_clip_len_${canonical}`);
        const resolved = saved ? Math.min(conf.maxSec, Math.max(conf.minSec, parseFloat(saved))) : conf.defaultSec;
        setClipDuration(resolved);
        localStorage.setItem('milimo_active_video_engine', canonical);

        if (persistToBackend) {
            setActiveVideoEngine(canonical);
            videoApi.setActiveVideoEngine(canonical)
                .then(() => {
                    toast(`Active video model switched to ${conf.label}`, 'info');
                })
                .catch((e) => {
                    console.error('Failed to persist active video engine:', e);
                });
        }
    }, []);

    useEffect(() => {
        let alive = true;
        const refreshModels = () => {
            videoApi.getVideoModels().then((reg) => { if (alive) setModelRegistry(reg); }).catch(() => {});
        };
        refreshModels();

        const applyActiveEngine = (overrideEngineOrModel?: string) => {
            if (overrideEngineOrModel) {
                const canonical = normalizeVideoEngine(overrideEngineOrModel);
                if (!alive) return;
                setActiveVideoEngine(canonical);
                selectEngine(canonical, false);
                return;
            }
            videoApi.getActiveVideoEngine()
                .then((r) => {
                    if (!alive) return;
                    const canonical = normalizeVideoEngine(r.engine || r.model_id);
                    setActiveVideoEngine(canonical);
                    selectEngine(canonical, false);
                })
                .catch(() => {});
        };

        applyActiveEngine();

        const handleActivatedEvent = (e: Event) => {
            const customEvt = e as CustomEvent<{ modelId?: string; engine?: string; category?: string }>;
            if (customEvt.detail?.category && customEvt.detail.category !== 'video') {
                return;
            }
            refreshModels();
            if (customEvt.detail?.engine || customEvt.detail?.modelId) {
                applyActiveEngine(customEvt.detail.engine || customEvt.detail.modelId);
            } else {
                applyActiveEngine();
            }
        };

        window.addEventListener(VIDEO_ACTIVE_MODEL_EVENT, handleActivatedEvent);
        return () => {
            alive = false;
            window.removeEventListener(VIDEO_ACTIVE_MODEL_EVENT, handleActivatedEvent);
        };
    }, [selectEngine]);

    const [videoStyle, setVideoStyle] = useState<string>('neon-cyberpunk');
    const [customStylePrompt, setCustomStylePrompt] = useState<string>('');
    const [timelineSeekTime, setTimelineSeekTime] = useState<number | null>(null);
    const [resolution, setResolution] = useState<'720p' | '1080p'>('720p');
    const [transitionStyle, setTransitionStyle] = useState<'beat_cut' | 'crossfade' | 'flash' | 'whip_pan' | 'glitch'>('beat_cut');
    const [isDeletingVideo, setIsDeletingVideo] = useState(false);

    const handleClipDurationChange = useCallback((val: number) => {
        const conf = MODEL_CONSTRAINTS[videoModel] || MODEL_CONSTRAINTS['wan_14b'];
        const clamped = Math.min(conf.maxSec, Math.max(conf.minSec, val));
        setClipDuration(clamped);
        localStorage.setItem(`milimo_video_clip_len_${videoModel}`, clamped.toString());
    }, [videoModel]);

    const handleResetDurationToMax = useCallback(() => {
        const conf = MODEL_CONSTRAINTS[videoModel] || MODEL_CONSTRAINTS['wan_14b'];
        setClipDuration(conf.maxSec);
        localStorage.setItem(`milimo_video_clip_len_${videoModel}`, conf.maxSec.toString());
    }, [videoModel]);

    const handleToggleCastMember = useCallback((cast: string) => {
        setVisibleCast(prev => prev.includes(cast) ? prev.filter(c => c !== cast) : [...prev, cast]);
    }, []);

    const handleZoomKeyframe = useCallback((clipIndex: number, url: string) => {
        setZoomKeyframe({ clipIndex, url });
    }, []);

    const handleSeekTimeline = useCallback((timeSec: number) => {
        setTimelineSeekTime(timeSec);
    }, []);

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

    // AI Director Treatment State
    const [directorTreatment, setDirectorTreatment] = useState<DirectorTreatment | null>(null);
    const [isGeneratingTreatment, setIsGeneratingTreatment] = useState(false);

    // Fetch existing Director Treatment when song changes
    useEffect(() => {
        if (!activeSong?.id) {
            setDirectorTreatment(null);
            return;
        }
        videoApi.getDirectorTreatment(activeSong.id)
            .then(res => {
                if (res?.treatment) {
                    setDirectorTreatment(res.treatment);
                    if (res.treatment.scenes && res.treatment.scenes.length > 0) {
                        const scenes = res.treatment.scenes;
                        setPlanResult({
                            status: 'ok',
                            job_id: activeSong.id,
                            total_clips: scenes.length,
                            vocal_clips_count: scenes.filter((c: any) => c.scene_type === 'VOCAL_PERFORMANCE' || c.is_vocal).length,
                            broll_clips_count: scenes.filter((c: any) => c.scene_type !== 'VOCAL_PERFORMANCE' && !c.is_vocal).length,
                            max_clip_duration: clipDuration,
                            model_name: videoModel,
                            clips: scenes.map((s: any, idx: number) => ({
                                clip_index: s.clip_index || (idx + 1),
                                start_time: s.start_time ?? (idx * 15),
                                end_time: s.end_time ?? ((idx + 1) * 15),
                                duration: s.duration ?? 15,
                                time_str: s.time_str || `${idx * 15}s - ${(idx + 1) * 15}s`,
                                is_vocal: s.is_vocal ?? (s.scene_type === 'VOCAL_PERFORMANCE'),
                                scene_type: s.scene_type || 'CINEMATIC_BROLL',
                                lyrics: s.lyrics || '',
                                prompt: s.prompt || '',
                                camera: s.camera || 'Cinematic tracking shot',
                                lighting: s.lighting || 'Atmospheric rim lighting',
                                section_label: s.section_label,
                                musical_energy: s.musical_energy,
                                visual_action: s.visual_action,
                                directors_note: s.directors_note
                            })),
                            treatment: res.treatment
                        });
                    }
                } else {
                    setDirectorTreatment(null);
                }
            })
            .catch(() => setDirectorTreatment(null));
    }, [activeSong?.id, clipDuration, videoModel]);

    // Fetch existing scene keyframes when song changes, and auto-hydrate timeline plan if needed
    useEffect(() => {
        if (!activeSong?.id) {
            setKeyframes({});
            return;
        }
        videoApi.getKeyframes(activeSong.id)
            .then(res => {
                if (res?.keyframes && Object.keys(res.keyframes).length > 0) {
                    const bustTime = Date.now();
                    const kfMap: Record<number, string> = {};
                    for (const [idx, url] of Object.entries(res.keyframes)) {
                        kfMap[Number(idx)] = (url as string).includes('?') ? `${url}&t=${bustTime}` : `${url}?t=${bustTime}`;
                    }
                    setKeyframes(kfMap);

                    // Ensure timeline is planned so keyframes can be viewed immediately
                    videoApi.planVideo(activeSong.id, {
                        model_name: videoModel,
                        visual_style: videoStyle,
                        custom_style_prompt: customStylePrompt,
                        aspect_ratio: aspectRatio,
                        max_clip_duration: clipDuration
                    }).then(plan => {
                        if (plan && plan.clips && plan.clips.length > 0) {
                            setPlanResult(prev => prev || plan);
                        }
                    }).catch(() => {});
                } else {
                    setKeyframes({});
                }
            })
            .catch(() => setKeyframes({}));
    }, [activeSong?.id]);

    const [activeTask, setActiveTask] = useState<VideoTaskStatus | null>(null);
    const [isRendering, setIsRendering] = useState(false);
    const [renderedVideoUrl, setRenderedVideoUrl] = useState<string | null>(null);
    const pollRef = useRef<number | undefined>(undefined);
    const kfPollRef = useRef<number | undefined>(undefined);

    // Modal states
    const [retakeModalOpen, setRetakeModalOpen] = useState(false);
    const [retakeClipIndex, setRetakeClipIndex] = useState<number | null>(null);
    const [isRetaking, setIsRetaking] = useState(false);
    const [zoomKeyframe, setZoomKeyframe] = useState<{ clipIndex: number; url: string } | null>(null);

    // Unmount cleanup to prevent leaking video polling intervals
    useEffect(() => {
        return () => {
            if (pollRef.current) {
                window.clearInterval(pollRef.current);
                pollRef.current = undefined;
            }
            if (kfPollRef.current) {
                window.clearInterval(kfPollRef.current);
                kfPollRef.current = undefined;
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
                custom_style_prompt: customStylePrompt,
                character_desc: characterPromptNote,
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
            if (plan.treatment) {
                setDirectorTreatment(plan.treatment);
            }
            toast(`Scene plan created: ${plan.total_clips} scenes ready for production.`, 'success');
        } catch (err: any) {
            console.error('Failed to plan video scenes:', err);
            toast(err?.response?.data?.detail || 'Failed to plan video scenes. Please ensure the track is completed.', 'error');
        } finally {
            setIsPlanning(false);
        }
    };

    // AI Visual Director Treatment Generator
    const handleGenerateDirectorTreatment = async () => {
        if (!activeSong) return;
        setIsGeneratingTreatment(true);
        try {
            const res = await videoApi.generateDirectorTreatment(activeSong.id, {
                visual_style: videoStyle,
                custom_style_prompt: customStylePrompt,
                character_desc: characterPromptNote,
                visible_cast: visibleCast,
                pacing_bias: pacingBias,
                auto_continue: autoContinue,
            });
            if (res.treatment) {
                setDirectorTreatment(res.treatment);
                toast(`AI Director treatment generated: "${res.treatment.concept_title}"`, 'success');
            }
            if (res.clips && res.clips.length > 0) {
                setPlanResult({
                    status: 'ok',
                    job_id: activeSong.id,
                    total_clips: res.clips.length,
                    vocal_clips_count: res.clips.filter(c => c.scene_type === 'VOCAL_PERFORMANCE').length,
                    broll_clips_count: res.clips.filter(c => c.scene_type !== 'VOCAL_PERFORMANCE').length,
                    max_clip_duration: clipDuration,
                    model_name: videoModel,
                    clips: res.clips,
                    treatment: res.treatment
                });
            }
        } catch (err: any) {
            console.error('Failed to generate director treatment:', err);
            toast(err?.response?.data?.detail || 'Failed to generate AI Director treatment.', 'error');
        } finally {
            setIsGeneratingTreatment(false);
        }
    };

    // Pre-Render Scene Keyframes for Storyboard Preview
    const handleGenerateKeyframes = async (forceRegenerate?: boolean) => {
        if (!activeSong) return;
        try {
            setIsGeneratingKeyframes(true);
            const isForce = forceRegenerate === true || (typeof forceRegenerate !== 'boolean' && Object.keys(keyframes).length > 0);
            if (isForce) {
                setKeyframes({});
            }

            // Eagerly plan timeline if not yet present so user sees scene cards immediately
            let currentClips = planResult?.clips;
            if (!currentClips || currentClips.length === 0) {
                try {
                    const eagerPlan = await videoApi.planVideo(activeSong.id, {
                        model_name: videoModel,
                        visual_style: videoStyle,
                        custom_style_prompt: customStylePrompt,
                        aspect_ratio: aspectRatio,
                        max_clip_duration: clipDuration
                    });
                    if (eagerPlan?.clips?.length) {
                        setPlanResult(eagerPlan);
                        currentClips = eagerPlan.clips;
                    }
                } catch (e) {
                    console.warn('Eager planning pre-keyframe generation failed; will use server segmentation:', e);
                }
            }

            // Start progressive polling so stills appear as soon as each is rendered on disk
            if (kfPollRef.current) {
                window.clearInterval(kfPollRef.current);
            }
            kfPollRef.current = window.setInterval(async () => {
                try {
                    const pollRes = await videoApi.getKeyframes(activeSong.id);
                    if (pollRes?.keyframes && Object.keys(pollRes.keyframes).length > 0) {
                        const bustTime = Date.now();
                        const kfMap: Record<number, string> = {};
                        for (const [idx, url] of Object.entries(pollRes.keyframes)) {
                            kfMap[Number(idx)] = (url as string).includes('?') ? `${url}&t=${bustTime}` : `${url}?t=${bustTime}`;
                        }
                        setKeyframes(prev => ({ ...prev, ...kfMap }));
                    }
                } catch {
                    // ignore polling errors
                }
            }, 3000);

            const res = await videoApi.generateKeyframes(
                activeSong.id,
                videoStyle,
                resolution,
                customStylePrompt,
                aspectRatio,
                isForce,
                currentClips
            );

            const activeJobId = activeSong.id;
            const taskId = res?.task_id;

            const onKeyframesCompleted = (finalKeyframes: any[]) => {
                if (finalKeyframes && finalKeyframes.length > 0) {
                    const bustTime = Date.now();
                    const kfMap: Record<number, string> = {};
                    for (const kf of finalKeyframes) {
                        if (kf.keyframe_url) {
                            const url = (kf.keyframe_url as string).includes('?')
                                ? `${kf.keyframe_url}&t=${bustTime}`
                                : `${kf.keyframe_url}?t=${bustTime}`;
                            kfMap[kf.clip_index] = url;
                        }
                    }
                    setKeyframes(kfMap);

                    // If timeline was not planned yet, automatically populate it from the returned keyframe scenes
                    if (!planResult || !planResult.clips || planResult.clips.length === 0) {
                        const clips: VideoClipSegment[] = finalKeyframes.map((kf: any, idx: number) => ({
                            clip_index: kf.clip_index || (idx + 1),
                            start_time: kf.start_time ?? (idx * 15),
                            end_time: kf.end_time ?? ((idx + 1) * 15),
                            duration: kf.duration ?? 15,
                            time_str: kf.time_str || `${idx * 15}s - ${(idx + 1) * 15}s`,
                            is_vocal: kf.is_vocal ?? (kf.scene_type === 'VOCAL_PERFORMANCE'),
                            scene_type: kf.scene_type || 'CINEMATIC_BROLL',
                            lyrics: kf.lyrics || '',
                            prompt: kf.prompt || '',
                            camera: kf.camera || 'Cinematic tracking shot',
                            lighting: kf.lighting || 'Atmospheric rim lighting',
                            section_label: kf.section_label,
                            musical_energy: kf.musical_energy,
                            visual_action: kf.visual_action,
                            directors_note: kf.directors_note
                        }));
                        setPlanResult({
                            status: 'ok',
                            job_id: activeJobId,
                            total_clips: clips.length,
                            vocal_clips_count: clips.filter(c => c.is_vocal).length,
                            broll_clips_count: clips.filter(c => !c.is_vocal).length,
                            max_clip_duration: clipDuration,
                            model_name: videoModel,
                            clips
                        });
                    }
                    toast(`Generated ${finalKeyframes.length} scene keyframes for review.`, 'success');
                }
            };

            if (taskId) {
                while (true) {
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    try {
                        const taskStatus: any = await videoApi.getVideoTaskStatus(taskId);
                        if (taskStatus.status === 'completed') {
                            onKeyframesCompleted(taskStatus.keyframes || []);
                            break;
                        } else if (taskStatus.status === 'cancelled') {
                            toast('Keyframe generation cancelled.', 'info');
                            break;
                        } else if (taskStatus.status === 'error') {
                            toast(taskStatus.error || 'Keyframe generation encountered an error.', 'error');
                            break;
                        }
                    } catch {
                        // transient network polling error, retry
                    }
                }
            } else if (res && res.keyframes) {
                onKeyframesCompleted(res.keyframes);
            }
        } catch (err: any) {
            console.error('Failed to generate keyframes:', err);
            toast('Failed to generate scene keyframes.', 'error');
        } finally {
            if (kfPollRef.current) {
                window.clearInterval(kfPollRef.current);
                kfPollRef.current = undefined;
            }
            setIsGeneratingKeyframes(false);
        }
    };

    const handleCancelKeyframes = async () => {
        if (!activeSong) return;
        try {
            await videoApi.cancelKeyframeGeneration(activeSong.id);
            toast('Keyframe generation cancelled.', 'info');
        } catch (err: any) {
            toast(err?.response?.data?.detail || 'Failed to cancel keyframe generation.', 'error');
        } finally {
            if (kfPollRef.current) {
                window.clearInterval(kfPollRef.current);
                kfPollRef.current = undefined;
            }
            setIsGeneratingKeyframes(false);
        }
    };

    const handleCancelVideoRender = async () => {
        if (!activeTask?.id) return;
        try {
            await videoApi.cancelVideoTask(activeTask.id);
            toast('Video rendering cancelled.', 'info');
            setActiveTask(prev => prev ? { ...prev, status: 'cancelled', step: 'cancelled' } : null);
        } catch (err: any) {
            toast(err?.response?.data?.detail || 'Failed to cancel video rendering.', 'error');
        } finally {
            if (pollRef.current) {
                window.clearInterval(pollRef.current);
                pollRef.current = undefined;
            }
            setIsRendering(false);
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
                custom_style_prompt: customStylePrompt,
                character_desc: characterPromptNote,
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
                scenes: planResult?.clips,
                clips: planResult?.clips,
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
                    } else if (status.status === 'cancelled') {
                        window.clearInterval(pollRef.current);
                        setIsRendering(false);
                        toast('Video rendering cancelled.', 'info');
                    } else if (status.status === 'error') {
                        window.clearInterval(pollRef.current);
                        setIsRendering(false);
                        toast(status.error || 'Video rendering encountered an error', 'error');
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
            const scenes = await videoApi.generateStoryboard(activeSong.id, videoStyle, customStylePrompt);
            if (scenes && scenes.length > 0) {
                // If no clips exist on timeline yet, sync storyboard scenes into planResult so timeline immediately displays them
                if (!planResult || !planResult.clips || planResult.clips.length === 0) {
                    const clips: any[] = scenes.map((s, idx) => ({
                        clip_index: idx + 1,
                        start_time: idx * 15,
                        end_time: (idx + 1) * 15,
                        duration: 15,
                        time_str: s.time || `${idx * 15}s - ${(idx + 1) * 15}s`,
                        is_vocal: s.is_vocal ?? false,
                        scene_type: (s.is_vocal ? 'VOCAL_PERFORMANCE' : 'CINEMATIC_BROLL') as any,
                        lyrics: s.lyrics || '',
                        prompt: s.prompt || '',
                        camera: s.camera || 'Slow cinematic tracking crane down',
                        lighting: s.lighting || 'Cyan and magenta anamorphic rim lighting'
                    }));
                    setPlanResult({
                        status: 'ok',
                        job_id: activeSong.id,
                        total_clips: clips.length,
                        vocal_clips_count: clips.filter(c => c.is_vocal).length,
                        broll_clips_count: clips.filter(c => !c.is_vocal).length,
                        max_clip_duration: 15,
                        model_name: videoModel,
                        clips
                    });
                }
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
                lighting,
                custom_style_prompt: customStylePrompt,
                aspect_ratio: aspectRatio,
            });
            if (res.keyframe_url) {
                const bustTime = Date.now();
                const url = res.keyframe_url.includes('?') ? `${res.keyframe_url}&t=${bustTime}` : `${res.keyframe_url}?t=${bustTime}`;
                setKeyframes(prev => ({ ...prev, [clipIndex]: url }));
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

    const handleReimagineScene = async (clipIndex: number, instruction?: string) => {
        if (!activeSong) return undefined;
        try {
            const currentScene = planResult?.clips?.find(c => c.clip_index === clipIndex);
            const res = await videoApi.reimagineScene(activeSong.id, clipIndex, {
                user_instruction: instruction,
                visual_style: videoStyle,
                character_desc: characterPromptNote,
                current_scene: currentScene,
            });
            if (res?.scene) {
                if (planResult?.clips) {
                    const updatedClips = planResult.clips.map(c => c.clip_index === clipIndex ? { ...c, ...res.scene } : c);
                    setPlanResult({ ...planResult, clips: updatedClips });
                }
                toast(`Scene #${clipIndex} re-imagined by AI Director!`, 'success');
                return res.scene;
            }
        } catch (err: any) {
            console.error('Failed to re-imagine scene:', err);
            toast(err?.response?.data?.detail || 'Failed to re-imagine scene.', 'error');
        }
        return undefined;
    };

    const activeRetakeSegment = useMemo(() => {
        if (retakeClipIndex === null || !planResult?.clips) return undefined;
        return planResult.clips.find(c => c.clip_index === retakeClipIndex);
    }, [retakeClipIndex, planResult?.clips]);

    const handleTogglePlayAudio = useCallback(() => {
        if (activeSong) onPlay(activeSong);
    }, [activeSong, onPlay]);

    const handleDownloadVideo = useCallback(() => {
        if (renderedVideoUrl) {
            api.downloadUrlAsFile(api.getAudioUrl(renderedVideoUrl), `${activeSong?.title || 'track'}_music_video.mp4`);
        }
    }, [renderedVideoUrl, activeSong?.title]);

    const handleSelectInspectorModel = useCallback((m: VideoModelKey) => {
        selectEngine(m, true);
    }, [selectEngine]);

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
                    resolution={resolution}
                    isPlaying={isPlaying}
                    playingSongId={playingSongId}
                    onTogglePlayAudio={handleTogglePlayAudio}
                    isPlanning={isPlanning}
                    onPlanScenes={handlePlanScenes}
                    isGeneratingKeyframes={isGeneratingKeyframes}
                    onGenerateKeyframes={handleGenerateKeyframes}
                    onCancelKeyframes={handleCancelKeyframes}
                    hasKeyframes={Object.keys(keyframes).length > 0}
                    isRendering={isRendering}
                    onRenderVideo={handleRenderAdvancedVideo}
                    onCancelRender={handleCancelVideoRender}
                    renderedVideoUrl={renderedVideoUrl}
                    onDownloadVideo={handleDownloadVideo}
                />

                {/* ZONE 2: DUAL WORKSPACE (Center Viewport + Right Inspector Dock) */}
                <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
                    {/* Center / Left Viewport */}
                    <div className="xl:col-span-7 2xl:col-span-7 space-y-4">
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
                            onCancelRender={handleCancelVideoRender}
                            isPlanning={isPlanning}
                            seekTime={timelineSeekTime}
                            onDismissTask={() => setActiveTask(null)}
                        />
                    </div>

                    {/* Right Tabbed Inspector Dock */}
                    <div className="xl:col-span-5 2xl:col-span-5 min-h-[500px]">
                        <VideoInspectorDock
                            directorTreatment={directorTreatment}
                            onGenerateTreatment={handleGenerateDirectorTreatment}
                            isGeneratingTreatment={isGeneratingTreatment}
                            videoModel={videoModel}
                            onSelectModel={handleSelectInspectorModel}
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
                            customStylePrompt={customStylePrompt}
                            onChangeCustomStylePrompt={setCustomStylePrompt}
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
                            onToggleCastMember={handleToggleCastMember}
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
                        aspectRatio={aspectRatio}
                        activeSong={activeSong}
                        onRetakeClip={handleOpenRetakeModal}
                        onZoomKeyframe={handleZoomKeyframe}
                        onSeekToTime={handleSeekTimeline}
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
                onReimagineScene={handleReimagineScene}
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
