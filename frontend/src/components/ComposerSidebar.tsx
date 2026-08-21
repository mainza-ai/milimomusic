import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Music,
    ChevronDown,
    ChevronUp,
    Sparkles,
    Wand2,
    ArrowRightCircle,
    Settings,
    Upload,
    Image as ImageIcon
} from 'lucide-react';

import { api, voiceApi, coverApi, API_BASE_URL, type Job, type LLMConfig, type VoiceProfile, type Project } from '../api';
import { LLMSettingsModal } from './LLMSettingsModal';
import { VoiceStudioModal } from './voice/VoiceStudioModal';
import { ModelsManagerModal } from './models/ModelsManagerModal';

interface ComposerSidebarProps {
    onGenerate: (data: CompositionData) => void;
    isGenerating: boolean;
    lyricsModels: string[];
    onGenerateLyrics: (topic: string, model: string, currentLyrics?: string, tags?: string) => Promise<string>;
    isGeneratingLyrics: boolean;
    currentJobId?: string;
    onCancel?: (jobId: string) => void;
    parentJob?: Job;
    onClearParentJob?: () => void;
    onRefreshModels?: () => void;
    onOpenTraining?: () => void;
    activeCheckpoint?: { name: string; id: string } | null;
    activeProject?: Project | null;
    onClearActiveProject?: () => void;
    producerPreset?: Partial<CompositionData> | null;
}

export interface CompositionData {
    title?: string;
    lyrics: string;
    topic: string;
    tags: string;
    durationMs: number;
    temperature: number;
    cfgScale: number;
    topk: number;
    llmModel: string;
    modelProvider?: string;
    voiceProfileId?: string;
    structuredCaption?: Record<string, string>;
    projectId?: string;
    seed?: number;
    isInstrumental?: boolean;
    coverImagePath?: string;
    imagePrompt?: string;
}

export const ComposerSidebar: React.FC<ComposerSidebarProps> = ({
    onGenerate,
    isGenerating,
    lyricsModels,
    onGenerateLyrics,
    isGeneratingLyrics,
    parentJob,
    onClearParentJob,
    onRefreshModels,
    activeProject,
    onClearActiveProject,
    producerPreset
}) => {
    // Accordion Expansion States
    const [openSections, setOpenSections] = useState<{ lyrics: boolean; sound: boolean; details: boolean }>({
        lyrics: true,
        sound: true,
        details: true
    });

    // Content State
    const [title, setTitle] = useState('');
    const [topic, setTopic] = useState('');
    const [style, setStyle] = useState('');
    const [lyrics, setLyrics] = useState('');
    const [isInstrumental, setIsInstrumental] = useState(false);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [isEnhancing, setIsEnhancing] = useState(false);

    // Cover Artwork State
    const [coverImagePath, setCoverImagePath] = useState<string>('');
    const [coverImagePrompt, setCoverImagePrompt] = useState<string>('');
    const [isUploadingCover, setIsUploadingCover] = useState(false);
    const [isGeneratingCover, setIsGeneratingCover] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // v2 Model & Voice State
    const [modelProvider, setModelProvider] = useState<string>('minimax_music3');
    const [voiceProfiles, setVoiceProfiles] = useState<VoiceProfile[]>([]);
    const [selectedVoiceProfile, setSelectedVoiceProfile] = useState<string>('');
    const [isVoiceStudioOpen, setIsVoiceStudioOpen] = useState(false);
    const [isModelsManagerOpen, setIsModelsManagerOpen] = useState(false);

    // Structured Caption fields (MiniMax Music 3)
    const [globalMetadata, setGlobalMetadata] = useState('Genre: Contemporary Pop\nMood: Energetic & Upbeat');
    const [vocalDetails, setVocalDetails] = useState('Lead Vocals: Clear, Expressive, Dynamic');
    const [arrangement, setArrangement] = useState('Instrumentation: Drums, Bass, Electric Guitar, Synth Leads');

    // Settings Modals
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [llmConfig, setLlmConfig] = useState<LLMConfig>({});

    const loadVoiceProfiles = async () => {
        try {
            const list = await voiceApi.listProfiles();
            setVoiceProfiles(list);
        } catch (e) {
            console.error('Failed to load voice profiles', e);
        }
    };

    const loadLlmConfig = async () => {
        try {
            const cfg = await api.getLLMConfig();
            setLlmConfig(cfg);
            onRefreshModels?.();
        } catch (e) {
            console.error("Failed to load LLM config", e);
        }
    };

    useEffect(() => {
        loadVoiceProfiles();
        loadLlmConfig();
    }, []);

    const [duration, setDuration] = useState(() => parseInt(localStorage.getItem('milimo_duration') || '60'));
    const [temperature] = useState(() => parseFloat(localStorage.getItem('milimo_temperature') || '1.0'));
    const [cfgScale] = useState(() => parseFloat(localStorage.getItem('milimo_cfg') || '2.0'));
    const [topk] = useState(() => parseInt(localStorage.getItem('milimo_topk') || '50'));
    const [seed] = useState<number | undefined>(() => {
        const saved = localStorage.getItem('milimo_seed');
        return saved ? parseInt(saved) : undefined;
    });
    const [isSeedLocked] = useState(() => localStorage.getItem('milimo_seed_locked') === 'true');

    const [lyricsModel, setLyricsModel] = useState(() => localStorage.getItem('milimo_lyrics_model') || (lyricsModels[0] || 'minimax-m3'));

    useEffect(() => {
        if (lyricsModels.length > 0 && (!lyricsModel || !lyricsModels.includes(lyricsModel))) {
            setLyricsModel(lyricsModels[0]);
        }
    }, [lyricsModels]);

    useEffect(() => {
        if (!producerPreset) return;
        if (producerPreset.title) setTitle(producerPreset.title);
        if (producerPreset.lyrics !== undefined) setLyrics(producerPreset.lyrics);
        if (producerPreset.topic) setTopic(producerPreset.topic);
        if (producerPreset.tags) setStyle(producerPreset.tags);
        if (producerPreset.structuredCaption?.global_metadata) setGlobalMetadata(producerPreset.structuredCaption.global_metadata);
        if (producerPreset.structuredCaption?.vocal_details) setVocalDetails(producerPreset.structuredCaption.vocal_details);
        if (producerPreset.structuredCaption?.arrangement) setArrangement(producerPreset.structuredCaption.arrangement);
        if (producerPreset.durationMs) setDuration(Math.round(producerPreset.durationMs / 1000));
        if (producerPreset.isInstrumental !== undefined) setIsInstrumental(producerPreset.isInstrumental);
        if (producerPreset.coverImagePath) setCoverImagePath(producerPreset.coverImagePath);
    }, [producerPreset]);

    useEffect(() => {
        localStorage.setItem('milimo_duration', duration.toString());
    }, [duration]);

    const handleEnhancePrompt = async () => {
        setIsEnhancing(true);
        try {
            // The enhancement is a professional structured-caption rewrite (official
            // MiniMax music-caption-rewriter workflow). If there is no concept yet,
            // spark one first via the inspiration endpoint.
            let currentTopic = topic || title || '';
            let currentStyle = style || '';
            if (!currentTopic) {
                const insp = await api.getInspiration(lyricsModel);
                currentTopic = insp.topic;
                currentStyle = insp.tags || currentStyle;
            }
            const result = await api.rewriteCaption(currentTopic, lyrics, currentStyle || undefined, lyricsModel);
            if (result.global_metadata) setGlobalMetadata(result.global_metadata);
            if (result.vocal_details) setVocalDetails(result.vocal_details);
            if (result.arrangement) setArrangement(result.arrangement);
            if (currentTopic && !topic) setTopic(currentTopic);
            if (currentStyle && !style) setStyle(currentStyle);
            if (!title && currentTopic) setTitle(currentTopic.slice(0, 30));
        } catch (e) {
            // LLM unreachable: keep the user's inputs; fall back to a minimal
            // structured default so the caption fields are never empty/lying.
            if (!globalMetadata) {
                setGlobalMetadata(`Genre: ${style || 'Contemporary Pop'}\nMood: Energetic & Dynamic`);
            }
        } finally {
            setIsEnhancing(false);
        }
    };

    const handleLyricsGen = async () => {
        if (!topic && !title) return;
        try {
            const genLyrics = await onGenerateLyrics(topic || title, lyricsModel, lyrics.trim(), style);
            setLyrics(genLyrics);
            setIsInstrumental(false);
        } catch (e: any) {
            alert("Lyrics Generation Failed: " + (e.message || "Unknown error"));
        }
    };

    const handleCoverFileUpload = async (file: File) => {
        try {
            setIsUploadingCover(true);
            const res = await coverApi.uploadCoverImage(file);
            const fullUrl = res.url.startsWith('http') ? res.url : `${API_BASE_URL}${res.url}`;
            setCoverImagePath(fullUrl);
        } catch (e) {
            console.error("Failed to upload cover image", e);
        } finally {
            setIsUploadingCover(false);
        }
    };

    const handleGenerateCoverArtwork = async () => {
        try {
            setIsGeneratingCover(true);
            const promptRes = await coverApi.generateCoverPrompt({
                title: title || topic || 'Studio Track',
                tags: style
            });
            const imgRes = await coverApi.generateCoverImage({ prompt: promptRes.prompt });
            const fullUrl = imgRes.url.startsWith('http') ? imgRes.url : `${API_BASE_URL}${imgRes.url}`;
            setCoverImagePath(fullUrl);
            setCoverImagePrompt(promptRes.prompt);
        } catch (e) {
            console.error("Failed to generate cover artwork", e);
        } finally {
            setIsGeneratingCover(false);
        }
    };

    const toggleSection = (section: 'lyrics' | 'sound' | 'details') => {
        setOpenSections(prev => ({ ...prev, [section]: !prev[section] }));
    };

    const handleSubmit = () => {
        const finalSeed = isSeedLocked && seed !== undefined ? seed : Math.floor(Math.random() * 2147483647);

        onGenerate({
            title: title.trim() || undefined,
            lyrics: isInstrumental ? '' : lyrics,
            topic: topic || title || 'Pop track',
            tags: style,
            durationMs: duration * 1000,
            temperature,
            cfgScale,
            topk,
            llmModel: lyricsModel,
            modelProvider: modelProvider,
            voiceProfileId: selectedVoiceProfile || undefined,
            structuredCaption: {
                global_metadata: globalMetadata,
                vocal_details: vocalDetails,
                arrangement: arrangement
            },
            projectId: activeProject?.id,
            seed: finalSeed,
            isInstrumental,
            coverImagePath: coverImagePath || undefined,
            imagePrompt: coverImagePrompt || undefined
        });
    };

    return (
        <div className="h-full flex flex-col bg-white/80 dark:bg-[#12141c]/90 backdrop-blur-2xl text-slate-800 dark:text-slate-200 select-none overflow-hidden w-full transition-colors duration-200">
            <input
                type="file"
                ref={fileInputRef}
                accept="image/*"
                onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleCoverFileUpload(file);
                }}
                className="hidden"
            />

            {parentJob && (
                <div className="bg-teal-500/10 border-b border-teal-500/20 p-3 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-teal-600 dark:text-teal-400 text-xs font-medium">
                        <ArrowRightCircle className="w-4 h-4" />
                        <span>Extending: {parentJob.title || "Untitled Track"}</span>
                    </div>
                    {onClearParentJob && (
                        <button onClick={onClearParentJob} className="text-teal-600 dark:text-teal-400 hover:text-teal-800 text-xs underline">Cancel</button>
                    )}
                </div>
            )}

            {activeProject && (
                <div className="bg-teal-500/10 border-b border-teal-500/20 px-3 py-2 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-teal-700 dark:text-teal-300 text-xs font-semibold truncate">
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/20 text-teal-700 dark:text-teal-300 font-bold truncate max-w-[150px]">
                            📁 {activeProject.name}
                        </span>
                    </div>
                    {onClearActiveProject && (
                        <button
                            onClick={onClearActiveProject}
                            className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 text-xs p-1"
                            title="Exit project context"
                        >
                            ✕
                        </button>
                    )}
                </div>
            )}

            <div className="p-4 border-b border-black/[0.06] dark:border-white/[0.08] bg-black/[0.02] dark:bg-white/[0.02] flex items-center justify-between">
                <div className="flex items-center space-x-2">
                    <h2 className="text-sm font-bold text-slate-900 dark:text-white">Compose</h2>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 font-semibold border border-teal-500/20">Studio 3.0</span>
                </div>
                <button onClick={() => setIsSettingsOpen(true)} className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-500">
                    <Settings size={15} />
                </button>
            </div>

            <LLMSettingsModal isOpen={isSettingsOpen} currentConfig={llmConfig} onConfigUpdate={loadLlmConfig} onClose={() => setIsSettingsOpen(false)} />
            <VoiceStudioModal isOpen={isVoiceStudioOpen} onClose={() => { setIsVoiceStudioOpen(false); loadVoiceProfiles(); }} />
            <ModelsManagerModal isOpen={isModelsManagerOpen} onClose={() => setIsModelsManagerOpen(false)} />

            <div className="flex-1 overflow-y-auto p-3.5 space-y-3">
                <div className="rounded-2xl border border-black/[0.06] dark:border-white/[0.08] bg-black/[0.015] dark:bg-white/[0.02] overflow-hidden transition-all">
                    <div className="p-3.5 flex items-center justify-between border-b border-black/[0.04] dark:border-white/[0.04]">
                        <button type="button" onClick={() => toggleSection('lyrics')} className="flex items-center gap-2 font-bold text-xs">
                            {openSections.lyrics ? <ChevronUp size={15} className="text-teal-500" /> : <ChevronDown size={15} className="text-slate-400" />}
                            <span>Lyrics</span>
                        </button>
                        <label className="flex items-center gap-2 cursor-pointer select-none">
                            <span className="text-[11px] font-semibold text-slate-500">Instrumental</span>
                            <div onClick={(e) => { e.stopPropagation(); setIsInstrumental(!isInstrumental); }} className={`w-9 h-5 rounded-full transition-colors relative flex items-center p-0.5 ${isInstrumental ? 'bg-teal-500' : 'bg-black/20'}`}>
                                <div className={`w-4 h-4 rounded-full bg-white shadow-sm transition-transform ${isInstrumental ? 'translate-x-4' : 'translate-x-0'}`} />
                            </div>
                        </label>
                    </div>
                    <AnimatePresence initial={false}>
                        {openSections.lyrics && (
                            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="p-3.5 space-y-3 overflow-hidden">
                                {isInstrumental ? (
                                    <div className="p-4 rounded-xl bg-teal-500/5 border border-teal-500/15 text-center">
                                        <Music size={20} className="mx-auto text-teal-500" />
                                        <p className="text-xs font-semibold text-teal-700">Instrumental Track Enabled</p>
                                    </div>
                                ) : (
                                    <>
                                        <div className="flex items-center gap-2">
                                            <button type="button" onClick={handleLyricsGen} disabled={isGeneratingLyrics || (!topic && !title)} className="flex-1 py-2 bg-gradient-to-r from-teal-500/15 to-cyan-500/15 rounded-xl text-xs font-bold flex items-center justify-center gap-1.5 shadow-sm disabled:opacity-50">
                                                <Sparkles size={13} />
                                                <span>{isGeneratingLyrics ? 'Writing...' : 'AI Co-Writer: Write'}</span>
                                            </button>
                                            <select value={lyricsModel} onChange={(e) => { setLyricsModel(e.target.value); localStorage.setItem('milimo_lyrics_model', e.target.value); }} className="apple-input py-1.5 px-2 text-[11px] font-mono max-w-[130px] truncate">
                                                {lyricsModels.map(m => <option key={m} value={m}>{m}</option>)}
                                            </select>
                                        </div>
                                        <textarea value={lyrics} onChange={(e) => setLyrics(e.target.value)} rows={6} placeholder="[Intro]..." className="w-full apple-input resize-none font-mono text-[11px] leading-relaxed p-2.5" />
                                    </>
                                )}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                <div className="rounded-2xl border border-black/[0.06] dark:border-white/[0.08] bg-black/[0.015] dark:bg-white/[0.02] overflow-hidden transition-all">
                    <div className="p-3.5 flex items-center justify-between border-b border-black/[0.04] dark:border-white/[0.04]">
                        <button type="button" onClick={() => toggleSection('sound')} className="flex items-center gap-2 font-bold text-xs">
                            {openSections.sound ? <ChevronUp size={15} className="text-teal-500" /> : <ChevronDown size={15} className="text-slate-400" />}
                            <span>Sound & Style</span>
                        </button>
                        <label className="flex items-center gap-2 cursor-pointer select-none">
                            <span className="text-[11px] font-semibold text-slate-500">Advanced</span>
                            <div onClick={(e) => { e.stopPropagation(); setShowAdvanced(!showAdvanced); }} className={`w-9 h-5 rounded-full transition-colors relative flex items-center p-0.5 ${showAdvanced ? 'bg-teal-500' : 'bg-black/20'}`}>
                                <div className={`w-4 h-4 rounded-full bg-white transition-transform ${showAdvanced ? 'translate-x-4' : 'translate-x-0'}`} />
                            </div>
                        </label>
                    </div>
                    <AnimatePresence initial={false}>
                        {openSections.sound && (
                            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="p-3.5 space-y-3.5 overflow-hidden">
                                <div className="space-y-1.5">
                                    <div className="flex items-center justify-between">
                                        <label className="text-[11px] font-bold uppercase text-slate-400">Description & Mood</label>
                                        <button type="button" onClick={handleEnhancePrompt} disabled={isEnhancing || !topic} className="text-[10px] font-bold text-teal-600 flex items-center gap-1 disabled:opacity-50">
                                            <Wand2 size={11} /> <span>Enhance</span>
                                        </button>
                                    </div>
                                    <input type="text" value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="Describe mood..." className="apple-input text-xs py-2 px-3" />
                                </div>
                                <input type="text" value={style} onChange={(e) => setStyle(e.target.value)} placeholder="Tags..." className="apple-input text-xs py-2 px-3 font-mono" />
                                {showAdvanced && (
                                    <div className="pt-2 border-t border-black/[0.06] space-y-3">
                                        <div className="grid grid-cols-2 gap-2">
                                            <select value={modelProvider} onChange={(e) => setModelProvider(e.target.value)} className="apple-input py-1.5 text-[11px] font-mono">
                                                <option value="minimax_music3">MiniMax Music 3</option>
                                                <option value="heartmula">HeartMuLa</option>
                                            </select>
                                            <select value={selectedVoiceProfile} onChange={(e) => { if (e.target.value === '__add_new__') setIsVoiceStudioOpen(true); else setSelectedVoiceProfile(e.target.value); }} className="apple-input py-1.5 text-[11px] font-mono">
                                                <option value="">Default AI Voice</option>
                                                {voiceProfiles.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                                                <option value="__add_new__">+ Train Voice...</option>
                                            </select>
                                        </div>
                                        <div className="space-y-1">
                                            <div className="flex justify-between text-xs"><span className="text-[11px]">Duration</span><span className="font-bold text-teal-500">{duration}s</span></div>
                                            <input type="range" min={5} max={300} step={5} value={duration} onChange={(e) => setDuration(parseInt(e.target.value))} className="w-full accent-teal-500" />
                                        </div>
                                    </div>
                                )}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                <div className="rounded-2xl border border-black/[0.06] dark:border-white/[0.08] bg-black/[0.015] dark:bg-white/[0.02] overflow-hidden transition-all">
                    <div className="p-3.5 flex items-center justify-between border-b border-black/[0.04] dark:border-white/[0.04]">
                        <button type="button" onClick={() => toggleSection('details')} className="flex items-center gap-2 font-bold text-xs">
                            {openSections.details ? <ChevronUp size={15} className="text-teal-500" /> : <ChevronDown size={15} className="text-slate-400" />}
                            <span>Details & Artwork</span>
                        </button>
                    </div>
                    <AnimatePresence initial={false}>
                        {openSections.details && (
                            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="p-3.5 space-y-3.5 overflow-hidden">
                                <input type="text" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Song Title" className="apple-input text-xs py-2 px-3" />
                                <div className="flex items-start gap-3">
                                    <div onClick={() => fileInputRef.current?.click()} className="w-20 h-20 rounded-xl bg-black/5 dark:bg-white/5 border border-dashed border-black/15 dark:border-white/15 flex items-center justify-center cursor-pointer overflow-hidden relative group">
                                        {coverImagePath ? (
                                            <img src={coverImagePath.startsWith('http') ? coverImagePath : `${API_BASE_URL}${coverImagePath}`} alt="Cover" className="w-full h-full object-cover rounded-xl" />
                                        ) : isUploadingCover || isGeneratingCover ? (
                                            <div className="w-5 h-5 rounded-full border-2 border-teal-500 border-t-transparent animate-spin" />
                                        ) : (
                                            <ImageIcon size={20} className="text-slate-400" />
                                        )}
                                    </div>
                                    <div className="flex-1 space-y-1.5">
                                        <button
                                            type="button"
                                            onClick={() => fileInputRef.current?.click()}
                                            disabled={isUploadingCover}
                                            className="w-full py-1.5 px-2.5 rounded-lg bg-black/5 dark:bg-white/5 text-[11px] font-semibold flex items-center justify-center gap-1.5"
                                        >
                                            <Upload size={12} />
                                            <span>{isUploadingCover ? 'Uploading...' : 'Upload image'}</span>
                                        </button>
                                        <button
                                            type="button"
                                            onClick={handleGenerateCoverArtwork}
                                            disabled={isGeneratingCover}
                                            className="w-full py-1.5 px-2.5 rounded-lg bg-teal-500/10 text-[11px] font-semibold text-teal-700 dark:text-teal-300 flex items-center justify-center gap-1.5"
                                        >
                                            <Sparkles size={12} />
                                            <span>{isGeneratingCover ? 'Generating...' : 'Prompt image'}</span>
                                        </button>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>

            <div className="p-4 border-t border-black/[0.06] bg-black/[0.02]">
                <button onClick={handleSubmit} disabled={isGenerating || (!topic && !title && !lyrics)} className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 disabled:opacity-50 text-slate-950 font-bold text-xs uppercase flex items-center justify-center gap-2">
                    <Sparkles size={16} />
                    <span>{isGenerating ? 'Synthesizing...' : 'Generate'}</span>
                </button>
            </div>
        </div>
    );
};
