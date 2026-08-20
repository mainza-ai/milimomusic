import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    Mic2,
    Music,
    ChevronDown,
    ChevronUp,
    Sparkles,
    Dices,
    Wand2,
    ArrowRightCircle,
    Settings,
    Mic,
    Cpu,
    Lock,
    Unlock,
    Clock,
    Sliders
} from 'lucide-react';

import { api, voiceApi, type Job, type LLMConfig, type VoiceProfile, type Project } from '../api';
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
    const [activeTab, setActiveTab] = useState<'sound' | 'lyrics'>('sound');
    const [topic, setTopic] = useState('');
    const [style, setStyle] = useState('');
    const [lyrics, setLyrics] = useState('');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [showStructuredCaption, setShowStructuredCaption] = useState(false);
    const [isEnhancing, setIsEnhancing] = useState(false);

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

    // Signal & Sampling Advanced State
    const [duration, setDuration] = useState(() => parseInt(localStorage.getItem('milimo_duration') || '60'));
    const [isEditingDuration, setIsEditingDuration] = useState(false);
    const [durationInputVal, setDurationInputVal] = useState(duration.toString());

    const [temperature, setTemperature] = useState(() => parseFloat(localStorage.getItem('milimo_temperature') || '1.0'));
    const [cfgScale, setCfgScale] = useState(() => parseFloat(localStorage.getItem('milimo_cfg') || '2.0'));
    const [topk, setTopk] = useState(() => parseInt(localStorage.getItem('milimo_topk') || '50'));
    const [topP, setTopP] = useState(() => parseFloat(localStorage.getItem('milimo_topp') || '0.95'));
    const [diffusionSteps, setDiffusionSteps] = useState(25);
    const [seed, setSeed] = useState<number | undefined>(() => {
        const saved = localStorage.getItem('milimo_seed');
        return saved ? parseInt(saved) : undefined;
    });
    const [isSeedLocked, setIsSeedLocked] = useState(() => localStorage.getItem('milimo_seed_locked') === 'true');
    const [audioFidelity, setAudioFidelity] = useState('48k_flac');

    const [lyricsModel, setLyricsModel] = useState(() => localStorage.getItem('milimo_lyrics_model') || (lyricsModels[0] || 'minimax-m3'));

    useEffect(() => {
        if (lyricsModels.length > 0 && (!lyricsModel || !lyricsModels.includes(lyricsModel))) {
            setLyricsModel(lyricsModels[0]);
        }
    }, [lyricsModels]);

    // Apply a producer preset (from the "Ask Producer" flow) to prefill the composer panel
    // so the producer's choices (lyrics, style/tags, structure) are visible and editable.
    useEffect(() => {
        if (!producerPreset) return;
        if (producerPreset.lyrics) setLyrics(producerPreset.lyrics);
        if (producerPreset.tags) setStyle(producerPreset.tags);
        if (producerPreset.structuredCaption?.global_metadata) setGlobalMetadata(producerPreset.structuredCaption.global_metadata);
        if (producerPreset.durationMs) setDuration(producerPreset.durationMs);
    }, [producerPreset]);

    // Keep durationInputVal in sync with duration
    useEffect(() => {
        setDurationInputVal(duration.toString());
        localStorage.setItem('milimo_duration', duration.toString());
    }, [duration]);

    const handleDurationBlur = () => {
        setIsEditingDuration(false);
        const parsed = parseInt(durationInputVal);
        if (!isNaN(parsed)) {
            const clamped = Math.max(5, Math.min(300, parsed));
            setDuration(clamped);
            setDurationInputVal(clamped.toString());
        } else {
            setDurationInputVal(duration.toString());
        }
    };

    const handleDurationKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            handleDurationBlur();
        } else if (e.key === 'Escape') {
            setIsEditingDuration(false);
            setDurationInputVal(duration.toString());
        }
    };

    const handleRandomizeSeed = () => {
        const newSeed = Math.floor(Math.random() * 2147483647);
        setSeed(newSeed);
        localStorage.setItem('milimo_seed', newSeed.toString());
    };

    const handleInspire = async () => {
        setIsEnhancing(true);
        try {
            const result = await api.getInspiration(lyricsModel);
            setTopic(result.topic);
            setStyle(result.tags);
            setGlobalMetadata(`Genre: ${result.tags}\nMood: Inspiring & Dynamic`);
        } catch (e) {
            setTopic("A neon journey through midnight rain");
            setStyle("Synthwave, Dark, Retro, Electronic");
        } finally {
            setIsEnhancing(false);
        }
    };

    const SECTION_TAGS = ['[Intro]', '[Verse 1]', '[Pre-Chorus]', '[Chorus]', '[Verse 2]', '[Bridge]', '[Solo]', '[Outro]'];

    const insertTag = (tag: string) => {
        setLyrics(prev => prev ? `${prev}\n\n${tag}\n` : `${tag}\n`);
    };

    const stylePills = [
        "Pop", "Synthwave", "R&B", "Rock", "Electronic", "Acoustic", "Cinematic", "Drums", "Bass", "Funk"
    ];

    const addStyle = (s: string) => {
        if (!style.includes(s)) {
            setStyle(prev => prev ? `${prev}, ${s}` : s);
        }
    };

    const handleLyricsGen = async () => {
        if (!topic) return;
        try {
            const genLyrics = await onGenerateLyrics(topic, lyricsModel, lyrics.trim(), style);
            setLyrics(genLyrics);
            setActiveTab('lyrics');
        } catch (e: any) {
            alert("Lyrics Generation Failed: " + (e.message || "Unknown error"));
        }
    };

    const handleEnhancePrompt = async () => {
        if (!topic) return;
        setIsEnhancing(true);
        try {
            const result = await api.enhancePrompt(topic, lyricsModel);
            if (result.topic) setTopic(result.topic);
            if (result.tags) setStyle(result.tags);
        } catch (e) {
            console.error("Enhance failed", e);
        } finally {
            setIsEnhancing(false);
        }
    };

    const handleSubmit = () => {
        const finalSeed = isSeedLocked && seed !== undefined ? seed : Math.floor(Math.random() * 2147483647);
        if (!isSeedLocked) {
            setSeed(finalSeed);
        }

        onGenerate({
            lyrics: lyrics,
            topic: topic,
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
            seed: finalSeed
        });
    };

    const formatDurationLabel = (sec: number) => {
        const mins = Math.floor(sec / 60);
        const remSec = sec % 60;
        if (mins > 0) {
            return `${mins}m ${remSec > 0 ? `${remSec}s` : ''} (${sec}s)`;
        }
        return `${sec}s`;
    };

    return (
        <div className="h-full flex flex-col bg-white/70 dark:bg-[#12141c]/85 backdrop-blur-2xl text-slate-800 dark:text-slate-200 select-none overflow-hidden w-full transition-colors duration-200">
            {/* Extension Mode Indicator */}
            {parentJob && (
                <div className="bg-teal-500/10 border-b border-teal-500/20 p-3 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-teal-600 dark:text-teal-400 text-xs font-medium">
                        <ArrowRightCircle className="w-4 h-4" />
                        <span>Extending: {parentJob.title || "Untitled Track"}</span>
                    </div>
                    {onClearParentJob && (
                        <button
                            onClick={onClearParentJob}
                            className="text-teal-600 dark:text-teal-400 hover:text-teal-800 text-xs underline"
                        >
                            Cancel
                        </button>
                    )}
                </div>
            )}

            {/* Active Project Banner */}
            {activeProject && (
                <div className="bg-teal-500/10 border-b border-teal-500/20 px-3 py-2 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-teal-700 dark:text-teal-300 text-xs font-semibold truncate">
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/20 text-teal-700 dark:text-teal-300 font-bold truncate max-w-[150px]">
                            📁 {activeProject.name}
                        </span>
                        <span className="text-[10px] text-slate-400 font-mono">
                            {activeProject.bpm} BPM • {activeProject.key_signature}
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

            {/* Sidebar Header */}
            <div className="p-4 border-b border-black/[0.06] dark:border-white/[0.08] bg-black/[0.02] dark:bg-white/[0.02] flex items-center justify-between">
                <div className="flex items-center space-x-2">
                    <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 dark:text-slate-200">
                        Composer
                    </h2>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 font-semibold border border-teal-500/20">
                        Studio Controls
                    </span>
                </div>

                <div className="flex items-center gap-1">
                    <button
                        onClick={handleInspire}
                        disabled={isEnhancing}
                        className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-500 dark:text-slate-400 hover:text-teal-600 dark:hover:text-teal-300 transition-colors"
                        title="Surprise Me (AI Inspiration)"
                    >
                        <Dices size={15} />
                    </button>
                    <button
                        onClick={() => setIsModelsManagerOpen(true)}
                        className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-500 dark:text-slate-400 hover:text-teal-600 dark:hover:text-teal-300 transition-colors"
                        title="Model Architecture Manager"
                    >
                        <Cpu size={15} />
                    </button>
                    <button
                        onClick={() => setIsSettingsOpen(true)}
                        className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-500 dark:text-slate-400 hover:text-teal-600 dark:hover:text-teal-300 transition-colors"
                        title="LLM Settings & Providers"
                    >
                        <Settings size={15} />
                    </button>
                </div>
            </div>

            {/* Models Modals */}
            <LLMSettingsModal
                isOpen={isSettingsOpen}
                currentConfig={llmConfig}
                onConfigUpdate={loadLlmConfig}
                onClose={() => {
                    setIsSettingsOpen(false);
                    loadLlmConfig();
                }}
            />
            <VoiceStudioModal
                isOpen={isVoiceStudioOpen}
                onClose={() => {
                    setIsVoiceStudioOpen(false);
                    loadVoiceProfiles();
                }}
            />
            <ModelsManagerModal
                isOpen={isModelsManagerOpen}
                onClose={() => setIsModelsManagerOpen(false)}
            />

            {/* Model & Voice Quick Pickers Bar */}
            <div className="px-4 py-2.5 bg-black/[0.02] dark:bg-white/[0.02] border-b border-black/[0.06] dark:border-white/[0.08] flex items-center justify-between gap-2">
                <div className="flex items-center space-x-1.5 flex-1 min-w-0">
                    <Cpu size={13} className="text-teal-500 flex-shrink-0" />
                    <select
                        value={modelProvider}
                        onChange={(e) => setModelProvider(e.target.value)}
                        className="apple-input py-1 text-[11px] font-mono flex-1 min-w-0"
                    >
                        <option value="minimax_music3">MiniMax Music 3 (48kHz DiT)</option>
                        <option value="heartmula">HeartMuLa (48kHz Transformer)</option>
                    </select>
                </div>

                <div className="flex items-center space-x-1.5 flex-1 min-w-0">
                    <Mic size={13} className="text-cyan-500 flex-shrink-0" />
                    <select
                        value={selectedVoiceProfile}
                        onChange={(e) => {
                            if (e.target.value === '__add_new__') {
                                setIsVoiceStudioOpen(true);
                            } else {
                                setSelectedVoiceProfile(e.target.value);
                            }
                        }}
                        className="apple-input py-1 text-[11px] font-mono flex-1 min-w-0"
                    >
                        <option value="">Default AI Voice</option>
                        {voiceProfiles.map(p => (
                            <option key={p.id} value={p.id}>👤 {p.name}</option>
                        ))}
                        <option value="__add_new__">+ Train New Voice...</option>
                    </select>
                </div>
            </div>

            {/* Tab Navigation (Sound vs Lyrics) */}
            <div className="flex border-b border-black/[0.06] dark:border-white/[0.08] bg-black/[0.01] dark:bg-white/[0.01]">
                <button
                    onClick={() => setActiveTab('sound')}
                    className={`flex-1 py-2.5 text-xs font-bold transition-all border-b-2 flex items-center justify-center space-x-1.5 ${
                        activeTab === 'sound'
                            ? 'border-teal-500 text-teal-600 dark:text-teal-400 bg-teal-500/5'
                            : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                    }`}
                >
                    <Music size={13} />
                    <span>Sound & Style</span>
                </button>
                <button
                    onClick={() => setActiveTab('lyrics')}
                    className={`flex-1 py-2.5 text-xs font-bold transition-all border-b-2 flex items-center justify-center space-x-1.5 ${
                        activeTab === 'lyrics'
                            ? 'border-teal-500 text-teal-600 dark:text-teal-400 bg-teal-500/5'
                            : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                    }`}
                >
                    <Mic2 size={13} />
                    <span>Lyrics & Structure</span>
                </button>
            </div>

            {/* Main Form Fields */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {activeTab === 'sound' && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                        {/* Topic / Prompt */}
                        <div className="space-y-1.5">
                            <div className="flex items-center justify-between">
                                <label className="text-xs font-bold text-slate-700 dark:text-slate-300">
                                    Track Concept & Mood
                                </label>
                                <button
                                    onClick={handleEnhancePrompt}
                                    disabled={isEnhancing || !topic}
                                    className="text-[10px] text-teal-600 dark:text-teal-400 hover:underline flex items-center gap-1 font-semibold disabled:opacity-40"
                                >
                                    <Wand2 size={11} />
                                    <span>Enhance with LLM</span>
                                </button>
                            </div>
                            <textarea
                                value={topic}
                                onChange={(e) => setTopic(e.target.value)}
                                placeholder="Describe the atmosphere, genre, tempo, instruments, and emotional arc..."
                                className="w-full h-20 apple-input resize-none text-xs leading-relaxed"
                            />
                        </div>

                        {/* Style / Tags */}
                        <div className="space-y-1.5">
                            <label className="text-xs font-bold text-slate-700 dark:text-slate-300">
                                Musical Style & Instrumentation
                            </label>
                            <textarea
                                value={style}
                                onChange={(e) => setStyle(e.target.value)}
                                placeholder="e.g. 'Funk, Slap Bass, Studio Acoustic Drums, Rhodes Piano, Vocal Hooks'"
                                className="w-full h-16 apple-input resize-none font-mono text-[11px]"
                            />
                            {/* Style Pills */}
                            <div className="flex flex-wrap gap-1 mt-1">
                                {stylePills.map(s => (
                                    <button
                                        key={s}
                                        onClick={() => addStyle(s)}
                                        className="text-[10px] bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 border border-black/[0.06] dark:border-white/5 rounded-md px-2 py-0.5 text-slate-600 dark:text-slate-400 hover:text-teal-600 dark:hover:text-teal-300 transition-all shadow-sm"
                                    >
                                        + {s}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Structured Captions Expander */}
                        <div className="border-t border-black/[0.06] dark:border-white/[0.08] pt-3">
                            <button
                                onClick={() => setShowStructuredCaption(!showStructuredCaption)}
                                className="flex items-center justify-between w-full text-xs font-bold text-teal-600 dark:text-teal-400 uppercase tracking-wider"
                            >
                                <span>Structured Caption Spec (MiniMax)</span>
                                {showStructuredCaption ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                            </button>

                            {showStructuredCaption && (
                                <div className="space-y-2.5 pt-3">
                                    <div>
                                        <label className="text-[10px] text-slate-500 dark:text-slate-400 block mb-1">Global Metadata</label>
                                        <input
                                            value={globalMetadata}
                                            onChange={(e) => setGlobalMetadata(e.target.value)}
                                            className="w-full apple-input font-mono text-[11px]"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-[10px] text-slate-500 dark:text-slate-400 block mb-1">Vocal Details</label>
                                        <input
                                            value={vocalDetails}
                                            onChange={(e) => setVocalDetails(e.target.value)}
                                            className="w-full apple-input font-mono text-[11px]"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-[10px] text-slate-500 dark:text-slate-400 block mb-1">Arrangement</label>
                                        <input
                                            value={arrangement}
                                            onChange={(e) => setArrangement(e.target.value)}
                                            className="w-full apple-input font-mono text-[11px]"
                                        />
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Advanced Signal & Sampling Controls */}
                        <div className="border-t border-black/[0.06] dark:border-white/[0.08] pt-3">
                            <button
                                onClick={() => setShowAdvanced(!showAdvanced)}
                                className="flex items-center justify-between w-full text-xs font-bold text-slate-600 dark:text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 uppercase tracking-wider"
                            >
                                <span className="flex items-center gap-1.5">
                                    <Sliders size={13} className="text-teal-500" />
                                    <span>Signal & Sampling Controls</span>
                                </span>
                                {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                            </button>

                            {showAdvanced && (
                                <div className="space-y-4 pt-3 text-xs animate-fade-in">
                                    {/* 1. Track Duration with Click-to-Edit */}
                                    <div className="space-y-1.5 bg-black/[0.02] dark:bg-white/[0.02] p-2.5 rounded-xl border border-black/[0.04] dark:border-white/5">
                                        <div className="flex items-center justify-between">
                                            <span className="font-bold text-slate-700 dark:text-slate-300 flex items-center gap-1">
                                                <Clock size={12} className="text-teal-500" />
                                                Generation Duration
                                            </span>
                                            {/* Click into time to edit actual duration */}
                                            {isEditingDuration ? (
                                                <div className="flex items-center gap-1">
                                                    <input
                                                        type="number"
                                                        min="5"
                                                        max="300"
                                                        autoFocus
                                                        value={durationInputVal}
                                                        onChange={(e) => setDurationInputVal(e.target.value)}
                                                        onBlur={handleDurationBlur}
                                                        onKeyDown={handleDurationKeyDown}
                                                        className="w-16 px-1.5 py-0.5 bg-teal-500/10 border border-teal-500 rounded text-right font-mono font-bold text-teal-700 dark:text-teal-300 text-xs focus:outline-none"
                                                    />
                                                    <span className="text-[10px] font-mono text-slate-400">sec</span>
                                                </div>
                                            ) : (
                                                <button
                                                    onClick={() => setIsEditingDuration(true)}
                                                    className="px-2 py-0.5 rounded-md bg-teal-500/10 hover:bg-teal-500/20 text-teal-700 dark:text-teal-300 font-mono font-bold border border-teal-500/20 transition-all text-xs"
                                                    title="Click to manually type exact duration"
                                                >
                                                    {formatDurationLabel(duration)} ✎
                                                </button>
                                            )}
                                        </div>

                                        <input
                                            type="range"
                                            min="5"
                                            max="300"
                                            step="1"
                                            value={duration}
                                            onChange={(e) => setDuration(Number(e.target.value))}
                                            className="w-full accent-teal-500 h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer"
                                        />

                                        {/* Quick Duration Presets */}
                                        <div className="flex flex-wrap gap-1 pt-1">
                                            {[15, 30, 60, 90, 120, 180, 300].map(s => (
                                                <button
                                                    key={s}
                                                    onClick={() => setDuration(s)}
                                                    className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold transition-all ${
                                                        duration === s
                                                            ? 'bg-teal-500 text-slate-950 font-bold shadow-sm'
                                                            : 'bg-black/[0.04] dark:bg-white/5 text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
                                                    }`}
                                                >
                                                    {s < 60 ? `${s}s` : s === 300 ? '5m' : `${s/60}m`}
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    {/* 2. Sampling Temperature */}
                                    <div className="space-y-1 bg-black/[0.02] dark:bg-white/[0.02] p-2.5 rounded-xl border border-black/[0.04] dark:border-white/5">
                                        <div className="flex justify-between font-mono">
                                            <span className="text-slate-700 dark:text-slate-300 font-bold">Temperature (Entropy)</span>
                                            <span className="text-teal-600 dark:text-teal-400 font-bold">{temperature.toFixed(2)}</span>
                                        </div>
                                        <input
                                            type="range"
                                            min="0.1"
                                            max="2.0"
                                            step="0.05"
                                            value={temperature}
                                            onChange={(e) => {
                                                const val = parseFloat(e.target.value);
                                                setTemperature(val);
                                                localStorage.setItem('milimo_temperature', val.toString());
                                            }}
                                            className="w-full accent-teal-500 h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer"
                                        />
                                        <div className="flex justify-between text-[9px] text-slate-400 font-mono pt-0.5">
                                            <button onClick={() => setTemperature(0.5)} className="hover:text-teal-500">0.5 Strict</button>
                                            <button onClick={() => setTemperature(0.8)} className="hover:text-teal-500">0.8 Studio</button>
                                            <button onClick={() => setTemperature(1.0)} className="hover:text-teal-500 font-bold">1.0 Standard</button>
                                            <button onClick={() => setTemperature(1.3)} className="hover:text-teal-500">1.3 Creative</button>
                                            <button onClick={() => setTemperature(1.6)} className="hover:text-teal-500">1.6 Wild</button>
                                        </div>
                                    </div>

                                    {/* 3. Classifier-Free Guidance (CFG Scale) */}
                                    <div className="space-y-1 bg-black/[0.02] dark:bg-white/[0.02] p-2.5 rounded-xl border border-black/[0.04] dark:border-white/5">
                                        <div className="flex justify-between font-mono">
                                            <span className="text-slate-700 dark:text-slate-300 font-bold">CFG Scale (Prompt Fidelity)</span>
                                            <span className="text-cyan-600 dark:text-cyan-400 font-bold">{cfgScale.toFixed(1)}</span>
                                        </div>
                                        <input
                                            type="range"
                                            min="1.0"
                                            max="5.0"
                                            step="0.1"
                                            value={cfgScale}
                                            onChange={(e) => {
                                                const val = parseFloat(e.target.value);
                                                setCfgScale(val);
                                                localStorage.setItem('milimo_cfg', val.toString());
                                            }}
                                            className="w-full accent-cyan-500 h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer"
                                        />
                                        <div className="flex justify-between text-[9px] text-slate-400 font-mono pt-0.5">
                                            <button onClick={() => setCfgScale(1.2)} className="hover:text-cyan-500">1.2 Loose</button>
                                            <button onClick={() => setCfgScale(1.5)} className="hover:text-cyan-500">1.5 Balanced</button>
                                            <button onClick={() => setCfgScale(2.0)} className="hover:text-cyan-500 font-bold">2.0 Faithful</button>
                                            <button onClick={() => setCfgScale(3.0)} className="hover:text-cyan-500">3.0 High</button>
                                        </div>
                                    </div>

                                    {/* 4. Top-K & Top-P Sampling Matrix */}
                                    <div className="grid grid-cols-2 gap-2 bg-black/[0.02] dark:bg-white/[0.02] p-2.5 rounded-xl border border-black/[0.04] dark:border-white/5">
                                        <div className="space-y-1">
                                            <div className="flex justify-between font-mono text-[11px]">
                                                <span className="text-slate-600 dark:text-slate-400 font-bold">Top-K</span>
                                                <span className="text-teal-600 dark:text-teal-400 font-bold">{topk}</span>
                                            </div>
                                            <input
                                                type="range"
                                                min="10"
                                                max="250"
                                                step="5"
                                                value={topk}
                                                onChange={(e) => {
                                                    const val = parseInt(e.target.value);
                                                    setTopk(val);
                                                    localStorage.setItem('milimo_topk', val.toString());
                                                }}
                                                className="w-full accent-teal-500 h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer"
                                            />
                                        </div>

                                        <div className="space-y-1">
                                            <div className="flex justify-between font-mono text-[11px]">
                                                <span className="text-slate-600 dark:text-slate-400 font-bold">Top-P (Nucleus)</span>
                                                <span className="text-teal-600 dark:text-teal-400 font-bold">{topP.toFixed(2)}</span>
                                            </div>
                                            <input
                                                type="range"
                                                min="0.5"
                                                max="1.0"
                                                step="0.01"
                                                value={topP}
                                                onChange={(e) => {
                                                    const val = parseFloat(e.target.value);
                                                    setTopP(val);
                                                    localStorage.setItem('milimo_topp', val.toString());
                                                }}
                                                className="w-full accent-teal-500 h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer"
                                            />
                                        </div>
                                    </div>

                                    {/* 5. Flow Matching DiT Steps */}
                                    <div className="space-y-1.5 bg-black/[0.02] dark:bg-white/[0.02] p-2.5 rounded-xl border border-black/[0.04] dark:border-white/5">
                                        <div className="flex justify-between font-mono text-[11px]">
                                            <span className="text-slate-700 dark:text-slate-300 font-bold">DiT Diffusion Steps</span>
                                            <span className="text-teal-600 dark:text-teal-400 font-bold">{diffusionSteps} steps</span>
                                        </div>
                                        <div className="grid grid-cols-3 gap-1">
                                            {[
                                                { steps: 15, label: '15 (Draft)' },
                                                { steps: 25, label: '25 (Studio)' },
                                                { steps: 35, label: '35 (Ultra)' }
                                            ].map(opt => (
                                                <button
                                                    key={opt.steps}
                                                    onClick={() => setDiffusionSteps(opt.steps)}
                                                    className={`py-1 rounded-lg text-[10px] font-mono font-bold transition-all ${
                                                        diffusionSteps === opt.steps
                                                            ? 'bg-teal-500 text-slate-950 shadow-sm'
                                                            : 'bg-black/[0.04] dark:bg-white/5 text-slate-500 hover:text-slate-900'
                                                    }`}
                                                >
                                                    {opt.label}
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    {/* 6. Seed & Reproducibility */}
                                    <div className="flex items-center justify-between gap-2 bg-black/[0.02] dark:bg-white/[0.02] p-2.5 rounded-xl border border-black/[0.04] dark:border-white/5">
                                        <div className="flex items-center gap-1.5">
                                            <button
                                                onClick={() => {
                                                    const next = !isSeedLocked;
                                                    setIsSeedLocked(next);
                                                    localStorage.setItem('milimo_seed_locked', next.toString());
                                                }}
                                                className={`p-1.5 rounded-lg border transition-all ${
                                                    isSeedLocked
                                                        ? 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20'
                                                        : 'bg-black/[0.04] dark:bg-white/5 text-slate-400 border-transparent'
                                                }`}
                                                title={isSeedLocked ? 'Seed Locked for Reproducible Renders' : 'Click to Lock Seed'}
                                            >
                                                {isSeedLocked ? <Lock size={12} /> : <Unlock size={12} />}
                                            </button>
                                            <span className="text-[11px] font-mono text-slate-600 dark:text-slate-400">
                                                Seed: <span className="font-bold text-slate-900 dark:text-slate-100">{seed ?? 'Random'}</span>
                                            </span>
                                        </div>

                                        <button
                                            onClick={handleRandomizeSeed}
                                            className="px-2 py-1 rounded-lg bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-[10px] font-mono font-bold flex items-center gap-1 text-slate-600 dark:text-slate-300 transition-colors"
                                        >
                                            <Dices size={11} />
                                            <span>Roll</span>
                                        </button>
                                    </div>

                                    {/* 7. Audio Output Fidelity */}
                                    <div className="space-y-1 bg-black/[0.02] dark:bg-white/[0.02] p-2.5 rounded-xl border border-black/[0.04] dark:border-white/5">
                                        <span className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider block mb-1">
                                            Audio Master Format
                                        </span>
                                        <div className="grid grid-cols-3 gap-1">
                                            {[
                                                { id: '48k_flac', label: '48kHz FLAC' },
                                                { id: '44k_wav', label: '44.1k WAV' },
                                                { id: '320k_mp3', label: '320k MP3' }
                                            ].map(f => (
                                                <button
                                                    key={f.id}
                                                    onClick={() => setAudioFidelity(f.id)}
                                                    className={`py-1 rounded-lg text-[10px] font-mono font-bold transition-all ${
                                                        audioFidelity === f.id
                                                            ? 'bg-cyan-500 text-slate-950 shadow-sm'
                                                            : 'bg-black/[0.04] dark:bg-white/5 text-slate-500 hover:text-slate-900'
                                                    }`}
                                                >
                                                    {f.label}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}

                {activeTab === 'lyrics' && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex flex-col gap-3">
                        {/* Section Tags Helper Bar */}
                        <div className="space-y-1">
                            <span className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                                Section Tags (MiniMax Structured)
                            </span>
                            <div className="flex flex-wrap gap-1">
                                {SECTION_TAGS.map(t => (
                                    <button
                                        key={t}
                                        onClick={() => insertTag(t)}
                                        className="text-[10px] font-mono bg-teal-500/10 hover:bg-teal-500/20 text-teal-700 dark:text-teal-300 border border-teal-500/20 rounded-md px-2 py-0.5 transition-colors shadow-sm"
                                    >
                                        + {t}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* LLM Model Selection Pill */}
                        <div className="flex items-center justify-between gap-2 p-2 bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/5 rounded-xl">
                            <span className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider flex items-center gap-1">
                                <Cpu size={11} className="text-teal-500" />
                                LLM Model
                            </span>
                            <select
                                value={lyricsModel}
                                onChange={(e) => {
                                    setLyricsModel(e.target.value);
                                    localStorage.setItem('milimo_lyrics_model', e.target.value);
                                }}
                                className="apple-input py-1 text-[11px] font-mono max-w-[210px] truncate"
                            >
                                {lyricsModels.map(m => (
                                    <option key={m} value={m}>{m}</option>
                                ))}
                            </select>
                        </div>

                        {/* Lyrics Generate CTA */}
                        <button
                            onClick={handleLyricsGen}
                            disabled={isGeneratingLyrics || !topic}
                            className="w-full py-2 bg-gradient-to-r from-teal-500/15 to-cyan-500/15 hover:from-teal-500/25 hover:to-cyan-500/25 text-teal-700 dark:text-teal-300 border border-teal-500/20 rounded-xl text-xs font-bold flex items-center justify-center gap-1.5 transition-all shadow-sm"
                        >
                            <Sparkles size={14} />
                            <span>{isGeneratingLyrics ? 'Generating Lyrics...' : 'AI Co-Writer: Write Lyrics'}</span>
                        </button>

                        <textarea
                            value={lyrics}
                            onChange={(e) => setLyrics(e.target.value)}
                            placeholder="[Intro]&#10;Echoes in the silence...&#10;&#10;[Verse 1]&#10;Stars are burning far away...&#10;&#10;[Chorus]&#10;We ignite the night..."
                            className="flex-1 w-full min-h-[220px] apple-input resize-none font-mono text-[11px] leading-relaxed"
                        />
                    </motion.div>
                )}
            </div>

            {/* Footer / Generate Button */}
            <div className="p-4 border-t border-black/[0.06] dark:border-white/[0.08] bg-black/[0.02] dark:bg-white/[0.02]">
                <button
                    onClick={handleSubmit}
                    disabled={isGenerating || !topic}
                    className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 disabled:opacity-50 text-slate-950 font-bold text-xs uppercase tracking-wider flex items-center justify-center space-x-2 shadow-lg shadow-teal-500/20 transition-all active:scale-[0.98]"
                >
                    <Sparkles size={16} />
                    <span>{isGenerating ? 'Synthesizing Master...' : 'Generate & Transcribe Track'}</span>
                </button>
            </div>
        </div>
    );
};
