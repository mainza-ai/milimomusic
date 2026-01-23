import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic2, Music, ChevronDown, ChevronUp, Sparkles, Plus, Dices, Wand2, ArrowRightCircle, RefreshCw, Settings, Send, MessageSquare } from 'lucide-react';
import { GradientButton } from './ui/GradientButton';

import { api, type Job, type LLMConfig } from '../api';
import { LLMSettingsModal } from './LLMSettingsModal';

interface ComposerSidebarProps {
    onGenerate: (data: CompositionData) => void;
    isGenerating: boolean;
    lyricsModels: string[];
    onGenerateLyrics: (topic: string, model: string, currentLyrics?: string, tags?: string) => Promise<string>;
    isGeneratingLyrics: boolean;
    currentJobId?: string;
    onCancel?: (jobId: string) => void;
    parentJob?: Job; // Phase 9: For extension (Full Job Object)
    onClearParentJob?: () => void; // To clear extension mode
    onRefreshModels?: () => void;
}

export interface CompositionData {
    lyrics: string;
    topic: string;
    tags: string; // New "Sound" description
    durationMs: number;
    temperature: number;
    cfgScale: number;
    topk: number;
    llmModel: string;
}

export const ComposerSidebar: React.FC<ComposerSidebarProps> = ({
    onGenerate,
    isGenerating,
    lyricsModels,
    onGenerateLyrics,
    isGeneratingLyrics,
    currentJobId,
    onCancel,
    parentJob, // Phase 9: For extension
    onClearParentJob, // To clear extension mode
    onRefreshModels
}) => {
    const [activeTab, setActiveTab] = useState<'sound' | 'lyrics'>('sound');
    const [topic, setTopic] = useState('');
    const [style, setStyle] = useState(''); // New Style Input
    const [lyrics, setLyrics] = useState('');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [isEnhancing, setIsEnhancing] = useState(false);

    // Lyrics Chat State
    const [chatInput, setChatInput] = useState('');
    const [chatMessages, setChatMessages] = useState<{ role: 'user' | 'ai'; content: string }[]>([]);
    const [isChatting, setIsChatting] = useState(false);
    const [showChat, setShowChat] = useState(false); // Toggle between full editor and split/chat mode

    // LLM Config State
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [llmConfig, setLlmConfig] = useState<LLMConfig>({});

    const loadLlmConfig = async () => {
        try {
            const cfg = await api.getLLMConfig();
            setLlmConfig(cfg);
            onRefreshModels?.();
        } catch (e) {
            console.error("Failed to load LLM config", e);
        }
    };

    // Initial Startup Logic: auto-detect Ollama
    useEffect(() => {
        const init = async () => {
            // 1. Load initial config
            let cfg: LLMConfig = {};
            try {
                cfg = await api.getLLMConfig();
                setLlmConfig(cfg);
                onRefreshModels?.();
            } catch (e) {
                console.error("Failed to load initial config", e);
                return;
            }

            // 2. Priority Check: If Ollama is running locally, we might want to default to it
            // ONLY if the user hasn't arguably "set" something else? 
            // The requirement was "automatically use an Ollama model by default if Ollama is running".
            // Implementation: We check availability. If available, we switch to it. 
            // Crucially, this ONLY happens on MOUNT (Refresh/App Start), not when user changes settings manually later.
            try {
                const ollamaModels = await api.fetchModels({ provider: 'ollama', ollama: cfg.ollama || { base_url: 'http://localhost:11434' } });

                if (ollamaModels.length > 0) {
                    console.log("Startup: Ollama detected running.");

                    // Determine target model
                    let targetModel = ollamaModels[0];
                    if (cfg.provider === 'ollama' && cfg.ollama?.model && ollamaModels.includes(cfg.ollama.model)) {
                        targetModel = cfg.ollama.model;
                    }

                    // We switch to Ollama IF:
                    // 1. Provider is not set at all
                    // 2. OR Provider is set to Ollama
                    // 3. OR (Optionally) we act aggressive and always switch on startup. 
                    // current implementation was aggressive. I will keep it aggressive FOR STARTUP only.

                    const newConfig = {
                        ...cfg,
                        provider: 'ollama',
                        ollama: {
                            ...cfg.ollama,
                            model: targetModel
                        }
                    };

                    // Only update if different
                    if (JSON.stringify(newConfig) !== JSON.stringify(cfg)) {
                        console.log("Startup: switching default provider to Ollama");
                        await api.updateLLMConfig(newConfig);
                        setLlmConfig(newConfig);
                        setLyricsModel(targetModel);
                        onRefreshModels?.();
                    }
                }
            } catch (e) {
                console.log("Startup: Ollama not detected, skipping auto-switch.");
            }
        };

        init();
    }, []);

    useEffect(() => {
        const handleLog = (e: any) => {
            const data = e.detail;
            if (data.msg) {
                setLogs(prev => {
                    const newLogs = [...prev, `> ${data.msg} `];
                    return newLogs.slice(-6); // Keep last 6 lines
                });
            }
        };
        if (isGenerating) {
            window.addEventListener('milimo_progress', handleLog);
        } else {
            setLogs([]); // Clear when done
        }
        return () => window.removeEventListener('milimo_progress', handleLog);
    }, [isGenerating]);

    // Advanced State
    const [duration, setDuration] = useState(() => parseInt(localStorage.getItem('milimo_duration') || '30'));
    const [temperature, setTemperature] = useState(() => parseFloat(localStorage.getItem('milimo_temperature') || '1.0'));
    const [cfgScale, setCfgScale] = useState(() => parseFloat(localStorage.getItem('milimo_cfg') || '1.5'));
    const [topk, setTopk] = useState(() => parseInt(localStorage.getItem('milimo_topk') || '50'));
    const [lyricsModel, setLyricsModel] = useState(() => localStorage.getItem('milimo_lyrics_model') || (lyricsModels[0] || 'llama3.2:3b-instruct-fp16'));

    // Save settings on change
    React.useEffect(() => {
        localStorage.setItem('milimo_duration', duration.toString());
        localStorage.setItem('milimo_temperature', temperature.toString());
        localStorage.setItem('milimo_cfg', cfgScale.toString());
        localStorage.setItem('milimo_topk', topk.toString());
        localStorage.setItem('milimo_lyrics_model', lyricsModel);
    }, [duration, temperature, cfgScale, topk, lyricsModel]);

    // Auto-select first available model when they load (if not set)
    React.useEffect(() => {
        // Prevent race condition: If the current selection matches our global config Intent, 
        // DO NOT reset it just because the list is stale or loading.
        const providerConfig = llmConfig.provider ? (llmConfig[llmConfig.provider as keyof LLMConfig] as any) : null;
        const globalModel = providerConfig?.model;
        if (globalModel && lyricsModel === globalModel) {
            return;
        }



        if (lyricsModels.length > 0 && !lyricsModels.includes(lyricsModel)) {
            setLyricsModel(lyricsModels[0]);
        }
    }, [lyricsModels, lyricsModel, llmConfig]);

    // Sync local lyricsModel with global config when config updates (e.g. user changes provider in settings)
    React.useEffect(() => {
        if (!llmConfig.provider) return;
        const providerConfig = llmConfig[llmConfig.provider as keyof LLMConfig] as any;
        const globalModel = providerConfig?.model;

        // If we have a configured model for the active provider, use it
        if (globalModel) {
            setLyricsModel(globalModel);
        }
    }, [llmConfig]);


    const handleInspire = async () => {
        setIsEnhancing(true);
        try {
            const result = await api.getInspiration(lyricsModel);
            setTopic(result.topic);
            setStyle(result.tags);
            // Default params for a new idea
            setDuration(30);
        } catch (e) {
            console.error(e);
            // Fallback
            setTopic("A cyberpunk detective chasing a suspect in rain");
            setStyle("Synthwave, Dark, Retro, Electronic");
        } finally {
            setIsEnhancing(false);
        }
    };

    // Curated HeartMuLa Tags
    const HEARTMULA_TAGS = [
        "Warm", "Reflection", "Pop", "Cafe", "R&B", "Keyboard", "Regret", "Drum machine",
        "Electric guitar", "Synthesizer", "Soft", "Energetic", "Electronic", "Self-discovery",
        "Sad", "Ballad", "Longing", "Meditation", "Faith", "Acoustic", "Peaceful", "Wedding",
        "Piano", "Strings", "Acoustic guitar", "Romantic", "Drums", "Emotional", "Walking",
        "Hope", "Hopeful", "Powerful", "Epic", "Driving", "Rock"
    ];

    const getRandomTags = (count: number) => {
        const shuffled = [...HEARTMULA_TAGS].sort(() => 0.5 - Math.random());
        return shuffled.slice(0, count);
    };

    const [stylePills, setStylePills] = useState<string[]>([]);
    const [isLoadingStyles, setIsLoadingStyles] = useState(false);

    useEffect(() => {
        // Initial load
        setStylePills(getRandomTags(12));
    }, []);

    const refreshStyles = () => {
        setIsLoadingStyles(true);
        // Simulate "loading" feel briefly
        setTimeout(() => {
            setStylePills(getRandomTags(12));
            setIsLoadingStyles(false);
        }, 300);
    };

    const addStyle = (s: string) => {
        if (style.includes(s)) return;
        setStyle(prev => prev ? `${prev}, ${s} ` : s);
    };

    const handleChatSubmit = async () => {
        if (!chatInput.trim()) return;
        const userMsg = chatInput;
        setChatInput('');

        // Optimistic UI
        setChatMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        setIsChatting(true);

        try {
            const result = await api.chatLyrics(lyrics, userMsg, lyricsModel, topic, style);

            setLyrics(result.lyrics); // Update editor
            setChatMessages(prev => [...prev, { role: 'ai', content: result.message || "I've updated the lyrics for you." }]);
        } catch (e: any) {
            console.error(e);
            setChatMessages(prev => [...prev, { role: 'ai', content: "Sorry, I encountered an error updating the lyrics." }]);
        } finally {
            setIsChatting(false);
        }
    };

    const handleLyricsGen = async () => {
        if (!topic) return;
        try {
            // Pass current lyrics as seed if they exist
            const seed = lyrics.trim();
            const genLyrics = await onGenerateLyrics(topic, lyricsModel, seed, style);
            setLyrics(genLyrics);
            setActiveTab('lyrics');
        } catch (e: any) {
            console.error(e);
            alert("Lyrics Generation Failed: " + (e.response?.data?.detail || e.message || "Unknown error. Is Ollama running?"));
        }
    };

    // Phase 9: Auto-populate from parentJob
    useEffect(() => {
        if (parentJob) {
            setTopic(parentJob.prompt);
            setLyrics(parentJob.lyrics || "");
            if (parentJob.tags) {
                setStyle(parentJob.tags);
            }
            setActiveTab('sound');
        }
    }, [parentJob]);

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
        onGenerate({
            lyrics: lyrics,
            topic: topic,
            tags: style,  // Use the SEPARATE style input
            durationMs: duration * 1000,
            temperature,
            cfgScale,
            topk,
            llmModel: lyricsModel
        });
    };

    return (
        <div className="h-full flex flex-col bg-white/60 backdrop-blur-3xl border-l border-white/50 shadow-2xl overflow-hidden glass-panel w-full md:w-[400px]">
            {/* Phase 9: Extension Mode Indicator */}
            {parentJob && (
                <div className="bg-[#00F0FF]/10 border-b border-[#00F0FF]/20 p-3 flex items-center justify-between animate-in slide-in-from-top-2">
                    <div className="flex items-center gap-2 text-[#00F0FF] text-sm font-medium">
                        <ArrowRightCircle className="w-4 h-4" />
                        <span>Extending: {parentJob.title || "Untitled Track"}</span>
                    </div>
                    {onClearParentJob && (
                        <button
                            onClick={onClearParentJob}
                            className="text-xs text-cyan-600 hover:text-cyan-800 hover:bg-cyan-500/10 px-2 py-1 rounded transition-colors"
                        >
                            Cancel
                        </button>
                    )}
                </div>
            )}

            {/* Header */}
            <div className="p-6 border-b border-white/30 bg-white/20 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <img src="/milimo_logo.png" alt="Milimo Music" className="w-10 h-10 object-contain drop-shadow-md" />
                    <div>
                        <h2 className="text-xl font-bold text-slate-800 tracking-tighter font-display leading-none">
                            MILIMO MUSIC
                        </h2>
                        <div className="flex items-center gap-2 mt-1">
                            <span className="text-[10px] font-mono text-cyan-600 uppercase tracking-widest bg-cyan-50 px-1.5 py-0.5 rounded-full border border-cyan-100/50">v3.1</span>
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    <button
                        onClick={handleInspire}
                        disabled={isEnhancing}
                        className="p-2 rounded-full hover:bg-white/50 text-slate-400 hover:text-cyan-600 transition-colors border border-transparent hover:border-white/50 shadow-sm hover:shadow-md group disabled:opacity-50"
                        title="Inspire Me (AI Brainstorm)"
                    >
                        {isEnhancing ? (
                            <span className="w-5 h-5 flex items-center justify-center animate-spin text-cyan-500">
                                <Wand2 className="w-4 h-4" />
                            </span>
                        ) : (
                            <Dices className="w-5 h-5 group-hover:rotate-180 transition-transform duration-500" />
                        )}
                    </button>
                    <button
                        onClick={() => {
                            setTopic('');
                            setLyrics('');
                            setStyle('');
                            setDuration(30);
                            setChatMessages([]);
                            setChatInput('');
                            setShowChat(false);
                        }}
                        className="p-2 rounded-full hover:bg-white/50 text-slate-400 hover:text-cyan-600 transition-colors border border-transparent hover:border-white/50 shadow-sm hover:shadow-md"
                        title="New Track (Reset Form)"
                    >
                        <Plus className="w-5 h-5" />
                    </button>
                    <button
                        onClick={() => setIsSettingsOpen(true)}
                        className="p-2 rounded-full hover:bg-white/50 text-slate-400 hover:text-cyan-600 transition-colors border border-transparent hover:border-white/50 shadow-sm hover:shadow-md"
                        title="LLM Settings"
                    >
                        <Settings className="w-5 h-5" />
                    </button>
                </div>
            </div>

            <LLMSettingsModal
                isOpen={isSettingsOpen}
                onClose={() => setIsSettingsOpen(false)}
                currentConfig={llmConfig}
                onConfigUpdate={loadLlmConfig}
            />

            {/* Tabs */}
            <div className="flex p-2 gap-2 border-b border-white/30 bg-slate-50/50">
                <button
                    onClick={() => setActiveTab('sound')}
                    className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all duration-200 ${activeTab === 'sound' ? 'bg-white shadow-sm ring-1 ring-black/5 text-cyan-700' : 'text-slate-500 hover:bg-white/40'} `}
                >
                    <div className="flex items-center justify-center gap-2">
                        <Music className="w-4 h-4" /> Sound
                    </div>
                </button>
                <button
                    onClick={() => setActiveTab('lyrics')}
                    className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all duration-200 ${activeTab === 'lyrics' ? 'bg-white shadow-sm ring-1 ring-black/5 text-cyan-700' : 'text-slate-500 hover:bg-white/40'} `}
                >
                    <div className="flex items-center justify-center gap-2">
                        <Mic2 className="w-4 h-4" /> Lyrics
                    </div>
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">

                {activeTab === 'sound' && (
                    <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="space-y-6">

                        {/* Topic Input */}
                        <div className="space-y-2">
                            <label className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                                <span className="w-1.5 h-1.5 rounded-full bg-cyan-500"></span>
                                Song Concept / Topic
                            </label>
                            <div className="relative">
                                <input
                                    value={topic}
                                    onChange={(e) => setTopic(e.target.value)}
                                    placeholder="e.g. 'A lost astronaut', 'Summer love'"
                                    className="w-full bg-white/60 rounded-sm border border-slate-200/60 px-3 py-2 pr-10 focus:ring-1 focus:ring-cyan-500/50 outline-none text-sm placeholder:text-slate-400 transition-all shadow-inner font-mono text-slate-700"
                                />
                                <button
                                    onClick={handleEnhancePrompt}
                                    disabled={isEnhancing || !topic}
                                    className="absolute right-1 top-1 p-1.5 rounded-sm hover:bg-cyan-100 text-slate-400 hover:text-cyan-600 transition-colors disabled:opacity-50"
                                    title="Enhance with AI (Magic Wand)"
                                >
                                    {isEnhancing ? <span className="animate-spin text-xs">⌛</span> : <Wand2 className="w-4 h-4" />}
                                </button>
                            </div>
                        </div>

                        {/* Style Input */}
                        <div className="space-y-2">
                            <div className="flex items-center justify-between">
                                <label className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                                    <span className="w-1.5 h-1.5 rounded-full bg-fuchsia-500"></span>
                                    Musical Style
                                </label>
                                <button
                                    onClick={refreshStyles}
                                    disabled={isLoadingStyles}
                                    className="text-[10px] text-cyan-600 hover:text-cyan-800 hover:bg-cyan-50 px-1.5 py-0.5 rounded transition-colors disabled:opacity-50 flex items-center gap-1"
                                    title="Get new style suggestions"
                                >
                                    {isLoadingStyles ? "Loading..." : "Refresh"}
                                    <RefreshCw className={`w-3 h-3 ${isLoadingStyles ? 'animate-spin' : ''}`} />
                                </button>
                            </div>
                            <textarea
                                value={style}
                                onChange={(e) => setStyle(e.target.value)}
                                placeholder="e.g. 'Pop, Electronic, Emotional, Rock'"
                                className="w-full h-20 bg-white/60 rounded-sm border border-slate-200/60 p-3 focus:ring-1 focus:ring-cyan-500/50 outline-none resize-none text-sm placeholder:text-slate-400 transition-all shadow-inner font-mono text-slate-700"
                            />
                            {/* Style Pills */}
                            <div className="flex flex-wrap gap-1.5 mt-2">
                                {stylePills.map(s => (
                                    <button
                                        key={s}
                                        onClick={() => addStyle(s)}
                                        className="text-[10px] font-mono bg-white/50 hover:bg-white border border-slate-200/50 hover:border-cyan-200 rounded-full px-2 py-0.5 text-slate-500 hover:text-cyan-600 transition-all"
                                    >
                                        + {s}
                                    </button>
                                ))}
                            </div>
                        </div>



                        {/* Advanced Toggles */}
                        <div className="border-t border-dashed border-slate-200 pt-4">
                            <button
                                onClick={() => setShowAdvanced(!showAdvanced)}
                                className="flex items-center justify-between w-full text-xs font-bold uppercase tracking-widest text-slate-400 hover:text-cyan-600 transition-colors font-mono"
                            >
                                <span>Signal Parameters</span>
                                {showAdvanced ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                            </button>

                            <AnimatePresence>
                                {showAdvanced && (
                                    <motion.div
                                        initial={{ height: 0, opacity: 0 }}
                                        animate={{ height: "auto", opacity: 1 }}
                                        exit={{ height: 0, opacity: 0 }}
                                        className="overflow-hidden"
                                    >
                                        <div className="space-y-6 pt-6">

                                            <div className="space-y-2">
                                                <div className="flex justify-between text-xs font-mono text-slate-500">
                                                    <span>DURATION</span>
                                                    <span className="bg-slate-100 px-1.5 rounded text-slate-700">{duration}s</span>
                                                </div>
                                                <input
                                                    type="range" min="10" max="300" step="5"
                                                    value={duration} onChange={(e) => setDuration(Number(e.target.value))}
                                                    className="w-full accent-cyan-500 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                                                />
                                            </div>

                                            <div className="space-y-2">
                                                <div className="flex justify-between text-xs font-mono text-slate-500">
                                                    <span>TEMP (Creativity)</span>
                                                    <span className="bg-slate-100 px-1.5 rounded text-slate-700">{temperature}</span>
                                                </div>
                                                <input
                                                    type="range" min="0.1" max="2.0" step="0.1"
                                                    value={temperature} onChange={(e) => setTemperature(Number(e.target.value))}
                                                    className="w-full accent-cyan-500 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                                                />
                                            </div>

                                            <div className="space-y-2">
                                                <div className="flex justify-between text-xs font-mono text-slate-500">
                                                    <span>CFG (Adherence)</span>
                                                    <span className="bg-slate-100 px-1.5 rounded text-slate-700">{cfgScale}</span>
                                                </div>
                                                <input
                                                    type="range" min="1.0" max="5.0" step="0.5"
                                                    value={cfgScale} onChange={(e) => setCfgScale(Number(e.target.value))}
                                                    className="w-full accent-cyan-500 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                                                />
                                            </div>

                                            <div className="space-y-2">
                                                <div className="flex justify-between text-xs font-mono text-slate-500">
                                                    <span>TOP-K (Variety)</span>
                                                    <span className="bg-slate-100 px-1.5 rounded text-slate-700">{topk}</span>
                                                </div>
                                                <input
                                                    type="range" min="10" max="100" step="10"
                                                    value={topk} onChange={(e) => setTopk(Number(e.target.value))}
                                                    className="w-full accent-cyan-500 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                                                />
                                            </div>

                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    </motion.div>
                )}

                {activeTab === 'lyrics' && (
                    <motion.div initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} className="h-full flex flex-col gap-3">

                        {/* Control Bar */}
                        <div className="flex items-center gap-2 bg-slate-50 p-1 rounded-md border border-slate-200">
                            <select
                                value={lyricsModel}
                                onChange={(e) => setLyricsModel(e.target.value)}
                                className="flex-1 bg-transparent text-xs font-mono text-slate-600 focus:outline-none"
                            >
                                {lyricsModel && !lyricsModels.includes(lyricsModel) && (
                                    <option key={lyricsModel} value={lyricsModel}>{lyricsModel} (Custom)</option>
                                )}
                                {lyricsModels.map(m => <option key={m} value={m}>{m}</option>)}
                            </select>

                            <button
                                onClick={() => setShowChat(!showChat)}
                                className={`p-1.5 rounded transition-colors ${showChat ? 'bg-cyan-100 text-cyan-700' : 'hover:bg-slate-200 text-slate-400'}`}
                                title={showChat ? "Hide Chat" : "Open AI Chat"}
                            >
                                <MessageSquare size={16} />
                            </button>
                        </div>

                        {/* Chat Interface (Collapsible) */}
                        <AnimatePresence>
                            {showChat && (
                                <motion.div
                                    initial={{ height: 0, opacity: 0, marginBottom: 0 }}
                                    animate={{ height: "auto", opacity: 1, marginBottom: 12 }}
                                    exit={{ height: 0, opacity: 0, marginBottom: 0 }}
                                    className="border border-cyan-100 bg-cyan-50/50 rounded-md overflow-hidden flex flex-col shadow-sm flex-shrink-0 max-h-[300px]"
                                >
                                    <div className="flex-1 overflow-y-auto p-3 space-y-3 min-h-[150px]">
                                        {chatMessages.length === 0 && (
                                            <div className="text-center py-4">
                                                <p className="text-xs text-cyan-800 font-bold mb-1">AI Co-Writer Ready</p>
                                                <p className="text-[10px] text-cyan-600/70">"Add a bridge", "Make it darker", "Rhyme with 'fire'"...</p>
                                            </div>
                                        )}
                                        {chatMessages.map((m, i) => (
                                            <div key={i} className={`text-xs p-2.5 rounded-lg max-w-[90%] leading-relaxed whitespace-pre-wrap ${m.role === 'user' ? 'bg-cyan-500 text-white ml-auto rounded-tr-none shadow-sm' : 'bg-white text-slate-700 mr-auto rounded-tl-none border border-slate-100 shadow-sm'}`}>
                                                {m.content}
                                            </div>
                                        ))}
                                        {isChatting && (
                                            <div className="flex gap-1 ml-2 p-2">
                                                <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                                                <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                                                <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                                            </div>
                                        )}
                                    </div>
                                    <div className="p-2 bg-white border-t border-cyan-100 flex gap-2 items-center">
                                        <input
                                            value={chatInput}
                                            onChange={(e) => setChatInput(e.target.value)}
                                            onKeyDown={(e) => e.key === 'Enter' && !isChatting && handleChatSubmit()}
                                            disabled={isChatting}
                                            placeholder="Tell the AI what to change..."
                                            className="flex-1 text-xs bg-transparent outline-none placeholder:text-slate-400 text-slate-700"
                                        />
                                        <button
                                            onClick={handleChatSubmit}
                                            disabled={isChatting || !chatInput.trim()}
                                            className="bg-cyan-500 hover:bg-cyan-600 text-white p-1.5 rounded-md disabled:opacity-50 transition-colors"
                                        >
                                            <Send size={14} />
                                        </button>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* Standard Controls (Hidden when chat is open to save space? No, keep them for "Generate New") */}
                        {!showChat && (
                            <button
                                onClick={handleLyricsGen}
                                disabled={isGeneratingLyrics || !topic}
                                className="w-full bg-cyan-100/80 hover:bg-cyan-200 text-cyan-800 text-xs font-mono font-bold rounded-sm px-3 py-1.5 transition-all shadow-md shadow-cyan-500/10 hover:shadow-cyan-500/20 flex items-center justify-center gap-1.5 disabled:opacity-50 disabled:shadow-none border border-cyan-200/50"
                            >
                                {isGeneratingLyrics ? <span className="animate-spin">⌛</span> : <Sparkles className="w-3.5 h-3.5 text-cyan-600" />}
                                GENERATE FROM SCRATCH
                            </button>
                        )}

                        <textarea
                            value={lyrics}
                            onChange={(e) => setLyrics(e.target.value)}
                            placeholder="Write your own lyrics from scratch, OR start typing a few lines and let the AI finish the song for you... otherwise, leave blank to use the generated lyrics."
                            className={`w-full bg-white/60 rounded-sm border border-slate-200/60 p-4 focus:ring-1 focus:ring-cyan-500/50 outline-none resize-none text-sm font-mono leading-relaxed placeholder:text-slate-400 shadow-inner text-slate-700 ${showChat ? 'h-64' : 'h-96'}`}
                        />
                        <p className="text-[10px] text-slate-400 text-center font-mono uppercase tracking-widest">
                            {showChat ? "AI Collaboration Active" : "Manual Override Enabled"}
                        </p>
                    </motion.div>
                )}

            </div>



            {/* Terminal Log Overlay */}
            <AnimatePresence>
                {isGenerating && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="bg-transparent font-mono text-[10px] text-cyan-600/80 p-3 pt-4 border-t border-cyan-100/50 shadow-inner overflow-hidden"
                    >
                        <div className="flex items-center gap-2 mb-1 text-xs text-cyan-700 font-bold uppercase tracking-wider">
                            <span className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse"></span>
                            Creative Process Live
                        </div>
                        <div className="flex flex-col gap-0.5 font-mono opacity-80">
                            {logs.map((log, i) => (
                                <div key={i} className="truncate">{log}</div>
                            ))}
                            {logs.length === 0 && <span className="opacity-50">Initializing tensor cores...</span>}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Footer Actions */}
            <div className="p-6 border-t border-white/30 bg-white/40 backdrop-blur-md flex flex-col gap-3">

                {/* Model Badge (Footer Priority) */}
                {llmConfig.provider && (
                    <div className="flex justify-center">
                        <button
                            onClick={() => setIsSettingsOpen(true)}
                            className={`text-[10px] font-mono px-3 py-1 rounded-full border shadow-sm flex items-center gap-2 transition-transform hover:scale-105 ${llmConfig.provider === 'ollama' ? 'bg-white/80 text-orange-700 border-orange-200' :
                                llmConfig.provider === 'openai' ? 'bg-white/80 text-green-700 border-green-200' :
                                    'bg-white/80 text-slate-600 border-slate-200'
                                }`}>
                            <span>{llmConfig.provider === 'ollama' ? 'Using Ollama' : 'Using Cloud AI'}</span>
                            <span className="w-1 h-1 rounded-full bg-current opacity-50"></span>
                            <span className="font-bold">
                                {llmConfig.provider === 'ollama' ? '🦙' : (llmConfig.provider === 'openai' ? '🤖' : '✨')}
                                {(llmConfig[llmConfig.provider as keyof LLMConfig] as any)?.model || 'No Model'}
                            </span>
                        </button>
                    </div>
                )}
                {isGenerating && currentJobId && onCancel ? (
                    <button
                        onClick={() => onCancel(currentJobId)}
                        className="w-full bg-red-500/10 hover:bg-red-500/20 text-red-600 border border-red-200 rounded-sm py-3 font-mono text-xs font-bold tracking-widest transition-all flex items-center justify-center gap-2"
                    >
                        <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
                        STOP GENERATION
                    </button>
                ) : (
                    <GradientButton
                        onClick={handleSubmit}
                        isLoading={isGenerating}
                        disabled={!topic || !style || !lyrics}
                        className="w-full shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/30 rounded-sm"
                    >
                        <Sparkles className="w-4 h-4 text-white" /> <span className="tracking-widest font-mono text-xs">GENERATE TRACK</span>
                    </GradientButton>
                )}
            </div>
        </div >
    );
};
