import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Save, CheckCircle2, AlertCircle, RefreshCw, Key, Globe, Cpu } from 'lucide-react';
import { type LLMConfig, api } from '../api';
import { Combobox } from './ui/Combobox';
import { useModalA11y } from './ui/primitives';

interface LLMSettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
    currentConfig: LLMConfig;
    onConfigUpdate: () => void;
}

export const LLMSettingsModal: React.FC<LLMSettingsModalProps> = ({
    isOpen,
    onClose,
    currentConfig,
    onConfigUpdate
}) => {
    const [activeTab, setActiveTab] = useState<string>('nvidia');
    const [config, setConfig] = useState<LLMConfig>(currentConfig);
    const [isSaving, setIsSaving] = useState(false);
    const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle');
    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [isLoadingModels, setIsLoadingModels] = useState(false);
    const panelRef = React.useRef<HTMLDivElement | null>(null);
    useModalA11y(isOpen, onClose, panelRef);

    useEffect(() => {
        if (isOpen) {
            setConfig(currentConfig);
            if (currentConfig.provider) {
                setActiveTab(currentConfig.provider);
            }
        }
    }, [isOpen, currentConfig]);

    // Handle generic model fetching
    const handleFetchModels = async () => {
        setIsLoadingModels(true);
        try {
            const tempConfig = {
                provider: activeTab,
                [activeTab]: config[activeTab as keyof LLMConfig]
            };
            const models = await api.fetchModels(tempConfig);
            setAvailableModels(models);

            const currentSection = config[activeTab as keyof LLMConfig];
            const currentModel = (typeof currentSection === 'object' && currentSection !== null) ? currentSection.model : undefined;

            if (models.length > 0) {
                const isCurrentValid = models.includes(currentModel || '');
                if (!currentModel || !isCurrentValid) {
                    handleChange(activeTab as keyof LLMConfig, 'model', models[0]);
                }
            }
        } catch (e) {
            console.error("Failed to fetch models", e);
        } finally {
            setIsLoadingModels(false);
        }
    };

    // Auto-fetch logic for all providers
    useEffect(() => {
        setAvailableModels([]);

        const fetch = async () => {
            setIsLoadingModels(true);
            try {
                const tempConfig = {
                    provider: activeTab,
                    [activeTab]: config[activeTab as keyof LLMConfig]
                };

                const configSection = config[activeTab as keyof LLMConfig] as any;
                const hasCredentials =
                    activeTab === 'ollama' ||
                    activeTab === 'lmstudio' ||
                    activeTab === 'omlx'
                        ? true
                        : !!(configSection?.has_key || configSection?.has_api_key || configSection?.api_key);

                if (hasCredentials) {
                    const models = await api.fetchModels(tempConfig);
                    setAvailableModels(models);
                }
            } catch (e) {
                console.error("Auto-fetch failed", e);
            } finally {
                setIsLoadingModels(false);
            }
        };

        if (isOpen) {
            fetch();
        }
    }, [activeTab, isOpen]);

    const handleSave = async () => {
        setIsSaving(true);
        setSaveStatus('idle');
        try {
            const updatedConfig = {
                ...config,
                provider: activeTab
            };

            await api.updateLLMConfig(updatedConfig);

            setSaveStatus('success');
            setTimeout(() => {
                setSaveStatus('idle');
                onConfigUpdate();
                onClose();
            }, 800);
        } catch (e) {
            console.error("Failed to save config", e);
            setSaveStatus('error');
        } finally {
            setIsSaving(false);
        }
    };

    const handleChange = (section: keyof LLMConfig, field: string, value: string) => {
        setConfig(prev => {
            const currentSection = prev[section];
            if (!currentSection || typeof currentSection === 'string') {
                return {
                    ...prev,
                    [section]: { [field]: value }
                };
            }

            return {
                ...prev,
                [section]: {
                    ...currentSection,
                    [field]: value
                }
            };
        });
    };

    const providers = [
        { id: 'nvidia', name: 'NVIDIA NIM', icon: '🟢', desc: 'Hosted Llama 3.1/3.3 & Nemotron' },
        { id: 'opencode', name: 'OpenCode Go', icon: '🚀', desc: 'Remote Cloud Engine' },
        { id: 'omlx', name: 'OMLX Local (Port 8787)', icon: '⚡', desc: 'Apple Silicon MLX Server' },
        { id: 'ollama', name: 'Ollama (Local)', icon: '🦙', desc: 'Local Ollama Instance' },
        { id: 'deepseek', name: 'DeepSeek', icon: '🐳', desc: 'DeepSeek Official API' },
        { id: 'openai', name: 'OpenAI (ChatGPT)', icon: '🤖', desc: 'GPT-4o & o1' },
        { id: 'gemini', name: 'Google Gemini', icon: '✨', desc: 'Gemini 1.5 & Flash' },
        { id: 'openrouter', name: 'OpenRouter', icon: '🌐', desc: 'Universal AI Gateway' },
        { id: 'lmstudio', name: 'LM Studio', icon: '🧪', desc: 'Local Inference Server' },
    ];

    return createPortal(
        <AnimatePresence>
            {isOpen && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 dark:bg-black/80 backdrop-blur-md p-4 animate-fade-in">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        ref={panelRef}
                        className="bg-white/95 dark:bg-[#141620]/95 rounded-3xl border border-black/[0.08] dark:border-white/10 shadow-apple-lg w-full max-w-4xl overflow-hidden flex flex-col max-h-[90vh] backdrop-blur-2xl"
                    >
                        {/* Header */}
                        <div className="flex items-center justify-between px-6 py-4 border-b border-black/[0.06] dark:border-white/10 bg-black/[0.02] dark:bg-[#181a24]">
                            <div className="flex items-center space-x-3">
                                <div className="w-8 h-8 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 text-teal-700 dark:text-teal-400 border border-teal-500/20 flex items-center justify-center font-bold text-sm">
                                    ⚙️
                                </div>
                                <div>
                                    <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                                        LLM Engine & Co-Writer Settings
                                    </h2>
                                    <p className="text-[11px] text-slate-500 dark:text-slate-400">
                                        Configure AI lyric generation, prompt enhancement, and style brainstorming.
                                    </p>
                                </div>
                            </div>
                            <button
                                onClick={onClose}
                                className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="flex flex-1 overflow-hidden">
                            {/* Provider Sidebar */}
                            <div className="w-72 border-r border-black/[0.06] dark:border-white/10 bg-black/[0.01] dark:bg-[#10121a] p-3 space-y-1 overflow-y-auto">
                                <span className="px-3 text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider block mb-1">
                                    Select Provider
                                </span>
                                {providers.map(p => (
                                    <button
                                        key={p.id}
                                        onClick={() => setActiveTab(p.id)}
                                        className={`w-full text-left px-3.5 py-2.5 rounded-2xl text-xs font-semibold transition-all flex items-center justify-between ${
                                            activeTab === p.id
                                                ? 'bg-white dark:bg-white/15 text-teal-700 dark:text-teal-300 shadow-apple-sm font-bold border border-teal-500/20'
                                                : 'text-slate-600 dark:text-slate-400 hover:bg-black/[0.03] dark:hover:bg-white/5'
                                        }`}
                                    >
                                        <div className="flex items-center space-x-2.5 truncate">
                                            <span className="text-base">{p.icon}</span>
                                            <div className="truncate">
                                                <div className="truncate">{p.name}</div>
                                                <div className="text-[10px] text-slate-400 dark:text-slate-500 font-normal">{p.desc}</div>
                                            </div>
                                        </div>
                                        {config.provider === p.id && (
                                            <span className="w-2 h-2 rounded-full bg-teal-500 flex-shrink-0"></span>
                                        )}
                                    </button>
                                ))}
                            </div>

                            {/* Provider Configuration Panel */}
                            <div className="flex-1 p-6 overflow-y-auto bg-white dark:bg-[#151722] text-slate-800 dark:text-slate-200 space-y-6">
                                <div className="flex items-center justify-between pb-3 border-b border-black/[0.06] dark:border-white/10">
                                    <div className="flex items-center space-x-2">
                                        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                                            {providers.find(p => p.id === activeTab)?.name} Configuration
                                        </h3>
                                    </div>
                                    {config.provider === activeTab ? (
                                        <span className="text-[10px] font-bold bg-teal-500/15 text-teal-700 dark:text-teal-300 px-2.5 py-0.5 rounded-full border border-teal-500/20">
                                            ACTIVE ENGINE
                                        </span>
                                    ) : (
                                        <span className="text-[10px] text-slate-400 dark:text-slate-500">
                                            Click "Save & Set Active" below to switch
                                        </span>
                                    )}
                                </div>

                                {/* NVIDIA NIM Panel */}
                                {activeTab === 'nvidia' && (
                                    <div className="space-y-4">
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                                                <Key size={12} />
                                                NVIDIA NIM API Key
                                            </label>
                                            <input
                                                type="password"
                                                value={config.nvidia?.api_key || ''}
                                                onChange={(e) => handleChange('nvidia', 'api_key', e.target.value)}
                                                className="w-full apple-input font-mono text-xs"
                                                placeholder={config.nvidia?.has_key ? '•••••••• (configured on backend)' : 'nvapi-...'}
                                            />
                                        </div>

                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                                                <Globe size={12} />
                                                API Base URL
                                            </label>
                                            <input
                                                type="text"
                                                value={config.nvidia?.base_url || 'https://integrate.api.nvidia.com/v1'}
                                                onChange={(e) => handleChange('nvidia', 'base_url', e.target.value)}
                                                className="w-full apple-input font-mono text-xs"
                                                placeholder="https://integrate.api.nvidia.com/v1"
                                            />
                                        </div>

                                        <div className="space-y-1">
                                            <div className="flex items-center justify-between">
                                                <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                                                    <Cpu size={12} />
                                                    Active Model ({availableModels.length} available)
                                                </label>
                                                <button
                                                    onClick={handleFetchModels}
                                                    disabled={isLoadingModels}
                                                    className="text-[11px] text-teal-600 dark:text-teal-400 hover:underline flex items-center gap-1"
                                                >
                                                    <RefreshCw size={11} className={isLoadingModels ? 'animate-spin' : ''} />
                                                    <span>Refresh Models</span>
                                                </button>
                                            </div>
                                            <Combobox
                                                value={config.nvidia?.model || 'deepseek-ai/deepseek-v4-flash-0731'}
                                                onChange={(val) => handleChange('nvidia', 'model', val)}
                                                options={availableModels}
                                                onRefresh={handleFetchModels}
                                                isLoading={isLoadingModels}
                                                placeholder="Select or enter NVIDIA NIM model..."
                                            />
                                        </div>

                                        <div className="bg-emerald-500/10 text-emerald-800 dark:text-emerald-300 p-3 rounded-2xl text-xs border border-emerald-500/20">
                                            Connected to <strong>NVIDIA NIM API</strong> with dynamic support for <code>deepseek-ai/deepseek-v4-flash-0731</code>, <code>deepseek-ai/deepseek-r1</code>, and 100+ hosted models.
                                        </div>
                                    </div>
                                )}

                                {/* OpenCode Go Panel */}
                                {activeTab === 'opencode' && (
                                    <div className="space-y-4">
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                                                <Key size={12} />
                                                OpenCode Go API Key
                                            </label>
                                            <input
                                                type="password"
                                                value={config.opencode?.api_key || ''}
                                                onChange={(e) => handleChange('opencode', 'api_key', e.target.value)}
                                                className="w-full apple-input font-mono text-xs"
                                                placeholder={config.opencode?.has_key ? '•••••••• (configured on backend)' : 'sk-...'}
                                            />
                                        </div>

                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                                                <Globe size={12} />
                                                API Base URL
                                            </label>
                                            <input
                                                type="text"
                                                value={config.opencode?.base_url || 'https://opencode.ai/zen/go/v1'}
                                                onChange={(e) => handleChange('opencode', 'base_url', e.target.value)}
                                                className="w-full apple-input font-mono text-xs"
                                                placeholder="https://opencode.ai/zen/go/v1"
                                            />
                                        </div>

                                        <div className="space-y-1">
                                            <div className="flex items-center justify-between">
                                                <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                                                    <Cpu size={12} />
                                                    Active Model ({availableModels.length} available)
                                                </label>
                                                <button
                                                    onClick={handleFetchModels}
                                                    disabled={isLoadingModels}
                                                    className="text-[11px] text-teal-600 dark:text-teal-400 hover:underline flex items-center gap-1"
                                                >
                                                    <RefreshCw size={11} className={isLoadingModels ? 'animate-spin' : ''} />
                                                    <span>Refresh Models</span>
                                                </button>
                                            </div>
                                            <Combobox
                                                value={config.opencode?.model || 'minimax-m3'}
                                                onChange={(val) => handleChange('opencode', 'model', val)}
                                                options={availableModels}
                                                onRefresh={handleFetchModels}
                                                isLoading={isLoadingModels}
                                                placeholder="Select OpenCode model..."
                                            />
                                        </div>

                                        <div className="bg-teal-500/10 text-teal-800 dark:text-teal-300 p-3 rounded-2xl text-xs border border-teal-500/20">
                                            Connected to <strong>OpenCode Go API</strong> with support for <code>minimax-m3</code>, <code>deepseek-v4-pro</code>, <code>qwen3.7-max</code>, and <code>kimi-k3</code>.
                                        </div>
                                    </div>
                                )}

                                {/* OMLX Local Server Panel */}
                                {activeTab === 'omlx' && (
                                    <div className="space-y-4">
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                                                <Globe size={12} />
                                                OMLX Server URL
                                            </label>
                                            <input
                                                type="text"
                                                value={config.omlx?.base_url || 'http://localhost:8787/v1'}
                                                onChange={(e) => handleChange('omlx', 'base_url', e.target.value)}
                                                className="w-full apple-input font-mono text-xs"
                                                placeholder="http://localhost:8787/v1"
                                            />
                                        </div>

                                        <div className="space-y-1">
                                            <div className="flex items-center justify-between">
                                                <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                                                    <Cpu size={12} />
                                                    Local MLX Model ({availableModels.length} loaded)
                                                </label>
                                                <button
                                                    onClick={handleFetchModels}
                                                    disabled={isLoadingModels}
                                                    className="text-[11px] text-teal-600 dark:text-teal-400 hover:underline flex items-center gap-1"
                                                >
                                                    <RefreshCw size={11} className={isLoadingModels ? 'animate-spin' : ''} />
                                                    <span>Refresh Models</span>
                                                </button>
                                            </div>
                                            <Combobox
                                                value={config.omlx?.model || 'Llama-3.2-3B-Instruct-bf16'}
                                                onChange={(val) => handleChange('omlx', 'model', val)}
                                                options={availableModels}
                                                onRefresh={handleFetchModels}
                                                isLoading={isLoadingModels}
                                                placeholder="Select local MLX model..."
                                            />
                                        </div>

                                        <div className="bg-amber-500/10 text-amber-800 dark:text-amber-300 p-3 rounded-2xl text-xs border border-amber-500/20">
                                            OMLX server running on <strong>localhost:8787</strong> with local Apple Silicon MLX models.
                                        </div>
                                    </div>
                                )}

                                {/* Ollama Panel */}
                                {activeTab === 'ollama' && (
                                    <div className="space-y-4">
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">Base URL</label>
                                            <input
                                                type="text"
                                                value={config.ollama?.base_url || 'http://localhost:11434'}
                                                onChange={(e) => handleChange('ollama', 'base_url', e.target.value)}
                                                className="w-full apple-input font-mono text-xs"
                                                placeholder="http://localhost:11434"
                                            />
                                        </div>
                                        <div className="space-y-1">
                                            <div className="flex items-center justify-between">
                                                <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">Model</label>
                                                <button onClick={handleFetchModels} disabled={isLoadingModels} className="text-[11px] text-teal-600 dark:text-teal-400 hover:underline">
                                                    Refresh
                                                </button>
                                            </div>
                                            <Combobox
                                                value={config.ollama?.model || ''}
                                                onChange={(val) => handleChange('ollama', 'model', val)}
                                                options={availableModels}
                                                onRefresh={handleFetchModels}
                                                isLoading={isLoadingModels}
                                                placeholder="Select model..."
                                            />
                                        </div>
                                    </div>
                                )}

                                {/* DeepSeek Panel */}
                                {activeTab === 'deepseek' && (
                                    <div className="space-y-4">
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">API Key</label>
                                            <input
                                                type="password"
                                                value={config.deepseek?.api_key || ''}
                                                onChange={(e) => handleChange('deepseek', 'api_key', e.target.value)}
                                                className="w-full apple-input font-mono text-xs"
                                                placeholder={config.deepseek?.has_key ? '•••••••• (configured on backend)' : 'sk-...'}
                                            />
                                        </div>
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">Model</label>
                                            <Combobox
                                                value={config.deepseek?.model || 'deepseek-chat'}
                                                onChange={(val) => handleChange('deepseek', 'model', val)}
                                                options={availableModels}
                                                onRefresh={handleFetchModels}
                                                isLoading={isLoadingModels}
                                                placeholder="deepseek-chat"
                                            />
                                        </div>
                                    </div>
                                )}

                                {/* OpenAI Panel */}
                                {activeTab === 'openai' && (
                                    <div className="space-y-4">
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">API Key</label>
                                            <input
                                                type="password"
                                                value={config.openai?.api_key || ''}
                                                onChange={(e) => handleChange('openai', 'api_key', e.target.value)}
                                                className="w-full apple-input font-mono text-xs"
                                                placeholder={config.openai?.has_key ? '•••••••• (configured on backend)' : 'sk-...'}
                                            />
                                        </div>
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">Model</label>
                                            <Combobox
                                                value={config.openai?.model || 'gpt-4o'}
                                                onChange={(val) => handleChange('openai', 'model', val)}
                                                options={availableModels}
                                                onRefresh={handleFetchModels}
                                                isLoading={isLoadingModels}
                                                placeholder="gpt-4o"
                                            />
                                        </div>
                                    </div>
                                )}

                                {/* Gemini Panel */}
                                {activeTab === 'gemini' && (
                                    <div className="space-y-4">
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">API Key</label>
                                            <input
                                                type="password"
                                                value={config.gemini?.api_key || ''}
                                                onChange={(e) => handleChange('gemini', 'api_key', e.target.value)}
                                                className="w-full apple-input font-mono text-xs"
                                                placeholder={config.gemini?.has_key ? '•••••••• (configured on backend)' : 'AIza...'}
                                            />
                                        </div>
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">Model</label>
                                            <Combobox
                                                value={config.gemini?.model || 'gemini-1.5-flash'}
                                                onChange={(val) => handleChange('gemini', 'model', val)}
                                                options={availableModels}
                                                onRefresh={handleFetchModels}
                                                isLoading={isLoadingModels}
                                                placeholder="gemini-1.5-flash"
                                            />
                                        </div>
                                    </div>
                                )}

                                {/* OpenRouter Panel */}
                                {activeTab === 'openrouter' && (
                                    <div className="space-y-4">
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">API Key</label>
                                            <input
                                                type="password"
                                                value={config.openrouter?.api_key || ''}
                                                onChange={(e) => handleChange('openrouter', 'api_key', e.target.value)}
                                                className="w-full apple-input font-mono text-xs"
                                                placeholder={config.openrouter?.has_key ? '•••••••• (configured on backend)' : 'sk-or-...'}
                                            />
                                        </div>
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">Model</label>
                                            <Combobox
                                                value={config.openrouter?.model || 'openai/gpt-3.5-turbo'}
                                                onChange={(val) => handleChange('openrouter', 'model', val)}
                                                options={availableModels}
                                                onRefresh={handleFetchModels}
                                                isLoading={isLoadingModels}
                                                placeholder="gpt-3.5-turbo"
                                            />
                                        </div>
                                    </div>
                                )}

                                {/* LM Studio Panel */}
                                {activeTab === 'lmstudio' && (
                                    <div className="space-y-4">
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">Base URL</label>
                                            <input
                                                type="text"
                                                value={config.lmstudio?.base_url || 'http://localhost:1234/v1'}
                                                onChange={(e) => handleChange('lmstudio', 'base_url', e.target.value)}
                                                className="w-full apple-input font-mono text-xs"
                                                placeholder="http://localhost:1234/v1"
                                            />
                                        </div>
                                        <div className="space-y-1">
                                            <label className="text-xs font-bold text-slate-600 dark:text-slate-400 uppercase tracking-wider">Model</label>
                                            <Combobox
                                                value={config.lmstudio?.model || 'local-model'}
                                                onChange={(val) => handleChange('lmstudio', 'model', val)}
                                                options={availableModels}
                                                onRefresh={handleFetchModels}
                                                isLoading={isLoadingModels}
                                                placeholder="local-model"
                                            />
                                        </div>
                                    </div>
                                )}

                                {/* Save Button */}
                                <div className="pt-4 flex items-center gap-3 border-t border-black/[0.06] dark:border-white/10">
                                    <button
                                        onClick={handleSave}
                                        disabled={isSaving}
                                        className="px-5 py-2.5 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 rounded-xl font-bold text-xs shadow-md shadow-teal-500/20 transition-all flex items-center gap-2 disabled:opacity-50 active:scale-95"
                                    >
                                        {isSaving ? <span className="animate-spin">⌛</span> : <Save className="w-4 h-4" />}
                                        <span>Save & Set Active Provider</span>
                                    </button>

                                    {saveStatus === 'success' && (
                                        <span className="text-teal-600 dark:text-teal-400 text-xs font-semibold flex items-center gap-1 animate-fade-in">
                                            <CheckCircle2 className="w-4 h-4" /> Configuration Saved & Activated!
                                        </span>
                                    )}
                                    {saveStatus === 'error' && (
                                        <span className="text-rose-600 dark:text-rose-400 text-xs font-semibold flex items-center gap-1 animate-fade-in">
                                            <AlertCircle className="w-4 h-4" /> Failed to save configuration
                                        </span>
                                    )}
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>,
        document.body
    );
};
