import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Save, CheckCircle2, AlertCircle } from 'lucide-react';
import { type LLMConfig, api } from '../api';
import { Combobox } from './ui/Combobox';

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
    const [activeTab, setActiveTab] = useState<string>('ollama');
    const [config, setConfig] = useState<LLMConfig>(currentConfig);
    const [isSaving, setIsSaving] = useState(false);
    const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle');
    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [isLoadingModels, setIsLoadingModels] = useState(false);

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

            // Auto-select first model if current is not in list (or empty)
            // But be careful not to overwrite user's custom entry if they just typed it
            // Assuming if they just fetched, they probably want to see valid models.
            // If current model is default (like "llama3") but not in list, pick first available.

            const currentSection = config[activeTab as keyof LLMConfig];
            // Safe access ensuring it's an object (ProviderConfig)
            const currentModel = (typeof currentSection === 'object' && currentSection !== null) ? currentSection.model : undefined;

            if (models.length > 0) {
                const isCurrentValid = models.includes(currentModel || '');
                // If invalid (empty) OR (invalid AND matches know default placeholder), pick first. 
                if (!currentModel || (!isCurrentValid && (currentModel === 'llama3' || currentModel === 'llama3.2:3b-instruct-fp16' || currentModel === 'gpt-4o' || currentModel === 'gemini-1.5-flash'))) {
                    handleChange(activeTab as keyof LLMConfig, 'model', models[0]);
                }
            }

        } catch (e) {
            console.error(e);
            alert("Failed to fetch models. Check credentials.");
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
                // Construct temp config
                const tempConfig = {
                    provider: activeTab,
                    [activeTab]: config[activeTab as keyof LLMConfig]
                };

                // Check if we have credentials before fetching
                const configSection = config[activeTab as keyof LLMConfig] as any;
                const hasCredentials = activeTab === 'ollama' || activeTab === 'lmstudio'
                    ? true // local providers usually okay without specific key, or base_url is default
                    : !!configSection?.api_key;

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
            // Update the active provider based on the tab
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
            }, 1000);
        } catch (e) {
            console.error("Failed to save config", e);
            setSaveStatus('error');
        } finally {
            setIsSaving(false);
        }
    };

    const handleChange = (section: keyof LLMConfig, field: string, value: string) => {
        setConfig(prev => {
            // Ensure we are working with an object property (not the 'provider' string)
            const currentSection = prev[section];
            // Fix for "Spread types may only be created from object types"
            // If it's a string (provider name) or undefined, we can't spread it.
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
        { id: 'ollama', name: 'Ollama (Local)', icon: '🦙' },
        { id: 'openai', name: 'OpenAI (ChatGPT)', icon: '🤖' },
        { id: 'gemini', name: 'Google Gemini', icon: '✨' },
        { id: 'deepseek', name: 'DeepSeek', icon: '🐳' },
        { id: 'openrouter', name: 'OpenRouter', icon: '🌐' },
        { id: 'lmstudio', name: 'LM Studio', icon: '🧪' },
    ];

    return createPortal(
        <AnimatePresence>
            {isOpen && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="bg-white rounded-lg shadow-2xl w-full max-w-4xl overflow-hidden flex flex-col max-h-[90vh]"
                    >
                        <div className="flex items-center justify-between p-4 border-b border-slate-100 bg-slate-50/50">
                            <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                                <span className="bg-cyan-100 p-1 rounded text-lg">⚙️</span>
                                LLM Provider Settings
                            </h2>
                            <button onClick={onClose} className="p-1 rounded-full hover:bg-slate-200 text-slate-500 transition-colors">
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="flex flex-1 overflow-hidden">
                            {/* Sidebar */}
                            <div className="w-1/3 border-r border-slate-100 bg-slate-50/30 p-2 space-y-1">
                                {providers.map(p => (
                                    <button
                                        key={p.id}
                                        onClick={() => setActiveTab(p.id)}
                                        className={`w-full text-left px-3 py-2.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${activeTab === p.id
                                            ? 'bg-cyan-50 text-cyan-700 ring-1 ring-cyan-200 shadow-sm'
                                            : 'text-slate-600 hover:bg-white hover:shadow-sm'
                                            }`}
                                    >
                                        <span>{p.icon}</span>
                                        {p.name}
                                        {config.provider === p.id && (
                                            <span className="ml-auto w-2 h-2 rounded-full bg-cyan-500"></span>
                                        )}
                                    </button>
                                ))}
                            </div>

                            {/* Content */}
                            <div className="flex-1 p-6 overflow-y-auto bg-white">
                                <div className="space-y-6">
                                    <div className="flex items-center justify-between mb-4">
                                        <h3 className="text-lg font-medium text-slate-800">
                                            {providers.find(p => p.id === activeTab)?.name} Configuration
                                        </h3>
                                        {config.provider === activeTab && (
                                            <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-bold border border-green-200">ACTIVE</span>
                                        )}
                                    </div>

                                    {activeTab === 'ollama' && (
                                        <div className="space-y-4">
                                            <div className="space-y-1">
                                                <label className="text-xs font-bold text-slate-500 uppercase">Base URL</label>
                                                <input
                                                    type="text"
                                                    value={config.ollama?.base_url || ''}
                                                    onChange={(e) => handleChange('ollama', 'base_url', e.target.value)}
                                                    className="w-full px-3 py-2 border border-slate-200 rounded-md focus:ring-2 focus:ring-cyan-500 focus:outline-none text-sm font-mono"
                                                    placeholder="http://localhost:11434"
                                                />
                                            </div>
                                            <div className="space-y-1">
                                                <label className="text-xs font-bold text-slate-500 uppercase">Model</label>
                                                <Combobox
                                                    value={config.ollama?.model || ''}
                                                    onChange={(val) => handleChange('ollama', 'model', val)}
                                                    options={availableModels}
                                                    onRefresh={handleFetchModels}
                                                    isLoading={isLoadingModels}
                                                    placeholder="Select model..."
                                                />
                                            </div>
                                            <div className="bg-blue-50 text-blue-700 p-3 rounded-md text-xs border border-blue-100">
                                                Run <code>ollama serve</code> in your terminal to use local models.
                                            </div>
                                        </div>
                                    )}

                                    {activeTab === 'openai' && (
                                        <div className="space-y-4">
                                            <div className="space-y-1">
                                                <label className="text-xs font-bold text-slate-500 uppercase">API Key</label>
                                                <input
                                                    type="password"
                                                    value={config.openai?.api_key || ''}
                                                    onChange={(e) => handleChange('openai', 'api_key', e.target.value)}
                                                    className="w-full px-3 py-2 border border-slate-200 rounded-md focus:ring-2 focus:ring-cyan-500 focus:outline-none text-sm font-mono"
                                                    placeholder="sk-..."
                                                />
                                            </div>
                                            <div className="space-y-1">
                                                <label className="text-xs font-bold text-slate-500 uppercase">Model</label>
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

                                    {activeTab === 'gemini' && (
                                        <div className="space-y-4">
                                            <div className="space-y-1">
                                                <label className="text-xs font-bold text-slate-500 uppercase">API Key</label>
                                                <input
                                                    type="password"
                                                    value={config.gemini?.api_key || ''}
                                                    onChange={(e) => handleChange('gemini', 'api_key', e.target.value)}
                                                    className="w-full px-3 py-2 border border-slate-200 rounded-md focus:ring-2 focus:ring-cyan-500 focus:outline-none text-sm font-mono"
                                                    placeholder="AIza..."
                                                />
                                            </div>
                                            <div className="space-y-1">
                                                <label className="text-xs font-bold text-slate-500 uppercase">Model</label>
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

                                    {activeTab === 'deepseek' && (
                                        <div className="space-y-4">
                                            <div className="space-y-1">
                                                <label className="text-xs font-bold text-slate-500 uppercase">API Key</label>
                                                <input
                                                    type="password"
                                                    value={config.deepseek?.api_key || ''}
                                                    onChange={(e) => handleChange('deepseek', 'api_key', e.target.value)}
                                                    className="w-full px-3 py-2 border border-slate-200 rounded-md focus:ring-2 focus:ring-cyan-500 focus:outline-none text-sm font-mono"
                                                    placeholder="sk-..."
                                                />
                                            </div>
                                            <div className="space-y-1">
                                                <label className="text-xs font-bold text-slate-500 uppercase">Model</label>
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

                                    {activeTab === 'openrouter' && (
                                        <div className="space-y-4">
                                            <div className="space-y-1">
                                                <label className="text-xs font-bold text-slate-500 uppercase">API Key</label>
                                                <input
                                                    type="password"
                                                    value={config.openrouter?.api_key || ''}
                                                    onChange={(e) => handleChange('openrouter', 'api_key', e.target.value)}
                                                    className="w-full px-3 py-2 border border-slate-200 rounded-md focus:ring-2 focus:ring-cyan-500 focus:outline-none text-sm font-mono"
                                                    placeholder="sk-or-..."
                                                />
                                            </div>
                                            <div className="space-y-1">
                                                <label className="text-xs font-bold text-slate-500 uppercase">Model</label>
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

                                    {activeTab === 'lmstudio' && (
                                        <div className="space-y-4">
                                            <div className="space-y-1">
                                                <label className="text-xs font-bold text-slate-500 uppercase">Base URL</label>
                                                <input
                                                    type="text"
                                                    value={config.lmstudio?.base_url || ''}
                                                    onChange={(e) => handleChange('lmstudio', 'base_url', e.target.value)}
                                                    className="w-full px-3 py-2 border border-slate-200 rounded-md focus:ring-2 focus:ring-cyan-500 focus:outline-none text-sm font-mono"
                                                    placeholder="http://localhost:1234/v1"
                                                />
                                            </div>
                                            <div className="space-y-1">
                                                <label className="text-xs font-bold text-slate-500 uppercase">Model</label>
                                                <Combobox
                                                    value={config.lmstudio?.model || 'local-model'}
                                                    onChange={(val) => handleChange('lmstudio', 'model', val)}
                                                    options={availableModels}
                                                    onRefresh={handleFetchModels}
                                                    isLoading={isLoadingModels}
                                                    placeholder="local-model"
                                                />
                                            </div>
                                            <div className="bg-yellow-50 text-yellow-800 p-3 rounded-md text-xs border border-yellow-200">
                                                Ensure LM Studio server is running and "Cross-Origin-Resource-Sharing (CORS)" is enabled in LM Studio settings.
                                            </div>
                                        </div>
                                    )}

                                    <div className="pt-4 flex items-center gap-2">
                                        <button
                                            onClick={handleSave}
                                            disabled={isSaving}
                                            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-md font-bold text-sm shadow-md transition-all flex items-center gap-2 disabled:opacity-50"
                                        >
                                            {isSaving ? <span className="animate-spin">⌛</span> : <Save className="w-4 h-4" />}
                                            Save & Set Active
                                        </button>

                                        {saveStatus === 'success' && (
                                            <span className="text-green-600 text-sm flex items-center gap-1 animate-in fade-in slide-in-from-left-2">
                                                <CheckCircle2 className="w-4 h-4" /> Saved!
                                            </span>
                                        )}
                                        {saveStatus === 'error' && (
                                            <span className="text-red-600 text-sm flex items-center gap-1 animate-in fade-in slide-in-from-left-2">
                                                <AlertCircle className="w-4 h-4" /> Failed to save
                                            </span>
                                        )}
                                    </div>

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
