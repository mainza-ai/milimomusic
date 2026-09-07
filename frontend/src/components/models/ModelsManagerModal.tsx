import React, { useState, useEffect, useRef } from 'react';
import { Cpu, Download, CheckCircle2, X, Activity, AlertTriangle, Loader2, Search, Trash2, ExternalLink, Globe, Sparkles } from 'lucide-react';
import { modelsApi, type ModelVariant, type HardwareProfile, type ModelDownloadStatus, type HuggingFaceSearchResult } from '../../api';

interface ModelsManagerModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export const ModelsManagerModal: React.FC<ModelsManagerModalProps> = ({ isOpen, onClose }) => {
    const [models, setModels] = useState<ModelVariant[]>([]);
    const [hardware, setHardware] = useState<HardwareProfile | null>(null);
    const [download, setDownload] = useState<ModelDownloadStatus | null>(null);
    const [downloadError, setDownloadError] = useState<string>('');
    const pollRef = useRef<number | undefined>(undefined);

    const [selectedTab, setSelectedTab] = useState<'audio' | 'image' | 'video' | 'search'>('audio');
    const [activatingId, setActivatingId] = useState<string | null>(null);

    // Hugging Face Search & Custom Model state
    const [searchQuery, setSearchQuery] = useState('music');
    const [searchPipeline, setSearchPipeline] = useState<string>('');
    const [searchResults, setSearchResults] = useState<HuggingFaceSearchResult[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [customRepoInput, setCustomRepoInput] = useState('');
    const [customCategory, setCustomCategory] = useState<'audio' | 'image' | 'video'>('audio');

    const loadData = async () => {
        try {
            const [treeData, hwData] = await Promise.all([
                modelsApi.getModelTree(),
                modelsApi.getHardwareProfile()
            ]);
            setModels(treeData);
            setHardware(hwData);
        } catch (e) {
            console.error('Failed to load model manager data', e);
        }
    };

    useEffect(() => {
        if (isOpen) {
            loadData();
        }
        return () => window.clearInterval(pollRef.current);
    }, [isOpen]);

    const handleDownload = async (repoId: string) => {
        setDownloadError('');
        try {
            const started = await modelsApi.startModelDownload(repoId);
            setDownload(started);
            window.clearInterval(pollRef.current);
            pollRef.current = window.setInterval(async () => {
                try {
                    const s = await modelsApi.getModelDownload(started.id);
                    setDownload(s);
                    if (['completed', 'cancelled', 'error'].includes(s.status)) {
                        window.clearInterval(pollRef.current);
                        if (s.status === 'completed') loadData();
                    }
                } catch { /* transient poll error */ }
            }, 800);
        } catch (e: any) {
            setDownloadError(e?.response?.data?.detail?.error?.message
                || e?.response?.data?.detail
                || e?.message
                || 'Download failed to start.');
        }
    };

    const handleActivateModel = async (modelId: string) => {
        try {
            setActivatingId(modelId);
            await modelsApi.selectActiveModel(modelId);
            await loadData();
        } catch (e: any) {
            console.error('Failed to activate model:', e);
        } finally {
            setActivatingId(null);
        }
    };

    const handleRunSearch = async (queryToUse?: string, pipelineToUse?: string) => {
        const q = queryToUse !== undefined ? queryToUse : searchQuery;
        const p = pipelineToUse !== undefined ? pipelineToUse : searchPipeline;
        try {
            setIsSearching(true);
            const res = await modelsApi.searchHuggingFace(q, p, 24);
            setSearchResults(res.models || []);
        } catch (err) {
            console.error('Failed to search Hugging Face:', err);
        } finally {
            setIsSearching(false);
        }
    };

    const handleDeleteCustom = async (modelId: string) => {
        try {
            await modelsApi.deleteCustomModel(modelId);
            await loadData();
        } catch (err) {
            console.error('Failed to delete custom model:', err);
        }
    };

    const handleDownloadCustomRepo = async () => {
        if (!customRepoInput.trim()) return;
        await handleDownload(customRepoInput.trim());
        setCustomRepoInput('');
    };

    const busy = !!download && ['queued', 'downloading'].includes(download.status);
    const pct = download?.progress_percent;

    const filteredModels = models.filter(m => (m.category || 'audio') === selectedTab);
    const audioCount = models.filter(m => (m.category || 'audio') === 'audio').length;
    const imageCount = models.filter(m => m.category === 'image').length;
    const videoCount = models.filter(m => m.category === 'video').length;

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 dark:bg-black/80 backdrop-blur-md animate-fade-in">
            <div className="bg-white/90 dark:bg-[#14161f]/95 border border-black/[0.08] dark:border-white/10 rounded-3xl w-full max-w-4xl overflow-hidden shadow-apple-lg flex flex-col max-h-[85vh] backdrop-blur-2xl">
                {/* Modal Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-black/[0.06] dark:border-white/10 bg-black/[0.02] dark:bg-[#181a24]">
                    <div className="flex items-center space-x-3">
                        <div className="w-9 h-9 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 text-teal-700 dark:text-teal-400 border border-teal-500/20 flex items-center justify-center">
                            <Cpu size={18} />
                        </div>
                        <div>
                            <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
                                Generation Models & Multi-Modal Hub
                            </h2>
                            <p className="text-xs text-slate-500 dark:text-slate-400">
                                Manage audio checkpoints, FLUX image diffusion, and Hailuo/Wan video engines.
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
                    >
                        <X size={18} />
                    </button>
                </div>

                {/* Modality Tabs */}
                <div className="flex items-center space-x-2 px-6 pt-4 border-b border-black/[0.06] dark:border-white/10">
                    <button
                        onClick={() => setSelectedTab('audio')}
                        className={`pb-3 px-3 text-xs font-bold border-b-2 transition-all flex items-center space-x-1.5 ${
                            selectedTab === 'audio'
                                ? 'border-teal-500 text-teal-600 dark:text-teal-400'
                                : 'border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                        }`}
                    >
                        <span>🎵 Audio & Music 3</span>
                        <span className="px-1.5 py-0.5 rounded-full bg-black/5 dark:bg-white/5 text-[10px]">
                            {audioCount}
                        </span>
                    </button>

                    <button
                        onClick={() => setSelectedTab('image')}
                        className={`pb-3 px-3 text-xs font-bold border-b-2 transition-all flex items-center space-x-1.5 ${
                            selectedTab === 'image'
                                ? 'border-teal-500 text-teal-600 dark:text-teal-400'
                                : 'border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                        }`}
                    >
                        <span>🎨 Image & Covers (FLUX)</span>
                        <span className="px-1.5 py-0.5 rounded-full bg-black/5 dark:bg-white/5 text-[10px]">
                            {imageCount}
                        </span>
                    </button>

                    <button
                        onClick={() => setSelectedTab('video')}
                        className={`pb-3 px-3 text-xs font-bold border-b-2 transition-all flex items-center space-x-1.5 ${
                            selectedTab === 'video'
                                ? 'border-teal-500 text-teal-600 dark:text-teal-400'
                                : 'border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                        }`}
                    >
                        <span>🎬 Video Studios (Hailuo/Wan)</span>
                        <span className="px-1.5 py-0.5 rounded-full bg-black/5 dark:bg-white/5 text-[10px]">
                            {videoCount}
                        </span>
                    </button>

                    <button
                        onClick={() => {
                            setSelectedTab('search');
                            if (searchResults.length === 0) {
                                handleRunSearch('music', '');
                            }
                        }}
                        className={`pb-3 px-3 text-xs font-bold border-b-2 transition-all flex items-center space-x-1.5 ${
                            selectedTab === 'search'
                                ? 'border-teal-500 text-teal-600 dark:text-teal-400'
                                : 'border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                        }`}
                    >
                        <Search size={13} />
                        <span>Hugging Face Hub</span>
                        <span className="px-1.5 py-0.5 rounded-full bg-teal-500/10 text-teal-600 dark:text-teal-400 text-[10px] font-bold">
                            Live
                        </span>
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {/* Hardware Profiler Card */}
                    {hardware && (
                        <div className="p-4 bg-gradient-to-r from-teal-500/10 via-cyan-500/5 to-sky-500/5 rounded-2xl border border-teal-500/20 flex items-center justify-between shadow-apple-sm">
                            <div className="flex items-center space-x-3">
                                <Activity className="text-teal-600 dark:text-teal-400 flex-shrink-0" size={20} />
                                <div>
                                    <div className="flex items-center space-x-2">
                                        <span className="text-xs font-bold text-slate-900 dark:text-slate-100">Hardware Detection:</span>
                                        <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-teal-500/20 text-teal-800 dark:text-teal-300 font-semibold uppercase">
                                            {hardware.hardware_tier.replace('_', ' ')}
                                        </span>
                                    </div>
                                    <p className="text-xs text-slate-600 dark:text-slate-300 mt-0.5">
                                        {hardware.processor} ({hardware.os_name} {hardware.architecture}) · {hardware.tier_description}
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Hugging Face Hub Search & Custom Registry View */}
                    {selectedTab === 'search' ? (
                        <div className="space-y-6">
                            {/* Search Header & Query Bar */}
                            <div className="p-5 rounded-2xl bg-gradient-to-r from-teal-500/10 via-cyan-500/5 to-transparent border border-teal-500/20 space-y-4">
                                <div>
                                    <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                                        <Globe size={16} className="text-teal-600 dark:text-teal-400" />
                                        <span>Search Hugging Face Hub</span>
                                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/20 text-teal-700 dark:text-teal-300">
                                            Live Discovery
                                        </span>
                                    </h3>
                                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                                        Search open-source audio, image diffusion (FLUX), and video models directly from the Hugging Face ecosystem.
                                    </p>
                                </div>

                                <div className="flex gap-2">
                                    <div className="relative flex-1">
                                        <Search size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
                                        <input
                                            type="text"
                                            value={searchQuery}
                                            onChange={(e) => setSearchQuery(e.target.value)}
                                            onKeyDown={(e) => e.key === 'Enter' && handleRunSearch()}
                                            placeholder="Search by keywords, tags, or author (e.g. music, flux, cogvideox, whisper)..."
                                            className="w-full pl-9 pr-3 py-2 text-xs rounded-xl bg-white/70 dark:bg-black/30 border border-black/10 dark:border-white/10 text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/40"
                                        />
                                    </div>
                                    <button
                                        onClick={() => handleRunSearch()}
                                        disabled={isSearching}
                                        className="px-4 py-2 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-all shadow-sm active:scale-95 disabled:opacity-50"
                                    >
                                        {isSearching ? <Loader2 size={13} className="animate-spin" /> : <Search size={13} />}
                                        <span>{isSearching ? 'Searching…' : 'Search Hub'}</span>
                                    </button>
                                </div>

                                {/* Pipeline filter chips */}
                                <div className="flex items-center gap-1.5 flex-wrap">
                                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mr-1">Filter:</span>
                                    {[
                                        { label: 'All Pipelines', pipeline: '' },
                                        { label: '🎵 Audio Generation', pipeline: 'text-to-audio' },
                                        { label: '🎨 Image (FLUX/Diffusion)', pipeline: 'text-to-image' },
                                        { label: '🎬 Video (Wan/CogVideo)', pipeline: 'text-to-video' }
                                    ].map(chip => (
                                        <button
                                            key={chip.label}
                                            onClick={() => {
                                                setSearchPipeline(chip.pipeline);
                                                handleRunSearch(searchQuery, chip.pipeline);
                                            }}
                                            className={`text-[11px] px-2.5 py-1 rounded-lg border transition-all ${
                                                searchPipeline === chip.pipeline
                                                    ? 'bg-teal-500/20 border-teal-500/40 text-teal-700 dark:text-teal-300 font-bold'
                                                    : 'bg-black/[0.02] dark:bg-white/[0.02] border-black/[0.05] dark:border-white/5 text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
                                            }`}
                                        >
                                            {chip.label}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Direct Hugging Face Repository Downloader */}
                            <div className="p-4 rounded-2xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.06] dark:border-white/5 space-y-3">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center space-x-2">
                                        <Sparkles size={14} className="text-cyan-500" />
                                        <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                                            Direct Repo Downloader
                                        </span>
                                    </div>
                                    <span className="text-[10px] text-slate-400">
                                        Paste any Hugging Face repo_id to download and register
                                    </span>
                                </div>
                                <div className="flex gap-2 flex-wrap sm:flex-nowrap">
                                    <input
                                        type="text"
                                        value={customRepoInput}
                                        onChange={(e) => setCustomRepoInput(e.target.value)}
                                        onKeyDown={(e) => e.key === 'Enter' && handleDownloadCustomRepo()}
                                        placeholder="e.g. black-forest-labs/FLUX.1-schnell or wan-video/Wan2.1-T2V-1.3B"
                                        className="flex-1 px-3 py-1.5 text-xs font-mono rounded-xl bg-white/70 dark:bg-black/30 border border-black/10 dark:border-white/10 text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/40 min-w-[200px]"
                                    />
                                    <select
                                        value={customCategory}
                                        onChange={(e) => setCustomCategory(e.target.value as any)}
                                        className="px-2.5 py-1.5 text-xs rounded-xl bg-white/70 dark:bg-black/30 border border-black/10 dark:border-white/10 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500/40"
                                    >
                                        <option value="audio">Audio Model</option>
                                        <option value="image">Image Diffusion</option>
                                        <option value="video">Video Generator</option>
                                    </select>
                                    <button
                                        onClick={handleDownloadCustomRepo}
                                        disabled={busy || !customRepoInput.trim()}
                                        className="px-3.5 py-1.5 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs flex items-center space-x-1.5 transition-all disabled:opacity-50 flex-shrink-0"
                                    >
                                        <Download size={13} />
                                        <span>Download</span>
                                    </button>
                                </div>
                            </div>

                            {/* Hugging Face Search Results */}
                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                                        Hub Results ({searchResults.length})
                                    </h4>
                                    {isSearching && (
                                        <span className="text-xs text-teal-500 flex items-center gap-1 font-mono">
                                            <Loader2 size={12} className="animate-spin" /> Querying HF Hub…
                                        </span>
                                    )}
                                </div>

                                {searchResults.length === 0 && !isSearching ? (
                                    <div className="text-center py-8 text-xs text-slate-400">
                                        No models found. Try another query or use the Direct Repo Downloader above.
                                    </div>
                                ) : (
                                    <div className="space-y-2.5">
                                        {searchResults.map((res) => {
                                            const isInstalled = models.some(m => m.repo_id === res.repo_id && m.is_installed);
                                            return (
                                                <div
                                                    key={res.repo_id}
                                                    className="p-4 rounded-xl border border-black/[0.06] dark:border-white/5 bg-white dark:bg-[#181a24] hover:border-teal-500/30 transition-all flex items-center justify-between gap-4"
                                                >
                                                    <div className="min-w-0 space-y-1">
                                                        <div className="flex items-center space-x-2 flex-wrap gap-y-1">
                                                            <span className="text-xs font-bold text-slate-900 dark:text-slate-100 font-mono">
                                                                {res.repo_id}
                                                            </span>
                                                            {res.pipeline_tag && (
                                                                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-black/[0.04] dark:bg-white/5 text-slate-600 dark:text-slate-400">
                                                                    {res.pipeline_tag}
                                                                </span>
                                                            )}
                                                            {res.size_formatted && res.size_formatted !== 'Unknown' && (
                                                                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-400 border border-teal-500/20 font-semibold">
                                                                    💾 {res.size_formatted}
                                                                </span>
                                                            )}
                                                        </div>
                                                        <div className="text-[11px] text-slate-500 flex items-center gap-3 flex-wrap">
                                                            <span>Author: <strong className="text-slate-700 dark:text-slate-300">{res.author}</strong></span>
                                                            {res.size_formatted && res.size_formatted !== 'Unknown' && (
                                                                <span>Size: <strong className="text-slate-700 dark:text-slate-300">{res.size_formatted}</strong></span>
                                                            )}
                                                            <span>⬇️ {res.downloads.toLocaleString()} downloads</span>
                                                            <span>❤️ {res.likes.toLocaleString()} likes</span>
                                                        </div>
                                                    </div>

                                                    <div className="flex items-center space-x-2 flex-shrink-0">
                                                        <a
                                                            href={`https://huggingface.co/${res.repo_id}`}
                                                            target="_blank"
                                                            rel="noreferrer"
                                                            className="p-1.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
                                                            title="View on Hugging Face"
                                                        >
                                                            <ExternalLink size={14} />
                                                        </a>

                                                        {isInstalled ? (
                                                            <span className="flex items-center space-x-1 text-xs font-semibold text-teal-700 dark:text-teal-400 bg-teal-500/10 px-2.5 py-1 rounded-xl border border-teal-500/20">
                                                                <CheckCircle2 size={13} />
                                                                <span>Installed</span>
                                                            </span>
                                                        ) : (
                                                            <button
                                                                onClick={() => handleDownload(res.repo_id)}
                                                                disabled={busy}
                                                                className="px-3 py-1.5 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs flex items-center space-x-1.5 transition-all shadow-sm active:scale-95 disabled:opacity-50"
                                                            >
                                                                {busy && download?.repo_id === res.repo_id ? (
                                                                    <Loader2 size={13} className="animate-spin" />
                                                                ) : (
                                                                    <Download size={13} />
                                                                )}
                                                                <span>Download{res.size_formatted && res.size_formatted !== 'Unknown' ? ` (${res.size_formatted})` : ''}</span>
                                                            </button>
                                                        )}
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                )}
                            </div>

                            {/* Installed Custom Models */}
                            {models.some(m => (m as any).is_custom) && (
                                <div className="space-y-3 pt-2">
                                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                                        Custom Downloaded Models
                                    </h4>
                                    <div className="space-y-2">
                                        {models.filter(m => (m as any).is_custom).map(m => (
                                            <div
                                                key={m.id}
                                                className="p-3.5 rounded-xl border border-teal-500/20 bg-teal-500/5 flex items-center justify-between"
                                            >
                                                <div>
                                                    <span className="text-xs font-bold text-slate-900 dark:text-slate-100">
                                                        {m.name}
                                                    </span>
                                                    <span className="ml-2 text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/20 text-teal-700 dark:text-teal-300">
                                                        {m.category || 'custom'}
                                                    </span>
                                                    {m.local_path && (
                                                        <p className="text-[10px] font-mono text-slate-400 truncate max-w-md mt-0.5">
                                                            {m.local_path}
                                                        </p>
                                                    )}
                                                </div>
                                                <button
                                                    onClick={() => handleDeleteCustom(m.id)}
                                                    className="p-1.5 text-rose-500 hover:bg-rose-500/10 rounded-lg transition-colors"
                                                    title="Delete Custom Model"
                                                >
                                                    <Trash2 size={14} />
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        /* Standard Model List */
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <h3 className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
                                    {selectedTab.toUpperCase()} Models ({filteredModels.length})
                                </h3>
                                <span className="text-[11px] text-slate-500">
                                    {selectedTab === 'audio'
                                        ? 'Smallest model auto-installed on fresh systems; switch variants freely.'
                                        : 'Visual models are strictly on-demand and downloaded only when initiated.'}
                                </span>
                            </div>

                            <div className="space-y-3">
                                {filteredModels.map(m => (
                                    <div
                                        key={m.id}
                                        className={`p-5 rounded-2xl border transition-all ${
                                            m.is_active
                                                ? 'bg-teal-500/5 border-teal-500/40 shadow-apple-md'
                                                : m.is_installed
                                                ? 'bg-white dark:bg-[#181a24] border-black/[0.06] dark:border-white/10 shadow-apple-sm'
                                                : 'bg-black/[0.02] dark:bg-[#151720]/60 border-black/[0.04] dark:border-white/5 opacity-80'
                                        }`}
                                    >
                                        <div className="flex items-start justify-between">
                                            <div>
                                                <div className="flex items-center space-x-2 flex-wrap gap-y-1">
                                                    <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">{m.name}</h4>
                                                    {m.is_active && (
                                                        <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-teal-500 text-slate-950 shadow-sm">
                                                            Active Engine
                                                        </span>
                                                    )}
                                                    {m.is_default && !m.is_active && (
                                                        <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-700 dark:text-cyan-300">
                                                            Default Tier
                                                        </span>
                                                    )}
                                                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-black/[0.04] dark:bg-white/5 text-slate-600 dark:text-slate-400 border border-black/[0.06] dark:border-white/5">
                                                        {m.quantization}
                                                    </span>
                                                </div>
                                                <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                                                    Architecture: {m.architecture} · License: {m.license}
                                                </p>
                                                <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
                                                    Hardware: {m.recommended_hardware}
                                                </p>
                                                {m.repo_id && (
                                                    <div className="text-[10px] text-teal-600 dark:text-teal-400 font-mono mt-1">
                                                        HuggingFace: {m.repo_id}
                                                    </div>
                                                )}
                                                {m.local_path && (
                                                    <div className="text-[10px] text-slate-500 dark:text-slate-400 font-mono mt-1 truncate max-w-lg">
                                                        Snapshot: {m.local_path}
                                                    </div>
                                                )}
                                            </div>

                                            <div className="flex flex-col items-end space-y-2 flex-shrink-0">
                                                <span className="text-xs font-mono font-semibold text-slate-700 dark:text-slate-300">
                                                    {m.size_gb} GB
                                                </span>

                                                {m.is_installed ? (
                                                    <div className="flex items-center space-x-2">
                                                        <span className="flex items-center space-x-1 text-xs font-semibold text-teal-700 dark:text-teal-400 bg-teal-500/10 px-2.5 py-1 rounded-xl border border-teal-500/20">
                                                            <CheckCircle2 size={13} />
                                                            <span>Ready</span>
                                                        </span>

                                                        {m.category === 'audio' && !m.is_active && (
                                                            <button
                                                                onClick={() => handleActivateModel(m.id)}
                                                                disabled={activatingId === m.id}
                                                                className="px-2.5 py-1 bg-black/5 dark:bg-white/5 hover:bg-teal-500/20 text-slate-700 dark:text-slate-200 hover:text-teal-600 dark:hover:text-teal-300 font-bold rounded-xl text-xs transition-all border border-black/5 dark:border-white/5"
                                                            >
                                                                {activatingId === m.id ? 'Activating…' : 'Activate'}
                                                            </button>
                                                        )}
                                                    </div>
                                                ) : (
                                                    <button
                                                        onClick={() => handleDownload(m.repo_id || m.id)}
                                                        disabled={busy || !m.repo_id}
                                                        className="px-3.5 py-1.5 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs flex items-center space-x-1.5 transition-all shadow-sm active:scale-95 disabled:opacity-50"
                                                    >
                                                        {busy ? <Loader2 size={13} className="animate-spin" /> : <Download size={13} />}
                                                        <span>{busy ? 'Downloading…' : 'Download'}</span>
                                                    </button>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                        {/* REAL download progress — server-tracked, per-file, cancellable */}
                        {download && (
                            <div className={`mx-6 mb-4 p-4 rounded-2xl border space-y-2 ${
                                download.status === 'error'
                                    ? 'bg-rose-500/10 border-rose-500/30'
                                    : download.status === 'completed'
                                    ? 'bg-teal-500/10 border-teal-500/30'
                                    : 'bg-black/[0.03] dark:bg-white/5 border-black/[0.06] dark:border-white/10'
                            }`}>
                                <div className="flex items-center justify-between gap-3">
                                    <span className="text-xs font-bold text-slate-800 dark:text-slate-200 truncate flex items-center gap-2">
                                        {download.status === 'error' && <AlertTriangle size={14} className="text-rose-500" />}
                                        {download.status === 'completed' && <CheckCircle2 size={14} className="text-teal-500" />}
                                        {download.status === 'downloading' && <Loader2 size={14} className="animate-spin text-teal-500" />}
                                        {download.repo_id}
                                    </span>
                                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500">
                                        {download.status}
                                        {pct !== null && ` · ${pct}%`}
                                    </span>
                                </div>
                                {pct !== null && download.status === 'downloading' && (
                                    <div className="w-full h-1.5 bg-black/[0.06] dark:bg-white/10 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-gradient-to-r from-teal-500 to-cyan-400 rounded-full transition-all duration-500"
                                            style={{ width: `${pct}%` }}
                                        />
                                    </div>
                                )}
                                {download.status === 'downloading' && download.current_file && (
                                    <p className="text-[10px] font-mono text-slate-400 truncate">{download.current_file} · file {download.files_done + 1}/{download.total_files}</p>
                                )}
                                {(download.status === 'error' || downloadError) && (
                                    <p className="text-[11px] text-rose-600 dark:text-rose-400">{download.error || downloadError}</p>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    };
