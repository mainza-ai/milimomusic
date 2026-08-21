import React, { useEffect, useRef, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Job } from '../api';
import { api } from '../api';
import { AudioPlayer } from './AudioPlayer';
import { Edit2, Check, Trash2, Search, Calendar, Heart, Sliders, Layers, Sparkles, Info, Clock } from 'lucide-react';

interface HistoryFeedProps {
    history: Job[];
    currentJobId: string | null;
    onRefresh: () => void;
    onExtend?: (job: Job) => void;
    onOpenWorkspace?: (job: Job) => void;
    onLoadMore?: () => void;
    hasMore?: boolean;
    onFilterChange: (status: string) => void;
    currentFilter: string;
    onSearch: (query: string) => void;
    searchQuery: string;
    isLoadingMore?: boolean;
    onToggleFavorite: (id: string) => void;
    onDelete?: (id: string) => void;
    onSelectTrack?: (job: Job) => void;
}

export const HistoryFeed: React.FC<HistoryFeedProps> = ({
    history,
    currentJobId,
    onRefresh,
    onOpenWorkspace,
    onFilterChange,
    currentFilter,
    onSearch,
    searchQuery,
    onToggleFavorite,
    onDelete,
    onSelectTrack
}) => {
    const scrollRef = useRef<HTMLDivElement>(null);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [tempTitle, setTempTitle] = useState("");
    const [localSearch, setLocalSearch] = useState(searchQuery);
    const [lyricsOpen, setLyricsOpen] = useState<Record<string, boolean>>({});

    useEffect(() => {
        setLocalSearch(searchQuery);
    }, [searchQuery]);

    useEffect(() => {
        if (currentJobId && scrollRef.current) {
            scrollRef.current.scrollTop = 0;
        }
    }, [currentJobId]);

    const handleRenameStart = (job: Job) => {
        setEditingId(job.id);
        setTempTitle(job.title || job.prompt || "Untitled");
    };

    const handleRenameSave = async (jobId: string) => {
        if (!tempTitle.trim()) return;
        try {
            await api.renameJob(jobId, tempTitle);
            setEditingId(null);
            onRefresh();
        } catch (e) {
            console.error("Rename failed", e);
        }
    };

    const handleDelete = async (jobId: string) => {
        if (onDelete) {
            onDelete(jobId);
            return;
        }
        if (!confirm("Are you sure you want to delete this track? This action cannot be undone.")) return;
        try {
            await api.deleteJob(jobId);
            onRefresh();
        } catch (e) {
            console.error("Delete failed", e);
        }
    };

    const getGroupLabel = (dateStr: string) => {
        const date = new Date(dateStr);
        const today = new Date();
        const yesterday = new Date();
        yesterday.setDate(yesterday.getDate() - 1);

        if (date.toDateString() === today.toDateString()) return "Today";
        if (date.toDateString() === yesterday.toDateString()) return "Yesterday";
        return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    };

    const groupedHistory = useMemo(() => {
        const groups: Record<string, Job[]> = {};
        history.forEach(job => {
            const date = new Date(job.created_at.endsWith("Z") ? job.created_at : job.created_at + "Z").toDateString();
            if (!groups[date]) groups[date] = [];
            groups[date].push(job);
        });
        return groups;
    }, [history]);

    const sortedGroupKeys = useMemo(() => {
        return Object.keys(groupedHistory).sort((a, b) => new Date(b).getTime() - new Date(a).getTime());
    }, [groupedHistory]);

    return (
        <div className="h-full flex flex-col bg-transparent text-slate-900 dark:text-slate-100 overflow-hidden select-none">
            {/* Filter and Search Bar */}
            <div className="p-5 border-b border-black/[0.06] dark:border-white/[0.08] bg-white/40 dark:bg-[#12141c]/50 backdrop-blur-xl space-y-3">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                        <span className="text-sm font-bold tracking-tight text-slate-900 dark:text-slate-100">
                            Sessions & History
                        </span>
                        <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20 font-semibold">
                            {history.length}
                        </span>
                    </div>

                    <div className="flex items-center space-x-1 bg-black/[0.04] dark:bg-white/5 p-1 rounded-xl border border-black/[0.06] dark:border-white/10">
                        {['all', 'completed', 'favorites'].map(status => (
                            <button
                                key={status}
                                onClick={() => onFilterChange(status)}
                                className={`px-3 py-1 text-xs font-semibold rounded-lg capitalize transition-all ${
                                    currentFilter === status
                                        ? 'bg-white dark:bg-white/20 text-teal-700 dark:text-teal-300 shadow-apple-sm font-bold'
                                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                                }`}
                            >
                                {status}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Search */}
                <div className="relative">
                    <Search className="absolute left-3.5 top-2.5 w-4 h-4 text-slate-400 dark:text-slate-500" />
                    <input
                        value={localSearch}
                        onChange={(e) => {
                            setLocalSearch(e.target.value);
                            onSearch(e.target.value);
                        }}
                        placeholder="Search sessions, tags, prompts..."
                        className="w-full pl-10 pr-4 py-2 bg-white/80 dark:bg-white/5 border border-black/[0.08] dark:border-white/10 rounded-xl text-xs text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/40 focus:border-teal-500 font-medium placeholder:text-slate-400 dark:placeholder:text-slate-500 shadow-sm"
                    />
                </div>
            </div>

            {/* List */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-6">
                {sortedGroupKeys.map(groupLabel => (
                    <div key={groupLabel}>
                        <div className="flex items-center gap-2 mb-3 text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                            <Calendar className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" />
                            <span>{getGroupLabel(groupLabel)}</span>
                            <div className="h-px bg-black/[0.06] dark:bg-white/10 flex-1 ml-2" />
                        </div>

                        <div className="space-y-3">
                            <AnimatePresence initial={false}>
                                {groupedHistory[groupLabel].map((job) => {
                                    const hasStemsAndMidi = Boolean(job.midi_path || job.notes_json);

                                    return (
                                        <motion.div
                                            key={job.id}
                                            initial={{ opacity: 0, y: 8 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            layout
                                            className={`p-4 rounded-2xl border transition-all ${
                                                job.id === currentJobId
                                                    ? 'bg-white dark:bg-[#181b26] border-teal-500/40 shadow-apple-md'
                                                    : 'bg-white/80 dark:bg-[#141620]/90 border-black/[0.06] dark:border-white/[0.08] hover:bg-white dark:hover:bg-[#181b26] hover:shadow-apple-sm'
                                            }`}
                                        >
                                            <div className="flex justify-between items-start mb-3">
                                                <div className="flex items-center gap-3">
                                                    {/* Track Icon / Logo */}
                                                    <div 
                                                        onClick={() => onSelectTrack?.(job)}
                                                        className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-sm shadow-sm overflow-hidden p-1 cursor-pointer hover:scale-105 transition-transform ${
                                                            job.status === 'completed'
                                                                ? 'bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20'
                                                                : job.status === 'failed'
                                                                ? 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20'
                                                                : 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20 animate-pulse'
                                                        }`}
                                                        title="Inspect Track Studio"
                                                    >
                                                        <img src="/milimo_logo.png" alt="Track" className="w-full h-full object-cover rounded-lg" onError={(e) => {
                                                            (e.target as HTMLElement).style.display = 'none';
                                                        }} />
                                                    </div>

                                                    <div className="flex-1 min-w-0">
                                                        <div className="flex items-center gap-2 flex-wrap">
                                                            {editingId === job.id ? (
                                                                <div className="flex items-center gap-1.5">
                                                                    <input
                                                                        autoFocus
                                                                        className="text-xs font-bold bg-white dark:bg-slate-900 border border-teal-500 rounded-lg px-2 py-0.5 text-slate-900 dark:text-slate-100"
                                                                        value={tempTitle}
                                                                        onChange={e => setTempTitle(e.target.value)}
                                                                        onKeyDown={e => e.key === 'Enter' && handleRenameSave(job.id)}
                                                                    />
                                                                    <button onClick={() => handleRenameSave(job.id)} className="p-1 bg-teal-500 text-slate-950 rounded-lg">
                                                                        <Check size={12} />
                                                                    </button>
                                                                </div>
                                                            ) : (
                                                                <div className="flex items-center gap-1.5">
                                                                    <h3
                                                                        onClick={() => onSelectTrack?.(job)}
                                                                        className="text-xs font-bold text-slate-900 dark:text-slate-100 hover:text-teal-600 dark:hover:text-teal-400 transition-colors cursor-pointer"
                                                                        title="Open Track Studio"
                                                                    >
                                                                        {job.title || job.prompt || "Untitled Master"}
                                                                    </h3>
                                                                    <button
                                                                        onClick={(e) => {
                                                                            e.stopPropagation();
                                                                            handleRenameStart(job);
                                                                        }}
                                                                        className="p-1 rounded-md text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
                                                                        title="Rename Track"
                                                                    >
                                                                        <Edit2 size={11} />
                                                                    </button>
                                                                </div>
                                                            )}

                                                            {hasStemsAndMidi && (
                                                                <span className="text-[9px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20 flex items-center gap-1 font-semibold">
                                                                    <Layers size={10} />
                                                                    Stems & MIDI
                                                                </span>
                                                            )}
                                                        </div>

                                                        <p
                                                            onClick={() => onSelectTrack?.(job)}
                                                            className="text-[11px] text-slate-500 dark:text-slate-400 truncate max-w-md mt-0.5 cursor-pointer hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
                                                            title="Open Track Studio"
                                                        >
                                                            {job.prompt}
                                                        </p>

                                                        {/* Sound Metadata Badges */}
                                                        <div className="flex items-center gap-1.5 mt-1.5 flex-wrap">
                                                            {job.status === 'completed' && onSelectTrack && (
                                                                <button
                                                                    onClick={() => onSelectTrack(job)}
                                                                    className="text-[9px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 hover:bg-teal-500/20 text-teal-700 dark:text-teal-300 border border-teal-500/20 flex items-center gap-1 font-semibold transition-colors cursor-pointer"
                                                                    title="Inspect Full Sound Details (BPM, Key, Stems, MIDI, Hyperparameters)"
                                                                >
                                                                    <Info size={10} />
                                                                    <span>Sound Details</span>
                                                                </button>
                                                            )}

                                                            {job.duration_ms && (
                                                                <span
                                                                    onClick={() => onSelectTrack?.(job)}
                                                                    className="text-[9px] font-mono px-1.5 py-0.5 rounded-md bg-black/[0.03] dark:bg-white/5 text-slate-500 dark:text-slate-400 flex items-center gap-1 cursor-pointer hover:text-teal-500 transition-colors"
                                                                    title="Track Duration"
                                                                >
                                                                    <Clock size={9} />
                                                                    <span>{Math.floor(job.duration_ms / 1000 / 60)}:{Math.floor((job.duration_ms / 1000) % 60).toString().padStart(2, '0')}</span>
                                                                </span>
                                                            )}

                                                            {job.model_provider && (
                                                                <span
                                                                    onClick={() => onSelectTrack?.(job)}
                                                                    className="text-[9px] font-mono px-1.5 py-0.5 rounded-md bg-black/[0.03] dark:bg-white/5 text-slate-500 dark:text-slate-400 truncate max-w-[120px] cursor-pointer hover:text-teal-500 transition-colors"
                                                                    title="Model Engine"
                                                                >
                                                                    {job.model_provider}
                                                                </span>
                                                            )}

                                                            {job.tags && (
                                                                <span
                                                                    onClick={() => onSelectTrack?.(job)}
                                                                    className="text-[9px] font-mono px-1.5 py-0.5 rounded-md bg-black/[0.03] dark:bg-white/5 text-teal-600 dark:text-teal-400 truncate max-w-[140px] cursor-pointer hover:underline"
                                                                    title={job.tags}
                                                                >
                                                                    {job.tags.split(',').slice(0, 2).join(', ')}
                                                                </span>
                                                            )}
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="flex items-center gap-1.5 flex-wrap justify-end">
                                                    <button
                                                        onClick={() => onToggleFavorite(job.id)}
                                                        className={`p-1.5 rounded-lg transition-colors ${
                                                            job.is_favorite ? "text-rose-500 bg-rose-500/10" : "text-slate-400 hover:text-rose-500"
                                                        }`}
                                                        title={job.is_favorite ? "Remove from Favorites" : "Add to Favorites"}
                                                    >
                                                        <Heart size={14} className={job.is_favorite ? "fill-current" : ""} />
                                                    </button>

                                                    {job.status === 'completed' && onSelectTrack && (
                                                        <button
                                                            onClick={() => onSelectTrack(job)}
                                                            className="px-2.5 py-1 rounded-xl bg-black/5 dark:bg-white/10 hover:bg-teal-500/20 text-slate-700 dark:text-slate-200 hover:text-teal-600 dark:hover:text-teal-300 font-bold text-[10px] flex items-center gap-1 transition-all border border-black/5 dark:border-white/10 active:scale-95 shadow-sm"
                                                            title="Inspect Track Studio (Stems, MIDI, Score, Provenance)"
                                                        >
                                                            <Sparkles size={11} className="text-teal-500" />
                                                            <span>Track Studio</span>
                                                        </button>
                                                    )}

                                                    {job.status === 'completed' && onOpenWorkspace && (
                                                        <button
                                                            onClick={() => onOpenWorkspace(job)}
                                                            className="px-3 py-1 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-[10px] flex items-center gap-1 transition-all shadow-sm active:scale-95"
                                                        >
                                                            <Sliders size={11} />
                                                            <span>DAW</span>
                                                        </button>
                                                    )}

                                                    <button
                                                        onClick={() => handleDelete(job.id)}
                                                        className="p-1.5 rounded-lg text-slate-400 hover:text-rose-500 hover:bg-rose-500/10 transition-colors"
                                                    >
                                                        <Trash2 size={13} />
                                                    </button>
                                                </div>
                                            </div>

                                            {(job.status === 'processing' || job.status === 'queued') && (
                                                <div className="mt-3 pt-3 border-t border-black/[0.04] dark:border-white/5 space-y-2">
                                                    <div className="flex items-center justify-between text-[11px] font-mono">
                                                        <span className="text-teal-600 dark:text-teal-400 font-bold flex items-center gap-1.5 animate-pulse">
                                                            <Sparkles size={13} className="animate-spin-slow text-teal-500" />
                                                            <span>{job.status === 'queued' ? 'Queued in Studio Pipeline...' : 'Synthesizing 48kHz audio & separating 4 stems...'}</span>
                                                        </span>
                                                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 font-bold">
                                                            {job.model_provider || 'MiniMax Music 3'}
                                                        </span>
                                                    </div>
                                                    <div className="w-full h-1.5 bg-black/[0.04] dark:bg-white/5 rounded-full overflow-hidden">
                                                        <div className="h-full bg-gradient-to-r from-teal-500 via-cyan-400 to-sky-500 rounded-full animate-pulse shadow-sm" style={{ width: '70%' }} />
                                                    </div>
                                                </div>
                                            )}

                                            {job.status === 'failed' && (
                                                <div className="mt-2 pt-2 border-t border-rose-500/20 text-[11px] text-rose-500 font-medium">
                                                    Generation Failed: {job.error_msg || 'Unknown error occurred during generation'}
                                                </div>
                                            )}

                                            {job.status === 'completed' && job.audio_path && (
                                                <div className="mt-2 pt-2 border-t border-black/[0.04] dark:border-white/5">
                                                    {job.lyrics && (
                                                        <div className="mb-2 rounded-xl bg-black/[0.02] dark:bg-white/[0.03] border border-black/[0.04] dark:border-white/5 p-3">
                                                            <button
                                                                onClick={() => setLyricsOpen(prev => ({ ...prev, [job.id]: !prev[job.id] }))}
                                                                className="w-full flex items-center justify-between text-[10px] font-mono font-bold text-teal-700 dark:text-teal-300 uppercase tracking-wider"
                                                            >
                                                                <span>Lyrics</span>
                                                                <span className="text-[9px] text-slate-400">
                                                                    {lyricsOpen[job.id] ? 'Hide ▲' : 'Show ▼'}
                                                                </span>
                                                            </button>
                                                            <pre
                                                                className={`mt-2 text-[11px] leading-relaxed text-slate-600 dark:text-slate-300 font-sans whitespace-pre-line ${
                                                                    lyricsOpen[job.id] ? '' : 'line-clamp-3'
                                                                }`}
                                                                style={lyricsOpen[job.id] ? undefined : { WebkitLineClamp: 3, overflow: 'hidden', display: '-webkit-box', WebkitBoxOrient: 'vertical' }}
                                                            >
                                                                {job.lyrics}
                                                            </pre>
                                                        </div>
                                                    )}
                                                    <AudioPlayer
                                                        audioUrl={job.audio_path}
                                                        title={job.title || job.prompt || "Untitled"}
                                                        jobId={job.id}
                                                    />
                                                </div>
                                            )}
                                        </motion.div>
                                    );
                                })}
                            </AnimatePresence>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};
