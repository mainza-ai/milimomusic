import React, { useEffect, useRef, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { api, type Job } from '../api';
import { AudioPlayer } from './AudioPlayer';
import { Music, AlertCircle, Disc, Edit2, Check, Trash2, ArrowRightCircle, Search, Calendar, Heart } from 'lucide-react';

interface HistoryFeedProps {
    history: Job[];
    currentJobId: string | null;
    onRefresh: () => void;
    onExtend?: (job: Job) => void;
    onLoadMore: () => void;
    hasMore: boolean;
    onFilterChange: (status: string) => void;
    currentFilter: string;
    onSearch: (query: string) => void;
    searchQuery: string;
    isLoadingMore?: boolean;
    onToggleFavorite: (id: string) => void;
}

export const HistoryFeed: React.FC<HistoryFeedProps> = ({
    history,
    currentJobId,
    onRefresh,
    onExtend,
    onLoadMore,
    hasMore,
    onFilterChange,
    currentFilter,
    onSearch,
    searchQuery,
    isLoadingMore,
    onToggleFavorite
}) => {
    const scrollRef = useRef<HTMLDivElement>(null);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [tempTitle, setTempTitle] = useState("");
    const [progress, setProgress] = useState(0);
    const [expandedLyrics, setExpandedLyrics] = useState<Set<string>>(new Set());

    const toggleLyrics = (id: string) => {
        const newSet = new Set(expandedLyrics);
        if (newSet.has(id)) {
            newSet.delete(id);
        } else {
            newSet.add(id);
        }
        setExpandedLyrics(newSet);
    };

    // Real-time Progress (SSE)
    useEffect(() => {
        const handleProgress = (e: any) => {
            const data = e.detail;
            if (data.progress) setProgress(data.progress);
        };
        window.addEventListener('milimo_progress', handleProgress);
        return () => window.removeEventListener('milimo_progress', handleProgress);
    }, []);

    // Auto-scroll to top only when A NEW job is added at the very top (currentJobId changes to something new)
    // We don't want to scroll to top when "Loading More" happens.
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
        if (!confirm("Are you sure you want to delete this track? This action cannot be undone.")) return;
        try {
            await api.deleteJob(jobId);
            onRefresh();
        } catch (e) {
            console.error("Delete failed", e);
            alert("Failed to delete track");
        }
    };

    // Group by Date using UTC forced timestamps
    const groupedHistory = useMemo(() => {
        const groups: { [key: string]: Job[] } = {};

        history.forEach(job => {
            // Ensure UTC
            const timeStr = job.created_at.endsWith("Z") ? job.created_at : job.created_at + "Z";
            const date = new Date(timeStr);
            const dateKey = date.toDateString(); // "Wed Jan 21 2026" (Local)

            if (!groups[dateKey]) groups[dateKey] = [];
            groups[dateKey].push(job);
        });
        return groups;
    }, [history]);

    // Order groups by Date Descending (Future -> Today -> Yesterday -> Older)
    const sortedGroupKeys = Object.keys(groupedHistory).sort((a, b) => {
        return new Date(b).getTime() - new Date(a).getTime();
    });

    const getGroupLabel = (dateKey: string) => {
        const today = new Date().toDateString();
        const yesterday = new Date(Date.now() - 86400000).toDateString();
        if (dateKey === today) return "Today";
        if (dateKey === yesterday) return "Yesterday";
        return dateKey; // "Thu Jan 15 2026"
    };

    const [localSearch, setLocalSearch] = useState(searchQuery);

    // Sync local search when prop changes (e.g. clear)
    useEffect(() => {
        setLocalSearch(searchQuery);
    }, [searchQuery]);

    // Search Debounce or Enter
    const handleSearchInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        setLocalSearch(e.target.value);
    };

    const handleSearchSubmit = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            onSearch(localSearch);
        }
    };

    return (
        <div className="h-full flex flex-col overflow-hidden relative">
            {/* Feed Header */}
            <div className="p-8 pb-4 flex flex-col gap-4 bg-white/50 backdrop-blur-sm border-b border-slate-200/50 z-10">
                <div className="flex items-end justify-between">
                    <div>
                        <h1 className="text-4xl font-bold text-slate-900 tracking-tighter">TRACK <span className="text-cyan-500 font-mono text-sm ml-2">HISTORY</span></h1>
                        <p className="text-slate-500 font-mono text-xs uppercase tracking-widest mt-1">Your Creations</p>
                    </div>
                    {/* Status Filters */}
                    <div className="flex bg-slate-100/50 p-1 rounded-sm gap-1">
                        {['all', 'favorites', 'completed', 'failed'].map(status => (
                            <button
                                key={status}
                                onClick={() => onFilterChange(status)}
                                className={`px-3 py-1 rounded-sm text-[10px] uppercase font-bold tracking-wide transition-all ${currentFilter === status
                                    ? 'bg-white shadow-sm text-cyan-600'
                                    : 'text-slate-400 hover:text-slate-600'
                                    }`}
                            >
                                {status}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Search Bar */}
                <div className="relative">
                    <Search className={`absolute left-3 top-2.5 w-4 h-4 text-slate-400 transition-colors ${localSearch ? 'text-cyan-500' : ''}`} />
                    <input
                        value={localSearch}
                        onChange={handleSearchInput}
                        onKeyDown={handleSearchSubmit}
                        onBlur={() => onSearch(localSearch)} // Auto-search on blur too for convenience
                        placeholder="Search tracks... (Press Enter)"
                        className="w-full pl-9 pr-4 py-2 bg-white border border-slate-200 rounded-sm text-sm focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400/20 transition-all font-mono placeholder:text-slate-300 text-slate-700"
                    />
                </div>
            </div>

            {/* List */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-8 pt-4 space-y-8 pb-32 scroll-smooth">
                {sortedGroupKeys.map(groupLabel => (
                    <div key={groupLabel}>
                        <div className="flex items-center gap-2 mb-4 opacity-50 sticky top-0 bg-slate-50/95 py-2 z-10 backdrop-blur-sm">
                            <Calendar className="w-3 h-3 text-slate-400" />
                            <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">{getGroupLabel(groupLabel)}</h3>
                            <div className="h-px bg-slate-200 flex-1 ml-2"></div>
                        </div>

                        <div className="space-y-4">
                            <AnimatePresence initial={false}>
                                {groupedHistory[groupLabel].map((job) => (
                                    <motion.div
                                        key={job.id}
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        layout
                                        transition={{ duration: 0.2 }}
                                    >
                                        <div className={`
                                            group relative overflow-hidden rounded-sm border transition-all duration-300
                                            ${job.id === currentJobId
                                                ? 'bg-white/80 border-cyan-400 shadow-xl shadow-cyan-500/10'
                                                : 'bg-white/40 border-slate-200/50 hover:bg-white/60 hover:border-slate-300 hover:shadow-lg'
                                            }
                                            backdrop-blur-md
                                        `}>
                                            <div className="p-5">
                                                {/* Header Row: Status, Title, Meta */}
                                                <div className="flex justify-between items-start mb-4">
                                                    <div className="flex items-center gap-4">
                                                        {/* Icon / Vinyl */}
                                                        <div className={`w-10 h-10 rounded-sm flex items-center justify-center shadow-inner ${job.status === 'completed' ? 'bg-gradient-to-br from-cyan-50 to-white text-cyan-600 border border-cyan-100' :
                                                            job.status === 'failed' ? 'bg-red-50 text-red-500' : 'bg-amber-50 text-amber-500 animate-pulse'
                                                            }`}>
                                                            {job.status === 'completed' ? <Disc className="w-5 h-5 animate-[spin_10s_linear_infinite]" /> : <Music className="w-5 h-5" />}
                                                        </div>

                                                        {/* Title & Prompt */}
                                                        <div className="flex-1 min-w-0">
                                                            {editingId === job.id ? (
                                                                <div className="flex items-center gap-2">
                                                                    <input
                                                                        autoFocus
                                                                        className="text-base font-bold bg-white border border-cyan-300 rounded-sm px-2 py-0.5 focus:outline-none focus:ring-2 ring-cyan-500 w-full font-mono text-slate-800"
                                                                        value={tempTitle}
                                                                        onChange={e => setTempTitle(e.target.value)}
                                                                        onKeyDown={e => e.key === 'Enter' && handleRenameSave(job.id)}
                                                                    />
                                                                    <button onClick={() => handleRenameSave(job.id)} className="p-1 bg-cyan-100 text-cyan-700 rounded-full hover:bg-cyan-200"><Check className="w-4 h-4" /></button>
                                                                </div>
                                                            ) : (
                                                                <div className="flex items-center gap-2 group/title cursor-pointer" onClick={() => handleRenameStart(job)}>
                                                                    <h3 className="text-base font-bold text-slate-800 line-clamp-1 hover:text-cyan-600 transition-colors">
                                                                        {job.title || job.prompt || "Untitled Specimen"}
                                                                    </h3>
                                                                    <Edit2 className="w-3 h-3 text-slate-300 opacity-0 group-hover/title:opacity-100 transition-opacity" />
                                                                </div>
                                                            )}

                                                            <p className="text-[10px] font-mono text-slate-500 line-clamp-1 mt-0.5 opacity-70">
                                                                {job.id.slice(0, 8)}... | {job.prompt}
                                                            </p>

                                                            {/* Clickable Tags */}
                                                            {job.tags && (
                                                                <div className="flex flex-wrap gap-1 mt-1.5">
                                                                    {job.tags.split(',').slice(0, 6).map((tag, idx) => (
                                                                        <button
                                                                            key={idx}
                                                                            onClick={(e) => {
                                                                                e.stopPropagation();
                                                                                onSearch(tag.trim());
                                                                            }}
                                                                            className="text-[9px] font-mono bg-slate-100 hover:bg-cyan-100 text-slate-500 hover:text-cyan-700 px-1.5 py-0.5 rounded-sm transition-colors border border-transparent hover:border-cyan-200"
                                                                            title={`Filter by ${tag.trim()}`}
                                                                        >
                                                                            #{tag.trim()}
                                                                        </button>
                                                                    ))}
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>

                                                    {/* Status Badge */}
                                                    <div className="text-right flex flex-col items-end gap-1">
                                                        <div className="flex items-center gap-2">
                                                            {/* Favorite Button */}
                                                            <button
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    onToggleFavorite(job.id);
                                                                }}
                                                                className={`p-1 rounded-sm transition-colors ${job.is_favorite
                                                                    ? "text-rose-500 hover:bg-rose-50"
                                                                    : "text-slate-300 hover:text-rose-500 hover:bg-rose-50"
                                                                    }`}
                                                                title={job.is_favorite ? "Remove from favorites" : "Add to favorites"}
                                                            >
                                                                <Heart className={`w-3.5 h-3.5 ${job.is_favorite ? "fill-current" : ""}`} />
                                                            </button>

                                                            <span className={`inline-flex items-center px-1.5 py-0.5 rounded-sm text-[9px] font-mono font-bold uppercase tracking-wide border ${job.status === 'completed' ? 'bg-teal-50 text-teal-700 border-teal-200' :
                                                                job.status === 'failed' ? 'bg-rose-50 text-rose-700 border-rose-200' :
                                                                    'bg-amber-50 text-amber-700 border-amber-200'
                                                                }`}>
                                                                {job.status === 'processing' && (
                                                                    <span className="w-1.5 h-1.5 bg-amber-500 rounded-full mr-1.5 animate-pulse" />
                                                                )}
                                                                {job.status}
                                                            </span>
                                                            <button
                                                                onClick={() => handleDelete(job.id)}
                                                                className="text-slate-300 hover:text-red-500 transition-colors"
                                                                title="Delete Track"
                                                            >
                                                                <Trash2 className="w-3 h-3" />
                                                            </button>
                                                        </div>

                                                        {job.status === 'completed' && onExtend && (
                                                            <button
                                                                onClick={() => onExtend(job)}
                                                                className="text-[9px] flex items-center gap-1 text-cyan-600 hover:text-cyan-500 font-mono transition-colors font-bold uppercase tracking-wider"
                                                            >
                                                                <ArrowRightCircle className="w-3 h-3" />
                                                                Extend
                                                            </button>
                                                        )}
                                                    </div>
                                                </div>

                                                {/* Progress Bar for Processing */}
                                                {job.status === 'processing' && (
                                                    <div className="w-full mt-2 mb-2">
                                                        <div className="h-0.5 bg-slate-100 rounded-full overflow-hidden">
                                                            <motion.div
                                                                className="h-full bg-cyan-500"
                                                                initial={{ width: "0%" }}
                                                                animate={{ width: `${Math.min(100, progress)}%` }}
                                                                transition={{ duration: 0.2 }}
                                                            />
                                                        </div>
                                                        <p className="text-[9px] text-cyan-600 font-mono mt-1 text-right animate-pulse">
                                                            Synthesizing... {progress}%
                                                        </p>
                                                    </div>
                                                )}

                                                {/* Audio Player */}
                                                {job.status === 'completed' && job.audio_path && (
                                                    <div className="mt-3 bg-slate-50/50 rounded-sm p-1.5 border border-slate-100">
                                                        <AudioPlayer audioUrl={job.audio_path} title={job.title || job.prompt || "Untitled"} jobId={job.id} />
                                                    </div>
                                                )}

                                                {/* Lyrics Dropdown */}
                                                {job.lyrics && (
                                                    <div className="mt-3 border-t border-slate-100 pt-2">
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                toggleLyrics(job.id);
                                                            }}
                                                            className="text-[10px] bg-slate-50 hover:bg-slate-100 text-slate-500 hover:text-cyan-600 px-2 py-1 rounded-sm w-full text-left flex items-center justify-between transition-colors font-mono"
                                                        >
                                                            <span>Lyrics Data</span>
                                                            <span className="opacity-70">{expandedLyrics.has(job.id) ? 'Hide' : 'Show'}</span>
                                                        </button>

                                                        {expandedLyrics.has(job.id) && (
                                                            <div className="mt-2 bg-slate-50 rounded-sm p-3 border border-slate-100 text-[10px] font-mono text-slate-600 whitespace-pre-wrap max-h-48 overflow-y-auto">
                                                                {job.lyrics}
                                                            </div>
                                                        )}
                                                    </div>
                                                )}

                                                {/* Error Message */}
                                                {job.status === 'failed' && (
                                                    <div className="mt-3 p-2 bg-rose-50/50 border border-rose-100 text-rose-600 text-[10px] rounded-sm flex items-center gap-2 font-mono">
                                                        <AlertCircle className="w-3 h-3 text-rose-500" />
                                                        {job.error_msg || "Generation sequence aborted."}
                                                    </div>
                                                )}

                                                {/* Timestamp Footer */}
                                                <div className="mt-2 text-[9px] text-slate-300 font-mono text-right">
                                                    {new Date(job.created_at).toLocaleTimeString()}
                                                </div>
                                            </div>
                                        </div>
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                        </div>
                    </div>
                ))}

                {history.length === 0 && (
                    <div className="h-64 flex flex-col items-center justify-center text-slate-300">
                        <div className="w-16 h-16 rounded-full border-2 border-slate-100 flex items-center justify-center mb-4">
                            <Music className="w-8 h-8 opacity-20" />
                        </div>
                        <p className="text-sm font-medium font-mono">No tracks found</p>
                        <p className="text-xs opacity-60 font-mono">Try changing filters or generate a new track.</p>
                    </div>
                )}

                {/* Load More Button */}
                {hasMore && (
                    <div className="flex justify-center pt-4">
                        <button
                            onClick={onLoadMore}
                            disabled={isLoadingMore}
                            className="text-xs font-bold uppercase tracking-widest text-slate-400 hover:text-cyan-600 transition-colors border-b border-dashed border-slate-300 hover:border-cyan-400 pb-0.5 disabled:opacity-50"
                        >
                            {isLoadingMore ? "Loading..." : "Load More History"}
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};
