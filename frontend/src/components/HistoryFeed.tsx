import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { api, type Job } from '../api';
import { AudioPlayer } from './AudioPlayer';
import { Music, AlertCircle, Clock, Disc, Edit2, Check, Trash2, ArrowRightCircle } from 'lucide-react'; // Added ArrowRightCircle

// ... (existing imports)



interface HistoryFeedProps {
    history: Job[];
    currentJobId: string | null;
    onRefresh: () => void;
    onExtend?: (job: Job) => void; // Phase 9
}

export const HistoryFeed: React.FC<HistoryFeedProps> = ({ history, currentJobId, onRefresh, onExtend }) => {
    const scrollRef = useRef<HTMLDivElement>(null);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [tempTitle, setTempTitle] = useState("");
    const [progress, setProgress] = useState(0);

    // Real-time Progress (SSE)
    useEffect(() => {
        const handleProgress = (e: any) => {
            const data = e.detail;
            // If this component was mapping multiple jobs, we'd check ID. 
            // But here we might be rendering many.
            // Actually, we can't easily hook a global listener to a list item without filtering.
            // Wait, the progress bar is inside the map.
            // The state 'progress' needs to be localized or we need a global store.
            // For simplicity, let's assume we only track the *current* active job's progress globally
            // and pass it down? Or simpler: Use a map of progresses if we had multiple.
            // But we only generate one at a time.
            if (data.progress) setProgress(data.progress);
        };
        window.addEventListener('milimo_progress', handleProgress);
        return () => window.removeEventListener('milimo_progress', handleProgress);
    }, []);

    // Auto-scroll to top when new job added
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = 0;
        }
    }, [history.length, currentJobId]);

    const handleRenameStart = (job: Job) => {
        setEditingId(job.id);
        setTempTitle(job.title || job.prompt || "Untitled");
    };

    const handleRenameSave = async (jobId: string) => {
        if (!tempTitle.trim()) return;
        try {
            await api.renameJob(jobId, tempTitle);
            setEditingId(null);
            // Hint: A full refresh might be needed, but optimistic UI (polling) will catch it eventually
        } catch (e) {
            console.error("Rename failed", e);
        }
    };

    const handleDelete = async (jobId: string) => {
        if (!confirm("Are you sure you want to delete this track? This action cannot be undone.")) return;
        try {
            await api.deleteJob(jobId);
            onRefresh(); // Immediate update
        } catch (e) {
            console.error("Delete failed", e);
            alert("Failed to delete track");
        }
    };

    return (
        <div className="h-full flex flex-col overflow-hidden relative">
            {/* Feed Header */}
            <div className="p-8 pb-6 flex items-end gap-4">
                <div>
                    <h1 className="text-4xl font-bold text-slate-900 tracking-tighter">TRACK <span className="text-cyan-500 font-mono text-sm ml-2">HISTORY</span></h1>
                    <p className="text-slate-500 font-mono text-xs uppercase tracking-widest mt-1">Your Creations</p>
                </div>
            </div>

            {/* List */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-8 pt-0 space-y-6 pb-32">
                <AnimatePresence initial={false}>
                    {history.map((job) => (
                        <motion.div
                            key={job.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            layout
                            transition={{ duration: 0.3, ease: "easeOut" }}
                        >
                            <div className={`
                                group relative overflow-hidden rounded-sm border transition-all duration-300
                                ${job.id === currentJobId
                                    ? 'bg-white/80 border-cyan-400 shadow-xl shadow-cyan-500/10'
                                    : 'bg-white/40 border-slate-200/50 hover:bg-white/60 hover:border-slate-300 hover:shadow-lg'
                                }
                                backdrop-blur-md
                            `}>
                                <div className="p-6">
                                    {/* Header Row: Status, Title, Meta */}
                                    <div className="flex justify-between items-start mb-4">
                                        <div className="flex items-center gap-4">
                                            {/* Icon / Vinyl */}
                                            <div className={`w-12 h-12 rounded-sm flex items-center justify-center shadow-inner ${job.status === 'completed' ? 'bg-gradient-to-br from-cyan-50 to-white text-cyan-600 border border-cyan-100' :
                                                job.status === 'failed' ? 'bg-red-50 text-red-500' : 'bg-amber-50 text-amber-500 animate-pulse'
                                                }`}>
                                                {job.status === 'completed' ? <Disc className="w-6 h-6 animate-[spin_10s_linear_infinite]" /> : <Music className="w-6 h-6" />}
                                            </div>

                                            {/* Title & Prompt */}
                                            <div>
                                                {editingId === job.id ? (
                                                    <div className="flex items-center gap-2">
                                                        <input
                                                            autoFocus
                                                            className="text-lg font-bold bg-white border border-cyan-300 rounded-sm px-2 py-0.5 focus:outline-none focus:ring-2 ring-cyan-500 w-full font-mono text-slate-800"
                                                            value={tempTitle}
                                                            onChange={e => setTempTitle(e.target.value)}
                                                            onKeyDown={e => e.key === 'Enter' && handleRenameSave(job.id)}
                                                        />
                                                        <button onClick={() => handleRenameSave(job.id)} className="p-1 bg-cyan-100 text-cyan-700 rounded-full hover:bg-cyan-200"><Check className="w-4 h-4" /></button>
                                                    </div>
                                                ) : (
                                                    <div className="flex items-center gap-2 group/title cursor-pointer" onClick={() => handleRenameStart(job)}>
                                                        <h3 className="text-lg font-bold text-slate-800 line-clamp-1 hover:text-cyan-600 transition-colors">
                                                            {job.title || job.prompt || "Untitled Specimen"}
                                                        </h3>
                                                        <Edit2 className="w-3 h-3 text-slate-300 opacity-0 group-hover/title:opacity-100 transition-opacity" />
                                                    </div>
                                                )}

                                                <p className="text-xs font-mono text-slate-500 line-clamp-1 mt-0.5 opacity-70">
                                                    ID: {job.id.slice(0, 8)}... | {job.prompt}
                                                </p>
                                            </div>
                                        </div>

                                        {/* Status Badge */}
                                        <div className="text-right">
                                            <div className="flex items-center justify-end gap-2 mb-1">
                                                <span className={`inline-flex items-center px-2 py-1 rounded-sm text-[10px] font-mono font-bold uppercase tracking-wide border ${job.status === 'completed' ? 'bg-teal-50 text-teal-700 border-teal-200' :
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
                                                    className="p-1 text-slate-300 hover:text-red-500 transition-colors"
                                                    title="Delete Track"
                                                >
                                                    <Trash2 className="w-3.5 h-3.5" />
                                                </button>
                                            </div>

                                            {/* Phase 9: Extend Button */}
                                            {job.status === 'completed' && onExtend && (
                                                <button
                                                    onClick={() => onExtend(job)}
                                                    className="mb-2 text-xs flex items-center gap-1 text-cyan-500 hover:text-cyan-400 font-mono transition-colors ml-auto border border-cyan-500/20 px-2 py-1 rounded bg-cyan-500/5 hover:bg-cyan-500/10"
                                                >
                                                    <ArrowRightCircle className="w-3 h-3" />
                                                    EXTEND
                                                </button>
                                            )}

                                            {/* Progress Bar for Processing */}
                                            {job.status === 'processing' && (
                                                <div className="w-32 ml-auto">
                                                    <div className="h-1 bg-slate-200 rounded-full overflow-hidden">
                                                        <motion.div
                                                            className="h-full bg-cyan-500"
                                                            initial={{ width: "0%" }}
                                                            animate={{ width: `${Math.min(100, progress)}%` }} // Use real progress state
                                                            transition={{ duration: 0.2 }} // Smooth out the jumps
                                                        />
                                                    </div>
                                                    <p className="text-[10px] text-cyan-600 font-mono mt-1 text-right animate-pulse">
                                                        Synthesizing... {progress}%
                                                    </p>
                                                </div>
                                            )}

                                            {job.status !== 'processing' && (
                                                <p className="text-[10px] text-slate-400 flex items-center justify-end gap-1 font-mono">
                                                    <Clock className="w-3 h-3" />
                                                    {new Date(job.created_at + "Z").toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}
                                                </p>
                                            )}
                                        </div>
                                    </div>

                                    {/* Audio Player */}
                                    {job.status === 'completed' && job.audio_path && (
                                        <div className="mt-4 bg-slate-50/50 rounded-sm p-2 border border-slate-100">
                                            <AudioPlayer audioUrl={job.audio_path} title={job.title || job.prompt || "Untitled"} jobId={job.id} />
                                        </div>
                                    )}

                                    {/* Error Message */}
                                    {job.status === 'failed' && (
                                        <div className="mt-4 p-3 bg-rose-50/50 border border-rose-100 text-rose-600 text-xs rounded-lg flex items-center gap-2 font-mono">
                                            <AlertCircle className="w-4 h-4 text-rose-500" />
                                            ERROR: {job.error_msg || "Generation sequence aborted."}
                                        </div>
                                    )}

                                    {/* Lyrics Accordion */}
                                    {job.lyrics && (
                                        <div className="mt-4">
                                            <details className="group/lyrics">
                                                <summary className="text-[10px] uppercase font-bold text-slate-400 cursor-pointer hover:text-cyan-600 transition-colors list-none flex items-center gap-1 font-mono">
                                                    <span>Lyric Data</span>
                                                    <span className="group-open/lyrics:rotate-180 transition-transform">▼</span>
                                                </summary>
                                                <div className="mt-2 text-sm text-slate-600 whitespace-pre-line font-mono bg-white/50 border border-slate-100 p-4 rounded-sm text-xs leading-relaxed">
                                                    {job.lyrics}
                                                </div>
                                            </details>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </AnimatePresence>

                {history.length === 0 && (
                    <div className="h-96 flex flex-col items-center justify-center text-slate-300">
                        <div className="w-24 h-24 rounded-full border-4 border-slate-100 flex items-center justify-center mb-6">
                            <Music className="w-10 h-10 opacity-20" />
                        </div>
                        <p className="text-sm font-medium font-mono">Awaiting input data...</p>
                        <p className="text-xs opacity-60 font-mono">Generate a track to begin.</p>
                    </div>
                )}
            </div>
        </div>
    );
};
