import React, { useState } from 'react';
import { type Job } from '../../api';
import { Play, Pause, Heart, Sliders, Search, Music, Disc, Sparkles, Trash2, Mic2, Copy, Check, X } from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';

interface SongsViewProps {
    songs: Job[];
    currentJobId?: string | null;
    onPlay: (job: Job) => void;
    onOpenWorkspace: (job: Job) => void;
    onToggleFavorite: (jobId: string) => void;
    onExtend: (job: Job) => void;
    onDelete?: (jobId: string) => void;
    onSelectTrack?: (job: Job) => void;
}

export const SongsView: React.FC<SongsViewProps> = ({
    songs,
    currentJobId,
    onPlay,
    onOpenWorkspace,
    onToggleFavorite,
    onExtend,
    onDelete,
    onSelectTrack
}) => {
    const [search, setSearch] = useState('');
    const [viewMode, setViewMode] = useState<'grid' | 'table'>('table');
    const [selectedTag, setSelectedTag] = useState<string>('all');
    const [selectedLyricsSong, setSelectedLyricsSong] = useState<Job | null>(null);
    const [copied, setCopied] = useState(false);

    const completedSongs = songs.filter(s => s.status === 'completed' && s.audio_path);

    const allTags = Array.from(
        new Set(
            completedSongs
                .flatMap(s => (s.tags ? s.tags.split(',').map(t => t.trim()) : []))
                .filter(Boolean)
        )
    );

    const filtered = completedSongs.filter(s => {
        const matchesSearch =
            (s.title || '').toLowerCase().includes(search.toLowerCase()) ||
            (s.prompt || '').toLowerCase().includes(search.toLowerCase()) ||
            (s.tags || '').toLowerCase().includes(search.toLowerCase());
        const matchesTag = selectedTag === 'all' || (s.tags && s.tags.includes(selectedTag));
        return matchesSearch && matchesTag;
    });

    return (
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
            {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                    <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white flex items-center gap-3">
                        <span className="p-2 rounded-2xl bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20">
                            🎵
                        </span>
                        <span>Song Library</span>
                    </h1>
                    <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
                        {completedSongs.length} studio tracks generated · Full 48kHz audio, stems, and MIDI notes
                    </p>
                </div>

                <div className="flex items-center space-x-3">
                    {/* View Switcher */}
                    <div className="flex items-center bg-black/[0.04] dark:bg-white/5 p-1 rounded-xl border border-black/[0.06] dark:border-white/10">
                        <button
                            onClick={() => setViewMode('table')}
                            title="List Table View"
                            aria-label="List Table View"
                            className={`px-3 py-1 text-xs font-semibold rounded-lg transition-all ${
                                viewMode === 'table'
                                    ? 'bg-white dark:bg-white/20 text-teal-600 dark:text-teal-300 shadow-sm'
                                    : 'text-slate-500 dark:text-slate-400'
                            }`}
                        >
                            List
                        </button>
                        <button
                            onClick={() => setViewMode('grid')}
                            title="Card Grid View"
                            aria-label="Card Grid View"
                            className={`px-3 py-1 text-xs font-semibold rounded-lg transition-all ${
                                viewMode === 'grid'
                                    ? 'bg-white dark:bg-white/20 text-teal-600 dark:text-teal-300 shadow-sm'
                                    : 'text-slate-500 dark:text-slate-400'
                            }`}
                        >
                            Grid
                        </button>
                    </div>
                </div>
            </div>

            {/* Filter & Search Bar */}
            <div className="flex flex-col sm:flex-row items-center gap-3">
                <div className="relative flex-1 w-full">
                    <Search size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
                    <input
                        type="text"
                        placeholder="Search song library by title, genre, prompt, or tags..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="w-full pl-10 pr-4 py-2.5 apple-input text-xs"
                    />
                </div>

                <div className="flex items-center gap-1.5 overflow-x-auto max-w-full pb-1 sm:pb-0">
                    <button
                        onClick={() => setSelectedTag('all')}
                        title="Show All Genres"
                        aria-label="Show All Genres"
                        className={`px-3 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
                            selectedTag === 'all'
                                ? 'bg-teal-500/15 text-teal-700 dark:text-teal-300 border border-teal-500/30'
                                : 'bg-black/[0.03] dark:bg-white/5 text-slate-600 dark:text-slate-400 border border-transparent'
                        }`}
                    >
                        All Genres
                    </button>
                    {allTags.slice(0, 6).map(tag => (
                        <button
                            key={tag}
                            onClick={() => setSelectedTag(tag)}
                            title={`Filter by ${tag}`}
                            aria-label={`Filter by ${tag}`}
                            className={`px-3 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
                                selectedTag === tag
                                    ? 'bg-teal-500/15 text-teal-700 dark:text-teal-300 border border-teal-500/30'
                                    : 'bg-black/[0.03] dark:bg-white/5 text-slate-600 dark:text-slate-400 border border-transparent'
                            }`}
                        >
                            {tag}
                        </button>
                    ))}
                </div>
            </div>

            {/* Content List or Grid */}
            {filtered.length === 0 ? (
                <div className="text-center py-16 space-y-3">
                    <Disc size={40} className="mx-auto text-slate-400 animate-spin-slow" />
                    <h3 className="text-sm font-bold text-slate-700 dark:text-slate-300">No tracks found</h3>
                    <p className="text-xs text-slate-500 dark:text-slate-400">Generate a song or adjust your search filter.</p>
                </div>
            ) : viewMode === 'table' ? (
                <div className="bg-white/70 dark:bg-[#141620]/80 rounded-2xl border border-black/[0.06] dark:border-white/10 shadow-apple-sm overflow-hidden backdrop-blur-xl">
                    <table className="w-full text-left text-xs">
                        <thead>
                            <tr className="border-b border-black/[0.06] dark:border-white/10 text-slate-400 dark:text-slate-500 uppercase font-mono text-[10px] tracking-wider">
                                <th className="py-3 px-4 w-12 text-center">#</th>
                                <th className="py-3 px-4">Title & Description</th>
                                <th className="py-3 px-4 hidden md:table-cell">Tags / Style</th>
                                <th className="py-3 px-4 hidden sm:table-cell">Stems & MIDI</th>
                                <th className="py-3 px-4 hidden lg:table-cell">Engine</th>
                                <th className="py-3 px-4 text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-black/[0.04] dark:divide-white/5">
                            {filtered.map((song) => {
                                const isCurrent = currentJobId === song.id;
                                return (
                                    <tr
                                        key={song.id}
                                        className={`hover:bg-black/[0.02] dark:hover:bg-white/[0.03] transition-colors group ${
                                            isCurrent ? 'bg-teal-500/5' : ''
                                        }`}
                                    >
                                        <td className="py-3 px-4 text-center text-slate-400 font-mono">
                                            <button
                                                onClick={() => onPlay(song)}
                                                className="w-7 h-7 mx-auto rounded-full bg-teal-500/10 hover:bg-teal-500 text-teal-700 dark:text-teal-300 hover:text-slate-950 flex items-center justify-center transition-all"
                                            >
                                                {isCurrent ? <Pause size={12} /> : <Play size={12} className="ml-0.5" />}
                                            </button>
                                        </td>
                                        <td className="py-3 px-4">
                                            <div 
                                                onClick={() => onSelectTrack?.(song)}
                                                className="font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2 cursor-pointer hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
                                            >
                                                <span>{song.title || song.prompt.slice(0, 40)}</span>
                                                {song.is_favorite && (
                                                    <Heart size={12} className="fill-rose-500 text-rose-500 flex-shrink-0" />
                                                )}
                                            </div>
                                            <div 
                                                onClick={() => onSelectTrack?.(song)}
                                                className="text-[11px] text-slate-500 dark:text-slate-400 truncate max-w-sm cursor-pointer"
                                            >
                                                {song.prompt}
                                            </div>
                                        </td>
                                        <td className="py-3 px-4 hidden md:table-cell">
                                            <span className="px-2 py-0.5 rounded-lg bg-black/[0.04] dark:bg-white/5 text-[11px] font-mono text-slate-600 dark:text-slate-400">
                                                {song.tags || 'Pop / Electronic'}
                                            </span>
                                        </td>
                                        <td className="py-3 px-4 hidden sm:table-cell">
                                            <div className="flex items-center gap-1.5">
                                                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20 font-bold flex items-center gap-1">
                                                    <Music size={10} />
                                                    MIDI Ready
                                                </span>
                                                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-700 dark:text-sky-300 border border-sky-500/20 font-bold">
                                                    4 Stems
                                                </span>
                                            </div>
                                        </td>
                                        <td className="py-3 px-4 hidden lg:table-cell">
                                            <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
                                                {song.model_provider || 'MiniMax Music 3'}
                                            </span>
                                        </td>
                                        <td className="py-3 px-4 text-right space-x-1.5">
                                            {onSelectTrack && (
                                                <button
                                                    onClick={() => onSelectTrack(song)}
                                                    className="px-2.5 py-1 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 font-bold rounded-lg text-[11px] transition-all inline-flex items-center gap-1"
                                                    title="Inspect Track Details & Stems"
                                                >
                                                    <span>Details</span>
                                                </button>
                                            )}
                                            {song.lyrics && (
                                                <button
                                                    onClick={() => setSelectedLyricsSong(song)}
                                                    className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 transition-colors inline-flex items-center"
                                                    title="View Lyrics"
                                                >
                                                    <Mic2 size={13} />
                                                </button>
                                            )}
                                            <button
                                                onClick={() => onToggleFavorite(song.id)}
                                                className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-rose-500 transition-colors"
                                                title="Favorite"
                                            >
                                                <Heart size={14} className={song.is_favorite ? 'fill-rose-500 text-rose-500' : ''} />
                                            </button>
                                            <button
                                                onClick={() => onOpenWorkspace(song)}
                                                className="px-2.5 py-1 bg-teal-500/10 hover:bg-teal-500 text-teal-700 dark:text-teal-300 hover:text-slate-950 font-bold rounded-lg text-[11px] transition-all inline-flex items-center gap-1"
                                                title="Open DAW Workspace"
                                            >
                                                <Sliders size={12} />
                                                <span>DAW</span>
                                            </button>
                                            <button
                                                onClick={() => onExtend(song)}
                                                className="px-2.5 py-1 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 font-bold rounded-lg text-[11px] transition-all inline-flex items-center gap-1"
                                                title="Extend Track"
                                            >
                                                <Sparkles size={12} />
                                                <span>Extend</span>
                                            </button>
                                            {onDelete && (
                                                <button
                                                    onClick={() => onDelete(song.id)}
                                                    className="p-1.5 rounded-lg hover:bg-rose-500/10 text-slate-400 hover:text-rose-500 transition-colors inline-flex items-center"
                                                    title="Delete Song"
                                                >
                                                    <Trash2 size={13} />
                                                </button>
                                            )}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {filtered.map(song => (
                        <GlassCard key={song.id} className="p-4 space-y-3 group hover:border-teal-500/40 transition-all">
                            <div className="relative aspect-video rounded-xl bg-gradient-to-br from-teal-500/20 to-cyan-500/20 flex items-center justify-center overflow-hidden border border-black/[0.06] dark:border-white/10">
                                <Disc size={32} className="text-teal-500 group-hover:scale-110 transition-transform" />
                                <button
                                    onClick={() => onPlay(song)}
                                    title={`Play ${song.title || 'track'}`}
                                    aria-label={`Play ${song.title || 'track'}`}
                                    className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity"
                                >
                                    <div className="w-10 h-10 rounded-full bg-teal-500 text-slate-950 flex items-center justify-center shadow-lg font-bold">
                                        <Play size={16} className="ml-0.5" />
                                    </div>
                                </button>
                            </div>

                            <div 
                                onClick={() => onSelectTrack?.(song)}
                                className="cursor-pointer group/title"
                            >
                                <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 group-hover/title:text-teal-600 dark:group-hover/title:text-teal-400 transition-colors truncate">
                                    {song.title || song.prompt.slice(0, 35)}
                                </h4>
                                <p className="text-[11px] text-slate-500 dark:text-slate-400 line-clamp-2 mt-0.5">
                                    {song.prompt}
                                </p>
                            </div>

                            <div className="flex items-center justify-between pt-2 border-t border-black/[0.06] dark:border-white/5">
                                <span className="text-[10px] font-mono text-teal-600 dark:text-teal-400 font-bold">
                                    MIDI + Stems
                                </span>
                                <div className="flex items-center gap-1.5">
                                    {song.lyrics && (
                                        <button
                                            onClick={() => setSelectedLyricsSong(song)}
                                            className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
                                            title="View Lyrics"
                                        >
                                            <Mic2 size={13} />
                                        </button>
                                    )}
                                    <button
                                        onClick={() => onOpenWorkspace(song)}
                                        className="p-1.5 rounded-lg bg-teal-500/10 hover:bg-teal-500 text-teal-700 dark:text-teal-300 hover:text-slate-950 transition-colors"
                                        title="Open DAW Workspace"
                                    >
                                        <Sliders size={13} />
                                    </button>
                                    <button
                                        onClick={() => onToggleFavorite(song.id)}
                                        className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-rose-500 transition-colors"
                                        title="Favorite"
                                    >
                                        <Heart size={13} className={song.is_favorite ? 'fill-rose-500 text-rose-500' : ''} />
                                    </button>
                                    {onDelete && (
                                        <button
                                            onClick={() => onDelete(song.id)}
                                            className="p-1.5 rounded-lg hover:bg-rose-500/10 text-slate-400 hover:text-rose-500 transition-colors"
                                            title="Delete Song"
                                        >
                                            <Trash2 size={13} />
                                        </button>
                                    )}
                                </div>
                            </div>
                        </GlassCard>
                    ))}
                </div>
            )}

            {/* Apple Music Style Lyrics Sheet Modal */}
            {selectedLyricsSong && (
                <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-md flex items-center justify-center p-4 animate-fade-in">
                    <div className="bg-white/95 dark:bg-[#12141c]/95 border border-black/[0.08] dark:border-white/10 rounded-3xl p-6 shadow-apple-2xl backdrop-blur-3xl w-full max-w-2xl max-h-[80vh] flex flex-col animate-scale-up">
                        {/* Modal Header */}
                        <div className="flex items-center justify-between border-b border-black/[0.06] dark:border-white/[0.08] pb-4 mb-4 flex-shrink-0">
                            <div className="flex items-center space-x-3 min-w-0">
                                <div className="p-2.5 rounded-2xl bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20">
                                    <Mic2 size={18} />
                                </div>
                                <div className="min-w-0">
                                    <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 truncate">
                                        {selectedLyricsSong.title || selectedLyricsSong.prompt}
                                    </h3>
                                    <p className="text-[11px] text-slate-500 dark:text-slate-400 font-mono mt-0.5">
                                        {selectedLyricsSong.tags || 'Track Lyrics'}
                                    </p>
                                </div>
                            </div>

                            <div className="flex items-center space-x-2">
                                <button
                                    onClick={() => {
                                        if (selectedLyricsSong.lyrics) {
                                            navigator.clipboard.writeText(selectedLyricsSong.lyrics);
                                            setCopied(true);
                                            setTimeout(() => setCopied(false), 2000);
                                        }
                                    }}
                                    className="px-3 py-1.5 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 text-xs font-semibold flex items-center gap-1.5 transition-colors"
                                >
                                    {copied ? <Check size={13} className="text-teal-500" /> : <Copy size={13} />}
                                    <span>{copied ? 'Copied' : 'Copy'}</span>
                                </button>
                                <button
                                    onClick={() => setSelectedLyricsSong(null)}
                                    className="p-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
                                >
                                    <X size={16} />
                                </button>
                            </div>
                        </div>

                        {/* Lyrics Body */}
                        <div className="flex-1 overflow-y-auto pr-2 select-text font-sans space-y-3">
                            {selectedLyricsSong.lyrics ? (
                                <pre className="text-xs sm:text-sm font-sans leading-relaxed text-slate-800 dark:text-slate-200 whitespace-pre-wrap">
                                    {selectedLyricsSong.lyrics}
                                </pre>
                            ) : (
                                <p className="text-xs text-slate-400 text-center py-12">No lyrics recorded for this track.</p>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
