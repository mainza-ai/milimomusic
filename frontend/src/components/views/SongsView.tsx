import React, { useState, useEffect, useMemo } from 'react';
import { type Job, type Project, projectApi } from '../../api';
import { Play, Pause, Heart, Sliders, Search, Music, Disc, Sparkles, Trash2, Mic2, Copy, Check, X, Layers, Info, Video, FolderKanban } from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';
import { AppFooter } from '../ui/AppFooter';

interface SongsViewProps {
    songs: Job[];
    currentJobId?: string | null;
    onPlay: (job: Job) => void;
    onOpenWorkspace: (job: Job) => void;
    onToggleFavorite: (jobId: string) => void;
    onExtend: (job: Job) => void;
    onDelete?: (jobId: string) => void;
    onSelectTrack?: (job: Job) => void;
    onOpenVideo?: (job: Job) => void;
}

export const SongsView: React.FC<SongsViewProps> = ({
    songs,
    currentJobId,
    onPlay,
    onOpenWorkspace,
    onToggleFavorite,
    onExtend,
    onDelete,
    onSelectTrack,
    onOpenVideo
}) => {
    const [search, setSearch] = useState('');
    const [viewMode, setViewMode] = useState<'table' | 'grid'>('table');
    const [selectedTag, setSelectedTag] = useState<string>('all');
    const [selectedLyricsSong, setSelectedLyricsSong] = useState<Job | null>(null);
    const [copied, setCopied] = useState(false);
    const [projects, setProjects] = useState<Project[]>([]);

    useEffect(() => {
        projectApi.listProjects().then(setProjects).catch(console.error);
    }, []);

    const projectMap = useMemo(() => {
        const map = new Map<string, Project>();
        projects.forEach(p => map.set(p.id, p));
        return map;
    }, [projects]);

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
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6 flex flex-col justify-between min-h-full">
            <div className="space-y-6">
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
                                    ? 'bg-teal-500/15 text-teal-700 dark:text-teal-300 border border-teal-500/30 font-bold'
                                    : 'bg-black/[0.03] dark:bg-white/5 text-slate-600 dark:text-slate-400 border border-transparent hover:bg-black/[0.06] dark:hover:bg-white/10'
                            }`}
                        >
                            All Genres
                        </button>
                        {allTags.slice(0, 8).map(tag => (
                            <button
                                key={tag}
                                onClick={() => setSelectedTag(tag)}
                                title={`Filter by ${tag}`}
                                aria-label={`Filter by ${tag}`}
                                className={`px-3 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
                                    selectedTag === tag
                                        ? 'bg-teal-500/15 text-teal-700 dark:text-teal-300 border border-teal-500/30 font-bold'
                                        : 'bg-black/[0.03] dark:bg-white/5 text-slate-600 dark:text-slate-400 border border-transparent hover:bg-black/[0.06] dark:hover:bg-white/10'
                                }`}
                            >
                                {tag}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Content List or Grid */}
                {filtered.length === 0 ? (
                    <div className="text-center py-20 space-y-3">
                        <Disc size={44} className="mx-auto text-slate-400 animate-spin-slow opacity-60" />
                        <h3 className="text-sm font-bold text-slate-700 dark:text-slate-300">No tracks found</h3>
                        <p className="text-xs text-slate-500 dark:text-slate-400">Generate a song or adjust your search filter.</p>
                    </div>
                ) : viewMode === 'table' ? (
                    <div className="bg-white/80 dark:bg-[#141620]/90 rounded-2xl border border-black/[0.06] dark:border-white/10 shadow-apple-sm overflow-hidden backdrop-blur-2xl">
                        <div className="overflow-x-auto">
                            <table className="w-full text-left text-xs min-w-[760px]">
                                <thead>
                                    <tr className="border-b border-black/[0.06] dark:border-white/10 text-slate-400 dark:text-slate-500 uppercase font-mono text-[10px] tracking-wider bg-black/[0.01] dark:bg-white/[0.02]">
                                        <th className="py-3 px-4 w-12 text-center">#</th>
                                        <th className="py-3 px-4">Track</th>
                                        <th className="py-3 px-4">Tags & Style</th>
                                        <th className="py-3 px-4">Stems & Notation</th>
                                        <th className="py-3 px-4">Engine</th>
                                        <th className="py-3 px-4 text-right">Actions</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-black/[0.04] dark:divide-white/5">
                                    {filtered.map(song => {
                                        const isCurrent = currentJobId === song.id;
                                        const tagsList = song.tags ? song.tags.split(',').map(t => t.trim()).filter(Boolean) : [];
                                        return (
                                            <tr
                                                key={song.id}
                                                className={`hover:bg-black/[0.02] dark:hover:bg-white/[0.03] transition-colors group ${
                                                    isCurrent ? 'bg-teal-500/5 dark:bg-teal-500/10' : ''
                                                }`}
                                            >
                                                {/* Play Trigger / Number */}
                                                <td className="py-3 px-4 text-center text-slate-400 font-mono w-12">
                                                    <button
                                                        onClick={() => onPlay(song)}
                                                        className="w-8 h-8 mx-auto rounded-xl bg-teal-500/10 hover:bg-teal-500 text-teal-700 dark:text-teal-300 hover:text-slate-950 flex items-center justify-center transition-all shadow-sm active:scale-95"
                                                        title={isCurrent ? "Pause Playback" : "Play Track"}
                                                    >
                                                        {isCurrent ? <Pause size={13} /> : <Play size={13} className="ml-0.5" />}
                                                    </button>
                                                </td>

                                                {/* Track Title & Prompt with Artwork */}
                                                <td className="py-3 px-4">
                                                    <div className="flex items-center gap-3">
                                                        {/* Artwork Thumbnail */}
                                                        <div
                                                            onClick={() => onSelectTrack?.(song)}
                                                            className="w-10 h-10 rounded-xl bg-gradient-to-tr from-teal-500/20 to-cyan-500/20 border border-black/[0.08] dark:border-white/10 p-0.5 flex-shrink-0 flex items-center justify-center relative overflow-hidden group/art cursor-pointer hover:scale-105 transition-transform"
                                                            title="Inspect Track Studio"
                                                        >
                                                            <img
                                                                src={song.cover_image_path ? (song.cover_image_path.startsWith('http') ? song.cover_image_path : song.cover_image_path) : '/milimo_logo.png'}
                                                                alt="Track"
                                                                className="w-full h-full object-cover rounded-lg"
                                                                onError={(e) => {
                                                                    (e.target as HTMLImageElement).src = '/milimo_logo.png';
                                                                }}
                                                            />
                                                            <Disc size={16} className="absolute text-teal-400 drop-shadow opacity-0 group-hover/art:opacity-100 transition-opacity" />
                                                        </div>

                                                        <div className="min-w-0 max-w-sm">
                                                            <div
                                                                onClick={() => onSelectTrack?.(song)}
                                                                className="font-bold text-slate-900 dark:text-slate-100 flex items-center gap-1.5 cursor-pointer hover:text-teal-600 dark:hover:text-teal-400 transition-colors truncate"
                                                                title="Open Track Studio"
                                                            >
                                                                <span className="truncate">{song.title || song.prompt.slice(0, 45)}</span>
                                                                {song.is_favorite && (
                                                                    <Heart size={12} className="fill-rose-500 text-rose-500 flex-shrink-0" />
                                                                )}
                                                                {song.project_id && projectMap.get(song.project_id) && (
                                                                    <span
                                                                        className="px-1.5 py-0.5 rounded-md bg-teal-500/10 text-[9px] font-mono text-teal-700 dark:text-teal-300 font-bold border border-teal-500/20 inline-flex items-center gap-1 flex-shrink-0"
                                                                        title={`Project: ${projectMap.get(song.project_id)!.name}`}
                                                                    >
                                                                        <FolderKanban size={9} />
                                                                        <span className="truncate max-w-[75px]">{projectMap.get(song.project_id)!.name}</span>
                                                                    </span>
                                                                )}
                                                            </div>
                                                            <p
                                                                onClick={() => onSelectTrack?.(song)}
                                                                className="text-[11px] text-slate-500 dark:text-slate-400 truncate cursor-pointer hover:text-slate-700 dark:hover:text-slate-200 mt-0.5"
                                                                title="Open Track Studio"
                                                            >
                                                                {song.prompt}
                                                            </p>
                                                        </div>
                                                    </div>
                                                </td>

                                                {/* Tags & Style (Horizontal Inline Micro-Pills) */}
                                                <td className="py-3 px-4">
                                                    <div className="flex items-center gap-1.5 flex-wrap max-w-xs">
                                                        {tagsList.length > 0 ? (
                                                            <>
                                                                {tagsList.slice(0, 2).map((t, idx) => (
                                                                    <span
                                                                        key={idx}
                                                                        className="px-2 py-0.5 rounded-md bg-black/[0.04] dark:bg-white/5 text-[10px] font-mono text-slate-700 dark:text-slate-300 border border-black/[0.04] dark:border-white/5 truncate max-w-[110px]"
                                                                        title={t}
                                                                    >
                                                                        {t}
                                                                    </span>
                                                                ))}
                                                                {tagsList.length > 2 && (
                                                                    <span
                                                                        className="px-1.5 py-0.5 rounded-md bg-teal-500/10 text-[10px] font-mono text-teal-700 dark:text-teal-300 font-bold border border-teal-500/20"
                                                                        title={tagsList.slice(2).join(', ')}
                                                                    >
                                                                        +{tagsList.length - 2}
                                                                    </span>
                                                                )}
                                                            </>
                                                        ) : (
                                                            <span className="text-[11px] text-slate-400 italic">Studio Master</span>
                                                        )}
                                                    </div>
                                                </td>

                                                {/* Stems & MIDI Badges */}
                                                <td className="py-3 px-4">
                                                    <div className="flex items-center gap-1.5">
                                                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20 font-bold flex items-center gap-1">
                                                            <Music size={10} />
                                                            MIDI
                                                        </span>
                                                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-cyan-500/10 text-cyan-700 dark:text-cyan-300 border border-cyan-500/20 font-bold flex items-center gap-1">
                                                            <Layers size={10} />
                                                            4 Stems
                                                        </span>
                                                    </div>
                                                </td>

                                                {/* Model Engine */}
                                                <td className="py-3 px-4">
                                                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-black/[0.04] dark:bg-white/5 text-slate-600 dark:text-slate-400 font-medium">
                                                        {song.model_provider || 'MiniMax Music 3'}
                                                    </span>
                                                </td>

                                                {/* Single-Line Non-Colliding Actions Toolbar */}
                                                <td className="py-3 px-4 text-right">
                                                    <div className="flex items-center justify-end gap-1 flex-nowrap">
                                                        {onSelectTrack && (
                                                            <button
                                                                onClick={() => onSelectTrack(song)}
                                                                className="px-2.5 py-1 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 font-bold rounded-xl text-[11px] transition-all flex items-center gap-1 border border-black/[0.04] dark:border-white/5 active:scale-95"
                                                                title="Inspect Sound Details & Stems"
                                                            >
                                                                <Info size={11} />
                                                                <span>Details</span>
                                                            </button>
                                                        )}
                                                        {song.lyrics && (
                                                            <button
                                                                onClick={() => setSelectedLyricsSong(song)}
                                                                className="p-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
                                                                title="View Lyrics"
                                                            >
                                                                <Mic2 size={14} />
                                                            </button>
                                                        )}
                                                        <button
                                                            onClick={() => onToggleFavorite(song.id)}
                                                            className="p-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-rose-500 transition-colors"
                                                            title="Favorite"
                                                        >
                                                            <Heart size={14} className={song.is_favorite ? 'fill-rose-500 text-rose-500' : ''} />
                                                        </button>
                                                        <button
                                                            onClick={() => onOpenWorkspace(song)}
                                                            className="px-3 py-1 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold rounded-xl text-[11px] transition-all flex items-center gap-1 shadow-sm active:scale-95"
                                                            title="Open DAW Multitrack Workspace"
                                                        >
                                                            <Sliders size={12} />
                                                            <span>DAW</span>
                                                        </button>
                                                        {onOpenVideo && (
                                                            <button
                                                                onClick={() => onOpenVideo(song)}
                                                                className="px-2.5 py-1 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-700 dark:text-cyan-300 font-bold rounded-xl text-[11px] transition-all flex items-center gap-1 border border-cyan-500/20 active:scale-95"
                                                                title="Create Music Video"
                                                            >
                                                                <Video size={12} />
                                                                <span>Video</span>
                                                            </button>
                                                        )}
                                                        <button
                                                            onClick={() => onExtend(song)}
                                                            className="p-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
                                                            title="Extend Audio Tail"
                                                        >
                                                            <Sparkles size={14} />
                                                        </button>
                                                        {onDelete && (
                                                            <button
                                                                onClick={() => onDelete(song.id)}
                                                                className="p-1.5 rounded-xl hover:bg-rose-500/10 text-slate-400 hover:text-rose-500 transition-colors"
                                                                title="Delete Track"
                                                            >
                                                                <Trash2 size={14} />
                                                            </button>
                                                        )}
                                                    </div>
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                        {filtered.map(song => (
                            <GlassCard key={song.id} className="p-4 space-y-3 group hover:border-teal-500/40 transition-all flex flex-col justify-between">
                                <div className="space-y-3">
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
                                        <div className="flex items-center gap-1.5">
                                            <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 group-hover/title:text-teal-600 dark:group-hover/title:text-teal-400 transition-colors truncate flex-1">
                                                {song.title || song.prompt.slice(0, 35)}
                                            </h4>
                                            {song.project_id && projectMap.get(song.project_id) && (
                                                <span
                                                    className="px-1.5 py-0.5 rounded-md bg-teal-500/10 text-[9px] font-mono text-teal-700 dark:text-teal-300 font-bold border border-teal-500/20 inline-flex items-center gap-1 flex-shrink-0"
                                                    title={`Project: ${projectMap.get(song.project_id)!.name}`}
                                                >
                                                    <FolderKanban size={9} />
                                                    <span className="truncate max-w-[65px]">{projectMap.get(song.project_id)!.name}</span>
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-[11px] text-slate-500 dark:text-slate-400 line-clamp-2 mt-0.5 leading-relaxed">
                                            {song.prompt}
                                        </p>
                                    </div>
                                </div>

                                <div className="pt-3 border-t border-black/[0.06] dark:border-white/5 space-y-2">
                                    <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 dark:text-slate-400">
                                        <span className="font-bold text-teal-600 dark:text-teal-400">MIDI + 4 Stems</span>
                                        <span>{song.model_provider || 'MiniMax M3'}</span>
                                    </div>
                                    <div className="flex items-center justify-between gap-1 pt-1">
                                        <button
                                            onClick={() => onSelectTrack?.(song)}
                                            className="px-2.5 py-1 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 font-bold rounded-xl text-[10px] transition-all"
                                        >
                                            Details
                                        </button>
                                        <div className="flex items-center gap-1">
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
                                                className="px-2.5 py-1 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-[10px] flex items-center gap-1 shadow-sm transition-all"
                                                title="Open DAW Workspace"
                                            >
                                                <Sliders size={11} />
                                                <span>DAW</span>
                                            </button>
                                            {onOpenVideo && (
                                                <button
                                                    onClick={() => onOpenVideo(song)}
                                                    className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-cyan-600 dark:hover:text-cyan-400 transition-colors"
                                                    title="Generate Music Video"
                                                >
                                                    <Video size={13} />
                                                </button>
                                            )}
                                            <button
                                                onClick={() => onToggleFavorite(song.id)}
                                                className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-rose-500 transition-colors"
                                                title="Favorite"
                                            >
                                                <Heart size={13} className={song.is_favorite ? 'fill-rose-500 text-rose-500' : ''} />
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </GlassCard>
                        ))}
                    </div>
                )}
            </div>

            {/* Global Creator Footer */}
            <AppFooter />

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
