import React, { useState } from 'react';
import { type Job } from '../../api';
import { Plus, ListMusic, Play, Music2, FolderPlus, Trash2 } from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';

interface PlaylistsViewProps {
    songs: Job[];
    onPlaySong: (job: Job) => void;
    onOpenWorkspace: (job: Job) => void;
}

interface Playlist {
    id: string;
    name: string;
    description: string;
    coverColor: string;
    songIds: string[];
    createdAt: string;
}

export const PlaylistsView: React.FC<PlaylistsViewProps> = ({
    songs,
    onPlaySong,
    onOpenWorkspace
}) => {
    const [playlists, setPlaylists] = useState<Playlist[]>(() => {
        const saved = localStorage.getItem('milimo_playlists');
        if (saved) {
            try { return JSON.parse(saved); } catch (e) {}
        }
        return [
            {
                id: 'pl-favorites',
                name: 'Favorites & Masterpieces',
                description: 'Curated studio tracks and favorite AI compositions',
                coverColor: 'from-rose-500 to-amber-500',
                songIds: songs.filter(s => s.is_favorite).map(s => s.id),
                createdAt: '2026-08-20'
            },
            {
                id: 'pl-synthwave',
                name: 'Midnight Synthwave Drive',
                description: 'Neon retro synth anthems, punchy drums, and 80s hooks',
                coverColor: 'from-cyan-500 to-sky-600',
                songIds: songs.filter(s => (s.tags || '').toLowerCase().includes('synth')).map(s => s.id),
                createdAt: '2026-08-20'
            },
            {
                id: 'pl-daw-masters',
                name: 'DAW Ready Multitracks',
                description: 'Full stems with transcribed MIDI and MusicXML scores',
                coverColor: 'from-teal-500 to-emerald-600',
                songIds: songs.slice(0, 4).map(s => s.id),
                createdAt: '2026-08-20'
            }
        ];
    });

    const [selectedPlaylistId, setSelectedPlaylistId] = useState<string | null>(playlists[0]?.id || null);
    const [isCreating, setIsCreating] = useState(false);
    const [newPlaylistName, setNewPlaylistName] = useState('');
    const [newPlaylistDesc, setNewPlaylistDesc] = useState('');

    const savePlaylists = (updated: Playlist[]) => {
        setPlaylists(updated);
        localStorage.setItem('milimo_playlists', JSON.stringify(updated));
    };

    const handleCreatePlaylist = (e: React.FormEvent) => {
        e.preventDefault();
        if (!newPlaylistName.trim()) return;

        const colors = [
            'from-teal-500 to-cyan-500',
            'from-cyan-500 to-blue-600',
            'from-rose-500 to-pink-600',
            'from-amber-500 to-orange-600',
            'from-emerald-500 to-teal-700'
        ];
        const randomColor = colors[Math.floor(Math.random() * colors.length)];

        const newPl: Playlist = {
            id: `pl-${Date.now()}`,
            name: newPlaylistName.trim(),
            description: newPlaylistDesc.trim() || 'Custom studio playlist',
            coverColor: randomColor,
            songIds: [],
            createdAt: new Date().toISOString().split('T')[0]
        };

        const updated = [...playlists, newPl];
        savePlaylists(updated);
        setSelectedPlaylistId(newPl.id);
        setNewPlaylistName('');
        setNewPlaylistDesc('');
        setIsCreating(false);
    };

    const handleDeletePlaylist = (id: string) => {
        const updated = playlists.filter(p => p.id !== id);
        savePlaylists(updated);
        if (selectedPlaylistId === id) {
            setSelectedPlaylistId(updated[0]?.id || null);
        }
    };

    const activePlaylist = playlists.find(p => p.id === selectedPlaylistId);
    const activePlaylistSongs = activePlaylist
        ? songs.filter(s => activePlaylist.songIds.includes(s.id))
        : [];

    return (
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
            {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                    <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white flex items-center gap-3">
                        <span className="p-2 rounded-2xl bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20">
                            📑
                        </span>
                        <span>Playlists & Albums</span>
                    </h1>
                    <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
                        Organize your studio tracks into albums, project playlists, and release collections
                    </p>
                </div>

                <button
                    onClick={() => setIsCreating(true)}
                    title="Create a new custom album or playlist"
                    aria-label="Create Playlist"
                    className="px-4 py-2.5 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-2 transition-all shadow-md shadow-teal-500/20 active:scale-95 self-start sm:self-auto"
                >
                    <Plus size={15} />
                    <span>Create Playlist</span>
                </button>
            </div>

            {/* Create Playlist Modal / Inline Form */}
            {isCreating && (
                <GlassCard className="p-5 border-teal-500/30 animate-fade-in space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                            <FolderPlus size={16} className="text-teal-500" />
                            Create New Playlist
                        </h3>
                        <button
                            onClick={() => setIsCreating(false)}
                            className="text-xs text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
                        >
                            Cancel
                        </button>
                    </div>

                    <form onSubmit={handleCreatePlaylist} className="space-y-3">
                        <input
                            type="text"
                            placeholder="Playlist Name e.g., 'Synthwave Album 2026' or 'Acoustic Sessions'"
                            value={newPlaylistName}
                            onChange={(e) => setNewPlaylistName(e.target.value)}
                            className="w-full apple-input text-xs"
                            required
                        />
                        <input
                            type="text"
                            placeholder="Description (optional)"
                            value={newPlaylistDesc}
                            onChange={(e) => setNewPlaylistDesc(e.target.value)}
                            className="w-full apple-input text-xs"
                        />
                        <div className="flex justify-end gap-2 pt-2">
                            <button
                                type="button"
                                onClick={() => setIsCreating(false)}
                                className="px-4 py-2 rounded-xl bg-black/[0.04] dark:bg-white/5 text-xs font-semibold text-slate-600 dark:text-slate-400"
                            >
                                Cancel
                            </button>
                            <button
                                type="submit"
                                className="px-4 py-2 rounded-xl bg-teal-500 text-slate-950 text-xs font-bold shadow-sm hover:bg-teal-400 transition-all"
                            >
                                Save Playlist
                            </button>
                        </div>
                    </form>
                </GlassCard>
            )}

            {/* Playlists Grid & Detail View */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Playlists List */}
                <div className="space-y-3">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 px-1">
                        All Playlists ({playlists.length})
                    </h3>
                    <div className="space-y-2">
                        {playlists.map(pl => {
                            const isSelected = selectedPlaylistId === pl.id;
                            return (
                                <div
                                    key={pl.id}
                                    onClick={() => setSelectedPlaylistId(pl.id)}
                                    className={`p-3.5 rounded-2xl border transition-all cursor-pointer flex items-center justify-between group ${
                                        isSelected
                                            ? 'bg-white dark:bg-white/15 border-teal-500/30 shadow-apple-sm'
                                            : 'bg-white/50 dark:bg-[#141620]/60 border-black/[0.06] dark:border-white/5 hover:border-black/[0.12] dark:hover:border-white/15'
                                    }`}
                                >
                                    <div className="flex items-center space-x-3 truncate">
                                        <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${pl.coverColor} flex items-center justify-center text-white shadow-sm flex-shrink-0`}>
                                            <ListMusic size={18} />
                                        </div>
                                        <div className="truncate">
                                            <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate">
                                                {pl.name}
                                            </h4>
                                            <p className="text-[10px] text-slate-500 dark:text-slate-400 truncate mt-0.5">
                                                {pl.songIds.length} tracks · {pl.description}
                                            </p>
                                        </div>
                                    </div>

                                    {pl.id !== 'pl-favorites' && (
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleDeletePlaylist(pl.id);
                                            }}
                                            className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg text-slate-400 hover:text-rose-500 transition-all"
                                            title="Delete Playlist"
                                        >
                                            <Trash2 size={13} />
                                        </button>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Selected Playlist Tracklist */}
                <div className="lg:col-span-2 space-y-4">
                    {activePlaylist ? (
                        <div className="bg-white/70 dark:bg-[#141620]/80 rounded-3xl border border-black/[0.06] dark:border-white/10 p-6 shadow-apple-sm backdrop-blur-xl space-y-6">
                            {/* Playlist Banner Header */}
                            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 pb-4 border-b border-black/[0.06] dark:border-white/10">
                                <div className={`w-20 h-20 rounded-2xl bg-gradient-to-br ${activePlaylist.coverColor} flex items-center justify-center text-white shadow-apple-md flex-shrink-0`}>
                                    <Music2 size={32} />
                                </div>
                                <div className="flex-1 min-w-0">
                                    <span className="text-[10px] font-bold uppercase tracking-wider text-teal-600 dark:text-teal-400">
                                        Studio Playlist
                                    </span>
                                    <h2 className="text-xl font-extrabold text-slate-900 dark:text-slate-100 truncate">
                                        {activePlaylist.name}
                                    </h2>
                                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                                        {activePlaylist.description}
                                    </p>
                                </div>
                                {activePlaylistSongs.length > 0 && (
                                    <button
                                        onClick={() => onPlaySong(activePlaylistSongs[0])}
                                        className="px-4 py-2 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-all shadow-md shadow-teal-500/20 active:scale-95"
                                    >
                                        <Play size={14} className="ml-0.5" />
                                        <span>Play All</span>
                                    </button>
                                )}
                            </div>

                            {/* Tracks in Playlist */}
                            {activePlaylistSongs.length === 0 ? (
                                <div className="text-center py-12 space-y-2">
                                    <ListMusic size={32} className="mx-auto text-slate-400" />
                                    <h4 className="text-xs font-bold text-slate-700 dark:text-slate-300">
                                        This playlist is currently empty
                                    </h4>
                                    <p className="text-[11px] text-slate-500 dark:text-slate-400">
                                        Add songs from the Song Library or create new tracks.
                                    </p>
                                </div>
                            ) : (
                                <div className="space-y-1 divide-y divide-black/[0.04] dark:divide-white/5">
                                    {activePlaylistSongs.map((song, idx) => (
                                        <div
                                            key={song.id}
                                            className="pt-2 pb-2 flex items-center justify-between hover:bg-black/[0.02] dark:hover:bg-white/[0.03] px-3 rounded-xl transition-colors group"
                                        >
                                            <div className="flex items-center space-x-3 truncate">
                                                <span className="text-xs font-mono text-slate-400 w-5">
                                                    {idx + 1}
                                                </span>
                                                <button
                                                    onClick={() => onPlaySong(song)}
                                                    className="w-7 h-7 rounded-full bg-teal-500/10 hover:bg-teal-500 text-teal-700 dark:text-teal-300 hover:text-slate-950 flex items-center justify-center transition-all flex-shrink-0"
                                                >
                                                    <Play size={11} className="ml-0.5" />
                                                </button>
                                                <div className="truncate">
                                                    <h5 className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate">
                                                        {song.title || song.prompt.slice(0, 35)}
                                                    </h5>
                                                    <span className="text-[10px] text-slate-500 dark:text-slate-400">
                                                        {song.tags || 'Pop / Electronic'}
                                                    </span>
                                                </div>
                                            </div>

                                            <div className="flex items-center space-x-2">
                                                <button
                                                    onClick={() => onOpenWorkspace(song)}
                                                    className="px-2.5 py-1 bg-teal-500/10 hover:bg-teal-500 text-teal-700 dark:text-teal-300 hover:text-slate-950 font-bold rounded-lg text-[10px] transition-all"
                                                >
                                                    DAW Edit
                                                </button>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="text-center py-16 text-slate-400">
                            Select a playlist to view tracks.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
