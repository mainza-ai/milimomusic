import React, { useState, useEffect } from 'react';
import { type Job, studioProfileApi } from '../../api';
import { Award, Mic, Layers, ShieldCheck, Heart } from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';
import { AppFooter } from '../ui/AppFooter';

interface ProfileViewProps {
    songs: Job[];
    onPlaySong: (job: Job) => void;
    onOpenWorkspace: (job: Job) => void;
    onSelectTrack?: (job: Job) => void;
}

export const ProfileView: React.FC<ProfileViewProps> = ({ songs, onPlaySong, onOpenWorkspace, onSelectTrack }) => {
    const [artistName, setArtistName] = useState('Mainza Kangombe');
    const [bio, setBio] = useState('Founder & Audio Architect. Exploring generative AI soundscapes, neural synthesis, and offline DAW workflows.');
    const [isEditing, setIsEditing] = useState(false);
    const [isSaving, setIsSaving] = useState(false);

    useEffect(() => {
        const fetchProfile = async () => {
            try {
                // Check if we need to migrate from localStorage
                const localName = localStorage.getItem('milimo_artist_name');
                const localBio = localStorage.getItem('milimo_artist_bio');

                const profile = await studioProfileApi.get();
                if (localName || localBio) {
                    const updated = await studioProfileApi.update({
                        artist_name: localName || profile.artist_name,
                        bio: localBio || profile.bio
                    });
                    setArtistName(updated.artist_name);
                    setBio(updated.bio);
                    localStorage.removeItem('milimo_artist_name');
                    localStorage.removeItem('milimo_artist_bio');
                } else {
                    setArtistName(profile.artist_name);
                    setBio(profile.bio);
                }
            } catch (err) {
                console.error('Error fetching studio profile:', err);
            }
        };
        fetchProfile();
    }, []);

    const completedSongs = songs.filter(s => s.status === 'completed');
    const favoriteCount = completedSongs.filter(s => s.is_favorite).length;

    const handleSave = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            setIsSaving(true);
            await studioProfileApi.update({
                artist_name: artistName,
                bio: bio
            });
            setIsEditing(false);
        } catch (err) {
            console.error('Failed to save studio profile:', err);
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
            {/* Profile Hero Card */}
            <GlassCard className="p-6 md:p-8 space-y-6 border-teal-500/20 relative overflow-hidden">
                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6">
                    {/* Apple-style Avatar Squircle */}
                    <div className="w-24 h-24 rounded-3xl bg-gradient-to-tr from-teal-500 via-cyan-500 to-sky-600 p-0.5 shadow-apple-lg flex-shrink-0">
                        <div className="w-full h-full bg-white dark:bg-[#141620] rounded-[22px] flex items-center justify-center text-teal-600 dark:text-teal-400 font-extrabold text-2xl">
                            {artistName.slice(0, 2).toUpperCase()}
                        </div>
                    </div>

                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                            <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white truncate">
                                {artistName}
                            </h1>
                            <span className="px-2.5 py-0.5 rounded-full bg-teal-500/15 text-teal-700 dark:text-teal-300 border border-teal-500/20 text-[10px] font-bold font-mono flex items-center gap-1">
                                <ShieldCheck size={11} />
                                STUDIO MASTER
                            </span>
                            <span className="px-2.5 py-0.5 rounded-full bg-sky-500/15 text-sky-700 dark:text-sky-300 border border-sky-500/20 text-[10px] font-bold font-mono">
                                LOCAL GPU
                            </span>
                        </div>

                        <p className="text-xs text-slate-600 dark:text-slate-400 mt-2 max-w-2xl leading-relaxed">
                            {bio}
                        </p>

                        <div className="flex items-center gap-4 mt-3 text-xs text-slate-500 dark:text-slate-400 font-mono">
                            <span>🎵 {completedSongs.length} Tracks Produced</span>
                            <span>❤️ {favoriteCount} Favorites</span>
                            <span>⚡ MiniMax Music 3</span>
                        </div>
                    </div>

                    <button
                        onClick={() => setIsEditing(!isEditing)}
                        className="px-4 py-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 font-bold text-xs transition-all border border-black/[0.06] dark:border-white/10"
                    >
                        {isEditing ? 'Cancel' : 'Edit Profile'}
                    </button>
                </div>

                {/* Edit Form */}
                {isEditing && (
                    <form onSubmit={handleSave} className="pt-4 border-t border-black/[0.06] dark:border-white/10 space-y-3 animate-fade-in">
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            <input
                                type="text"
                                value={artistName}
                                onChange={(e) => setArtistName(e.target.value)}
                                placeholder="Artist Name"
                                className="apple-input text-xs"
                                required
                            />
                            <input
                                type="text"
                                value={bio}
                                onChange={(e) => setBio(e.target.value)}
                                placeholder="Bio / Philosophy"
                                className="apple-input text-xs"
                            />
                        </div>
                        <div className="flex justify-end">
                            <button
                                type="submit"
                                disabled={isSaving}
                                className="px-4 py-1.5 bg-teal-500 hover:bg-teal-400 disabled:opacity-50 text-slate-950 font-bold text-xs rounded-xl shadow-sm transition-all"
                            >
                                {isSaving ? 'Saving...' : 'Save Changes'}
                            </button>
                        </div>
                    </form>
                )}
            </GlassCard>

            {/* Badges & Milestones */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <GlassCard className="p-4 flex items-center space-x-3">
                    <div className="w-10 h-10 rounded-2xl bg-teal-500/10 text-teal-600 dark:text-teal-400 flex items-center justify-center font-bold">
                        <Award size={20} />
                    </div>
                    <div>
                        <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100">Studio Pioneer</h4>
                        <p className="text-[10px] text-slate-400">Milimo Music v2 Architecture</p>
                    </div>
                </GlassCard>

                <GlassCard className="p-4 flex items-center space-x-3">
                    <div className="w-10 h-10 rounded-2xl bg-amber-500/10 text-amber-600 dark:text-amber-400 flex items-center justify-center font-bold">
                        <Mic size={20} />
                    </div>
                    <div>
                        <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100">Voice Identity Lab</h4>
                        <p className="text-[10px] text-slate-400">RVC v2 & SVC Enabled</p>
                    </div>
                </GlassCard>

                <GlassCard className="p-4 flex items-center space-x-3">
                    <div className="w-10 h-10 rounded-2xl bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 flex items-center justify-center font-bold">
                        <Layers size={20} />
                    </div>
                    <div>
                        <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100">Score & MIDI Studio</h4>
                        <p className="text-[10px] text-slate-400">Note-Level Transcription Ready</p>
                    </div>
                </GlassCard>
            </div>

            {/* User Tracks Showcase */}
            <div className="space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 px-1">
                    Featured Studio Creations ({completedSongs.length})
                </h3>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {completedSongs.slice(0, 6).map(song => (
                        <GlassCard key={song.id} className="p-4 space-y-3 group hover:border-teal-500/40 transition-all">
                            <div className="flex items-center justify-between">
                                <span className="text-[10px] font-mono text-teal-600 dark:text-teal-400 font-bold">
                                    {song.tags || 'Pop / Electronic'}
                                </span>
                                {song.is_favorite && <Heart size={12} className="fill-rose-500 text-rose-500" />}
                            </div>

                            <div
                                onClick={() => onSelectTrack?.(song)}
                                className="cursor-pointer group/title"
                                title="Open Track Studio"
                            >
                                <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate group-hover/title:text-teal-600 dark:group-hover/title:text-teal-400 transition-colors">
                                    {song.title || song.prompt.slice(0, 35)}
                                </h4>
                                <p className="text-[11px] text-slate-500 dark:text-slate-400 line-clamp-2 mt-0.5">
                                    {song.prompt}
                                </p>
                            </div>

                            <div className="flex items-center justify-between pt-2 border-t border-black/[0.06] dark:border-white/5 gap-1.5 flex-wrap">
                                <button
                                    onClick={() => onPlaySong(song)}
                                    className="px-2.5 py-1 bg-teal-500/10 hover:bg-teal-500 text-teal-700 dark:text-teal-300 hover:text-slate-950 font-bold rounded-lg text-[10px] transition-all"
                                >
                                    Play
                                </button>
                                {onSelectTrack && (
                                    <button
                                        onClick={() => onSelectTrack(song)}
                                        className="px-2.5 py-1 bg-black/5 dark:bg-white/5 hover:bg-teal-500/20 text-slate-700 dark:text-slate-200 hover:text-teal-600 dark:hover:text-teal-300 font-bold rounded-lg text-[10px] transition-all border border-black/5 dark:border-white/5"
                                        title="Open Track Studio"
                                    >
                                        Studio
                                    </button>
                                )}
                                <button
                                    onClick={() => onOpenWorkspace(song)}
                                    className="px-2.5 py-1 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 font-bold rounded-lg text-[10px] transition-all"
                                >
                                    DAW
                                </button>
                            </div>
                        </GlassCard>
                    ))}
                </div>
            </div>

            {/* Global Creator Footer */}
            <AppFooter />
        </div>
    );
};
