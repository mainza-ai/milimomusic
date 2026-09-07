import React, { useState, useEffect, useRef } from 'react';
import { Mic, Upload, Trash2, X, AlertTriangle, ShieldCheck, Play, Pause, Activity } from 'lucide-react';
import { voiceApi, type VoiceProfile } from '../../api';

interface VoiceStudioModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export const VoiceStudioModal: React.FC<VoiceStudioModalProps> = ({ isOpen, onClose }) => {
    const [profiles, setProfiles] = useState<VoiceProfile[]>([]);
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [consentConfirmed, setConsentConfirmed] = useState(false);
    const [f0Method, setF0Method] = useState('rmvpe');
    const [file, setFile] = useState<File | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [playingProfileId, setPlayingProfileId] = useState<string | null>(null);
    const audioRef = useRef<HTMLAudioElement | null>(null);

    const loadProfiles = async () => {
        try {
            const list = await voiceApi.listProfiles();
            setProfiles(list);
        } catch (e) {
            console.error('Failed to load voice profiles', e);
        }
    };

    useEffect(() => {
        if (isOpen) {
            loadProfiles();
        } else {
            if (audioRef.current) {
                audioRef.current.pause();
                audioRef.current = null;
            }
            setPlayingProfileId(null);
        }
    }, [isOpen]);

    useEffect(() => {
        return () => {
            if (audioRef.current) {
                audioRef.current.pause();
                audioRef.current = null;
            }
        };
    }, []);

    const togglePlayPreview = (profile: VoiceProfile) => {
        if (!profile.sample_audio_path) return;

        if (playingProfileId === profile.id) {
            audioRef.current?.pause();
            setPlayingProfileId(null);
        } else {
            if (audioRef.current) {
                audioRef.current.pause();
            }
            const audio = new Audio(profile.sample_audio_path);
            audioRef.current = audio;
            setPlayingProfileId(profile.id);
            audio.play().catch((err) => {
                console.warn('Audio preview playback error:', err);
                setPlayingProfileId(null);
            });
            audio.onended = () => setPlayingProfileId(null);
            audio.onerror = () => setPlayingProfileId(null);
        }
    };

    const handleCreate = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!name || !consentConfirmed) return;

        setIsSubmitting(true);
        try {
            await voiceApi.createProfile({
                name,
                description,
                consent_confirmed: consentConfirmed,
                f0_method: f0Method,
                audio_file: file || undefined
            });
            setName('');
            setDescription('');
            setConsentConfirmed(false);
            setFile(null);
            loadProfiles();
        } catch (err: any) {
            const msg = err.response?.data?.detail?.error?.message || err.response?.data?.detail || err.message;
            alert('Failed to create voice profile: ' + msg);
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleDelete = async (id: string) => {
        if (!confirm('Are you sure you want to delete this voice profile?')) return;
        try {
            if (playingProfileId === id && audioRef.current) {
                audioRef.current.pause();
                setPlayingProfileId(null);
            }
            await voiceApi.deleteProfile(id);
            loadProfiles();
        } catch (e) {
            console.error('Failed to delete voice profile', e);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 dark:bg-black/80 backdrop-blur-md animate-fade-in">
            <div className="bg-white/90 dark:bg-[#14161f]/95 border border-black/[0.08] dark:border-white/10 rounded-3xl w-full max-w-3xl overflow-hidden shadow-apple-lg flex flex-col max-h-[90vh] backdrop-blur-2xl">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-black/[0.06] dark:border-white/10 bg-black/[0.02] dark:bg-[#181a24]">
                    <div className="flex items-center space-x-3">
                        <div className="w-9 h-9 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 text-teal-700 dark:text-teal-400 border border-teal-500/20 flex items-center justify-center">
                            <Mic size={18} />
                        </div>
                        <div>
                            <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
                                Voice Identity & Training Studio
                            </h2>
                            <p className="text-xs text-slate-500 dark:text-slate-400">
                                Create and manage vocal identity profiles for offline Singing Voice Conversion (SVC).
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

                {/* Body */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {/* Add Profile Form */}
                    <form onSubmit={handleCreate} className="p-5 bg-black/[0.02] dark:bg-[#181a24] border border-black/[0.06] dark:border-white/10 rounded-2xl space-y-4">
                        <h3 className="text-xs font-bold text-slate-900 dark:text-slate-200 uppercase tracking-wider">
                            Train New Voice Identity
                        </h3>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                                    Voice Name *
                                </label>
                                <input
                                    type="text"
                                    required
                                    placeholder="e.g., Acoustic Folk Singer"
                                    value={name}
                                    onChange={(e) => setName(e.target.value)}
                                    className="w-full apple-input"
                                />
                            </div>

                            <div>
                                <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                                    Pitch Extraction Method (F0)
                                </label>
                                <select
                                    value={f0Method}
                                    onChange={(e) => setF0Method(e.target.value)}
                                    className="w-full apple-input"
                                >
                                    <option value="rmvpe">RMVPE (High Quality Vocal Pitch)</option>
                                    <option value="crepe">CREPE (Harmonic Accurate)</option>
                                    <option value="harvest">Harvest (Robust)</option>
                                    <option value="pm">PM (Fast)</option>
                                </select>
                            </div>
                        </div>

                        <div>
                            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                                Description
                            </label>
                            <input
                                type="text"
                                placeholder="e.g., Warm tenor with subtle vibrato"
                                value={description}
                                onChange={(e) => setDescription(e.target.value)}
                                className="w-full apple-input"
                            />
                        </div>

                        {/* File Upload */}
                        <div>
                            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                                Solo Vocal Dataset (.wav / .mp3 / .zip)
                            </label>
                            <label className="flex flex-col items-center justify-center p-4 border border-dashed border-black/20 dark:border-white/20 hover:border-teal-500/50 rounded-2xl cursor-pointer bg-white/50 dark:bg-[#12141c] transition-colors shadow-sm">
                                <Upload size={18} className="text-teal-600 dark:text-teal-400 mb-1" />
                                <span className="text-xs text-slate-600 dark:text-slate-300 font-medium">
                                    {file ? file.name : "Select or drag clean vocal audio (1-10 mins recommended)"}
                                </span>
                                <input
                                    type="file"
                                    accept="audio/*,.zip"
                                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                                    className="hidden"
                                />
                            </label>
                        </div>

                        {/* Mandatory Consent Checkbox */}
                        <div className="p-3.5 bg-amber-500/10 border border-amber-500/20 rounded-xl flex items-start space-x-3">
                            <AlertTriangle size={18} className="text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
                            <label className="text-xs text-slate-700 dark:text-slate-300 flex items-start space-x-2 cursor-pointer">
                                <input
                                    type="checkbox"
                                    required
                                    checked={consentConfirmed}
                                    onChange={(e) => setConsentConfirmed(e.target.checked)}
                                    className="mt-0.5 accent-teal-500 rounded"
                                />
                                <span>
                                    <strong>Mandatory Legal Consent:</strong> I verify that I own the rights or have explicit permission to use and clone this voice for AI musical generation.
                                </span>
                            </label>
                        </div>

                        <div className="flex justify-end">
                            <button
                                type="submit"
                                disabled={isSubmitting || !name || !consentConfirmed}
                                className="px-4 py-2 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 disabled:opacity-50 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-all shadow-md shadow-teal-500/20"
                            >
                                <ShieldCheck size={14} />
                                <span>{isSubmitting ? "Extracting F0 & Training Profile..." : "Create Voice Profile"}</span>
                            </button>
                        </div>
                    </form>

                    {/* Existing Profiles List */}
                    <div className="space-y-3">
                        <h3 className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
                            Available Voice Profiles ({profiles.length})
                        </h3>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {profiles.map((p) => (
                                <div
                                    key={p.id}
                                    className="p-4 bg-white dark:bg-[#181a24] border border-black/[0.06] dark:border-white/10 rounded-2xl flex flex-col justify-between shadow-apple-sm space-y-3"
                                >
                                    <div className="flex items-start justify-between">
                                        <div className="flex items-center space-x-3">
                                            {p.sample_audio_path ? (
                                                <button
                                                    type="button"
                                                    onClick={() => togglePlayPreview(p)}
                                                    className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all ${
                                                        playingProfileId === p.id
                                                            ? 'bg-teal-500 text-slate-950 shadow-md shadow-teal-500/30'
                                                            : 'bg-teal-500/10 dark:bg-teal-500/20 text-teal-700 dark:text-teal-400 hover:bg-teal-500/20'
                                                    }`}
                                                    title={playingProfileId === p.id ? 'Stop Preview' : 'Play Voice Preview'}
                                                >
                                                    {playingProfileId === p.id ? <Pause size={15} /> : <Play size={15} className="ml-0.5" />}
                                                </button>
                                            ) : (
                                                <div className="w-9 h-9 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 text-teal-700 dark:text-teal-400 flex items-center justify-center font-bold text-xs p-1">
                                                    🎤
                                                </div>
                                            )}
                                            <div>
                                                <div className="flex items-center space-x-1.5">
                                                    <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100">{p.name}</h4>
                                                    {p.is_default && (
                                                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-400 font-semibold border border-teal-500/20">
                                                            Default
                                                        </span>
                                                    )}
                                                </div>
                                                <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                                                    {p.description || "Custom singing voice"}
                                                </p>
                                            </div>
                                        </div>

                                        {!p.is_default && (
                                            <button
                                                onClick={() => handleDelete(p.id)}
                                                className="p-1.5 rounded-lg text-slate-400 hover:text-rose-500 hover:bg-rose-500/10 transition-colors"
                                                title="Delete Profile"
                                            >
                                                <Trash2 size={13} />
                                            </button>
                                        )}
                                    </div>

                                    {/* Acoustic Timbre Chips */}
                                    <div className="flex flex-wrap items-center gap-1.5 pt-1 border-t border-black/[0.04] dark:border-white/5 text-[10px]">
                                        <span className="px-2 py-0.5 rounded-md bg-black/[0.03] dark:bg-white/[0.04] text-slate-600 dark:text-slate-400 font-medium">
                                            {p.f0_method.toUpperCase()}
                                        </span>
                                        {p.acoustic_features?.median_f0_hz && (
                                            <span className="px-2 py-0.5 rounded-md bg-teal-500/10 text-teal-700 dark:text-teal-400 font-medium flex items-center gap-1">
                                                <Activity size={10} />
                                                {p.acoustic_features.median_f0_hz} Hz
                                            </span>
                                        )}
                                        {p.acoustic_features?.spectral_centroid_hz && (
                                            <span className="px-2 py-0.5 rounded-md bg-purple-500/10 text-purple-700 dark:text-purple-400 font-medium">
                                                {Math.round(p.acoustic_features.spectral_centroid_hz)} Hz timbre
                                            </span>
                                        )}
                                        {p.dataset_files && p.dataset_files.length > 0 && (
                                            <span className="px-2 py-0.5 rounded-md bg-blue-500/10 text-blue-700 dark:text-blue-400 font-medium">
                                                {p.dataset_files.length} audio file{p.dataset_files.length > 1 ? 's' : ''}
                                            </span>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
