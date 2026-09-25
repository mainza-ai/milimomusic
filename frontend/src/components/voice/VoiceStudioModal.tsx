import React, { useState, useEffect, useRef } from 'react';
import {
    Mic,
    Upload,
    Trash2,
    X,
    AlertTriangle,
    ShieldCheck,
    Play,
    Pause,
    Activity,
    Sparkles,
    Loader2,
    CheckCircle2,
    Music,
    Sliders
} from 'lucide-react';
import { api, voiceApi, trainingApi, type VoiceProfile, type YuE2TrainingJob } from '../../api';
import { toast } from '../../utils/toast';

interface VoiceStudioModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export const VoiceStudioModal: React.FC<VoiceStudioModalProps> = ({ isOpen, onClose }) => {
    const [activeTab, setActiveTab] = useState<'voice_profiles' | 'yue2_studio'>('voice_profiles');

    // Voice Profiles State
    const [profiles, setProfiles] = useState<VoiceProfile[]>([]);
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [consentConfirmed, setConsentConfirmed] = useState(false);
    const [f0Method, setF0Method] = useState('rmvpe');
    const [file, setFile] = useState<File | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [playingProfileId, setPlayingProfileId] = useState<string | null>(null);
    const audioRef = useRef<HTMLAudioElement | null>(null);

    // YuE2 Training Studio State
    const [yue2Mode, setYue2Mode] = useState<'auto' | 'guided'>('auto');
    const [datasetName, setDatasetName] = useState('my_custom_music');
    const [loraRank, setLoraRank] = useState<number>(32);
    const [activeYue2Job, setActiveYue2Job] = useState<YuE2TrainingJob | null>(null);
    const [isStartingYue2, setIsStartingYue2] = useState(false);
    const [auditionPlayingKey, setAuditionPlayingKey] = useState<string | null>(null);
    const auditionAudioRef = useRef<HTMLAudioElement | null>(null);

    // Unmount cleanup to stop any active audio preview
    useEffect(() => {
        return () => {
            if (audioRef.current) {
                audioRef.current.pause();
                audioRef.current = null;
            }
            if (auditionAudioRef.current) {
                auditionAudioRef.current.pause();
                auditionAudioRef.current = null;
            }
        };
    }, []);

    const loadProfiles = async () => {
        try {
            const list = await voiceApi.listProfiles();
            setProfiles(list);
        } catch (e) {
            console.error('Failed to load voice profiles', e);
            toast('Failed to load voice profiles', 'error');
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

    const togglePlayPreview = (profile: VoiceProfile) => {
        if (!profile.sample_audio_path) return;

        if (playingProfileId === profile.id) {
            audioRef.current?.pause();
            setPlayingProfileId(null);
        } else {
            if (audioRef.current) {
                audioRef.current.pause();
            }
            const fullUrl = api.getAudioUrl(profile.sample_audio_path);
            const audio = new Audio(fullUrl);
            audioRef.current = audio;
            setPlayingProfileId(profile.id);
            audio.play().catch((err) => {
                console.warn('Audio preview playback error:', err);
                toast('Audio preview playback error', 'error');
                setPlayingProfileId(null);
            });
            audio.onended = () => setPlayingProfileId(null);
            audio.onerror = () => {
                toast('Failed to load preview audio', 'error');
                setPlayingProfileId(null);
            };
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
            toast('Voice profile created successfully', 'success');
            loadProfiles();
        } catch (err: any) {
            const msg = err.response?.data?.detail?.error?.message || err.response?.data?.detail || err.message;
            toast('Failed to create voice profile: ' + msg, 'error');
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleDelete = async (id: string) => {
        if (!window.confirm('Are you sure you want to delete this voice profile?')) return;
        try {
            if (playingProfileId === id && audioRef.current) {
                audioRef.current.pause();
                setPlayingProfileId(null);
            }
            await voiceApi.deleteProfile(id);
            toast('Voice profile deleted', 'info');
            loadProfiles();
        } catch (e) {
            console.error('Failed to delete voice profile', e);
            toast('Failed to delete voice profile', 'error');
        }
    };

    // YuE2 Job Polling
    useEffect(() => {
        if (!activeYue2Job || activeYue2Job.status === 'completed' || activeYue2Job.status === 'failed') return;
        const timer = window.setInterval(async () => {
            try {
                const updated = await trainingApi.getJob(activeYue2Job.job_id);
                setActiveYue2Job(updated);
            } catch {
                // Ignore transient polling error
            }
        }, 2000);
        return () => window.clearInterval(timer);
    }, [activeYue2Job]);

    const handleStartYue2Training = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!datasetName.trim()) return;

        setIsStartingYue2(true);
        try {
            const job = await trainingApi.createJob(datasetName.trim(), yue2Mode, loraRank);
            setActiveYue2Job(job);
            toast(`YuE2 Training Job started in ${yue2Mode.toUpperCase()} mode!`, 'success');
        } catch (err: any) {
            const msg = err.response?.data?.detail?.error?.message || err.response?.data?.detail || err.message;
            toast('Failed to start training: ' + msg, 'error');
        } finally {
            setIsStartingYue2(false);
        }
    };

    const toggleAuditionPreview = (key: string, relPath: string) => {
        if (!relPath) return;
        if (auditionPlayingKey === key) {
            auditionAudioRef.current?.pause();
            setAuditionPlayingKey(null);
        } else {
            if (auditionAudioRef.current) {
                auditionAudioRef.current.pause();
            }
            const fullUrl = trainingApi.getAuditionAudioUrl(relPath);
            const audio = new Audio(fullUrl);
            auditionAudioRef.current = audio;
            setAuditionPlayingKey(key);
            audio.play().catch((err) => {
                console.warn('Audition playback error:', err);
                toast('Audition audio unavailable', 'error');
                setAuditionPlayingKey(null);
            });
            audio.onended = () => setAuditionPlayingKey(null);
            audio.onerror = () => {
                toast('Failed to load audition artifact', 'error');
                setAuditionPlayingKey(null);
            };
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md">
            <div className="w-full max-w-2xl bg-white dark:bg-[#12141c] border border-black/[0.08] dark:border-white/10 rounded-3xl shadow-apple-2xl flex flex-col max-h-[90vh] overflow-hidden animate-fade-in">
                {/* Header */}
                <div className="px-6 py-5 border-b border-black/[0.06] dark:border-white/10 flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                        <div className="w-9 h-9 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400 flex items-center justify-center">
                            <Mic size={20} />
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
                        aria-label="Close modal"
                        title="Close"
                        className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
                    >
                        <X size={18} />
                    </button>
                </div>

                {/* Mode Tabs */}
                <div className="px-6 pt-3 pb-0 border-b border-black/[0.06] dark:border-white/10 flex items-center space-x-2 bg-black/[0.01] dark:bg-white/[0.02]">
                    <button
                        type="button"
                        onClick={() => setActiveTab('voice_profiles')}
                        className={`pb-2.5 px-3 text-xs font-bold border-b-2 transition-all flex items-center gap-1.5 ${
                            activeTab === 'voice_profiles'
                                ? 'border-teal-500 text-teal-600 dark:text-teal-400'
                                : 'border-transparent text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                        }`}
                    >
                        <Mic size={14} />
                        <span>Voice Identity & SVC</span>
                    </button>
                    <button
                        type="button"
                        onClick={() => setActiveTab('yue2_studio')}
                        className={`pb-2.5 px-3 text-xs font-bold border-b-2 transition-all flex items-center gap-1.5 ${
                            activeTab === 'yue2_studio'
                                ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400'
                                : 'border-transparent text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                        }`}
                    >
                        <Sparkles size={14} className="text-indigo-400" />
                        <span>YuE2 "My Music" Studio</span>
                        <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 font-bold border border-indigo-500/20">
                            48kHz Stereo
                        </span>
                    </button>
                </div>

                {/* Body */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {activeTab === 'voice_profiles' ? (
                        <>
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
                             title="Submit">
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
                </>
            ) : (
                <div className="space-y-6">
                    {/* Mode Selector & Configuration */}
                    <div className="p-5 bg-black/[0.02] dark:bg-[#181a24] border border-black/[0.06] dark:border-white/10 rounded-2xl space-y-4">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                            <div>
                                <h3 className="text-xs font-bold text-slate-900 dark:text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
                                    <Sparkles size={14} className="text-indigo-400" />
                                    <span>YuE2 Foundation Training Mode</span>
                                </h3>
                                <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                                    Train custom 48 kHz stereo style & instrumental LoRA adapters
                                </p>
                            </div>
                            <div className="flex bg-black/[0.04] dark:bg-white/5 p-1 rounded-xl border border-black/[0.06] dark:border-white/10">
                                <button
                                    type="button"
                                    onClick={() => setYue2Mode('auto')}
                                    className={`px-3 py-1 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 ${
                                        yue2Mode === 'auto'
                                            ? 'bg-indigo-500 text-white shadow-sm'
                                            : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                                    }`}
                                >
                                    <span>⚡ Auto Mode</span>
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setYue2Mode('guided')}
                                    className={`px-3 py-1 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 ${
                                        yue2Mode === 'guided'
                                            ? 'bg-indigo-500 text-white shadow-sm'
                                            : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                                    }`}
                                >
                                    <span>🛠️ Guided Mode (4 Stages)</span>
                                </button>
                            </div>
                        </div>

                        <form onSubmit={handleStartYue2Training} className="space-y-4 pt-2 border-t border-black/[0.04] dark:border-white/5">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                                        Dataset Identifier *
                                    </label>
                                    <input
                                        type="text"
                                        required
                                        placeholder="e.g., ambient_guitar_48k"
                                        value={datasetName}
                                        onChange={(e) => setDatasetName(e.target.value)}
                                        className="w-full apple-input"
                                    />
                                </div>
                                <div>
                                    <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                                        LoRA Rank / Capacity
                                    </label>
                                    <select
                                        value={loraRank}
                                        onChange={(e) => setLoraRank(parseInt(e.target.value, 10))}
                                        className="w-full apple-input"
                                    >
                                        <option value={16}>Rank 16 (Fast, Lightweight)</option>
                                        <option value={32}>Rank 32 (Standard Recommended)</option>
                                        <option value={64}>Rank 64 (Deep Expressive)</option>
                                    </select>
                                </div>
                            </div>

                            <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-xs text-indigo-300 space-y-1">
                                <div className="font-bold flex items-center gap-1.5 text-indigo-200">
                                    <Sliders size={13} />
                                    {yue2Mode === 'auto' ? 'Auto Mode: Hands-Off 1-Click Pipeline' : 'Guided Mode: 4-Stage Audition Studio'}
                                </div>
                                <p className="text-[11px] text-slate-400">
                                    {yue2Mode === 'auto'
                                        ? 'Automatically executes tokenizer adaptation (100 steps), AR style training (200 steps), generates auditory audition checks, and saves the final LoRA in the background.'
                                        : 'Step-by-step 4-stage pipeline with auditory before/after reconstruction checks (original ground truth vs baseline vs adapted decoder) and test song auditions at checkpoints.'}
                                </p>
                            </div>

                            <div className="flex justify-end">
                                <button
                                    type="submit"
                                    disabled={isStartingYue2 || !datasetName.trim()}
                                    className="px-4 py-2 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500 disabled:opacity-50 text-white font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-all shadow-md shadow-indigo-500/20"
                                >
                                    {isStartingYue2 ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
                                    <span>{isStartingYue2 ? "Launching YuE2 Job..." : "Start YuE2 Training"}</span>
                                </button>
                            </div>
                        </form>
                    </div>

                    {/* Active or Completed YuE2 Job HUD */}
                    {activeYue2Job && (
                        <div className="p-5 bg-white dark:bg-[#181a24] border border-black/[0.06] dark:border-white/10 rounded-2xl space-y-4 shadow-apple-sm">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center space-x-2">
                                    <span className="font-mono text-xs font-bold text-slate-800 dark:text-slate-200">
                                        Job: {activeYue2Job.job_id}
                                    </span>
                                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                                        activeYue2Job.status === 'completed'
                                            ? 'bg-teal-500/20 text-teal-300 border border-teal-500/30'
                                            : activeYue2Job.status === 'failed'
                                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                                            : 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                                    }`}>
                                        {activeYue2Job.status}
                                    </span>
                                </div>
                                <span className="text-[11px] font-mono text-slate-400">
                                    Rank {activeYue2Job.rank || 32} · {activeYue2Job.mode.toUpperCase()}
                                </span>
                            </div>

                            {/* Stage Progress */}
                            <div className="space-y-1.5">
                                <div className="flex justify-between text-xs">
                                    <span className="font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                                        {activeYue2Job.status === 'processing' && <Loader2 size={12} className="animate-spin text-indigo-400" />}
                                        {activeYue2Job.status === 'completed' && <CheckCircle2 size={12} className="text-teal-400" />}
                                        <span>Stage {activeYue2Job.current_stage} of {activeYue2Job.total_stages}:</span>
                                        <span className="text-indigo-400 font-bold">
                                            {activeYue2Job.current_stage === 1 && 'Dataset Review & Excerpts'}
                                            {activeYue2Job.current_stage === 2 && 'Real-Audio Tokenizer/Decoder Adaptation'}
                                            {activeYue2Job.current_stage === 3 && 'AR Song Style LoRA Training'}
                                            {activeYue2Job.current_stage === 4 && 'Auditions & Output Finalization'}
                                        </span>
                                    </span>
                                    <span className="font-mono text-slate-400">
                                        Step {activeYue2Job.current_step}/{activeYue2Job.total_steps}
                                    </span>
                                </div>
                                <div className="w-full h-1.5 bg-black/[0.06] dark:bg-white/10 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500"
                                        style={{
                                            width: `${Math.min(100, Math.round((activeYue2Job.current_step / Math.max(1, activeYue2Job.total_steps)) * 100))}%`
                                        }}
                                    />
                                </div>
                            </div>

                            {/* Stage 2 Auditory Reconstruction Auditions (Before vs After) */}
                            {activeYue2Job.reconstruction_auditions && Object.keys(activeYue2Job.reconstruction_auditions).length > 0 && (
                                <div className="pt-3 border-t border-black/[0.04] dark:border-white/5 space-y-2">
                                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                                        <Music size={13} className="text-teal-400" />
                                        <span>Stage 2 Auditory Reconstruction Auditions (Real Audio vs Adapted)</span>
                                    </h4>
                                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                                        {Object.entries(activeYue2Job.reconstruction_auditions).map(([key, path]) => {
                                            const isPlaying = auditionPlayingKey === `recon_${key}`;
                                            const label = key === 'original' ? '1. Ground Truth' : key === 'before_adaptation' ? '2. Base Reconstruction' : '3. Adapted Decoder';
                                            return (
                                                <div key={key} className="p-2.5 rounded-xl bg-black/[0.02] dark:bg-white/5 border border-black/[0.04] dark:border-white/10 flex items-center justify-between">
                                                    <div className="space-y-0.5">
                                                        <div className="text-[11px] font-bold text-slate-800 dark:text-slate-200">{label}</div>
                                                        <div className="text-[9px] font-mono text-slate-400 truncate max-w-[110px]">{path.split('/').pop()}</div>
                                                    </div>
                                                    <button
                                                        type="button"
                                                        onClick={() => toggleAuditionPreview(`recon_${key}`, path)}
                                                        className={`p-2 rounded-lg transition-all ${
                                                            isPlaying
                                                                ? 'bg-indigo-500 text-white shadow-md'
                                                                : 'bg-indigo-500/10 text-indigo-400 hover:bg-indigo-500/20'
                                                        }`}
                                                        title={isPlaying ? 'Pause' : 'Play Audition'}
                                                    >
                                                        {isPlaying ? <Pause size={13} /> : <Play size={13} className="ml-0.5" />}
                                                    </button>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* Stage 4 Checkpoint Test Auditions */}
                            {activeYue2Job.test_song_auditions && Object.keys(activeYue2Job.test_song_auditions).length > 0 && (
                                <div className="pt-3 border-t border-black/[0.04] dark:border-white/5 space-y-2">
                                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                                        <Sparkles size={13} className="text-purple-400" />
                                        <span>Stage 4 Test Song Checkpoint Auditions</span>
                                    </h4>
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                        {Object.entries(activeYue2Job.test_song_auditions).map(([step, path]) => {
                                            const isPlaying = auditionPlayingKey === `test_${step}`;
                                            return (
                                                <div key={step} className="p-2.5 rounded-xl bg-black/[0.02] dark:bg-white/5 border border-black/[0.04] dark:border-white/10 flex items-center justify-between">
                                                    <div>
                                                        <div className="text-[11px] font-bold text-slate-800 dark:text-slate-200">Checkpoint Step {step}</div>
                                                        <div className="text-[9px] font-mono text-slate-400 truncate max-w-[130px]">{path.split('/').pop()}</div>
                                                    </div>
                                                    <button
                                                        type="button"
                                                        onClick={() => toggleAuditionPreview(`test_${step}`, path)}
                                                        className={`p-2 rounded-lg transition-all ${
                                                            isPlaying
                                                                ? 'bg-purple-500 text-white shadow-md'
                                                                : 'bg-purple-500/10 text-purple-400 hover:bg-purple-500/20'
                                                        }`}
                                                        title={isPlaying ? 'Pause' : 'Play Test Song'}
                                                    >
                                                        {isPlaying ? <Pause size={13} /> : <Play size={13} className="ml-0.5" />}
                                                    </button>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* Output LoRA Artifact */}
                            {activeYue2Job.output_lora_path && (
                                <div className="p-3 rounded-xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-between">
                                    <div>
                                        <div className="text-xs font-bold text-teal-400 flex items-center gap-1.5">
                                            <CheckCircle2 size={14} />
                                            <span>YuE2 LoRA Ready for Production Generation</span>
                                        </div>
                                        <div className="text-[10px] font-mono text-teal-300/80 mt-0.5">
                                            {activeYue2Job.output_lora_path}
                                        </div>
                                    </div>
                                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-teal-500/20 text-teal-300 font-bold">
                                        Active LoRA
                                    </span>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    </div>
</div>
    );
};
