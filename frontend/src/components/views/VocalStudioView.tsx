import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
    Mic,
    Sliders,
    Users,
    Music,
    Upload,
    Trash2,
    Play,
    Pause,
    Activity,
    ShieldCheck,
    AlertTriangle,
    CheckCircle2,
    Radio,
    ArrowLeft,
    Loader2
} from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';
import { AppFooter } from '../ui/AppFooter';
import { VocalBoothRecorder } from '../voice/VocalBoothRecorder';
import { VocalDSPRack } from '../voice/VocalDSPRack';
import { VocalAuditionPlayer } from '../voice/VocalAuditionPlayer';
import {
    api,
    voiceApi,
    trackApi,
    type VoiceProfile,
    type Job
} from '../../api';
import { toast } from '../../utils/toast';

export type VocalStudioMode = 'conversion' | 'library' | 'booth';

interface VocalStudioViewProps {
    initialTrack?: Job | null;
    initialStemPath?: string;
    onOpenWorkspace?: (job: Job) => void;
    isModal?: boolean;
    onCloseModal?: () => void;
}

export const VocalStudioView: React.FC<VocalStudioViewProps> = ({
    initialTrack,
    initialStemPath,
    onOpenWorkspace,
    isModal = false,
    onCloseModal
}) => {
    // Mode State
    const [mode, setMode] = useState<VocalStudioMode>('conversion');

    // Library Songs & Active Track Selection
    const [songs, setSongs] = useState<Job[]>([]);
    const [, setIsLoadingSongs] = useState(false);
    const [selectedTrack, setSelectedTrack] = useState<Job | null>(initialTrack || null);

    // Voice Profiles State
    const [profiles, setProfiles] = useState<VoiceProfile[]>([]);
    const [selectedProfileId, setSelectedProfileId] = useState<string>('default_aria');
    const [playingProfileId, setPlayingProfileId] = useState<string | null>(null);
    const profileAudioRef = useRef<HTMLAudioElement | null>(null);

    // DSP Rack Parameters
    const [pitchShift, setPitchShift] = useState<number>(0);
    const [dryWet, setDryWet] = useState<number>(100);
    const [formantPreserve, setFormantPreserve] = useState<boolean>(true);
    const [f0Method, setF0Method] = useState<string>('rmvpe');
    const [isConverting, setIsConverting] = useState<boolean>(false);

    // Audition Outputs
    const [convertedVocalUrl, setConvertedVocalUrl] = useState<string | undefined>(undefined);
    const [remixedMasterUrl, setRemixedMasterUrl] = useState<string | undefined>(undefined);

    // New Profile Creation State
    const [newName, setNewName] = useState('');
    const [newDesc, setNewDesc] = useState('');
    const [consentConfirmed, setConsentConfirmed] = useState(false);
    const [uploadFile, setUploadFile] = useState<File | null>(null);
    const [isSubmittingProfile, setIsSubmittingProfile] = useState(false);

    // Load available songs
    const loadSongs = async () => {
        setIsLoadingSongs(true);
        try {
            const raw = await api.getHistory(50, 0, 'all', '');
            const completed = (raw || []).filter((j) => (j.status || '').toLowerCase() === 'completed');
            setSongs(completed);
            if (!selectedTrack && completed.length > 0) {
                setSelectedTrack(completed[0]);
            }
        } catch (e) {
            console.error('Failed to load songs:', e);
        } finally {
            setIsLoadingSongs(false);
        }
    };

    // Load available voice profiles
    const loadProfiles = async () => {
        try {
            const list = await voiceApi.listProfiles();
            setProfiles(list);
            if (list.length > 0 && !selectedProfileId) {
                setSelectedProfileId(list[0].id);
            }
        } catch (e) {
            console.error('Failed to load profiles:', e);
            toast('Failed to load voice profiles', 'error');
        }
    };

    useEffect(() => {
        loadSongs();
        loadProfiles();
    }, []);

    // Sync initial track update if prop changes
    useEffect(() => {
        if (initialTrack) {
            setSelectedTrack(initialTrack);
        }
    }, [initialTrack]);

    // Cleanup audio playback on unmount
    useEffect(() => {
        return () => {
            if (profileAudioRef.current) {
                profileAudioRef.current.pause();
                profileAudioRef.current = null;
            }
        };
    }, []);

    // Resolve isolated vocal stem path
    const originalVocalStemUrl = useMemo(() => {
        if (!selectedTrack) return undefined;
        if (initialStemPath) return initialStemPath;
        try {
            if (selectedTrack.stems_json) {
                const stems = typeof selectedTrack.stems_json === 'string'
                    ? JSON.parse(selectedTrack.stems_json)
                    : selectedTrack.stems_json;
                if (stems?.vocals) return stems.vocals;
            }
        } catch {}
        return undefined;
    }, [selectedTrack, initialStemPath]);

    const hasVocalStem = !!originalVocalStemUrl;

    // Toggle Preview Playback of Profile
    const togglePlayProfile = (p: VoiceProfile) => {
        if (!p.sample_audio_path) return;
        if (playingProfileId === p.id) {
            profileAudioRef.current?.pause();
            setPlayingProfileId(null);
        } else {
            if (profileAudioRef.current) {
                profileAudioRef.current.pause();
            }
            const fullUrl = api.getAudioUrl(p.sample_audio_path);
            const audio = new Audio(fullUrl);
            profileAudioRef.current = audio;
            setPlayingProfileId(p.id);
            audio.play().catch(() => setPlayingProfileId(null));
            audio.onended = () => setPlayingProfileId(null);
            audio.onerror = () => setPlayingProfileId(null);
        }
    };

    // Apply Singing Voice Conversion
    const handleConvertVocals = async () => {
        if (!selectedTrack || !hasVocalStem || !selectedProfileId) return;
        setIsConverting(true);
        try {
            const derivative = await trackApi.voiceConvertTrack(selectedTrack.id, selectedProfileId, {
                pitch_shift: pitchShift,
                dry_wet: dryWet,
                formant_preserve: formantPreserve
            });

            // Extract converted vocal and master URLs
            let convVocal: string | undefined = undefined;
            try {
                const stems = typeof derivative.stems_json === 'string'
                    ? JSON.parse(derivative.stems_json)
                    : derivative.stems_json;
                convVocal = stems?.vocals;
            } catch {}

            setConvertedVocalUrl(convVocal);
            setRemixedMasterUrl(derivative.audio_path);
            toast('Singing voice conversion completed successfully!', 'success');
        } catch (err: any) {
            console.error('Voice conversion failed:', err);
            const msg = err.response?.data?.detail || err.message;
            toast(`Voice conversion failed: ${msg}`, 'error');
        } finally {
            setIsConverting(false);
        }
    };

    // Handle profile creation
    const handleCreateProfile = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!newName || !consentConfirmed) return;
        setIsSubmittingProfile(true);
        try {
            const created = await voiceApi.createProfile({
                name: newName,
                description: newDesc,
                consent_confirmed: consentConfirmed,
                f0_method: f0Method,
                audio_file: uploadFile || undefined
            });
            setNewName('');
            setNewDesc('');
            setConsentConfirmed(false);
            setUploadFile(null);
            toast(`Voice profile '${created.name}' created!`, 'success');
            await loadProfiles();
            setSelectedProfileId(created.id);
            setMode('conversion');
        } catch (err: any) {
            const msg = err.response?.data?.detail || err.message;
            toast(`Failed to create voice profile: ${msg}`, 'error');
        } finally {
            setIsSubmittingProfile(false);
        }
    };

    // Handle voice captured from live Vocal Booth
    const handleVocalBoothCaptured = (file: File, durationSec: number) => {
        setUploadFile(file);
        setNewName(`Vocal Take (${Math.round(durationSec)}s)`);
        setNewDesc(`Live microphone recording captured in Vocal Booth (${Math.round(durationSec)}s)`);
        setMode('library');
        toast('Vocal take captured! Review details and confirm legal consent below.', 'info');
    };

    const handleDeleteProfile = async (id: string) => {
        if (!window.confirm('Are you sure you want to delete this voice profile?')) return;
        try {
            await voiceApi.deleteProfile(id);
            toast('Voice profile deleted', 'info');
            loadProfiles();
            if (selectedProfileId === id) {
                setSelectedProfileId('default_aria');
            }
        } catch (err: any) {
            toast('Failed to delete profile', 'error');
        }
    };

    return (
        <div className={`flex flex-col justify-between ${isModal ? 'h-full overflow-y-auto' : 'min-h-full p-4 md:p-6 pb-28 sm:pb-32 space-y-6'}`}>
            <div className="space-y-6 max-w-[1600px] mx-auto w-full">
                {/* ZONE 1: TOP MASTER STUDIO BAR */}
                <GlassCard className="p-4 border border-black/[0.08] dark:border-white/10 space-y-3">
                    <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                        {/* Title & Navigation */}
                        <div className="flex items-center space-x-3">
                            {isModal && onCloseModal && (
                                <button
                                    onClick={onCloseModal}
                                    className="p-2 rounded-xl bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 text-slate-700 dark:text-slate-200 transition-colors"
                                    title="Back"
                                >
                                    <ArrowLeft size={16} />
                                </button>
                            )}
                            <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-teal-500 to-cyan-500 text-slate-950 flex items-center justify-center shadow-md shadow-teal-500/20">
                                <Mic size={20} />
                            </div>
                            <div>
                                <h2 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
                                    <span>AI Vocal Studio & Singing Voice Conversion</span>
                                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-600 dark:text-teal-400 font-bold border border-teal-500/20">
                                        RVC v2 + Neural SVC
                                    </span>
                                </h2>
                                <p className="text-xs text-slate-500 dark:text-slate-400">
                                    Clone singer identities, perform singing voice conversion, and master remixed tracks
                                </p>
                            </div>
                        </div>

                        {/* Mode Segmented Controls */}
                        <div className="flex items-center p-1 rounded-xl bg-black/[0.04] dark:bg-white/5 border border-black/[0.06] dark:border-white/10">
                            <button
                                type="button"
                                onClick={() => setMode('conversion')}
                                className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                                    mode === 'conversion'
                                        ? 'bg-teal-500 text-slate-950 shadow-sm'
                                        : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white'
                                }`}
                            >
                                <Sliders size={13} />
                                <span>Voice Conversion</span>
                            </button>

                            <button
                                type="button"
                                onClick={() => setMode('library')}
                                className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                                    mode === 'library'
                                        ? 'bg-teal-500 text-slate-950 shadow-sm'
                                        : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white'
                                }`}
                            >
                                <Users size={13} />
                                <span>Voice Identities ({profiles.length})</span>
                            </button>

                            <button
                                type="button"
                                onClick={() => setMode('booth')}
                                className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                                    mode === 'booth'
                                        ? 'bg-rose-500 text-white shadow-sm'
                                        : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white'
                                }`}
                            >
                                <Radio size={13} className={mode === 'booth' ? 'animate-pulse' : ''} />
                                <span>Live Vocal Booth</span>
                            </button>
                        </div>
                    </div>

                    {/* Source Track Selector Row */}
                    <div className="pt-2 border-t border-black/[0.04] dark:border-white/5 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                            <Music size={14} className="text-teal-500 flex-shrink-0" />
                            <span className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex-shrink-0">
                                Source Song:
                            </span>
                            <select
                                value={selectedTrack?.id || ''}
                                onChange={(e) => {
                                    const match = songs.find((s) => s.id === e.target.value);
                                    if (match) setSelectedTrack(match);
                                }}
                                className="apple-input text-xs py-1.5 px-3 max-w-sm font-medium"
                            >
                                {songs.map((song) => (
                                    <option key={song.id} value={song.id}>
                                        {song.title || 'Untitled Track'} ({Math.round((song.duration_ms || 180000) / 1000)}s)
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Stem Status Telemetry Badges */}
                        <div className="flex items-center gap-2">
                            {hasVocalStem ? (
                                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20">
                                    <CheckCircle2 size={12} />
                                    <span>Vocal Stem Isolated</span>
                                </span>
                            ) : (
                                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                                    <AlertTriangle size={12} />
                                    <span>No Vocal Stem Found</span>
                                </span>
                            )}
                        </div>
                    </div>
                </GlassCard>

                {/* ZONE 2: DUAL WORKSPACE BASED ON ACTIVE MODE */}
                {mode === 'conversion' && (
                    <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
                        {/* Left Side: Available Voice Identities (5 cols) */}
                        <div className="xl:col-span-5 space-y-4">
                            <GlassCard className="p-5 border border-black/[0.08] dark:border-white/10 space-y-4">
                                <div className="flex items-center justify-between pb-2 border-b border-black/[0.06] dark:border-white/10">
                                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800 dark:text-slate-200 flex items-center gap-2">
                                        <Users size={14} className="text-teal-500" />
                                        <span>Target Voice Identities</span>
                                    </h4>
                                    <button
                                        type="button"
                                        onClick={() => setMode('library')}
                                        className="text-[11px] font-bold text-teal-600 dark:text-teal-400 hover:underline"
                                    >
                                        + Train New
                                    </button>
                                </div>

                                <div className="space-y-2.5 max-h-[520px] overflow-y-auto pr-1">
                                    {profiles.map((p) => {
                                        const isSelected = p.id === selectedProfileId;
                                        return (
                                            <div
                                                key={p.id}
                                                onClick={() => setSelectedProfileId(p.id)}
                                                className={`p-3.5 rounded-2xl border transition-all cursor-pointer flex flex-col justify-between space-y-2.5 ${
                                                    isSelected
                                                        ? 'bg-teal-500/10 border-teal-500/50 shadow-apple-sm'
                                                        : 'bg-black/[0.02] dark:bg-white/[0.02] border-black/[0.06] dark:border-white/10 hover:border-black/20 dark:hover:border-white/20'
                                                }`}
                                            >
                                                <div className="flex items-start justify-between">
                                                    <div className="flex items-center space-x-3">
                                                        {p.sample_audio_path ? (
                                                            <button
                                                                type="button"
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    togglePlayProfile(p);
                                                                }}
                                                                className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all ${
                                                                    playingProfileId === p.id
                                                                        ? 'bg-teal-500 text-slate-950 shadow-md shadow-teal-500/30'
                                                                        : 'bg-teal-500/15 text-teal-700 dark:text-teal-400 hover:bg-teal-500/25'
                                                                }`}
                                                                title="Audition Profile Preview"
                                                            >
                                                                {playingProfileId === p.id ? <Pause size={14} /> : <Play size={14} className="ml-0.5" />}
                                                            </button>
                                                        ) : (
                                                            <div className="w-9 h-9 rounded-xl bg-teal-500/10 text-teal-700 dark:text-teal-400 flex items-center justify-center text-xs font-bold">
                                                                🎤
                                                            </div>
                                                        )}
                                                        <div>
                                                            <div className="flex items-center space-x-1.5">
                                                                <h5 className="text-xs font-bold text-slate-900 dark:text-white">
                                                                    {p.name}
                                                                </h5>
                                                                {p.is_default && (
                                                                    <span className="text-[9px] px-1.5 py-0.2 rounded-full bg-teal-500/20 text-teal-700 dark:text-teal-400 font-semibold">
                                                                        Default
                                                                    </span>
                                                                )}
                                                            </div>
                                                            <p className="text-[11px] text-slate-500 dark:text-slate-400 line-clamp-1 mt-0.5">
                                                                {p.description || 'Singing Voice Model'}
                                                            </p>
                                                        </div>
                                                    </div>

                                                    <input
                                                        type="radio"
                                                        name="selected_profile"
                                                        checked={isSelected}
                                                        onChange={() => setSelectedProfileId(p.id)}
                                                        className="mt-1 accent-teal-500"
                                                    />
                                                </div>

                                                {/* Acoustic Feature Chips */}
                                                <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-black/[0.04] dark:border-white/5 text-[10px]">
                                                    <span className="px-2 py-0.5 rounded-md bg-black/[0.04] dark:bg-white/[0.04] text-slate-600 dark:text-slate-400 font-mono font-medium">
                                                        {p.f0_method.toUpperCase()}
                                                    </span>
                                                    {p.acoustic_features?.median_f0_hz && (
                                                        <span className="px-2 py-0.5 rounded-md bg-teal-500/10 text-teal-700 dark:text-teal-400 font-medium flex items-center gap-1">
                                                            <Activity size={10} />
                                                            {p.acoustic_features.median_f0_hz} Hz
                                                        </span>
                                                    )}
                                                    {p.acoustic_features?.timbre_profile && (
                                                        <span className="px-2 py-0.5 rounded-md bg-purple-500/10 text-purple-700 dark:text-purple-400 font-medium">
                                                            {p.acoustic_features.timbre_profile.replace('_', ' ')}
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </GlassCard>
                        </div>

                        {/* Right Side: Vocal DSP Rack (7 cols) */}
                        <div className="xl:col-span-7 space-y-4">
                            <VocalDSPRack
                                selectedTrack={selectedTrack}
                                selectedStemPath={originalVocalStemUrl}
                                hasVocalStem={hasVocalStem}
                                voiceProfiles={profiles}
                                selectedProfileId={selectedProfileId}
                                onSelectProfileId={setSelectedProfileId}
                                pitchShift={pitchShift}
                                onChangePitchShift={setPitchShift}
                                dryWet={dryWet}
                                onChangeDryWet={setDryWet}
                                formantPreserve={formantPreserve}
                                onChangeFormantPreserve={setFormantPreserve}
                                f0Method={f0Method}
                                onChangeF0Method={setF0Method}
                                isConverting={isConverting}
                                onConvertVocals={handleConvertVocals}
                            />
                        </div>
                    </div>
                )}

                {/* ZONE 2 (Alternate): VOICE IDENTITY LIBRARY MANAGEMENT */}
                {mode === 'library' && (
                    <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
                        {/* Left: Create / Train Voice Profile */}
                        <div className="xl:col-span-5 space-y-4">
                            <form onSubmit={handleCreateProfile}>
                                <GlassCard className="p-5 border border-black/[0.08] dark:border-white/10 space-y-4">
                                    <div className="flex items-center justify-between pb-2 border-b border-black/[0.06] dark:border-white/10">
                                        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800 dark:text-slate-200">
                                            Train New Voice Identity
                                        </h4>
                                        <button
                                            type="button"
                                            onClick={() => setMode('booth')}
                                            className="px-2.5 py-1 rounded-lg bg-rose-500/10 text-rose-500 font-bold text-[10px] flex items-center gap-1 border border-rose-500/20"
                                        >
                                            <Radio size={11} />
                                            <span>Record with Mic</span>
                                        </button>
                                    </div>

                                    <div>
                                        <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1">
                                            Voice Name *
                                        </label>
                                        <input
                                            type="text"
                                            required
                                            placeholder="e.g. Luna (Indie Alto)"
                                            value={newName}
                                            onChange={(e) => setNewName(e.target.value)}
                                            className="w-full apple-input"
                                        />
                                    </div>

                                    <div>
                                        <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1">
                                            Description / Vocal Character
                                        </label>
                                        <input
                                            type="text"
                                            placeholder="e.g. Breathy vocal timbre with soft vibrato"
                                            value={newDesc}
                                            onChange={(e) => setNewDesc(e.target.value)}
                                            className="w-full apple-input"
                                        />
                                    </div>

                                    <div>
                                        <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1">
                                            Vocal Dataset Audio (.wav / .mp3 / .zip)
                                        </label>
                                        <label className="flex flex-col items-center justify-center p-4 border border-dashed border-black/20 dark:border-white/20 hover:border-teal-500/50 rounded-2xl cursor-pointer bg-black/[0.02] dark:bg-white/[0.02] transition-colors">
                                            <Upload size={18} className="text-teal-600 dark:text-teal-400 mb-1" />
                                            <span className="text-xs text-slate-600 dark:text-slate-300 font-medium">
                                                {uploadFile ? uploadFile.name : 'Select or drop solo vocal audio file'}
                                            </span>
                                            <input
                                                type="file"
                                                accept="audio/*,.zip"
                                                onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                                                className="hidden"
                                            />
                                        </label>
                                    </div>

                                    {/* Mandatory Consent Verification */}
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
                                                <strong>Mandatory Legal Consent:</strong> I verify that I own the rights or have explicit permission to clone and convert this singing voice.
                                            </span>
                                        </label>
                                    </div>

                                    <button
                                        type="submit"
                                        disabled={isSubmittingProfile || !newName || !consentConfirmed}
                                        className="w-full py-3 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 disabled:opacity-50 text-slate-950 font-bold text-xs flex items-center justify-center space-x-2 shadow-md shadow-teal-500/20 transition-all"
                                    >
                                        {isSubmittingProfile ? (
                                            <>
                                                <Loader2 size={14} className="animate-spin" />
                                                <span>Extracting F0 & Acoustic Features...</span>
                                            </>
                                        ) : (
                                            <>
                                                <ShieldCheck size={14} />
                                                <span>Train & Save Voice Identity</span>
                                            </>
                                        )}
                                    </button>
                                </GlassCard>
                            </form>
                        </div>

                        {/* Right: Existing Profiles Grid */}
                        <div className="xl:col-span-7 space-y-4">
                            <GlassCard className="p-5 border border-black/[0.08] dark:border-white/10 space-y-4">
                                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800 dark:text-slate-200">
                                    Saved Voice Profiles ({profiles.length})
                                </h4>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                    {profiles.map((p) => (
                                        <div
                                            key={p.id}
                                            className="p-4 rounded-2xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.06] dark:border-white/10 flex flex-col justify-between space-y-3"
                                        >
                                            <div className="flex items-start justify-between">
                                                <div className="flex items-center space-x-3">
                                                    {p.sample_audio_path ? (
                                                        <button
                                                            type="button"
                                                            onClick={() => togglePlayProfile(p)}
                                                            className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all ${
                                                                playingProfileId === p.id
                                                                    ? 'bg-teal-500 text-slate-950 shadow-md shadow-teal-500/30'
                                                                    : 'bg-teal-500/10 text-teal-700 dark:text-teal-400 hover:bg-teal-500/20'
                                                            }`}
                                                        >
                                                            {playingProfileId === p.id ? <Pause size={14} /> : <Play size={14} className="ml-0.5" />}
                                                        </button>
                                                    ) : (
                                                        <div className="w-9 h-9 rounded-xl bg-teal-500/10 text-teal-700 dark:text-teal-400 flex items-center justify-center text-xs font-bold">
                                                            🎤
                                                        </div>
                                                    )}
                                                    <div>
                                                        <h5 className="text-xs font-bold text-slate-900 dark:text-white">
                                                            {p.name}
                                                        </h5>
                                                        <p className="text-[11px] text-slate-500 dark:text-slate-400 line-clamp-1">
                                                            {p.description || 'Vocal Model'}
                                                        </p>
                                                    </div>
                                                </div>

                                                {!p.is_default && (
                                                    <button
                                                        type="button"
                                                        onClick={() => handleDeleteProfile(p.id)}
                                                        className="p-1.5 rounded-lg text-slate-400 hover:text-rose-500 hover:bg-rose-500/10 transition-colors"
                                                        title="Delete Profile"
                                                    >
                                                        <Trash2 size={13} />
                                                    </button>
                                                )}
                                            </div>

                                            {/* Feature Chips */}
                                            <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-black/[0.04] dark:border-white/5 text-[10px]">
                                                <span className="px-2 py-0.5 rounded-md bg-black/[0.04] dark:bg-white/[0.04] text-slate-600 dark:text-slate-400 font-mono font-medium">
                                                    {p.f0_method.toUpperCase()}
                                                </span>
                                                {p.acoustic_features?.median_f0_hz && (
                                                    <span className="px-2 py-0.5 rounded-md bg-teal-500/10 text-teal-700 dark:text-teal-400 font-medium flex items-center gap-1">
                                                        <Activity size={10} />
                                                        {p.acoustic_features.median_f0_hz} Hz
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </GlassCard>
                        </div>
                    </div>
                )}

                {/* ZONE 2 (Alternate): LIVE VOCAL BOOTH (MICROPHONE) */}
                {mode === 'booth' && (
                    <div className="max-w-2xl mx-auto space-y-4">
                        <VocalBoothRecorder
                            onAudioCaptured={handleVocalBoothCaptured}
                            onCancel={() => setMode('conversion')}
                        />
                    </div>
                )}

                {/* ZONE 3: FULL-WIDTH AUDITION & A/B COMPARISON TRANSPORT */}
                <VocalAuditionPlayer
                    track={selectedTrack}
                    originalVocalUrl={originalVocalStemUrl}
                    convertedVocalUrl={convertedVocalUrl}
                    remixedMasterUrl={remixedMasterUrl}
                    onOpenInDAW={onOpenWorkspace}
                />
            </div>

            {/* App Footer */}
            {!isModal && <AppFooter />}
        </div>
    );
};
