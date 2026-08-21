import React, { useState, useEffect, useRef } from 'react';
import {
    type Job,
    type VoiceProfile,
    trackApi,
    voiceApi,
    API_BASE_URL
} from '../../api';
import { useAudioEngine } from '../../context/AudioEngineContext';
import {
    ArrowLeft,
    Play,
    Pause,
    Heart,
    Sliders,
    Sparkles,
    Download,
    Package,
    Music,
    Mic,
    FileText,
    Copy,
    Check,
    RefreshCw,
    Edit3,
    GitFork,
    Layers,
    Zap,
    Cpu,
    FileCode,
    Volume2,
    VolumeX,
    RotateCcw,
    RotateCw,
    SkipBack,
    SkipForward,
    Shuffle,
    Repeat,
    Repeat1,
    Gauge
} from 'lucide-react';

interface TrackDetailViewProps {
    track: Job;
    onBack: () => void;
    onPlay?: (job: Job) => void;
    isPlaying?: boolean;
    playingSongId?: string | null;
    onOpenWorkspace: (job: Job) => void;
    onExtend: (job: Job) => void;
    onReroll: (preset: any) => void;
    onToggleFavorite: (jobId: string) => void;
    onTrackUpdated?: (job: Job) => void;
    allJobs?: Job[];
    onSelectTrack?: (job: Job) => void;
}

export const TrackDetailView: React.FC<TrackDetailViewProps> = ({
    track: initialTrack,
    onBack,
    onOpenWorkspace,
    onExtend,
    onReroll,
    onToggleFavorite,
    onTrackUpdated,
    allJobs = [],
    onSelectTrack
}) => {
    const {
        currentTrack: engineTrack,
        isPlaying: engineIsPlaying,
        currentTime: engineCurrentTime,
        duration: engineDuration,
        volume: engineVolume,
        isMuted: engineIsMuted,
        togglePlay: engineTogglePlay,
        playTrack: enginePlayTrack,
        seek: engineSeek,
        prevTrackOrRestart: enginePrevTrackOrRestart,
        nextTrack: engineNextTrack,
        isShuffle: engineIsShuffle,
        toggleShuffle: engineToggleShuffle,
        repeatMode: engineRepeatMode,
        setRepeatMode: engineSetRepeatMode,
        playbackRate: enginePlaybackRate,
        setPlaybackRate: engineSetPlaybackRate,
        setVolume: engineSetVolume,
        toggleMute: engineToggleMute
    } = useAudioEngine();

    const [timeMode, setTimeMode] = useState<'elapsed' | 'remaining'>('elapsed');
    const [isSpeedOpen, setIsSpeedOpen] = useState(false);
    const [track, setTrack] = useState<Job>(initialTrack);
    const [activeTab, setActiveTab] = useState<'stems' | 'score' | 'lyrics' | 'provenance' | 'lineage'>('stems');
    const [isEditingTitle, setIsEditingTitle] = useState(false);
    const [titleInput, setTitleInput] = useState(track.title || track.prompt);
    const [copiedSeed, setCopiedSeed] = useState(false);
    const [copiedLyrics, setCopiedLyrics] = useState(false);
    const [selectedVoiceProfile, setSelectedVoiceProfile] = useState<string>('');
    const [voiceProfiles, setVoiceProfiles] = useState<VoiceProfile[]>([]);
    const [isConvertingVoice, setIsConvertingVoice] = useState(false);
    const [pitchShift, setPitchShift] = useState<number>(0);
    const [formantPreserve, setFormantPreserve] = useState<boolean>(true);
    const [dryWet, setDryWet] = useState<number>(100);
    const [stemSourceMode, setStemSourceMode] = useState<'muscriptor' | 'htdemucs'>('muscriptor');

    // Dedicated Stem Audition Audio Node
    const stemAudioRef = useRef<HTMLAudioElement | null>(null);
    const [playingStemKey, setPlayingStemKey] = useState<string | null>(null);

    // Stem playback preview states
    const [soloStem, setSoloStem] = useState<string | null>(null);
    const [mutedStems, setMutedStems] = useState<Set<string>>(new Set());

    const isCurrentPlaying = engineIsPlaying && engineTrack?.id === track.id;
    const currentTime = engineTrack?.id === track.id ? engineCurrentTime : 0;
    const duration = (engineTrack?.id === track.id && engineDuration)
        ? engineDuration
        : (track.duration_ms ? track.duration_ms / 1000 : 60);

    const formatTime = (seconds: number) => {
        if (isNaN(seconds) || seconds < 0) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const handleToggleMasterPlay = () => {
        if (stemAudioRef.current && playingStemKey) {
            stemAudioRef.current.pause();
            setPlayingStemKey(null);
        }
        engineTogglePlay(track);
    };

    const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
        const targetTime = parseFloat(e.target.value);
        if (engineTrack?.id === track.id) {
            engineSeek(targetTime);
        } else {
            enginePlayTrack(track).then(() => engineSeek(targetTime));
        }
    };

    const handleStemPlay = (stemKey: string, stemPath?: string) => {
        if (!stemPath) return;
        const fullUrl = stemPath.startsWith('http') ? stemPath : `${API_BASE_URL}${stemPath}`;
        const stemAudio = stemAudioRef.current;
        if (!stemAudio) return;

        if (playingStemKey === stemKey) {
            stemAudio.pause();
            setPlayingStemKey(null);
        } else {
            if (isCurrentPlaying) {
                engineTogglePlay();
            }
            stemAudio.src = fullUrl;
            stemAudio.play().then(() => {
                setPlayingStemKey(stemKey);
            }).catch(console.error);
        }
    };

    useEffect(() => {
        setTrack(initialTrack);
        setTitleInput(initialTrack.title || initialTrack.prompt);
    }, [initialTrack]);

    useEffect(() => {
        voiceApi.listProfiles().then(setVoiceProfiles).catch(console.error);
    }, []);

    // Parse metadata safely
    const stemsData = track.stems_json ? (() => {
        try { return JSON.parse(track.stems_json); } catch { return {}; }
    })() : {};

    const notesData = track.notes_json ? (() => {
        try { return JSON.parse(track.notes_json); } catch { return []; }
    })() : [];

    const beatGrid = track.beat_grid_json ? (() => {
        try { return JSON.parse(track.beat_grid_json); } catch { return {}; }
    })() : {};

    const timedLyrics = track.timed_lyrics_json ? (() => {
        try { return JSON.parse(track.timed_lyrics_json); } catch { return []; }
    })() : [];

    const structuredCaption = track.structured_caption_json ? (() => {
        try { return JSON.parse(track.structured_caption_json); } catch { return {}; }
    })() : {};

    const [isRealigningLyrics, setIsRealigningLyrics] = useState(false);
    const [lyricsDisplayMode, setLyricsDisplayMode] = useState<'karaoke' | 'raw'>('karaoke');

    const activeLineIndex = (() => {
        if (timedLyrics.length === 0) return -1;
        for (let i = 0; i < timedLyrics.length; i++) {
            const line = timedLyrics[i];
            const nextLine = timedLyrics[i + 1];
            const lineStart = line.start;
            const lineEnd = nextLine ? nextLine.start : (line.end || line.start + 6);
            if (currentTime >= lineStart && currentTime < lineEnd) {
                return i;
            }
        }
        if (currentTime >= timedLyrics[timedLyrics.length - 1].start) {
            return timedLyrics.length - 1;
        }
        return -1;
    })();

    const handleRealignLyrics = async () => {
        setIsRealigningLyrics(true);
        try {
            const res = await trackApi.realignLyrics(track.id, track.lyrics);
            setTrack(res.job);
            onTrackUpdated?.(res.job);
        } catch (e: any) {
            console.error('Failed to realign lyrics', e);
        } finally {
            setIsRealigningLyrics(false);
        }
    };

    const handleSaveTitle = async () => {
        if (!titleInput.trim()) return;
        try {
            const updated = await trackApi.updateTrackMetadata(track.id, { title: titleInput.trim() });
            setTrack(updated);
            onTrackUpdated?.(updated);
            setIsEditingTitle(false);
        } catch (e) {
            console.error('Failed to update title', e);
        }
    };

    const handleCopySeed = () => {
        if (track.seed !== undefined && track.seed !== null) {
            navigator.clipboard.writeText(track.seed.toString());
            setCopiedSeed(true);
            setTimeout(() => setCopiedSeed(false), 2000);
        }
    };

    const handleCopyLyrics = () => {
        if (track.lyrics) {
            navigator.clipboard.writeText(track.lyrics);
            setCopiedLyrics(true);
            setTimeout(() => setCopiedLyrics(false), 2000);
        }
    };

    const handleApplyVoiceConvert = async () => {
        if (!selectedVoiceProfile) return;
        setIsConvertingVoice(true);
        try {
            const derivative = await trackApi.voiceConvertTrack(track.id, selectedVoiceProfile);
            onTrackUpdated?.(derivative);
            if (onSelectTrack) onSelectTrack(derivative);
        } catch (e: any) {
            alert('Voice Conversion Error: ' + (e.response?.data?.detail || e.message));
        } finally {
            setIsConvertingVoice(false);
        }
    };

    // Stems list derivation
    const master4Stems = [
        { key: 'vocals', label: 'Vocals', icon: '🎤', path: stemsData.vocals },
        { key: 'drums', label: 'Drums', icon: '🥁', path: stemsData.drums },
        { key: 'bass', label: 'Bass', icon: '🎸', path: stemsData.bass },
        { key: 'other', label: 'Other Instruments', icon: '🎹', path: stemsData.other }
    ].filter(s => !!s.path);

    const instrumentParts = stemsData.instrumental_parts ? Object.entries(stemsData.instrumental_parts).map(([name, path]) => ({
        key: name,
        label: name,
        icon: name.toLowerCase().includes('drum') ? '🥁' : name.toLowerCase().includes('bass') ? '🎸' : name.toLowerCase().includes('vocal') ? '🎤' : '🎹',
        path: path as string
    })) : [];

    const activeStemsList = stemSourceMode === 'muscriptor' && instrumentParts.length > 0 ? instrumentParts : master4Stems;

    // Lineage derivation
    const parentTrack = track.parent_job_id ? allJobs.find(j => j.id === track.parent_job_id) : null;
    const derivativeTracks = allJobs.filter(j => j.parent_job_id === track.id);

    return (
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 md:p-8 space-y-6 max-w-6xl mx-auto w-full min-w-0 animate-fade-in">
            {/* Top Navigation & Breadcrumb */}
            <div className="flex items-center justify-between gap-4">
                <button
                    onClick={onBack}
                    className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-xs font-semibold text-slate-700 dark:text-slate-300 transition-colors shadow-sm"
                >
                    <ArrowLeft size={14} />
                    <span>Back to Library</span>
                </button>

                <div className="flex items-center gap-2">
                    <a
                        href={trackApi.getStudioPackUrl(track.id)}
                        download
                        className="px-3.5 py-1.5 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs flex items-center gap-1.5 shadow-md shadow-teal-500/20 active:scale-[0.98] transition-all"
                    >
                        <Package size={14} />
                        <span>Download Studio Pack (.zip)</span>
                    </a>

                    <button
                        onClick={() => onOpenWorkspace(track)}
                        className="px-3.5 py-1.5 rounded-xl bg-black/[0.05] dark:bg-white/10 hover:bg-teal-500/15 text-teal-700 dark:text-teal-300 border border-teal-500/20 font-bold text-xs flex items-center gap-1.5 transition-all"
                    >
                        <Sliders size={14} />
                        <span>Open in DAW</span>
                    </button>
                </div>
            </div>

            {/* Master Track Hero Command Bar */}
            <div className="bg-white/80 dark:bg-[#141620]/90 rounded-3xl border border-black/[0.08] dark:border-white/10 shadow-apple-lg backdrop-blur-2xl p-5 sm:p-6 md:p-7 space-y-6 relative overflow-hidden">
                {/* Dedicated Stem Audition Element (Master Audio is managed globally by AudioEngineContext) */}
                <audio
                    ref={stemAudioRef}
                    onEnded={() => setPlayingStemKey(null)}
                />

                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-5 md:gap-6">
                    {/* Artwork Container */}
                    <div className="relative w-28 h-28 sm:w-32 sm:h-32 md:w-36 md:h-36 rounded-2xl overflow-hidden bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 shadow-apple-md flex-shrink-0 group">
                        {track.cover_image_path ? (
                            <img
                                src={track.cover_image_path.startsWith('http') ? track.cover_image_path : `${API_BASE_URL}${track.cover_image_path}`}
                                alt={track.title || 'Cover Artwork'}
                                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                            />
                        ) : (
                            <div className="w-full h-full flex flex-col items-center justify-center text-slate-400">
                                <Music size={36} className="opacity-40 mb-1" />
                                <span className="text-[10px] font-mono">No Artwork</span>
                            </div>
                        )}
                        <button
                            onClick={handleToggleMasterPlay}
                            className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-white backdrop-blur-[2px] cursor-pointer"
                        >
                            {isCurrentPlaying ? <Pause size={28} /> : <Play size={28} className="ml-1" />}
                        </button>
                    </div>

                    {/* Track Metadata & Controls */}
                    <div className="flex-1 min-w-0 space-y-2.5 w-full">
                        <div className="flex items-start justify-between gap-3">
                            <div className="flex-1 min-w-0">
                                {isEditingTitle ? (
                                    <div className="flex items-center gap-2 max-w-md">
                                        <input
                                            type="text"
                                            value={titleInput}
                                            onChange={(e) => setTitleInput(e.target.value)}
                                            onKeyDown={(e) => e.key === 'Enter' && handleSaveTitle()}
                                            className="apple-input text-base font-bold py-1 px-2.5 flex-1"
                                            autoFocus
                                        />
                                        <button
                                            onClick={handleSaveTitle}
                                            className="px-2.5 py-1 rounded-lg bg-teal-500 text-slate-950 font-bold text-xs"
                                        >
                                            Save
                                        </button>
                                        <button
                                            onClick={() => setIsEditingTitle(false)}
                                            className="p-1 text-slate-400 hover:text-slate-600 text-xs"
                                        >
                                            ✕
                                        </button>
                                    </div>
                                ) : (
                                    <div className="flex items-center gap-2 group">
                                        <h1 className="text-xl sm:text-2xl md:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white truncate">
                                            {track.title || track.prompt}
                                        </h1>
                                        <button
                                            onClick={() => setIsEditingTitle(true)}
                                            className="opacity-0 group-hover:opacity-100 p-1 text-slate-400 hover:text-teal-600 dark:hover:text-teal-300 transition-opacity"
                                            title="Edit Track Title"
                                        >
                                            <Edit3 size={14} />
                                        </button>
                                    </div>
                                )}
                                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 line-clamp-2 leading-relaxed">
                                    {track.prompt}
                                </p>
                            </div>

                            <button
                                onClick={() => onToggleFavorite(track.id)}
                                className={`p-2 rounded-xl border transition-all ${
                                    track.is_favorite
                                        ? 'bg-rose-500/10 border-rose-500/20 text-rose-500'
                                        : 'bg-black/[0.03] dark:bg-white/5 border-transparent text-slate-400 hover:text-rose-500'
                                }`}
                                title="Favorite Track"
                            >
                                <Heart size={16} className={track.is_favorite ? 'fill-rose-500' : ''} />
                            </button>
                        </div>

                        {/* Technical Metadata Chips */}
                        <div className="flex flex-wrap items-center gap-1.5 pt-1">
                            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 font-bold border border-teal-500/20">
                                ⚡ {beatGrid.bpm ? `${Math.round(beatGrid.bpm)} BPM` : '120 BPM'}
                            </span>
                            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-700 dark:text-cyan-300 font-semibold border border-cyan-500/20">
                                🎼 {beatGrid.beats_per_bar ? `${beatGrid.beats_per_bar}/4 Time` : '4/4 Time'}
                            </span>
                            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-700 dark:text-amber-300 font-semibold border border-amber-500/20">
                                ⏱️ {Math.round((track.duration_ms || 60000) / 1000)}s
                            </span>
                            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-black/[0.04] dark:bg-white/5 text-slate-600 dark:text-slate-400 border border-black/[0.06] dark:border-white/5">
                                🔊 -14.0 LUFS Master
                            </span>
                            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-black/[0.04] dark:bg-white/5 text-slate-600 dark:text-slate-400 border border-black/[0.06] dark:border-white/5">
                                🎛️ {track.model_provider || 'MiniMax Music 3'}
                            </span>
                        </div>

                        {/* Primary Action Buttons */}
                        <div className="flex flex-wrap items-center gap-2 pt-2">
                            <button
                                onClick={handleToggleMasterPlay}
                                className="px-4 py-2 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs flex items-center gap-2 shadow-md shadow-teal-500/20 transition-all active:scale-[0.98] cursor-pointer"
                            >
                                {isCurrentPlaying ? <Pause size={14} /> : <Play size={14} className="ml-0.5" />}
                                <span>{isCurrentPlaying ? 'Pause Master' : 'Play Master Audio'}</span>
                            </button>

                            <button
                                onClick={() => onExtend(track)}
                                className="px-3.5 py-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 font-bold text-xs flex items-center gap-1.5 transition-colors border border-black/[0.06] dark:border-white/5"
                            >
                                <Sparkles size={13} />
                                <span>Extend Outro</span>
                            </button>

                            <button
                                onClick={() => onReroll({
                                    topic: track.prompt,
                                    tags: track.tags,
                                    lyrics: track.lyrics,
                                    structuredCaption: structuredCaption,
                                    seed: track.seed,
                                    durationMs: track.duration_ms
                                })}
                                className="px-3.5 py-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 font-bold text-xs flex items-center gap-1.5 transition-colors border border-black/[0.06] dark:border-white/5"
                            >
                                <RefreshCw size={13} />
                                <span>Remix / Re-roll</span>
                            </button>
                        </div>
                    </div>
                </div>

                {/* Master Audio Waveform & Transport Controls */}
                <div className="pt-4 border-t border-black/[0.06] dark:border-white/10 space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
                        {/* Transport Buttons & Timecode */}
                        <div className="flex items-center space-x-1 sm:space-x-1.5 flex-shrink-0">
                            {/* Shuffle */}
                            <button
                                onClick={engineToggleShuffle}
                                className={`p-1.5 rounded-xl transition-colors hidden sm:block ${
                                    engineIsShuffle
                                        ? 'text-teal-600 dark:text-teal-400 bg-teal-500/10'
                                        : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                                }`}
                                title={`Shuffle: ${engineIsShuffle ? 'On' : 'Off'}`}
                            >
                                <Shuffle size={14} />
                            </button>

                            {/* Return to Start / Previous Track Button */}
                            <button
                                onClick={() => {
                                    if (engineTrack?.id === track.id) {
                                        enginePrevTrackOrRestart();
                                    } else {
                                        enginePlayTrack(track);
                                    }
                                }}
                                className="p-1.5 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
                                title="Return to Start / Previous Track (|<<)"
                            >
                                <SkipBack size={15} />
                            </button>

                            {/* Rewind 10s */}
                            <button
                                onClick={() => {
                                    if (engineTrack?.id === track.id) {
                                        engineSeek(Math.max(0, currentTime - 10));
                                    }
                                }}
                                className="p-1.5 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
                                title="Rewind 10s (J)"
                            >
                                <RotateCcw size={14} />
                            </button>

                            {/* Play/Pause Hero Button */}
                            <button
                                onClick={handleToggleMasterPlay}
                                className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold flex items-center justify-center shadow-sm shadow-teal-500/20 active:scale-95 transition-transform"
                                title={isCurrentPlaying ? 'Pause Master (Space / K)' : 'Play Master (Space / K)'}
                            >
                                {isCurrentPlaying ? <Pause size={16} /> : <Play size={16} className="ml-0.5" />}
                            </button>

                            {/* Advance 10s */}
                            <button
                                onClick={() => {
                                    if (engineTrack?.id === track.id) {
                                        engineSeek(Math.min(duration, currentTime + 10));
                                    }
                                }}
                                className="p-1.5 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
                                title="Advance 10s (L)"
                            >
                                <RotateCw size={14} />
                            </button>

                            {/* Next Track */}
                            <button
                                onClick={engineNextTrack}
                                className="p-1.5 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
                                title="Next Track (>>|)"
                            >
                                <SkipForward size={15} />
                            </button>

                            {/* Repeat / Loop */}
                            <button
                                onClick={() => {
                                    const next = engineRepeatMode === 'off' ? 'all' : engineRepeatMode === 'all' ? 'one' : 'off';
                                    engineSetRepeatMode(next);
                                }}
                                className={`p-1.5 rounded-xl transition-colors hidden sm:block ${
                                    engineRepeatMode !== 'off'
                                        ? 'text-teal-600 dark:text-teal-400 bg-teal-500/10'
                                        : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                                }`}
                                title={`Repeat Mode: ${engineRepeatMode}`}
                            >
                                {engineRepeatMode === 'one' ? <Repeat1 size={14} /> : <Repeat size={14} />}
                            </button>

                            {/* Timecode Toggle */}
                            <button
                                onClick={() => setTimeMode(timeMode === 'elapsed' ? 'remaining' : 'elapsed')}
                                className="flex items-center gap-1.5 pl-2 select-none hover:opacity-80 transition-opacity"
                                title="Toggle Elapsed / Remaining Time"
                            >
                                <span className="font-bold text-teal-600 dark:text-teal-400">
                                    {timeMode === 'elapsed'
                                        ? formatTime(currentTime)
                                        : `-${formatTime(Math.max(0, duration - currentTime))}`}
                                </span>
                                <span className="text-slate-400">/</span>
                                <span className="text-slate-500 dark:text-slate-400">
                                    {formatTime(duration || (track.duration_ms || 60000) / 1000)}
                                </span>
                            </button>
                        </div>

                        {/* Animated Equalizer Waves */}
                        <div className="hidden lg:flex items-center gap-0.5 h-4 px-2">
                            {Array.from({ length: 16 }).map((_, i) => (
                                <div
                                    key={i}
                                    className={`w-1 bg-teal-500 rounded-full transition-all duration-150 ${
                                        isCurrentPlaying ? 'animate-pulse' : 'opacity-30'
                                    }`}
                                    style={{
                                        height: isCurrentPlaying
                                            ? `${Math.max(20, Math.sin(i * 0.8 + currentTime * 5) * 80 + 20)}%`
                                            : '25%'
                                    }}
                                />
                            ))}
                        </div>

                        {/* Right Controls: Speed & Volume Slider */}
                        <div className="flex items-center gap-2 flex-shrink-0">
                            {/* Speed Selector */}
                            <div className="relative">
                                <button
                                    onClick={() => setIsSpeedOpen(!isSpeedOpen)}
                                    className="px-2 py-1 rounded-lg bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-[10px] font-mono font-bold text-slate-700 dark:text-slate-300 flex items-center gap-1 transition-colors"
                                    title="Playback Speed"
                                >
                                    <Gauge size={11} className="text-teal-500" />
                                    <span>{enginePlaybackRate}x</span>
                                </button>

                                {isSpeedOpen && (
                                    <div className="absolute bottom-full mb-2 right-0 bg-white dark:bg-[#181a24] border border-black/[0.08] dark:border-white/10 rounded-xl shadow-apple-lg p-1 space-y-1 z-50 animate-fade-in">
                                        {[0.75, 1.0, 1.25, 1.5, 2.0].map((s) => (
                                            <button
                                                key={s}
                                                onClick={() => {
                                                    engineSetPlaybackRate(s);
                                                    setIsSpeedOpen(false);
                                                }}
                                                className={`w-full px-3 py-1 text-left text-xs font-mono rounded-lg transition-colors ${
                                                    enginePlaybackRate === s
                                                        ? 'bg-teal-500/15 text-teal-700 dark:text-teal-300 font-bold'
                                                        : 'text-slate-600 dark:text-slate-400 hover:bg-black/5 dark:hover:bg-white/5'
                                                }`}
                                            >
                                                {s}x
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>

                            {/* Volume */}
                            <button
                                onClick={engineToggleMute}
                                className="text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
                            >
                                {engineIsMuted || engineVolume === 0 ? <VolumeX size={14} /> : <Volume2 size={14} />}
                            </button>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.05"
                                value={engineIsMuted ? 0 : engineVolume}
                                onChange={(e) => engineSetVolume(parseFloat(e.target.value))}
                                className="w-16 accent-teal-500 h-1 bg-black/10 dark:bg-white/10 rounded-lg cursor-pointer"
                            />
                        </div>
                    </div>

                    {/* Draggable Scrubber */}
                    <input
                        type="range"
                        min="0"
                        max={duration || (track.duration_ms || 60000) / 1000}
                        step="0.1"
                        value={currentTime}
                        onChange={handleSeek}
                        className="w-full accent-teal-500 h-2 bg-black/[0.06] dark:bg-white/10 rounded-lg cursor-pointer"
                    />
                </div>
            </div>

            {/* Apple Segmented Tab Switcher */}
            <div className="flex items-center p-1 rounded-2xl bg-black/[0.04] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 overflow-x-auto">
                {[
                    { id: 'stems', label: 'Stems Matrix', icon: Layers, count: activeStemsList.length },
                    { id: 'score', label: 'Score & MIDI', icon: Music, count: notesData.length },
                    { id: 'lyrics', label: 'Vocal & Lyrics', icon: Mic, count: timedLyrics.length },
                    { id: 'provenance', label: 'AI Provenance', icon: Cpu },
                    { id: 'lineage', label: 'Version Tree', icon: GitFork, count: derivativeTracks.length + (parentTrack ? 1 : 0) }
                ].map(tab => {
                    const Icon = tab.icon;
                    const isActive = activeTab === tab.id;
                    return (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id as any)}
                            className={`flex-1 py-2 px-3 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-2 whitespace-nowrap ${
                                isActive
                                    ? 'bg-white dark:bg-white/15 text-teal-700 dark:text-teal-300 shadow-apple-sm'
                                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                            }`}
                        >
                            <Icon size={14} className={isActive ? 'text-teal-500' : 'text-slate-400'} />
                            <span>{tab.label}</span>
                            {tab.count !== undefined && (
                                <span className={`text-[10px] px-1.5 py-0.2 rounded-full font-mono ${
                                    isActive ? 'bg-teal-500/20 text-teal-700 dark:text-teal-300' : 'bg-black/5 dark:bg-white/5 text-slate-400'
                                }`}>
                                    {tab.count}
                                </span>
                            )}
                        </button>
                    );
                })}
            </div>

            {/* TAB CONTENT PANES */}
            <div className="space-y-4">
                {/* 1. STEMS MATRIX TAB */}
                {activeTab === 'stems' && (
                    <div className="space-y-4 animate-fade-in">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-2">
                            <div>
                                <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                                    <span>Multitrack Stems Matrix</span>
                                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20">
                                        Dual-Engine Ready
                                    </span>
                                </h3>
                                <p className="text-xs text-slate-500 dark:text-slate-400">
                                    Solo, mute, audition, and download uncompressed 48kHz WAV audio stems.
                                </p>
                            </div>

                            {/* Stem Source Switcher */}
                            {instrumentParts.length > 0 && (
                                <div className="flex items-center bg-black/[0.04] dark:bg-white/5 p-1 rounded-xl border border-black/[0.06] dark:border-white/10 text-xs font-semibold">
                                    <button
                                        onClick={() => setStemSourceMode('muscriptor')}
                                        className={`px-3 py-1 rounded-lg transition-all ${
                                            stemSourceMode === 'muscriptor'
                                                ? 'bg-white dark:bg-white/20 text-teal-700 dark:text-teal-300 font-bold shadow-sm'
                                                : 'text-slate-500'
                                        }`}
                                    >
                                        Dynamic Instrument Parts ({instrumentParts.length})
                                    </button>
                                    <button
                                        onClick={() => setStemSourceMode('htdemucs')}
                                        className={`px-3 py-1 rounded-lg transition-all ${
                                            stemSourceMode === 'htdemucs'
                                                ? 'bg-white dark:bg-white/20 text-teal-700 dark:text-teal-300 font-bold shadow-sm'
                                                : 'text-slate-500'
                                        }`}
                                    >
                                        4 Master Stems (HTDemucs)
                                    </button>
                                </div>
                            )}
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                            {activeStemsList.map((stem) => {
                                const isMuted = mutedStems.has(stem.key);
                                const isSolo = soloStem === stem.key;
                                return (
                                    <div
                                        key={stem.key}
                                        className="p-4 rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/10 shadow-apple-sm backdrop-blur-xl flex flex-col justify-between space-y-3"
                                    >
                                        <div className="flex items-center justify-between gap-2">
                                            <div className="flex items-center gap-2.5">
                                                <div className="w-8 h-8 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400 flex items-center justify-center font-bold text-sm">
                                                    {stem.icon}
                                                </div>
                                                <div>
                                                    <h4 className="text-xs font-bold text-slate-900 dark:text-white capitalize">
                                                        {stem.label}
                                                    </h4>
                                                    <span className="text-[10px] font-mono text-slate-400">
                                                        48kHz Stereo WAV
                                                    </span>
                                                </div>
                                            </div>

                                            <div className="flex items-center gap-1.5">
                                                <button
                                                    onClick={() => handleStemPlay(stem.key, stem.path)}
                                                    className={`p-1.5 rounded-lg border transition-all ${
                                                        playingStemKey === stem.key
                                                            ? 'bg-teal-500 text-slate-950 border-teal-400 shadow-sm'
                                                            : 'bg-teal-500/10 text-teal-600 dark:text-teal-400 border-teal-500/20 hover:bg-teal-500 hover:text-slate-950'
                                                    }`}
                                                    title={playingStemKey === stem.key ? 'Pause Stem' : 'Audition Isolated Stem'}
                                                >
                                                    {playingStemKey === stem.key ? <Pause size={12} /> : <Play size={12} className="ml-0.5" />}
                                                </button>

                                                <button
                                                    onClick={() => setSoloStem(isSolo ? null : stem.key)}
                                                    className={`w-7 h-7 rounded-lg font-mono font-bold text-[11px] flex items-center justify-center border transition-all ${
                                                        isSolo
                                                            ? 'bg-amber-500 text-slate-950 border-amber-400 shadow-sm'
                                                            : 'bg-black/[0.03] dark:bg-white/5 border-black/10 dark:border-white/10 text-slate-500 hover:text-amber-500'
                                                    }`}
                                                    title="Solo Stem"
                                                >
                                                    S
                                                </button>
                                                <button
                                                    onClick={() => {
                                                        const next = new Set(mutedStems);
                                                        if (next.has(stem.key)) next.delete(stem.key);
                                                        else next.add(stem.key);
                                                        setMutedStems(next);
                                                    }}
                                                    className={`w-7 h-7 rounded-lg font-mono font-bold text-[11px] flex items-center justify-center border transition-all ${
                                                        isMuted
                                                            ? 'bg-rose-500 text-white border-rose-400 shadow-sm'
                                                            : 'bg-black/[0.03] dark:bg-white/5 border-black/10 dark:border-white/10 text-slate-500 hover:text-rose-500'
                                                    }`}
                                                    title="Mute Stem"
                                                >
                                                    M
                                                </button>
                                            </div>
                                        </div>

                                        {/* Waveform graphic bar */}
                                        <div className="h-8 rounded-xl bg-black/[0.03] dark:bg-white/5 border border-black/[0.04] dark:border-white/5 p-1 flex items-center justify-between gap-0.5 overflow-hidden">
                                            {Array.from({ length: 40 }).map((_, i) => (
                                                <div
                                                    key={i}
                                                    className="flex-1 bg-teal-500/30 rounded-full transition-all"
                                                    style={{ height: `${Math.max(15, Math.sin(i * 0.4) * 80 + 20)}%` }}
                                                />
                                            ))}
                                        </div>

                                        {/* Action: Download Stem WAV */}
                                        <div className="flex items-center justify-between pt-1 border-t border-black/[0.04] dark:border-white/5 text-[11px]">
                                            <span className="text-slate-400 font-mono">
                                                Broadcast Ready
                                            </span>
                                            {stem.path && (
                                                <a
                                                    href={stem.path.startsWith('http') ? stem.path : `${API_BASE_URL}${stem.path}`}
                                                    download={`${track.title || 'track'}_${stem.key}.wav`}
                                                    className="text-teal-600 dark:text-teal-400 hover:underline font-bold flex items-center gap-1"
                                                >
                                                    <Download size={12} />
                                                    <span>Download WAV</span>
                                                </a>
                                            )}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}

                {/* 2. NEURAL SCORE & MIDI TAB */}
                {activeTab === 'score' && (
                    <div className="space-y-4 animate-fade-in">
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                            <div className="p-3.5 rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/10 shadow-apple-sm text-center">
                                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                                    Total Notes
                                </span>
                                <span className="text-xl font-extrabold text-teal-600 dark:text-teal-400 font-mono">
                                    {notesData.length}
                                </span>
                            </div>

                            <div className="p-3.5 rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/10 shadow-apple-sm text-center">
                                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                                    Pitch Range
                                </span>
                                <span className="text-xl font-extrabold text-slate-800 dark:text-slate-200 font-mono">
                                    {notesData.length > 0 ? `${notesData[0].note_name || 'C2'} – ${notesData[notesData.length - 1].note_name || 'G5'}` : 'C2 – C6'}
                                </span>
                            </div>

                            <div className="p-3.5 rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/10 shadow-apple-sm text-center">
                                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                                    Engraved Clefs
                                </span>
                                <span className="text-xl font-extrabold text-cyan-600 dark:text-cyan-400 font-mono">
                                    Treble & Bass
                                </span>
                            </div>

                            <div className="p-3.5 rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/10 shadow-apple-sm text-center">
                                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                                    Quantization Grid
                                </span>
                                <span className="text-xl font-extrabold text-amber-600 dark:text-amber-400 font-mono">
                                    1/16 Beat Alignment
                                </span>
                            </div>
                        </div>

                        {/* Export Center */}
                        <div className="p-5 rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/10 shadow-apple-sm space-y-4">
                            <h4 className="text-xs font-bold text-slate-900 dark:text-white uppercase tracking-wider">
                                Export Neural Scores & Transcriptions
                            </h4>
                            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                {track.midi_path && (
                                    <a
                                        href={track.midi_path.startsWith('http') ? track.midi_path : `${API_BASE_URL}${track.midi_path}`}
                                        download={`${track.title || 'track'}.mid`}
                                        className="p-3.5 rounded-xl bg-black/[0.03] dark:bg-white/5 hover:bg-teal-500/10 border border-black/[0.06] dark:border-white/5 flex items-center justify-between group transition-colors"
                                    >
                                        <div className="flex items-center gap-2.5">
                                            <Music size={18} className="text-teal-500" />
                                            <div>
                                                <div className="text-xs font-bold text-slate-900 dark:text-white">Multi-Track MIDI (.mid)</div>
                                                <div className="text-[10px] text-slate-400 font-mono">Logic, Ableton, FL Studio</div>
                                            </div>
                                        </div>
                                        <Download size={14} className="text-slate-400 group-hover:text-teal-500" />
                                    </a>
                                )}

                                {track.musicxml_path && (
                                    <a
                                        href={track.musicxml_path.startsWith('http') ? track.musicxml_path : `${API_BASE_URL}${track.musicxml_path}`}
                                        download={`${track.title || 'track'}.musicxml`}
                                        className="p-3.5 rounded-xl bg-black/[0.03] dark:bg-white/5 hover:bg-teal-500/10 border border-black/[0.06] dark:border-white/5 flex items-center justify-between group transition-colors"
                                    >
                                        <div className="flex items-center gap-2.5">
                                            <FileCode size={18} className="text-cyan-500" />
                                            <div>
                                                <div className="text-xs font-bold text-slate-900 dark:text-white">W3C MusicXML (.musicxml)</div>
                                                <div className="text-[10px] text-slate-400 font-mono">MuseScore, Sibelius, Finale</div>
                                            </div>
                                        </div>
                                        <Download size={14} className="text-slate-400 group-hover:text-cyan-500" />
                                    </a>
                                )}

                                {notesData.length > 0 && (
                                    <button
                                        onClick={() => {
                                            const blob = new Blob([JSON.stringify(notesData, null, 2)], { type: 'application/json' });
                                            const url = URL.createObjectURL(blob);
                                            const a = document.createElement('a');
                                            a.href = url;
                                            a.download = `${track.title || 'track'}_notes.json`;
                                            a.click();
                                        }}
                                        className="p-3.5 rounded-xl bg-black/[0.03] dark:bg-white/5 hover:bg-teal-500/10 border border-black/[0.06] dark:border-white/5 flex items-center justify-between group transition-colors text-left"
                                    >
                                        <div className="flex items-center gap-2.5">
                                            <FileText size={18} className="text-amber-500" />
                                            <div>
                                                <div className="text-xs font-bold text-slate-900 dark:text-white">Notes Array JSON (.json)</div>
                                                <div className="text-[10px] text-slate-400 font-mono">Timestamps, pitch & velocities</div>
                                            </div>
                                        </div>
                                        <Download size={14} className="text-slate-400 group-hover:text-amber-500" />
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {/* 3. VOCAL & LYRICS TAB */}
                {activeTab === 'lyrics' && (
                    <div className="space-y-4 animate-fade-in">
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                            {/* Karaoke / Lyrics Viewer */}
                            <div className="lg:col-span-2 p-5 rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/10 shadow-apple-sm space-y-4">
                                <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-black/[0.04] dark:border-white/5">
                                    <div className="flex items-center gap-2">
                                        <Mic size={16} className="text-teal-500" />
                                        <h4 className="text-xs font-bold text-slate-900 dark:text-white uppercase tracking-wider">
                                            Synchronized Karaoke & Lyrics
                                        </h4>
                                        {timedLyrics.length > 0 && (
                                            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 font-semibold border border-teal-500/20">
                                                Acoustic Timed
                                            </span>
                                        )}
                                    </div>

                                    <div className="flex items-center gap-2">
                                        {/* View Mode Toggle */}
                                        {timedLyrics.length > 0 && (
                                            <div className="flex items-center bg-black/[0.04] dark:bg-white/5 p-0.5 rounded-lg text-[10px] font-bold">
                                                <button
                                                    onClick={() => setLyricsDisplayMode('karaoke')}
                                                    className={`px-2 py-1 rounded-md transition-all ${
                                                        lyricsDisplayMode === 'karaoke'
                                                            ? 'bg-white dark:bg-white/20 text-teal-700 dark:text-teal-300 shadow-sm'
                                                            : 'text-slate-400'
                                                    }`}
                                                >
                                                    Karaoke
                                                </button>
                                                <button
                                                    onClick={() => setLyricsDisplayMode('raw')}
                                                    className={`px-2 py-1 rounded-md transition-all ${
                                                        lyricsDisplayMode === 'raw'
                                                            ? 'bg-white dark:bg-white/20 text-teal-700 dark:text-teal-300 shadow-sm'
                                                            : 'text-slate-400'
                                                    }`}
                                                >
                                                    Text
                                                </button>
                                            </div>
                                        )}

                                        {/* Realign Acoustics Button */}
                                        <button
                                            onClick={handleRealignLyrics}
                                            disabled={isRealigningLyrics || !track.lyrics}
                                            className="px-2.5 py-1 bg-black/[0.04] dark:bg-white/5 hover:bg-teal-500/10 text-slate-600 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400 text-xs font-semibold rounded-xl flex items-center gap-1 transition-all disabled:opacity-40"
                                            title="Recompute acoustic forced alignment on vocal stem"
                                        >
                                            <RefreshCw size={12} className={isRealigningLyrics ? 'animate-spin text-teal-500' : ''} />
                                            <span>{isRealigningLyrics ? 'Aligning...' : 'Re-Align'}</span>
                                        </button>

                                        {/* Download LRC */}
                                        <a
                                            href={`${API_BASE_URL}/tracks/${track.id}/lrc`}
                                            download={`${track.title || 'lyrics'}.lrc`}
                                            className="px-2.5 py-1 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-600 dark:text-slate-300 text-xs font-semibold rounded-xl flex items-center gap-1 transition-all"
                                            title="Download Synchronized LRC File"
                                        >
                                            <FileText size={12} className="text-teal-500" />
                                            <span>.LRC</span>
                                        </a>

                                        {/* Copy Text */}
                                        <button
                                            onClick={handleCopyLyrics}
                                            className="px-2.5 py-1 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-600 dark:text-slate-300 text-xs font-semibold rounded-xl flex items-center gap-1 transition-all"
                                        >
                                            {copiedLyrics ? <Check size={12} className="text-teal-500" /> : <Copy size={12} />}
                                            <span>{copiedLyrics ? 'Copied' : 'Copy'}</span>
                                        </button>
                                    </div>
                                </div>

                                {timedLyrics.length > 0 && lyricsDisplayMode === 'karaoke' ? (
                                    <div className="space-y-3 max-h-80 overflow-y-auto pr-2 custom-scrollbar text-center py-2">
                                        {timedLyrics.map((line: any, idx: number) => {
                                            const isActive = idx === activeLineIndex;
                                            const isSection = line.is_section || (line.text.startsWith('[') && line.text.endsWith(']'));

                                            if (isSection) {
                                                return (
                                                    <div key={idx} className="py-2">
                                                        <span className="text-[11px] font-mono font-bold uppercase tracking-widest text-teal-600 dark:text-teal-400 bg-teal-500/10 px-3 py-1 rounded-full border border-teal-500/20">
                                                            {line.text}
                                                        </span>
                                                    </div>
                                                );
                                            }

                                            return (
                                                <div
                                                    key={idx}
                                                    onClick={() => engineSeek(line.start)}
                                                    className={`cursor-pointer transition-all duration-300 px-4 py-2 rounded-2xl ${
                                                        isActive
                                                            ? 'bg-teal-500/15 dark:bg-teal-500/20 text-teal-900 dark:text-teal-200 font-black text-base sm:text-lg scale-[1.01] shadow-apple-sm'
                                                            : 'text-slate-400 dark:text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 text-sm font-medium'
                                                    }`}
                                                >
                                                    {isActive && line.words && line.words.length > 0 ? (
                                                        <span className="inline-flex flex-wrap justify-center gap-1.5">
                                                            {line.words.map((w: any, wIdx: number) => {
                                                                const isWordSung = currentTime >= w.start;
                                                                return (
                                                                    <span
                                                                        key={wIdx}
                                                                        className={`transition-colors duration-150 ${
                                                                            isWordSung
                                                                                ? 'text-teal-700 dark:text-teal-300 font-black'
                                                                                : 'text-slate-400 dark:text-slate-500 opacity-60'
                                                                        }`}
                                                                    >
                                                                        {w.word}
                                                                    </span>
                                                                );
                                                            })}
                                                        </span>
                                                    ) : (
                                                        <span>{line.text}</span>
                                                    )}
                                                    {isActive && (
                                                        <span className="text-[10px] font-mono block opacity-60 mt-0.5">
                                                            {formatTime(line.start)}
                                                        </span>
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </div>
                                ) : track.lyrics ? (
                                    <div className="font-mono text-xs leading-loose whitespace-pre-wrap max-h-80 overflow-y-auto p-4 rounded-2xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/5">
                                        {track.lyrics}
                                    </div>
                                ) : (
                                    <div className="text-center py-12 text-slate-400 text-xs">
                                        Instrumental Track (No vocal lyrics generated)
                                    </div>
                                )}
                            </div>

                            {/* Vocal Conversion (SVC) Studio */}
                            <div className="p-5 rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/10 shadow-apple-sm space-y-4">
                                <h4 className="text-xs font-bold text-slate-900 dark:text-white uppercase tracking-wider flex items-center gap-2">
                                    <Zap size={14} className="text-amber-500" />
                                    <span>Singing Voice Conversion</span>
                                </h4>
                                <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                                    Re-voice the isolated vocal stem with an offline trained neural voice profile.
                                </p>

                                <div className="space-y-3">
                                    <div className="space-y-1">
                                        <label className="text-[10px] font-bold uppercase text-slate-400 block">
                                            Target Voice Profile
                                        </label>
                                        <select
                                            value={selectedVoiceProfile}
                                            onChange={(e) => setSelectedVoiceProfile(e.target.value)}
                                            className="apple-input py-2 px-3 text-xs font-mono w-full"
                                        >
                                            <option value="">Choose Voice Profile...</option>
                                            {voiceProfiles.map(p => (
                                                <option key={p.id} value={p.id}>👤 {p.name}</option>
                                            ))}
                                        </select>
                                    </div>

                                    {/* Pitch Shift Slider */}
                                    <div className="space-y-1">
                                        <div className="flex items-center justify-between text-[10px] font-mono">
                                            <span className="text-slate-400 font-bold uppercase">Pitch Shift</span>
                                            <span className="text-amber-500 font-bold">{pitchShift > 0 ? `+${pitchShift}` : pitchShift} semitones</span>
                                        </div>
                                        <input
                                            type="range"
                                            min="-12"
                                            max="12"
                                            step="1"
                                            value={pitchShift}
                                            onChange={(e) => setPitchShift(parseInt(e.target.value))}
                                            className="w-full accent-amber-500 cursor-pointer h-1.5 rounded-lg bg-black/10 dark:bg-white/10"
                                        />
                                    </div>

                                    {/* Dry / Wet Blend Slider */}
                                    <div className="space-y-1">
                                        <div className="flex items-center justify-between text-[10px] font-mono">
                                            <span className="text-slate-400 font-bold uppercase">Wet / Dry Blend</span>
                                            <span className="text-amber-500 font-bold">{dryWet}% Wet</span>
                                        </div>
                                        <input
                                            type="range"
                                            min="0"
                                            max="100"
                                            step="5"
                                            value={dryWet}
                                            onChange={(e) => setDryWet(parseInt(e.target.value))}
                                            className="w-full accent-amber-500 cursor-pointer h-1.5 rounded-lg bg-black/10 dark:bg-white/10"
                                        />
                                    </div>

                                    {/* Formant Preservation Switch */}
                                    <div className="flex items-center justify-between py-1">
                                        <span className="text-[10px] font-bold uppercase text-slate-400">Formant Preservation</span>
                                        <button
                                            type="button"
                                            onClick={() => setFormantPreserve(!formantPreserve)}
                                            className={`w-9 h-5 rounded-full transition-colors relative ${
                                                formantPreserve ? 'bg-amber-500' : 'bg-black/20 dark:bg-white/20'
                                            }`}
                                        >
                                            <div
                                                className={`w-3.5 h-3.5 rounded-full bg-white transition-transform absolute top-0.75 left-0.75 ${
                                                    formantPreserve ? 'translate-x-4' : 'translate-x-0'
                                                }`}
                                            />
                                        </button>
                                    </div>

                                    <button
                                        onClick={handleApplyVoiceConvert}
                                        disabled={!selectedVoiceProfile || isConvertingVoice}
                                        className="w-full py-2.5 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-1.5 shadow-md shadow-amber-500/20 disabled:opacity-50 transition-all"
                                    >
                                        <Zap size={13} />
                                        <span>{isConvertingVoice ? 'Synthesizing Voice...' : 'Apply Voice Conversion'}</span>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* 4. AI GENERATION PROVENANCE TAB */}
                {activeTab === 'provenance' && (
                    <div className="space-y-4 animate-fade-in">
                        <div className="p-5 rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/10 shadow-apple-sm space-y-4">
                            <div className="flex items-center justify-between pb-2 border-b border-black/[0.04] dark:border-white/5">
                                <h4 className="text-xs font-bold text-slate-900 dark:text-white uppercase tracking-wider flex items-center gap-2">
                                    <Cpu size={14} className="text-teal-500" />
                                    <span>Generation Hyperparameters & Provenance</span>
                                </h4>
                                <button
                                    onClick={handleCopySeed}
                                    className="text-[11px] font-mono font-bold text-teal-600 dark:text-teal-400 hover:underline flex items-center gap-1"
                                >
                                    {copiedSeed ? <Check size={12} /> : <Copy size={12} />}
                                    <span>Seed: {track.seed ?? 'Random'}</span>
                                </button>
                            </div>

                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                <div className="p-3 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/5">
                                    <span className="text-[10px] text-slate-400 font-mono block">Model Engine</span>
                                    <span className="text-xs font-bold text-slate-800 dark:text-slate-200 font-mono">{track.model_provider || 'minimax_music3'}</span>
                                </div>
                                <div className="p-3 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/5">
                                    <span className="text-[10px] text-slate-400 font-mono block">CFG Scale</span>
                                    <span className="text-xs font-bold text-teal-600 dark:text-teal-400 font-mono">{track.cfg_scale ?? 1.5}</span>
                                </div>
                                <div className="p-3 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/5">
                                    <span className="text-[10px] text-slate-400 font-mono block">Temperature</span>
                                    <span className="text-xs font-bold text-teal-600 dark:text-teal-400 font-mono">{track.temperature ?? 1.0}</span>
                                </div>
                                <div className="p-3 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/5">
                                    <span className="text-[10px] text-slate-400 font-mono block">Top-K</span>
                                    <span className="text-xs font-bold text-teal-600 dark:text-teal-400 font-mono">{track.topk ?? 50}</span>
                                </div>
                            </div>

                            {/* Structured Caption Breakdown */}
                            {structuredCaption && Object.keys(structuredCaption).length > 0 && (
                                <div className="space-y-2 pt-2">
                                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                                        Structured Caption Breakdown (MiniMax Format)
                                    </span>
                                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 font-mono text-[11px]">
                                        <div className="p-3 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/5">
                                            <div className="font-bold text-teal-600 dark:text-teal-400 mb-1">[Global Metadata]</div>
                                            <div className="text-slate-600 dark:text-slate-300 whitespace-pre-wrap">{structuredCaption.global_metadata || 'None'}</div>
                                        </div>
                                        <div className="p-3 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/5">
                                            <div className="font-bold text-cyan-600 dark:text-cyan-400 mb-1">[Vocal Details]</div>
                                            <div className="text-slate-600 dark:text-slate-300 whitespace-pre-wrap">{structuredCaption.vocal_details || 'None'}</div>
                                        </div>
                                        <div className="p-3 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/5">
                                            <div className="font-bold text-amber-600 dark:text-amber-400 mb-1">[Arrangement]</div>
                                            <div className="text-slate-600 dark:text-slate-300 whitespace-pre-wrap">{structuredCaption.arrangement || 'None'}</div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* 5. VERSION HISTORY & LINEAGE TAB */}
                {activeTab === 'lineage' && (
                    <div className="space-y-4 animate-fade-in">
                        <div className="p-5 rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/10 shadow-apple-sm space-y-4">
                            <h4 className="text-xs font-bold text-slate-900 dark:text-white uppercase tracking-wider flex items-center gap-2">
                                <GitFork size={14} className="text-teal-500" />
                                <span>Track Version Lineage & Derivatives</span>
                            </h4>

                            {parentTrack ? (
                                <div className="p-3.5 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/5 flex items-center justify-between">
                                    <div>
                                        <span className="text-[10px] font-mono text-teal-600 dark:text-teal-400 font-bold block">
                                            Parent Origin Track
                                        </span>
                                        <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                                            {parentTrack.title || parentTrack.prompt}
                                        </span>
                                    </div>
                                    {onSelectTrack && (
                                        <button
                                            onClick={() => onSelectTrack(parentTrack)}
                                            className="px-3 py-1 rounded-lg bg-black/5 dark:bg-white/10 hover:bg-teal-500 text-slate-700 dark:text-slate-200 hover:text-slate-950 text-xs font-bold transition-all"
                                        >
                                            View Parent
                                        </button>
                                    )}
                                </div>
                            ) : (
                                <div className="text-xs text-slate-400">
                                    This track is an original master (no prior parent track).
                                </div>
                            )}

                            {derivativeTracks.length > 0 && (
                                <div className="space-y-2 pt-2">
                                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                                        Derivative Iterations ({derivativeTracks.length})
                                    </span>
                                    <div className="space-y-1.5">
                                        {derivativeTracks.map(d => (
                                            <div
                                                key={d.id}
                                                className="p-3 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/5 flex items-center justify-between hover:border-teal-500/30 transition-colors"
                                            >
                                                <div className="truncate pr-2">
                                                    <span className="text-xs font-bold text-slate-800 dark:text-slate-200 truncate block">
                                                        {d.title || d.prompt}
                                                    </span>
                                                    <span className="text-[10px] font-mono text-slate-400">
                                                        {d.id.slice(0, 8)} • {d.voice_profile_id ? `SVC: ${d.voice_profile_id}` : 'Extended Outro'}
                                                    </span>
                                                </div>
                                                {onSelectTrack && (
                                                    <button
                                                        onClick={() => onSelectTrack(d)}
                                                        className="px-2.5 py-1 rounded-lg bg-teal-500/10 text-teal-700 dark:text-teal-300 hover:bg-teal-500 hover:text-slate-950 text-xs font-bold transition-all flex-shrink-0"
                                                    >
                                                        Inspect
                                                    </button>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};
