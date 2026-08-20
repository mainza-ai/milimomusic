import React, { useState, useRef, useEffect } from 'react';
import {
    Headphones,
    Sliders,
    Music,
    FileText,
    Download,
    X,
    Play,
    Pause,
    RotateCcw,
    RotateCw,
    Volume2,
    VolumeX,
    Layers,
    CheckCircle2,
    Mic2,
    Copy
} from 'lucide-react';
import { API_BASE_URL } from '../../api';
import type { Job, TimedLine, StemsMap, NoteEvent } from '../../api';
import { ArrangeTimeline } from './ArrangeTimeline';
import { PianoRoll } from './PianoRoll';
import { NotationViewer } from './NotationViewer';
import { MultitrackMixer } from './MultitrackMixer';

export type WorkspaceMode = 'listen' | 'arrange' | 'pianoroll' | 'notation' | 'mix' | 'lyrics';

export interface StemChannel {
    id: string;
    name: string;
    color: string;
    volume: number; // 0 - 100
    pan: number; // -50 to +50
    isMuted: boolean;
    isSolo: boolean;
    audioUrl?: string;
    /** General MIDI program (instrument) number, for per-instrument (MuScriptor) parts. */
    midiProgram?: number;
}

interface SessionWorkspaceProps {
    job: Job;
    onClose?: () => void;
}

export const SessionWorkspace: React.FC<SessionWorkspaceProps> = ({ job, onClose }) => {
    const [mode, setMode] = useState<WorkspaceMode>('listen');
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(job.duration_ms ? job.duration_ms / 1000 : 60);
    const [masterVolume, setMasterVolume] = useState(0.9);
    const [isMasterMuted, setIsMasterMuted] = useState(false);

    // Multitrack Stem Channels
    const parsedStems: StemsMap = job.stems_json
        ? typeof job.stems_json === 'string'
            ? JSON.parse(job.stems_json)
            : job.stems_json
        : {};

    // DAW playback channels — TWO genuine stem sources the user can switch between:
    //   1. Per-Instrument (MuScriptor) -> dynamic, one channel per distinct instrument
    //      from the transcription (DEFAULT — this is the true "dynamic stems" the DAW
    //      was built around; solo/mute isolates each real part).
    //   2. htDemucs -> real neural source-separation of the master into
    //      vocals/drums/bass/other (4 master stems).
    // Both are real, production data. A missing stem is omitted rather than shown as a
    // silent phantom channel. The user picks which source the DAW reflects.
    const realStemDefs: { id: string; name: string; audio?: string; volume: number; pan: number }[] = [
        { id: 'vocals', name: '🎤 Vocals', audio: parsedStems.vocals, volume: 85, pan: 0 },
        { id: 'drums', name: '🥁 Drums', audio: parsedStems.drums, volume: 90, pan: 0 },
        { id: 'bass', name: '🎸 Bass', audio: parsedStems.bass, volume: 88, pan: 0 },
        { id: 'other', name: '🎹 Instruments', audio: parsedStems.other, volume: 82, pan: -10 }
    ];

    const instrumentalParts: Record<string, string> = parsedStems.instrumental_parts || {};
    const instrumentPrograms: Record<string, number> = parsedStems.instrument_programs || {};
    const PART_COLORS = [
        'from-teal-500 to-cyan-500',
        'from-amber-500 to-orange-500',
        'from-pink-500 to-rose-500',
        'from-sky-500 to-blue-500',
        'from-violet-500 to-purple-500',
        'from-emerald-500 to-green-500',
        'from-rose-500 to-red-500',
        'from-indigo-500 to-blue-600'
    ];
    const partEntries = Object.entries(instrumentalParts);
    const partChannels: StemChannel[] = partEntries.map(([name, audio], i) => ({
        id: `part-${name}`,
        name,
        color: PART_COLORS[i % PART_COLORS.length],
        volume: 85,
        pan: 0,
        isMuted: false,
        isSolo: false,
        audioUrl: `${API_BASE_URL}${audio}`,
        midiProgram: instrumentPrograms[name]
    }));

    // DEFAULT to the dynamic per-instrument parts when present (that's the true
    // "dynamic stems" experience); fall back to the 4 master stems only when a
    // track has no per-instrument parts yet.
    const defaultSource: 'htdemucs' | 'muscriptor' =
        partEntries.length > 0 ? 'muscriptor' : 'htdemucs';

    const realChannels: StemChannel[] = realStemDefs
        .filter(def => def.audio)
        .map(def => ({
            id: def.id,
            name: def.name,
            color: def.id === 'vocals'
                ? 'from-pink-500 to-rose-500'
                : def.id === 'drums'
                ? 'from-amber-500 to-orange-500'
                : def.id === 'bass'
                ? 'from-cyan-500 to-blue-500'
                : 'from-teal-500 to-emerald-500',
            volume: def.volume,
            pan: def.pan,
            isMuted: false,
            isSolo: false,
            audioUrl: def.audio ? `${API_BASE_URL}${def.audio}` : undefined
        }));

    const [stemSource, setStemSource] = useState<'htdemucs' | 'muscriptor'>(defaultSource);
    // Active channels are whichever source is currently selected.
    const initialChannels: StemChannel[] =
        stemSource === 'muscriptor' && partChannels.length > 0 ? partChannels : realChannels;
    const [stemChannels, setStemChannels] = useState<StemChannel[]>(initialChannels);

    const switchStemSource = (source: 'htdemucs' | 'muscriptor') => {
        setStemSource(source);
        // Pause any currently-playing audio and drop stale refs from the other
        // source so only the newly-selected source can ever be heard.
        Object.keys(stemAudioRefs.current).forEach(id => {
            const el = stemAudioRefs.current[id];
            if (el) { try { el.pause(); } catch { /* ignore */ } }
            delete stemAudioRefs.current[id];
        });
        setLoadedStemIds({});
        // Rebuild the channel set for the new source and reset all mute/solo state.
        if (source === 'muscriptor' && partChannels.length > 0) {
            setStemChannels(partChannels.map(c => ({ ...c, isMuted: false, isSolo: false })));
        } else {
            setStemChannels(realChannels.map(c => ({ ...c, isMuted: false, isSolo: false })));
        }
        setIsPlaying(false);
    };

    const hasDualSources = partChannels.length > 0 && (parsedStems.vocals || parsedStems.drums || parsedStems.bass || parsedStems.other);

    // Master Audio Player (Master Mix / Clock)
    const masterAudioRef = useRef<HTMLAudioElement | null>(null);

    // Stem Audio Players for Multitrack Sync
    const stemAudioRefs = useRef<Record<string, HTMLAudioElement>>({});

    // Track which stem audio elements actually loaded. If a stem 404s / fails, we fall
    // back to the master mix instead of muting the master for a dead stem.
    const [loadedStemIds, setLoadedStemIds] = useState<Record<string, boolean>>({});
    const hasLoadedStems = Object.values(loadedStemIds).some(Boolean);
    const onStemLoaded = (id: string) => setLoadedStemIds(prev => ({ ...prev, [id]: true }));

    const timedLyrics: TimedLine[] = job.timed_lyrics_json
        ? typeof job.timed_lyrics_json === 'string'
            ? JSON.parse(job.timed_lyrics_json)
            : job.timed_lyrics_json
        : [];

    const notes: NoteEvent[] = job.notes_json
        ? typeof job.notes_json === 'string'
            ? JSON.parse(job.notes_json)
            : job.notes_json
        : [];

    // Sync volume & true isolated solo/mute across stem players
    useEffect(() => {
        const hasSolo = stemChannels.some(s => s.isSolo);

        stemChannels.forEach(stem => {
            const el = stemAudioRefs.current[stem.id];
            if (el) {
                let effectiveVolume = 0;
                if (isMasterMuted) {
                    effectiveVolume = 0;
                } else if (hasSolo) {
                    // Only soloed tracks that are not muted emit sound
                    effectiveVolume = stem.isSolo && !stem.isMuted ? (stem.volume / 100) * masterVolume : 0;
                } else {
                    // Muted tracks are completely silent (0)
                    effectiveVolume = !stem.isMuted ? (stem.volume / 100) * masterVolume : 0;
                }
                el.volume = Math.max(0, Math.min(1, effectiveVolume));
            }
        });

        // CRITICAL: In multitrack mode the master is muted so the stems are heard exclusively.
        // We only mute it when at least one stem actually LOADED; if every stem failed
        // (404/missing), fall back to the master mix so playback is never dead-silent.
        if (masterAudioRef.current) {
            if (hasLoadedStems) {
                masterAudioRef.current.volume = 0; // Stems carry 100% of the multitrack audio
            } else {
                masterAudioRef.current.volume = isMasterMuted ? 0 : masterVolume;
            }
        }
    }, [stemChannels, masterVolume, isMasterMuted, hasLoadedStems]);

    const togglePlay = () => {
        if (isPlaying) {
            pauseAll();
        } else {
            playAll();
        }
    };

    const playAll = () => {
        // Only play the ACTIVE source's channels (never stale refs from the other
        // stem source). The master acts purely as the transport clock.
        stemChannels.forEach(stem => {
            const el = stemAudioRefs.current[stem.id];
            if (el) {
                el.currentTime = masterAudioRef.current?.currentTime || currentTime;
                el.play().catch(console.error);
            }
        });
        if (masterAudioRef.current) {
            masterAudioRef.current.currentTime = stemChannels.length ? (stemAudioRefs.current[stemChannels[0].id]?.currentTime ?? currentTime) : currentTime;
            // Master is muted by the volume-sync effect when stems are active; play it
            // only so the clock/transport runs.
            masterAudioRef.current.play().catch(console.error);
        }
        setIsPlaying(true);
    };

    const pauseAll = () => {
        if (masterAudioRef.current) {
            masterAudioRef.current.pause();
        }
        // Pause ALL known refs (covers any cleanup gap) — safe to iterate the map.
        Object.keys(stemAudioRefs.current).forEach(id => {
            const el = stemAudioRefs.current[id];
            if (el) el.pause();
        });
        setIsPlaying(false);
    };

    const handleTimeUpdate = () => {
        if (masterAudioRef.current) {
            const time = masterAudioRef.current.currentTime;
            setCurrentTime(time);

            // Sync only the ACTIVE source's stem audio elements if drift exceeds 50ms.
            stemChannels.forEach(stem => {
                const el = stemAudioRefs.current[stem.id];
                if (el && Math.abs(el.currentTime - time) > 0.05) {
                    el.currentTime = time;
                }
            });
        }
    };

    const handleLoadedMetadata = () => {
        if (masterAudioRef.current && masterAudioRef.current.duration) {
            setDuration(masterAudioRef.current.duration);
        }
    };

    const handleSeek = (time: number) => {
        const clamped = Math.max(0, Math.min(duration, time));
        if (masterAudioRef.current) {
            masterAudioRef.current.currentTime = clamped;
        }
        Object.values(stemAudioRefs.current).forEach(el => {
            if (el) el.currentTime = clamped;
        });
        setCurrentTime(clamped);
    };

    const handleRewind = (seconds: number = 5) => {
        handleSeek(currentTime - seconds);
    };

    const handleAdvance = (seconds: number = 5) => {
        handleSeek(currentTime + seconds);
    };

    const handleChannelVolumeChange = (id: string, volume: number) => {
        setStemChannels(prev => prev.map(c => c.id === id ? { ...c, volume } : c));
    };

    const handleChannelPanChange = (id: string, pan: number) => {
        setStemChannels(prev => prev.map(c => c.id === id ? { ...c, pan } : c));
    };

    const handleChannelToggleMute = (id: string) => {
        setStemChannels(prev => prev.map(c => c.id === id ? { ...c, isMuted: !c.isMuted } : c));
    };

    const handleChannelToggleSolo = (id: string) => {
        setStemChannels(prev => prev.map(c => c.id === id ? { ...c, isSolo: !c.isSolo } : c));
    };

    const handleExport = (format: string) => {
        window.open(`${API_BASE_URL}/transcribe/export/${job.id}/${format}`, '_blank');
    };

    const formatTime = (seconds: number = 0) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div className="flex flex-col h-full bg-[#fbfbfd] dark:bg-[#090b10] text-slate-900 dark:text-slate-100 overflow-hidden select-none transition-colors duration-200">
            {/* Master Audio Element (Used as Clock / Transport Master) */}
            {job.audio_path && (
                <audio
                    ref={masterAudioRef}
                    src={job.audio_path.startsWith('http') ? job.audio_path : `${API_BASE_URL}${job.audio_path}`}
                    onTimeUpdate={handleTimeUpdate}
                    onLoadedMetadata={handleLoadedMetadata}
                    onEnded={() => setIsPlaying(false)}
                />
            )}

            {/* Individual Stem Audio Elements for Isolated Multitrack Playback */}
            {stemChannels.map(stem => (
                stem.audioUrl ? (
                    <audio
                        key={stem.id}
                        ref={el => {
                            if (el) {
                                stemAudioRefs.current[stem.id] = el;
                            } else {
                                // Element unmounted (e.g. source switch) — drop the
                                // stale ref so it never plays alongside the active set.
                                delete stemAudioRefs.current[stem.id];
                            }
                        }}
                        src={stem.audioUrl}
                        onLoadedMetadata={() => onStemLoaded(stem.id)}
                        onError={() => setLoadedStemIds(prev => ({ ...prev, [stem.id]: false }))}
                    />
                ) : null
            ))}

            {/* Top Workspace Header Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 px-4 sm:px-6 py-3 border-b border-black/[0.06] dark:border-white/[0.08] bg-white/70 dark:bg-[#12141c]/80 backdrop-blur-2xl flex-shrink-0 z-20 shadow-apple-sm">
                <div className="flex items-center space-x-3 min-w-0">
                    <div className="w-8 h-8 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 text-teal-700 dark:text-teal-300 border border-teal-500/20 flex items-center justify-center font-bold text-xs p-1 flex-shrink-0">
                        <img src="/milimo_logo.png" alt="Logo" className="w-full h-full object-cover rounded-lg" onError={(e) => {
                            (e.target as HTMLElement).style.display = 'none';
                        }} />
                    </div>
                    <div className="min-w-0">
                        <div className="flex items-center space-x-2 truncate">
                            <h1 className="text-xs sm:text-sm font-bold text-slate-900 dark:text-slate-100 truncate">
                                {job.title || job.prompt || "Producer Session Specimen"}
                            </h1>
                            <span className="hidden sm:inline text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20 font-semibold flex-shrink-0">
                                48kHz Stereo FLAC
                            </span>
                        </div>
                        <p className="text-[10px] text-slate-500 dark:text-slate-400 font-mono truncate max-w-xs sm:max-w-md">
                            {hasLoadedStems
                                ? `${stemChannels.length} ${stemSource === 'muscriptor' ? 'Dynamic Instruments (MuScriptor)' : 'Master Stems (HTDemucs)'} · Active`
                                : "Master Audio"} · {notes.length} MIDI Notes
                        </p>
                    </div>
                </div>

                {/* Stem Source Selector — user chooses between the real neural
                    separation (vocals/drums/bass/other) and the MuScriptor
                    per-instrument parts. Both are genuine data. */}
                {hasDualSources && (
                    <div className="flex items-center bg-black/[0.04] dark:bg-[#181a24] p-1 rounded-xl border border-black/[0.06] dark:border-white/10 space-x-1 shadow-sm">
                        <button
                            onClick={() => switchStemSource('muscriptor')}
                            title="Use the dynamic per-instrument parts (one channel per instrument MuScriptor detected)"
                            aria-label="Dynamic Instruments"
                            className={`px-2.5 py-1.5 rounded-lg text-[10px] sm:text-xs font-semibold transition-all whitespace-nowrap ${
                                stemSource === 'muscriptor'
                                    ? 'bg-white dark:bg-white/20 text-teal-700 dark:text-teal-300 font-bold shadow-apple-sm'
                                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-black/[0.03] dark:hover:bg-white/5'
                            }`}
                        >
                            {partChannels.length} Instruments
                        </button>
                        <button
                            onClick={() => switchStemSource('htdemucs')}
                            title="Use the neural source-separated master group (Vocals / Drums / Bass / Other)"
                            aria-label="Master Source Group"
                            className={`px-2.5 py-1.5 rounded-lg text-[10px] sm:text-xs font-semibold transition-all whitespace-nowrap ${
                                stemSource === 'htdemucs'
                                    ? 'bg-white dark:bg-white/20 text-teal-700 dark:text-teal-300 font-bold shadow-apple-sm'
                                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-black/[0.03] dark:hover:bg-white/5'
                            }`}
                        >
                            Vocals / Drums / Bass / Other
                        </button>
                    </div>
                )}

                {/* Apple Segmented Mode Switcher */}
                <div className="flex items-center bg-black/[0.04] dark:bg-[#181a24] p-1 rounded-xl border border-black/[0.06] dark:border-white/10 space-x-1 shadow-sm overflow-x-auto max-w-full">
                    {[
                        { id: 'listen', label: 'Listen', icon: Headphones },
                        { id: 'arrange', label: 'Arrange', icon: Layers },
                        { id: 'pianoroll', label: 'Piano Roll', icon: Music },
                        { id: 'notation', label: 'Notation', icon: FileText },
                        { id: 'mix', label: 'Mix', icon: Sliders },
                        { id: 'lyrics', label: 'Lyrics', icon: Mic2 }
                    ].map((tab) => {
                        const Icon = tab.icon;
                        const isActive = mode === tab.id;
                        return (
                            <button
                                key={tab.id}
                                onClick={() => setMode(tab.id as WorkspaceMode)}
                                title={`Switch to ${tab.label} Mode`}
                                aria-label={`Switch to ${tab.label} Mode`}
                                className={`flex items-center space-x-1.5 px-2.5 sm:px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all whitespace-nowrap ${
                                    isActive
                                        ? 'bg-white dark:bg-white/20 text-teal-700 dark:text-teal-300 font-bold shadow-apple-sm'
                                        : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-black/[0.03] dark:hover:bg-white/5'
                                }`}
                            >
                                <Icon size={14} className="flex-shrink-0" />
                                <span className="hidden xs:inline">{tab.label}</span>
                            </button>
                        );
                    })}
                </div>

                {/* Right Actions & Export Dropdown */}
                <div className="flex items-center space-x-2 flex-shrink-0">
                    <div className="relative group">
                        <button
                            title="Export Multi-Track MIDI, MusicXML Score, or Timed Lyrics"
                            aria-label="Export DAW Assets"
                            className="px-3 sm:px-3.5 py-1.5 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-all shadow-md shadow-teal-500/20 active:scale-95"
                        >
                            <Download size={13} />
                            <span className="hidden sm:inline">Export DAW Assets</span>
                            <span className="sm:hidden">Export</span>
                        </button>
                        <div className="absolute right-0 mt-1.5 w-52 bg-white dark:bg-[#181a24] border border-black/[0.08] dark:border-white/10 rounded-2xl shadow-apple-lg p-1.5 hidden group-hover:block z-50 animate-fade-in">
                            <button
                                onClick={() => handleExport('midi')}
                                title="Download .mid MIDI file"
                                className="w-full text-left px-3 py-2 text-xs font-semibold rounded-xl hover:bg-teal-500/10 text-slate-800 dark:text-slate-200 hover:text-teal-600 dark:hover:text-teal-300 flex items-center justify-between"
                            >
                                <span>Multi-Track MIDI (.mid)</span>
                                <Music size={12} className="text-teal-500" />
                            </button>
                            <button
                                onClick={() => handleExport('musicxml')}
                                title="Download W3C MusicXML Sheet Music Score"
                                className="w-full text-left px-3 py-2 text-xs font-semibold rounded-xl hover:bg-teal-500/10 text-slate-800 dark:text-slate-200 hover:text-teal-600 dark:hover:text-teal-300 flex items-center justify-between"
                            >
                                <span>MusicXML Sheet (.musicxml)</span>
                                <FileText size={12} className="text-cyan-500" />
                            </button>
                            <button
                                onClick={() => handleExport('lrc')}
                                title="Download Karaoke Synchronized Lyrics"
                                className="w-full text-left px-3 py-2 text-xs font-semibold rounded-xl hover:bg-teal-500/10 text-slate-800 dark:text-slate-200 hover:text-teal-600 dark:hover:text-teal-300 flex items-center justify-between"
                            >
                                <span>Timed Lyrics (.lrc)</span>
                                <CheckCircle2 size={12} className="text-amber-500" />
                            </button>
                        </div>
                    </div>

                    {onClose && (
                        <button
                            onClick={onClose}
                            title="Close Workspace (Return to Overview)"
                            aria-label="Close Workspace"
                            className="p-1.5 rounded-xl text-slate-500 hover:text-slate-900 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
                        >
                            <X size={16} />
                        </button>
                    )}
                </div>
            </div>

            {/* Mode Canvas Content */}
            <div className="flex-1 overflow-hidden relative">
                {mode === 'listen' && (
                    <div className="h-full flex flex-col items-center justify-center p-8 space-y-6">
                        {/* Artwork with Apple App Icon styling */}
                        <div className="relative group">
                            <div className="w-52 h-52 rounded-3xl bg-gradient-to-tr from-teal-500/20 via-cyan-500/20 to-sky-500/20 border border-black/[0.08] dark:border-white/10 shadow-apple-lg flex items-center justify-center p-6 backdrop-blur-xl">
                                <img src="/milimo_logo.png" alt="Artwork" className="w-full h-full object-contain filter drop-shadow-md" />
                            </div>
                        </div>

                        {/* Title & Info */}
                        <div className="text-center max-w-lg space-y-1">
                            <h2 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white font-sans">
                                {job.title || "Untitled Master Composition"}
                            </h2>
                            <p className="text-xs text-slate-500 dark:text-slate-400 font-mono">
                                {job.prompt}
                            </p>
                            <div className="flex items-center justify-center gap-2 pt-2">
                                <span className="text-[11px] font-mono font-bold text-teal-600 dark:text-teal-400 bg-teal-500/10 px-2.5 py-1 rounded-full border border-teal-500/20">
                                    48kHz Stereo FLAC
                                </span>
                                <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400 bg-black/5 dark:bg-white/5 px-2.5 py-1 rounded-full">
                                    {notes.length} MIDI Notes Transcribed
                                </span>
                            </div>
                        </div>

                        {/* Synchronized Karaoke Lyrics Stream */}
                        {timedLyrics.length > 0 && (
                            <div className="h-24 w-full max-w-xl bg-white/70 dark:bg-[#12141c]/70 border border-black/[0.06] dark:border-white/10 rounded-2xl p-4 flex flex-col items-center justify-center text-center space-y-1 overflow-hidden shadow-apple-sm backdrop-blur-xl">
                                {timedLyrics.map((line, idx) => {
                                    const isCurrent = currentTime >= line.start && currentTime <= line.end;
                                    return (
                                        <p
                                            key={idx}
                                            className={`text-xs transition-all duration-300 font-mono ${
                                                isCurrent
                                                    ? 'text-teal-600 dark:text-teal-300 font-bold text-sm scale-105'
                                                    : 'text-slate-400 dark:text-slate-500 opacity-60'
                                            }`}
                                        >
                                            {line.text}
                                        </p>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                )}

                {mode === 'arrange' && (
                    <ArrangeTimeline
                        job={job}
                        stemChannels={stemChannels}
                        currentTime={currentTime}
                        duration={duration}
                        onSeek={handleSeek}
                        onToggleMute={handleChannelToggleMute}
                        onToggleSolo={handleChannelToggleSolo}
                    />
                )}

                {mode === 'pianoroll' && (
                    <PianoRoll
                        job={job}
                        currentTime={currentTime}
                        duration={duration}
                        onSeek={handleSeek}
                        isPlaying={isPlaying}
                    />
                )}

                {mode === 'notation' && (
                    <NotationViewer
                        job={job}
                        currentTime={currentTime}
                        onSeek={handleSeek}
                    />
                )}

                {mode === 'mix' && (
                    <MultitrackMixer
                        job={job}
                        stemChannels={stemChannels}
                        onVolumeChange={handleChannelVolumeChange}
                        onPanChange={handleChannelPanChange}
                        onToggleMute={handleChannelToggleMute}
                        onToggleSolo={handleChannelToggleSolo}
                        masterVolume={masterVolume}
                        onMasterVolumeChange={setMasterVolume}
                        isPlaying={isPlaying}
                    />
                )}

                {mode === 'lyrics' && (
                    <div className="h-full flex flex-col p-6 max-w-4xl mx-auto overflow-hidden animate-fade-in">
                        {/* Header */}
                        <div className="flex items-center justify-between border-b border-black/[0.06] dark:border-white/[0.08] pb-4 mb-4 flex-shrink-0">
                            <div>
                                <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                                    <Mic2 size={18} className="text-teal-500" />
                                    <span>{job.title || job.prompt || "Song Lyrics"}</span>
                                </h2>
                                <p className="text-xs text-slate-500 dark:text-slate-400 font-mono mt-0.5">
                                    {timedLyrics.length > 0 ? "✨ Real-Time Synchronized Lyrics & Karaoke" : "Full Structured Lyrics"}
                                </p>
                            </div>

                            {job.lyrics && (
                                <button
                                    onClick={() => {
                                        navigator.clipboard.writeText(job.lyrics || '');
                                        alert("Lyrics copied to clipboard!");
                                    }}
                                    className="px-3 py-1.5 rounded-xl bg-teal-500/10 hover:bg-teal-500/20 text-teal-700 dark:text-teal-300 border border-teal-500/20 text-xs font-semibold flex items-center gap-1.5 transition-colors"
                                >
                                    <Copy size={13} />
                                    <span>Copy Text</span>
                                </button>
                            )}
                        </div>

                        {/* Lyrics Container */}
                        <div className="flex-1 overflow-y-auto pr-2 space-y-4 font-sans select-text">
                            {timedLyrics.length > 0 ? (
                                timedLyrics.map((line, idx) => {
                                    const isCurrent = currentTime >= line.start && currentTime <= line.end;
                                    const isPast = currentTime > line.end;
                                    const isSection = line.text.startsWith('[') && line.text.endsWith(']');

                                    if (isSection) {
                                        return (
                                            <div key={idx} className="pt-4 pb-1">
                                                <span className="text-xs font-mono font-bold uppercase tracking-widest text-teal-600 dark:text-teal-400 bg-teal-500/10 px-3 py-1 rounded-full border border-teal-500/20">
                                                    {line.text}
                                                </span>
                                            </div>
                                        );
                                    }

                                    return (
                                        <div
                                            key={idx}
                                            onClick={() => handleSeek(line.start)}
                                            className={`cursor-pointer transition-all duration-300 rounded-2xl px-4 py-2.5 ${
                                                isCurrent
                                                    ? 'bg-teal-500/15 dark:bg-teal-500/20 text-teal-900 dark:text-teal-200 font-extrabold text-lg sm:text-xl scale-[1.01] shadow-apple-sm'
                                                    : isPast
                                                    ? 'text-slate-500 dark:text-slate-400 font-medium text-base hover:text-teal-600 dark:hover:text-teal-400'
                                                    : 'text-slate-400 dark:text-slate-500 font-normal text-base hover:text-slate-800 dark:hover:text-slate-200'
                                            }`}
                                        >
                                            <div className="flex items-center justify-between gap-4">
                                                <span>{line.text}</span>
                                                <span className="text-xs font-mono text-slate-400 opacity-60">
                                                    {formatTime(line.start)}
                                                </span>
                                            </div>
                                        </div>
                                    );
                                })
                            ) : job.lyrics ? (
                                <pre className="text-sm font-sans leading-relaxed text-slate-800 dark:text-slate-200 whitespace-pre-wrap">
                                    {job.lyrics}
                                </pre>
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center text-center p-12 text-slate-400 space-y-3">
                                    <Mic2 size={32} className="opacity-40" />
                                    <p className="text-sm">No lyrics found for this session track.</p>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {/* Bottom Transport Control Bar */}
            <div className="px-6 py-3 border-t border-black/[0.06] dark:border-white/[0.08] bg-white/80 dark:bg-[#12141c]/90 backdrop-blur-2xl flex flex-wrap items-center justify-between gap-4 z-20 shadow-apple-md">
                {/* Transport Buttons */}
                <div className="flex items-center space-x-2">
                    <button
                        onClick={() => handleRewind(5)}
                        className="p-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 transition-transform active:scale-95"
                        title="Rewind 5s"
                    >
                        <RotateCcw size={16} />
                    </button>

                    <button
                        onClick={togglePlay}
                        className="w-10 h-10 rounded-2xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold flex items-center justify-center shadow-apple-md active:scale-95 transition-transform"
                        title={isPlaying ? "Pause" : "Play"}
                    >
                        {isPlaying ? <Pause size={18} /> : <Play size={18} className="ml-0.5" />}
                    </button>

                    <button
                        onClick={() => handleAdvance(5)}
                        className="p-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 transition-transform active:scale-95"
                        title="Advance 5s"
                    >
                        <RotateCw size={16} />
                    </button>

                    {/* Timecode */}
                    <div className="flex items-center space-x-1 font-mono text-xs text-slate-700 dark:text-slate-300 ml-2">
                        <span className="font-bold">{formatTime(currentTime)}</span>
                        <span className="text-slate-400">/</span>
                        <span className="text-slate-500">{formatTime(duration)}</span>
                    </div>
                </div>

                {/* Timeline Scrubber */}
                <div className="flex-1 max-w-xl mx-auto flex items-center space-x-3">
                    <input
                        type="range"
                        min="0"
                        max={duration || 100}
                        step="0.1"
                        value={currentTime}
                        onChange={(e) => handleSeek(parseFloat(e.target.value))}
                        title={`Seek playhead: ${formatTime(currentTime)} / ${formatTime(duration)}`}
                        aria-label="Timeline Scrubber"
                        className="w-full h-2 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-teal-500"
                    />
                </div>

                {/* Master Volume */}
                <div className="flex items-center space-x-2">
                    <button
                        onClick={() => setIsMasterMuted(!isMasterMuted)}
                        title={isMasterMuted ? "Unmute Master Volume" : "Mute Master Volume"}
                        aria-label={isMasterMuted ? "Unmute Master Volume" : "Mute Master Volume"}
                        className="text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
                    >
                        {isMasterMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
                    </button>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={isMasterMuted ? 0 : masterVolume}
                        onChange={(e) => {
                            setIsMasterMuted(false);
                            setMasterVolume(parseFloat(e.target.value));
                        }}
                        title={`Master Volume: ${Math.round(masterVolume * 100)}%`}
                        aria-label="Master Volume Slider"
                        className="w-24 h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-teal-500"
                    />
                    <span className="text-[10px] font-mono text-slate-400 w-8">
                        {isMasterMuted ? '0%' : `${Math.round(masterVolume * 100)}%`}
                    </span>
                </div>
            </div>
        </div>
    );
};
