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
    Copy,
    SkipBack,
    Repeat
} from 'lucide-react';
import { API_BASE_URL, getStemMeta } from '../../api';
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
    const [isLooping, setIsLooping] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(job.duration_ms ? job.duration_ms / 1000 : 60);
    const [masterVolume, setMasterVolume] = useState(0.9);
    const [isMasterMuted, setIsMasterMuted] = useState(false);

    // Beat Grid & Meter
    const beatGrid = job.beat_grid_json
        ? typeof job.beat_grid_json === 'string'
            ? JSON.parse(job.beat_grid_json)
            : job.beat_grid_json
        : {};
    const bpm = beatGrid.bpm || 120;
    const barDuration = (60 / bpm) * (beatGrid.beats_per_bar || 4);

    // Multitrack Stem Channels
    const parsedStems: StemsMap = job.stems_json
        ? typeof job.stems_json === 'string'
            ? JSON.parse(job.stems_json)
            : job.stems_json
        : {};

    // DAW playback channels — TWO genuine stem sources the user can switch between:
    //   1. Per-Instrument (MuScriptor) -> dynamic, one channel per distinct instrument
    //      from the transcription (DEFAULT).
    //   2. Neural Source Separation -> real neural source-separation of the master into
    //      dynamic stems (Vocals, Drums, Bass, Guitar, Piano, Other, etc.).
    const reservedStemKeys = new Set([
        'instrumental_parts',
        'instrument_programs',
        'stems_source',
        'sources_available',
        'default_source'
    ]);

    const realChannels: StemChannel[] = Object.entries(parsedStems)
        .filter(([k, v]) => !reservedStemKeys.has(k) && typeof v === 'string' && !!v)
        .map(([k, audio]) => {
            const meta = getStemMeta(k);
            return {
                id: k,
                name: `${meta.icon} ${meta.label}`,
                color: meta.gradient,
                volume: k === 'drums' ? 90 : k === 'bass' ? 88 : 85,
                pan: 0,
                isMuted: false,
                isSolo: false,
                audioUrl: `${API_BASE_URL}${audio}`
            };
        });

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

    // DEFAULT to dynamic per-instrument parts when present, else neural stems
    const defaultSource: 'neural' | 'muscriptor' =
        partEntries.length > 0 ? 'muscriptor' : 'neural';

    const [stemSource, setStemSource] = useState<'neural' | 'muscriptor'>(defaultSource);
    // Active channels are whichever source is currently selected.
    const initialChannels: StemChannel[] =
        stemSource === 'muscriptor' && partChannels.length > 0 ? partChannels : realChannels;
    const [stemChannels, setStemChannels] = useState<StemChannel[]>(initialChannels);

    const switchStemSource = (source: 'neural' | 'muscriptor') => {
        setStemSource(source);
        // Stop any in-flight playback and tear down the old source's graph nodes
        // so only the newly-selected source can ever be heard.
        stopSources();
        isPlayingRef.current = false;
        setIsPlaying(false);
        Object.keys(stemGainRefs.current).forEach(id => {
            try { stemGainRefs.current[id].disconnect(); } catch { /* ignore */ }
            delete stemGainRefs.current[id];
        });
        Object.keys(stemPanRefs.current).forEach(id => {
            try { stemPanRefs.current[id].disconnect(); } catch { /* ignore */ }
            delete stemPanRefs.current[id];
        });
        // Drop the decoded stem buffers of the deactivated source and let the
        // [stemSource] effect re-decode the newly-selected channels.
        Object.keys(bufCacheRef.current).forEach(id => {
            if (id !== '__master__') delete bufCacheRef.current[id];
        });
        decodeStartedRef.current = false;
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

    // ── Production-grade Web Audio multitrack TRANSPORT ─────────────────────
    // Unlike mixed <audio> elements (each with its own independent clock, which
    // drift and need glitch-prone seeks to resync), every stem and the master are
    // decoded into AudioBuffers and played through AudioBufferSourceNodes, all
    // scheduled against the SINGLE AudioContext.currentTime master clock. Every
    // source is started with the same (when, offset) so the whole transport is
    // sample-accurate and can NEVER drift — there is no seek-correction loop at
    // all. Mixing is done on the audio thread via gain/panner nodes (no DOM
    // writes → no clicks). Reading `currentTime` of the decode/scheduler is O(1).
    const audioCtxRef = useRef<AudioContext | null>(null);
    const masterGainRef = useRef<GainNode | null>(null);        // global fader (master volume + mute)
    const masterMixGainRef = useRef<GainNode | null>(null);     // master-mix channel (0 when stems active)
    const stemGainRefs = useRef<Record<string, GainNode>>({});
    const stemPanRefs = useRef<Record<string, StereoPannerNode>>({});
    const bufCacheRef = useRef<Record<string, AudioBuffer>>({}); // decoded buffers by id/url
    const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
    const isPlayingRef = useRef(false);
    const isLoopingRef = useRef(false);
    const currentTimeRef = useRef(0);
    const durationRef = useRef(duration);
    const playStartClockRef = useRef(0);
    const playStartPosRef = useRef(0);
    const rafRef = useRef(0);
    const preparedIdsRef = useRef<Set<string>>(new Set()); // buffers that have a URL to load
    const decodeStartedRef = useRef(false);   // avoid double decode under StrictMode
    const UI_TICK_MS = 80;                    // min gap between playhead re-renders (~12Hz)
    const lastUiTickRef = useRef(0);

    // Track which stems actually decoded/loaded. If a stem 404s / fails to decode,
    // we fall back to the master mix instead of running a dead-stem multitrack.
    const [loadedStemIds, setLoadedStemIds] = useState<Record<string, boolean>>({});
    const hasLoadedStems = Object.values(loadedStemIds).some(Boolean);

    const timedLyrics: TimedLine[] = job.timed_lyrics_json
        ? typeof job.timed_lyrics_json === 'string'
            ? JSON.parse(job.timed_lyrics_json)
            : job.timed_lyrics_json
        : [];

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

    const notes: NoteEvent[] = job.notes_json
        ? typeof job.notes_json === 'string'
            ? JSON.parse(job.notes_json)
            : job.notes_json
        : [];

    const ensureAudioContext = (): AudioContext | null => {
        if (!audioCtxRef.current) {
            const Ctor: typeof AudioContext =
                window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
            if (!Ctor) return null;
            audioCtxRef.current = new Ctor();
        }
        buildGraph();
        return audioCtxRef.current;
    };

    // Build (once) the routing graph: per-channel Gain→Panner into a master gain.
    const buildGraph = () => {
        const ctx = audioCtxRef.current;
        if (!ctx) return;
        if (!masterGainRef.current) {
            const master = ctx.createGain();
            master.gain.value = 1;
            master.connect(ctx.destination);
            masterGainRef.current = master;
        }
        if (!masterMixGainRef.current) {
            const g = ctx.createGain();
            g.gain.value = 1;
            g.connect(masterGainRef.current);
            masterMixGainRef.current = g;
        }
    };

    // Compute + apply the full mix state to all nodes (smooth ramps → no zipper).
    const applyMixParams = () => {
        const ctx = audioCtxRef.current;
        if (!ctx || !masterGainRef.current) return;
        const hasSolo = stemChannels.some(s => s.isSolo);
        stemChannels.forEach(stem => {
            const gain = stemGainRefs.current[stem.id];
            const pan = stemPanRefs.current[stem.id];
            if (!gain) return;
            let perStem = 0;
            if (hasSolo) perStem = stem.isSolo && !stem.isMuted ? stem.volume / 100 : 0;
            else perStem = !stem.isMuted ? stem.volume / 100 : 0;
            gain.gain.setTargetAtTime(Math.max(0, Math.min(1, perStem)), ctx.currentTime, 0.015);
            if (pan) pan.pan.setTargetAtTime(Math.max(-1, Math.min(1, stem.pan / 50)), ctx.currentTime, 0.015);
        });
        // Global fader + master-mix fallback.
        masterGainRef.current.gain.setTargetAtTime(
            isMasterMuted ? 0 : Math.max(0, Math.min(1, masterVolume)), ctx.currentTime, 0.015
        );
        if (masterMixGainRef.current) {
            masterMixGainRef.current.gain.setTargetAtTime(hasLoadedStems ? 0 : 1, ctx.currentTime, 0.015);
        }
    };

    // Ensure a per-stem routing (gain + panner) exists for the given channel.
    const ensureStemNodes = (id: string): GainNode | null => {
        const ctx = audioCtxRef.current;
        if (!ctx) { ensureAudioContext(); }
        if (!audioCtxRef.current || !masterGainRef.current) return null;
        if (!stemGainRefs.current[id]) {
            const gain = audioCtxRef.current.createGain();
            const pan = audioCtxRef.current.createStereoPanner();
            gain.connect(pan);
            pan.connect(masterGainRef.current);
            stemGainRefs.current[id] = gain;
            stemPanRefs.current[id] = pan;
        }
        return stemGainRefs.current[id];
    };

    // ── Decoding ────────────────────────────────────────────────────────────
    const getMasterUrl = (): string | null => job.audio_path
        ? (job.audio_path.startsWith('http') ? job.audio_path : `${API_BASE_URL}${job.audio_path}`)
        : null;

    const fetchAudio = (url: string): Promise<ArrayBuffer> =>
        fetch(url).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.arrayBuffer(); });

    // Decode one track to an AudioBuffer and cache it. Resolves null on failure.
    const decodeToBuffer = async (key: string, url: string): Promise<AudioBuffer | null> => {
        const ctx = audioCtxRef.current || ensureAudioContext();
        if (!ctx) return null;
        if (bufCacheRef.current[key]) return bufCacheRef.current[key];
        try {
            const data = await fetchAudio(url);
            const buf = await ctx.decodeAudioData(data);
            bufCacheRef.current[key] = buf;
            return buf;
        } catch (e) {
            console.warn('Failed to decode audio:', key, e);
            return null;
        }
    };

    // Pre-decode the master + all active stems so first play is instant. Runs once
    // (guarded) and records which ids succeeded / which have URLs to prepare.
    const prepareBuffers = async () => {
        const ctx = audioCtxRef.current || ensureAudioContext();
        if (!ctx) return;
        const jobs: { id: string; url: string }[] = [];
        if (job.audio_path) {
            const url = job.audio_path.startsWith('http') ? job.audio_path : `${API_BASE_URL}${job.audio_path}`;
            jobs.push({ id: '__master__', url });
        }
        stemChannels.forEach(stem => {
            if (stem.audioUrl) {
                jobs.push({ id: stem.id, url: stem.audioUrl });
                preparedIdsRef.current.add(stem.id);
            }
        });
        const results = await Promise.all(jobs.map(j => decodeToBuffer(j.id, j.url).then(b => ({ id: j.id, b }))));
        // Track which stems loaded so the UI shows multitrack vs master-fallback.
        const loaded: Record<string, boolean> = {};
        results.forEach(r => { if (r.b) loaded[r.id] = true; });
        const masterLen = bufCacheRef.current['__master__']?.duration;
        if (masterLen) setDuration(masterLen);
        setLoadedStemIds(() => {
            // Only clear ids that are no longer prepared; keep failure markers.
            const next: Record<string, boolean> = {};
            preparedIdsRef.current.forEach(id => { next[id] = !!loaded[id]; });
            return next;
        });
    };

    // The current transport position, derived from the master clock (sample-accurate).
    const getPosition = (): number => {
        const ctx = audioCtxRef.current;
        if (!ctx || !isPlayingRef.current) return currentTimeRef.current;
        const elapsed = ctx.currentTime - playStartClockRef.current;
        return Math.max(0, playStartPosRef.current + elapsed);
    };

    // Stop + drop every currently scheduled source.
    const stopSources = () => {
        activeSourcesRef.current.forEach(src => {
            try { src.stop(); } catch { /* already stopped */ }
            try { src.disconnect(); } catch { /* ignore */ }
        });
        activeSourcesRef.current.clear();
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
        rafRef.current = 0;
    };

    // Schedule the master + every loaded stem at the current transport position.
    // ALL sources are started with the same (when, offset), so they are sample-locked.
    const scheduleAll = async () => {
        const ctx = audioCtxRef.current || ensureAudioContext();
        if (!ctx || !masterGainRef.current) return;
        stopSources();
        const startAt = ctx.currentTime;
        const pos = Math.max(0, Math.min(durationRef.current, currentTimeRef.current));

        const masterBuf = bufCacheRef.current['__master__'];
        if (masterBuf && pos < masterBuf.duration) {
            const src = ctx.createBufferSource();
            src.buffer = masterBuf;
            src.connect(masterMixGainRef.current!);
            const offset = Math.min(Math.max(pos, 0), Math.max(0, masterBuf.duration - 0.02));
            src.start(startAt, offset, Math.max(0, masterBuf.duration - offset));
            activeSourcesRef.current.add(src);
        }
        stemChannels.forEach(stem => {
            const buf = bufCacheRef.current[stem.id];
            if (!buf || pos >= buf.duration) return;
            const gain = ensureStemNodes(stem.id);
            if (!gain) return;
            const src = ctx.createBufferSource();
            src.buffer = buf;
            src.connect(gain);
            const offset = Math.min(Math.max(pos, 0), Math.max(0, buf.duration - 0.02));
            src.start(startAt, offset, Math.max(0, buf.duration - offset));
            activeSourcesRef.current.add(src);
        });

        playStartClockRef.current = startAt;
        playStartPosRef.current = pos;

        // The stems' gain/panner nodes are created lazily by ensureStemNodes above.
        // Re-apply the current mix state so the graph always reflects the UI.
        applyMixParams();
    };

    // Throttled playhead loop derived from the master clock, decoupled from re-renders.
    const startPlayheadLoop = () => {
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
        const tick = () => {
            if (!isPlayingRef.current) {
                rafRef.current = 0;
                return;
            }
            const pos = getPosition();
            const dur = durationRef.current;
            if (pos >= dur) {
                if (isLoopingRef.current) {
                    currentTimeRef.current = 0;
                    setCurrentTime(0);
                    scheduleAll();
                    return;
                }
                pauseAll();
                currentTimeRef.current = dur;
                setCurrentTime(dur);
                return;
            }
            const now = performance.now();
            if (now - lastUiTickRef.current >= UI_TICK_MS) {
                lastUiTickRef.current = now;
                currentTimeRef.current = pos;
                setCurrentTime(pos);
            }
            rafRef.current = requestAnimationFrame(tick);
        };
        rafRef.current = requestAnimationFrame(tick);
    };

    // Push mix state (vol/mute/solo/pan/master) into the graph nodes.
    useEffect(() => {
        applyMixParams();
    }, [stemChannels, masterVolume, isMasterMuted, hasLoadedStems]); // eslint-disable-line react-hooks/exhaustive-deps

    // Keep the playback-scheduler's duration mirror in sync with the UI duration
    // (which becomes exact once the master buffer decodes).
    useEffect(() => {
        durationRef.current = duration;
    }, [duration]);

    // Build the graph and pre-decode the master + active stems so first play is
    // instant.
    useEffect(() => {
        ensureAudioContext();
        if (decodeStartedRef.current) return;
        decodeStartedRef.current = true;
        prepareBuffers();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [stemSource]);

    // Close the shared AudioContext and tear down all nodes + sources on unmount
    // so nothing leaks if the workspace is closed.
    useEffect(() => {
        return () => {
            stopSources();
            Object.keys(stemGainRefs.current).forEach(id => {
                try { stemGainRefs.current[id].disconnect(); } catch { /* ignore */ }
            });
            Object.keys(stemPanRefs.current).forEach(id => {
                try { stemPanRefs.current[id].disconnect(); } catch { /* ignore */ }
            });
            if (audioCtxRef.current) {
                audioCtxRef.current.close().catch(console.error);
                audioCtxRef.current = null;
                masterGainRef.current = null;
                masterMixGainRef.current = null;
            }
            stemGainRefs.current = {};
            stemPanRefs.current = {};
            decodeStartedRef.current = false;
            preparedIdsRef.current = new Set();
        };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const togglePlay = () => {
        if (isPlaying) {
            pauseAll();
        } else {
            playAll();
        }
    };

    const playAll = async () => {
        const ctx = audioCtxRef.current || ensureAudioContext();
        if (!ctx) return;
        
        // 1. Ensure AudioContext is actively running (browser autoplay / resume policy)
        if (ctx.state === 'suspended') {
            try {
                await ctx.resume();
            } catch (e) {
                console.error('AudioContext resume failed:', e);
            }
        }

        // 2. Auto-reset to 0 if starting playback at or past track duration
        if (currentTimeRef.current >= durationRef.current - 0.05) {
            currentTimeRef.current = 0;
            setCurrentTime(0);
        }

        // 3. Decode any buffer that isn't cached yet
        try {
            const jobs: { id: string; url: string }[] = [];
            const mUrl = getMasterUrl();
            if (mUrl && !bufCacheRef.current['__master__']) jobs.push({ id: '__master__', url: mUrl });
            stemChannels.forEach(stem => {
                if (stem.audioUrl && !bufCacheRef.current[stem.id]) {
                    jobs.push({ id: stem.id, url: stem.audioUrl });
                    preparedIdsRef.current.add(stem.id);
                }
            });
            if (jobs.length) {
                const results = await Promise.all(jobs.map(j => decodeToBuffer(j.id, j.url).then(b => ({ id: j.id, b }))));
                const loaded: Record<string, boolean> = {};
                results.forEach(r => { if (r.b) loaded[r.id] = true; });
                setLoadedStemIds(() => {
                    const next: Record<string, boolean> = {};
                    preparedIdsRef.current.forEach(id => { next[id] = loaded[id] || !!bufCacheRef.current[id]; });
                    return next;
                });
                const m = bufCacheRef.current['__master__'];
                if (m) setDuration(m.duration);
            }

            // 4. Schedule sources on hardware master clock
            await scheduleAll();
            isPlayingRef.current = true;
            setIsPlaying(true);
            lastUiTickRef.current = 0;
            startPlayheadLoop();
        } catch (err) {
            console.error('Playback initialization error:', err);
        }
    };

    const pauseAll = () => {
        const finalPos = Math.max(0, Math.min(durationRef.current, getPosition()));
        currentTimeRef.current = finalPos;
        setCurrentTime(finalPos);
        stopSources();
        isPlayingRef.current = false;
        setIsPlaying(false);
    };

    const handleSeek = (time: number) => {
        const clamped = Math.max(0, Math.min(durationRef.current || duration, time));
        currentTimeRef.current = clamped;
        setCurrentTime(clamped);
        if (isPlayingRef.current) {
            scheduleAll().catch(console.error);
        }
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
            {/* No <audio> elements here: every stem and the master AI decode into
                AudioBuffers and play through Web Audio (AudioBufferSourceNode), all
                scheduled against one AudioContext.currentTime master clock. */}

            {/* Top Workspace Header Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 px-4 sm:px-6 py-3 border-b border-black/[0.06] dark:border-white/[0.08] bg-white/70 dark:bg-[#12141c]/80 backdrop-blur-2xl flex-shrink-0 z-20 shadow-apple-sm">
                <div className="flex items-center space-x-3 min-w-0">
                    <div className="w-8 h-8 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 text-teal-700 dark:text-teal-300 border border-teal-500/20 flex items-center justify-center font-bold text-xs p-0.5 flex-shrink-0 overflow-hidden">
                        <img 
                            src={job.cover_image_path ? (job.cover_image_path.startsWith('http') ? job.cover_image_path : `${API_BASE_URL}${job.cover_image_path}`) : '/milimo_logo.png'} 
                            alt="Logo" 
                            className="w-full h-full object-cover rounded-lg" 
                            onError={(e) => {
                                (e.target as HTMLImageElement).src = '/milimo_logo.png';
                            }} 
                        />
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
                            onClick={() => switchStemSource('neural')}
                            title="Use the neural source-separated master group"
                            aria-label="Neural Stems"
                            className={`px-2.5 py-1.5 rounded-lg text-[10px] sm:text-xs font-semibold transition-all whitespace-nowrap ${
                                stemSource === 'neural'
                                    ? 'bg-white dark:bg-white/20 text-teal-700 dark:text-teal-300 font-bold shadow-apple-sm'
                                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-black/[0.03] dark:hover:bg-white/5'
                            }`}
                        >
                            {realChannels.length} Neural Stems
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
                            <div className="w-52 h-52 rounded-3xl bg-gradient-to-tr from-teal-500/20 via-cyan-500/20 to-sky-500/20 border border-black/[0.08] dark:border-white/10 shadow-apple-lg flex items-center justify-center p-2 backdrop-blur-xl overflow-hidden">
                                <img 
                                    src={job.cover_image_path ? (job.cover_image_path.startsWith('http') ? job.cover_image_path : `${API_BASE_URL}${job.cover_image_path}`) : '/milimo_logo.png'} 
                                    alt="Artwork" 
                                    className="w-full h-full object-cover rounded-2xl filter drop-shadow-md" 
                                    onError={(e) => { (e.target as HTMLImageElement).src = '/milimo_logo.png'; }}
                                />
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
                        {timedLyrics.length > 0 && activeLineIndex !== -1 && (
                            <div className="h-20 w-full max-w-xl bg-white/70 dark:bg-[#12141c]/70 border border-black/[0.06] dark:border-white/10 rounded-2xl p-3 flex flex-col items-center justify-center text-center space-y-0.5 overflow-hidden shadow-apple-sm backdrop-blur-xl">
                                {activeLineIndex > 0 && (
                                    <p className="text-[11px] font-mono text-slate-400 dark:text-slate-500 opacity-50 truncate max-w-md">
                                        {timedLyrics[activeLineIndex - 1]?.text}
                                    </p>
                                )}
                                <p className="text-xs sm:text-sm font-bold text-teal-600 dark:text-teal-300 font-mono scale-105 transition-all duration-200">
                                    {timedLyrics[activeLineIndex]?.text}
                                </p>
                                {activeLineIndex < timedLyrics.length - 1 && (
                                    <p className="text-[11px] font-mono text-slate-400 dark:text-slate-500 opacity-50 truncate max-w-md">
                                        {timedLyrics[activeLineIndex + 1]?.text}
                                    </p>
                                )}
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

                            <div className="flex items-center gap-2">
                                <a
                                    href={`${API_BASE_URL}/tracks/${job.id}/lrc`}
                                    download={`${job.title || 'lyrics'}.lrc`}
                                    className="px-3 py-1.5 rounded-xl bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 text-xs font-semibold flex items-center gap-1.5 transition-colors"
                                >
                                    <FileText size={13} className="text-teal-500" />
                                    <span>Download .LRC</span>
                                </a>

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
                        </div>

                        {/* Lyrics Container */}
                        <div className="flex-1 overflow-y-auto pr-2 space-y-4 font-sans select-text">
                            {timedLyrics.length > 0 ? (
                                timedLyrics.map((line, idx) => {
                                    const isCurrent = idx === activeLineIndex;
                                    const isPast = idx < activeLineIndex;
                                    const isSection = (line as any).is_section || (line.text.startsWith('[') && line.text.endsWith(']'));

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
                                                {isCurrent && line.words && line.words.length > 0 ? (
                                                    <span className="inline-flex flex-wrap gap-1.5">
                                                        {line.words.map((w: any, wIdx: number) => {
                                                            const isWordSung = currentTime >= w.start;
                                                            return (
                                                                <span
                                                                    key={wIdx}
                                                                    className={`transition-colors duration-150 ${
                                                                        isWordSung
                                                                            ? 'text-teal-800 dark:text-teal-200 font-black'
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
                <div className="flex items-center space-x-1 sm:space-x-1.5 flex-shrink-0">
                    {/* Return to Zero / Start */}
                    <button
                        onClick={() => handleSeek(0)}
                        className="p-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 transition-transform active:scale-95"
                        title="Return to Zero / Start (|<<) (Home)"
                        aria-label="Return to Zero"
                    >
                        <SkipBack size={15} />
                    </button>

                    {/* Rewind 1 Bar */}
                    <button
                        onClick={() => handleSeek(currentTime - barDuration)}
                        className="px-2 py-1.5 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-[11px] font-mono font-bold text-slate-700 dark:text-slate-300 transition-transform active:scale-95"
                        title="Step Back 1 Measure / Bar"
                    >
                        -1 Bar
                    </button>

                    {/* Rewind 10s */}
                    <button
                        onClick={() => handleRewind(10)}
                        className="p-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 transition-transform active:scale-95"
                        title="Rewind 10s (J)"
                    >
                        <RotateCcw size={15} />
                    </button>

                    {/* Play / Pause Hero Button */}
                    <button
                        onClick={togglePlay}
                        className="w-10 h-10 rounded-2xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold flex items-center justify-center shadow-apple-md active:scale-95 transition-transform"
                        title={isPlaying ? "Pause (Space / K)" : "Play (Space / K)"}
                    >
                        {isPlaying ? <Pause size={18} /> : <Play size={18} className="ml-0.5" />}
                    </button>

                    {/* Advance 10s */}
                    <button
                        onClick={() => handleAdvance(10)}
                        className="p-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 transition-transform active:scale-95"
                        title="Advance 10s (L)"
                    >
                        <RotateCw size={15} />
                    </button>

                    {/* Advance 1 Bar */}
                    <button
                        onClick={() => handleSeek(currentTime + barDuration)}
                        className="px-2 py-1.5 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-[11px] font-mono font-bold text-slate-700 dark:text-slate-300 transition-transform active:scale-95"
                        title="Step Forward 1 Measure / Bar"
                    >
                        +1 Bar
                    </button>

                    {/* Timecode */}
                    <div className="flex items-center space-x-1 font-mono text-xs text-slate-700 dark:text-slate-300 ml-2">
                        <span className="font-bold text-teal-600 dark:text-teal-400">{formatTime(currentTime)}</span>
                        <span className="text-slate-400">/</span>
                        <span className="text-slate-500">{formatTime(duration)}</span>
                    </div>

                    {/* Loop Toggle Button */}
                    <button
                        onClick={() => setIsLooping(!isLooping)}
                        className={`p-1.5 rounded-xl transition-colors ml-1 ${
                            isLooping
                                ? 'text-teal-600 dark:text-teal-400 bg-teal-500/10'
                                : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                        }`}
                        title={`Loop Playback: ${isLooping ? 'On' : 'Off'}`}
                    >
                        <Repeat size={14} />
                    </button>

                    <span className="hidden sm:inline text-[10px] font-mono px-2 py-0.5 rounded-md bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20 font-bold ml-1">
                        {Math.round(bpm)} BPM
                    </span>
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
                <div className="flex items-center space-x-2 flex-shrink-0">
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
                        className="w-20 sm:w-24 h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-teal-500"
                    />
                    <span className="text-[10px] font-mono text-slate-400 w-8">
                        {isMasterMuted ? '0%' : `${Math.round(masterVolume * 100)}%`}
                    </span>
                </div>
            </div>
        </div>
    );
};
