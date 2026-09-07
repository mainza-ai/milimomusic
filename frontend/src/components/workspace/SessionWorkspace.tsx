import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
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
import { pushHotkeyScope, isTextEntryTarget, hasModifier } from '../../utils/hotkeyScope';
import { safeJsonParse } from '../../utils/safeJsonParse';
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
    // A-B loop region (seconds). null = unset; loop wraps at loopEnd when both set.
    const [loopStart, setLoopStart] = useState<number | null>(null);
    const [loopEnd, setLoopEnd] = useState<number | null>(null);
    const loopRef = useRef<{ start: number | null; end: number | null }>({ start: null, end: null });
    loopRef.current = { start: loopStart, end: loopEnd };
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(job.duration_ms ? job.duration_ms / 1000 : 60);
    const [masterVolume, setMasterVolume] = useState(0.9);
    const [isMasterMuted, setIsMasterMuted] = useState(false);
    const [isExportOpen, setIsExportOpen] = useState(false);
    const exportMenuRef = useRef<HTMLDivElement | null>(null);

    // Close the export menu on outside click or Escape.
    useEffect(() => {
        if (!isExportOpen) return;
        const onPointerDown = (e: MouseEvent) => {
            if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) {
                setIsExportOpen(false);
            }
        };
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') setIsExportOpen(false);
        };
        document.addEventListener('mousedown', onPointerDown);
        document.addEventListener('keydown', onKeyDown);
        return () => {
            document.removeEventListener('mousedown', onPointerDown);
            document.removeEventListener('keydown', onKeyDown);
        };
    }, [isExportOpen]);

    // ── Parsed job payloads — memoized ──────────────────────────────────────
    // These were parsed INLINE on every render; during playback the workspace
    // re-renders ~12Hz, re-parsing potentially megabyte note/lyric JSON each
    // tick. That alone could stall the main thread.
    const beatGrid = useMemo<{ bpm?: number; beats_per_bar?: number }>(
        () => safeJsonParse(job.beat_grid_json, {}, 'beat_grid_json'), [job.beat_grid_json]);
    const bpm = beatGrid.bpm || 120;
    const beatsPerBar = Number(beatGrid.beats_per_bar) > 0 ? Number(beatGrid.beats_per_bar) : 4;
    const barDuration = (60 / bpm) * beatsPerBar;

    const parsedStems: StemsMap = useMemo(
        () => safeJsonParse(job.stems_json, {} as StemsMap, 'stems_json'), [job.stems_json]);

    const timedLyrics = useMemo<TimedLine[]>(
        () => safeJsonParse<TimedLine[]>(job.timed_lyrics_json, [], 'timed_lyrics_json'), [job.timed_lyrics_json]);

    const notes = useMemo<NoteEvent[]>(
        () => safeJsonParse<NoteEvent[]>(job.notes_json, [], 'notes_json'), [job.notes_json]);

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

    const realChannels: StemChannel[] = useMemo(() => Object.entries(parsedStems)
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
        }), [parsedStems]);

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
    const partChannels: StemChannel[] = useMemo(() => Object.entries(instrumentalParts).map(([name, audio], i) => ({
        id: `part-${name}`,
        name,
        color: PART_COLORS[i % PART_COLORS.length],
        volume: 85,
        pan: 0,
        isMuted: false,
        isSolo: false,
        audioUrl: `${API_BASE_URL}${audio}`,
        midiProgram: instrumentPrograms[name]
    })), [instrumentalParts, instrumentPrograms]);

    // DEFAULT to dynamic per-instrument parts when present, else neural stems
    const defaultSource: 'neural' | 'muscriptor' =
        partEntries.length > 0 ? 'muscriptor' : 'neural';

    const [stemSource, setStemSource] = useState<'neural' | 'muscriptor'>(defaultSource);
    // Active channels are whichever source is currently selected.
    const initialChannels: StemChannel[] =
        stemSource === 'muscriptor' && partChannels.length > 0 ? partChannels : realChannels;
    const [stemChannels, setStemChannels] = useState<StemChannel[]>(initialChannels);

    // Mix memory per stem source: toggling between MuScriptor parts and neural
    // stems used to DESTROY the user's fader/mute/solo moves. Now each source
    // remembers its own mix.
    const mixMemoryRef = useRef<Record<string, StemChannel[]>>({});

    const switchStemSource = (source: 'neural' | 'muscriptor') => {
        setStemSource(source);
        // Remember the outgoing source's mix exactly as the user left it.
        mixMemoryRef.current[stemSource] = stemChannels;
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
        Object.keys(stemAnalyserRefs.current).forEach(id => {
            try { stemAnalyserRefs.current[id].disconnect(); } catch { /* ignore */ }
            delete stemAnalyserRefs.current[id];
        });
        Object.keys(bufCacheRef.current).forEach(id => {
            if (id !== '__master__') delete bufCacheRef.current[id];
        });
        setStemPeaks({});
        setStemDurations({});
        decodeStartedRef.current = null;
        setLoadedStemIds({});
        // Restore the incoming source's remembered mix, or build a fresh one.
        const remembered = mixMemoryRef.current[source];
        if (remembered && remembered.length > 0) {
            setStemChannels(remembered.map(c => ({ ...c })));
        } else if (source === 'muscriptor' && partChannels.length > 0) {
            setStemChannels(partChannels.map(c => ({ ...c, isMuted: false, isSolo: false })));
        } else {
            setStemChannels(realChannels.map(c => ({ ...c, isMuted: false, isSolo: false })));
        }
        setIsPlaying(false);
    };

    // ── Per-track session persistence ────────────────────────────────────────
    // Mixer levels, stem source and loop region survive refresh — session
    // recall is table stakes in every commercial DAW. The active MODE is
    // deliberately excluded: the workspace always opens on its home (Listen).
    const wsKey = `milimo_ws_${job.id}`;
    const wsHydratedRef = useRef(false);

    useEffect(() => {
        if (wsHydratedRef.current) return;
        wsHydratedRef.current = true;
        try {
            const raw = localStorage.getItem(wsKey);
            if (!raw) return;
            const s = JSON.parse(raw);
            if (typeof s.masterVolume === 'number') setMasterVolume(s.masterVolume);
            if (typeof s.isMasterMuted === 'boolean') setIsMasterMuted(s.isMasterMuted);
            // NOTE: `mode` is intentionally NOT restored — entering the DAW must
            // always land on its home (Listen) view, never teleport into an
            // editor because a previous visit ended elsewhere. Mix levels,
            // stem source and loop region still recall per track.
            if ((s.stemSource === 'neural' || s.stemSource === 'muscriptor') && Array.isArray(s.channels?.[s.stemSource])) {
                setStemSource(s.stemSource);
                setStemChannels((s.channels[s.stemSource] as StemChannel[]).map(c => ({ ...c })));
                mixMemoryRef.current[s.stemSource] = s.channels[s.stemSource];
            }
            if (typeof s.loopStart === 'number') setLoopStart(s.loopStart);
            if (typeof s.loopEnd === 'number') setLoopEnd(s.loopEnd);
        } catch (e) {
            console.warn('Workspace session restore failed', e);
        }
    }, [wsKey]);

    useEffect(() => {
        const t = window.setTimeout(() => {
            try {
                localStorage.setItem(wsKey, JSON.stringify({
                    masterVolume,
                    isMasterMuted,
                    stemSource,
                    channels: { [stemSource]: stemChannels },
                    loopStart,
                    loopEnd
                }));
            } catch { /* storage quota — non-fatal */ }
        }, 400);
        return () => window.clearTimeout(t);
    }, [masterVolume, isMasterMuted, stemSource, stemChannels, loopStart, loopEnd, wsKey]);

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
    const [masteredPath, setMasteredPath] = useState<string | undefined>(job.mastered_path);
    const [masterAuditionMode, setMasterAuditionMode] = useState<'original' | 'mastered'>(
        job.mastered_path ? 'mastered' : 'original'
    );
    const audioCtxRef = useRef<AudioContext | null>(null);
    const masterGainRef = useRef<GainNode | null>(null);        // global fader (master volume + mute)
    const masterMixGainRef = useRef<GainNode | null>(null);     // master-mix channel
    const masterOrigGainRef = useRef<GainNode | null>(null);     // unmastered original mix bus
    const masterPostGainRef = useRef<GainNode | null>(null);     // Matchering reference mastered bus
    const stemGainRefs = useRef<Record<string, GainNode>>({});
    const stemPanRefs = useRef<Record<string, StereoPannerNode>>({});
    // Post-gain AnalyserNode taps feeding the mixer's REAL peak meters.
    const stemAnalyserRefs = useRef<Record<string, AnalyserNode>>({});
    const masterAnalyserRef = useRef<AnalyserNode | null>(null);
    const bufCacheRef = useRef<Record<string, AudioBuffer>>({}); // decoded buffers by id/url
    const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
    const isPlayingRef = useRef(false);
    const isLoopingRef = useRef(false);
    // Keep the scheduler-facing mirror in sync with the toggle state.
    // (This sync was missing entirely — the loop flag stayed false forever,
    // so neither full-track nor A-B looping ever engaged.)
    useEffect(() => {
        isLoopingRef.current = isLooping;
    }, [isLooping]);
    const currentTimeRef = useRef(0);
    const durationRef = useRef(duration);
    const playStartClockRef = useRef(0);
    const playStartPosRef = useRef(0);
    const rafRef = useRef(0);
    const preparedIdsRef = useRef<Set<string>>(new Set()); // buffers that have a URL to load
    const decodeStartedRef = useRef<string | null>(null);   // last source decoded (avoids StrictMode double decode)
    const UI_TICK_MS = 80;                    // min gap between playhead re-renders (~12Hz)
    const lastUiTickRef = useRef(0);

    // Track which stems actually decoded/loaded. If a stem 404s / fails to decode,
    // we fall back to the master mix instead of running a dead-stem multitrack.
    const [loadedStemIds, setLoadedStemIds] = useState<Record<string, boolean>>({});
    const hasLoadedStems = Object.values(loadedStemIds).some(Boolean);

    // Real waveform peaks (normalized 0..1) + real durations per stem, computed
    // from the decoded AudioBuffers — consumed by the Arrange timeline for true
    // clip widths and genuine amplitude waveforms (no simulated bars).
    const [stemPeaks, setStemPeaks] = useState<Record<string, number[]>>({});
    const [stemDurations, setStemDurations] = useState<Record<string, number>>({});

    const computePeaks = (buffer: AudioBuffer, buckets: number = 96): number[] => {
        const data = buffer.getChannelData(0);
        const block = Math.max(1, Math.floor(data.length / buckets));
        const peaks: number[] = [];
        let max = 0;
        for (let i = 0; i < buckets; i++) {
            let peak = 0;
            const start = i * block;
            for (let j = 0; j < block; j += 16) { // stride-sample: plenty for a display waveform
                const v = Math.abs(data[start + j] || 0);
                if (v > peak) peak = v;
            }
            if (peak > max) max = peak;
            peaks.push(peak);
        }
        return max > 0 ? peaks.map(p => p / max) : peaks;
    };

    const refreshStemVisuals = () => {
        const nextPeaks: Record<string, number[]> = {};
        const nextDurations: Record<string, number> = {};
        Object.entries(bufCacheRef.current).forEach(([id, buf]) => {
            if (id === '__master__') return;
            nextPeaks[id] = computePeaks(buf);
            nextDurations[id] = buf.duration;
        });
        setStemPeaks(nextPeaks);
        setStemDurations(nextDurations);
    };

    const isSectionLine = (l?: TimedLine) => {
        if (!l) return false;
        return Boolean(l.is_section || (l.text.startsWith('[') && l.text.endsWith(']')));
    };

    const activeLineIndex = (() => {
        if (timedLyrics.length === 0) return -1;
        let lastSungIdx = -1;
        for (let i = 0; i < timedLyrics.length; i++) {
            const line = timedLyrics[i];
            if (isSectionLine(line)) continue;
            lastSungIdx = i;

            // Find next non-section line to define upper time boundary
            let nextSungStart = line.end || (line.start + 5.0);
            for (let j = i + 1; j < timedLyrics.length; j++) {
                if (!isSectionLine(timedLyrics[j])) {
                    nextSungStart = timedLyrics[j].start;
                    break;
                }
            }

            if (currentTime >= line.start && currentTime < nextSungStart) {
                return i;
            }
        }
        if (lastSungIdx !== -1 && currentTime >= (timedLyrics[lastSungIdx]?.start ?? 0)) {
            return lastSungIdx;
        }
        return -1;
    })();

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
            // Parallel tap for the mixer's real master meter (no routing impact).
            if (!masterAnalyserRef.current) {
                const analyser = ctx.createAnalyser();
                analyser.fftSize = 512;
                master.connect(analyser);
                masterAnalyserRef.current = analyser;
            }
            masterGainRef.current = master;
        }
        if (!masterMixGainRef.current) {
            const g = ctx.createGain();
            g.gain.value = 1;
            g.connect(masterGainRef.current);
            masterMixGainRef.current = g;
        }
        if (!masterOrigGainRef.current) {
            const g = ctx.createGain();
            g.gain.value = 1;
            g.connect(masterMixGainRef.current);
            masterOrigGainRef.current = g;
        }
        if (!masterPostGainRef.current) {
            const g = ctx.createGain();
            g.gain.value = 0;
            g.connect(masterGainRef.current);
            masterPostGainRef.current = g;
        }
    };

    // Compute + apply the full mix state to all nodes (smooth ramps → no zipper).
    const applyMixParams = () => {
        const ctx = audioCtxRef.current;
        if (!ctx || !masterGainRef.current) return;

        const hasMastered = Boolean(masteredPath || job.mastered_path);
        const isMasteredAudition = hasMastered && masterAuditionMode === 'mastered';

        const hasSolo = stemChannels.some(s => s.isSolo);
        stemChannels.forEach(stem => {
            const gain = stemGainRefs.current[stem.id];
            const pan = stemPanRefs.current[stem.id];
            if (!gain) return;
            let perStem = 0;
            if (isMasteredAudition) {
                // When auditioning reference master (B), multitrack stems are silenced
                perStem = 0;
            } else if (hasSolo) {
                perStem = stem.isSolo && !stem.isMuted ? stem.volume / 100 : 0;
            } else {
                perStem = !stem.isMuted ? stem.volume / 100 : 0;
            }
            gain.gain.setTargetAtTime(Math.max(0, Math.min(1, perStem)), ctx.currentTime, 0.015);
            if (pan) pan.pan.setTargetAtTime(Math.max(-1, Math.min(1, stem.pan / 50)), ctx.currentTime, 0.015);
        });

        // Global fader + master-mix fallback.
        masterGainRef.current.gain.setTargetAtTime(
            isMasterMuted ? 0 : Math.max(0, Math.min(1, masterVolume)), ctx.currentTime, 0.015
        );
        if (masterMixGainRef.current) {
            masterMixGainRef.current.gain.setTargetAtTime(hasLoadedStems && !isMasteredAudition ? 0 : 1, ctx.currentTime, 0.015);
        }
        if (masterOrigGainRef.current) {
            // Original master plays if no stems are loaded AND we are in original audition mode
            const origLevel = (!isMasteredAudition && !hasLoadedStems) ? 1 : 0;
            masterOrigGainRef.current.gain.setTargetAtTime(origLevel, ctx.currentTime, 0.015);
        }
        if (masterPostGainRef.current) {
            // Mastered track plays when in 'mastered' audition mode
            const postLevel = isMasteredAudition ? 1 : 0;
            masterPostGainRef.current.gain.setTargetAtTime(postLevel, ctx.currentTime, 0.015);
        }
    };

    useEffect(() => {
        applyMixParams();
    }, [masterAuditionMode]);

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
        // Parallel post-gain tap so the mixer meter shows the REAL audible
        // level (mute/solo/volume/pan all included).
        if (!stemAnalyserRefs.current[id]) {
            const analyser = audioCtxRef.current.createAnalyser();
            analyser.fftSize = 512;
            stemGainRefs.current[id].connect(analyser);
            stemAnalyserRefs.current[id] = analyser;
        }
        return stemGainRefs.current[id];
    };

    // ── Decoding ────────────────────────────────────────────────────────────
    const getMasterUrl = (): string | null => {
        const effPath = (masterAuditionMode === 'mastered' && (masteredPath || job.mastered_path))
            ? (masteredPath || job.mastered_path)
            : job.audio_path;
        if (!effPath) return null;
        return effPath.startsWith('http') ? effPath : `${API_BASE_URL}${effPath}`;
    };

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
            jobs.push({ id: '__master_orig__', url });
        }
        const effMastered = masteredPath || job.mastered_path;
        if (effMastered) {
            const url = effMastered.startsWith('http') ? effMastered : `${API_BASE_URL}${effMastered}`;
            jobs.push({ id: '__master_post__', url });
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
        refreshStemVisuals();
    };

    // The current transport position, derived from the master clock (sample-accurate).
    // Stable identity (reads refs only) so the piano roll can poll it per rAF
    // and schedule synth notes ON the audio clock — never chasing UI ticks.
    const getPosition = useCallback((): number => {
        const ctx = audioCtxRef.current;
        if (!ctx || !isPlayingRef.current) return currentTimeRef.current;
        const elapsed = ctx.currentTime - playStartClockRef.current;
        return Math.max(0, playStartPosRef.current + elapsed);
    }, []);

    // Stop + drop every currently scheduled source.
    // NOTE: this deliberately does NOT touch rafRef. The UI playhead loop is a
    // SEPARATE lifecycle from audio nodes — cancelling it here froze the
    // playhead forever on every seek-while-playing (audio kept playing).
    // Loop termination happens via isPlayingRef=false (tick self-exits) or
    // startPlayheadLoop's own re-schedule.
    const stopSources = () => {
        activeSourcesRef.current.forEach(src => {
            try { src.stop(); } catch { /* already stopped */ }
            try { src.disconnect(); } catch { /* ignore */ }
        });
        activeSourcesRef.current.clear();
    };

    // Schedule the master + every loaded stem at the current transport position.
    // ALL sources are started with the same (when, offset), so they are sample-locked.
    const scheduleAll = async () => {
        const ctx = audioCtxRef.current || ensureAudioContext();
        if (!ctx || !masterGainRef.current) return;
        stopSources();
        const startAt = ctx.currentTime;
        const pos = Math.max(0, Math.min(durationRef.current, currentTimeRef.current));

        const masterOrigBuf = bufCacheRef.current['__master_orig__'] || bufCacheRef.current['__master__'];
        if (masterOrigBuf && pos < masterOrigBuf.duration) {
            const src = ctx.createBufferSource();
            src.buffer = masterOrigBuf;
            src.connect(masterOrigGainRef.current || masterMixGainRef.current!);
            const offset = Math.min(Math.max(pos, 0), Math.max(0, masterOrigBuf.duration - 0.02));
            src.start(startAt, offset, Math.max(0, masterOrigBuf.duration - offset));
            activeSourcesRef.current.add(src);
        }

        const masterPostBuf = bufCacheRef.current['__master_post__'];
        if (masterPostBuf && pos < masterPostBuf.duration) {
            const src = ctx.createBufferSource();
            src.buffer = masterPostBuf;
            src.connect(masterPostGainRef.current || masterGainRef.current!);
            const offset = Math.min(Math.max(pos, 0), Math.max(0, masterPostBuf.duration - 0.02));
            src.start(startAt, offset, Math.max(0, masterPostBuf.duration - offset));
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
    // CRITICAL: every branch MUST re-queue the next frame except a true stop.
    // (A wrap used to `return` without rescheduling — killing the loop on the
    // first A-B/end wrap: audio kept playing but the playhead froze forever.)
    const startPlayheadLoop = () => {
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
        const tick = () => {
            if (!isPlayingRef.current) {
                rafRef.current = 0;
                return;
            }
            const pos = getPosition();
            const dur = durationRef.current;
            const lStart = loopRef.current.start;
            const lEnd = loopRef.current.end;

            if (isLoopingRef.current && lStart !== null && lEnd !== null && lEnd > lStart && pos >= lEnd) {
                // A-B wrap
                currentTimeRef.current = lStart;
                setCurrentTime(lStart);
                void scheduleAll();
            } else if (pos >= dur) {
                if (isLoopingRef.current) {
                    // Full-track wrap
                    currentTimeRef.current = 0;
                    setCurrentTime(0);
                    void scheduleAll();
                } else {
                    // Terminal end-of-track: the ONLY branch allowed to stop.
                    pauseAll();
                    currentTimeRef.current = dur;
                    setCurrentTime(dur);
                    rafRef.current = 0;
                    return;
                }
            } else {
                const now = performance.now();
                if (now - lastUiTickRef.current >= UI_TICK_MS) {
                    lastUiTickRef.current = now;
                    currentTimeRef.current = pos;
                    setCurrentTime(pos);
                }
            }
            rafRef.current = requestAnimationFrame(tick);
        };
        rafRef.current = requestAnimationFrame(tick);
    };

    // ── Transport watchdog ──────────────────────────────────────────────────
    // While playing, the playhead's rAF chain MUST be alive. If anything ever
    // silently kills it again (the way the old stopSources did on seek), this
    // restarts it within ~500ms and leaves a console breadcrumb so the cause
    // is observable instead of a mystery frozen playhead.
    useEffect(() => {
        const iv = window.setInterval(() => {
            if (isPlayingRef.current && !rafRef.current) {
                console.warn('[Milimo DAW] transport watchdog: playhead loop was dead while playing — restarted.');
                startPlayheadLoop();
            }
        }, 500);
        return () => window.clearInterval(iv);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

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
    // instant. Guard is per-source: session restore may change stemSource after
    // mount, and the restored source MUST be decoded too.
    useEffect(() => {
        ensureAudioContext();
        if (decodeStartedRef.current === stemSource) return;
        decodeStartedRef.current = stemSource;
        prepareBuffers();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [stemSource]);

    // Close the shared AudioContext and tear down all nodes + sources on unmount
    // so nothing leaks if the workspace is closed.
    useEffect(() => {
        return () => {
            // Halt the UI playhead loop FIRST — it must not fire after unmount
            // (stopSources no longer cancels it; see note there).
            isPlayingRef.current = false;
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
            rafRef.current = 0;
            stopSources();
            Object.keys(stemGainRefs.current).forEach(id => {
                try { stemGainRefs.current[id].disconnect(); } catch { /* ignore */ }
            });
            Object.keys(stemPanRefs.current).forEach(id => {
                try { stemPanRefs.current[id].disconnect(); } catch { /* ignore */ }
            });
            Object.keys(stemAnalyserRefs.current).forEach(id => {
                try { stemAnalyserRefs.current[id].disconnect(); } catch { /* ignore */ }
            });
            if (audioCtxRef.current) {
                audioCtxRef.current.close().catch(console.error);
                audioCtxRef.current = null;
                masterGainRef.current = null;
                masterMixGainRef.current = null;
                masterAnalyserRef.current = null;
            }
            stemGainRefs.current = {};
            stemPanRefs.current = {};
            stemAnalyserRefs.current = {};
            decodeStartedRef.current = null;
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
                refreshStemVisuals();
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

    // Scrubbing used to tear down and recreate every source on EVERY input
    // step — machine-gun clicks. Now: state updates instantly, but the
    // expensive reschedule is throttled with a trailing call.
    const lastScheduleAtRef = useRef(0);
    const scheduleTrailingRef = useRef<number | undefined>(undefined);

    const handleSeek = (time: number) => {
        const clamped = Math.max(0, Math.min(durationRef.current || duration, time));
        currentTimeRef.current = clamped;
        setCurrentTime(clamped);
        if (isPlayingRef.current) {
            const now = performance.now();
            if (now - lastScheduleAtRef.current > 200) {
                lastScheduleAtRef.current = now;
                scheduleAll().catch(console.error);
            } else {
                window.clearTimeout(scheduleTrailingRef.current);
                scheduleTrailingRef.current = window.setTimeout(() => {
                    if (isPlayingRef.current) {
                        lastScheduleAtRef.current = performance.now();
                        scheduleAll().catch(console.error);
                    }
                }, 220);
            }
        }
    };

    const handleRewind = (seconds: number = 5) => {
        handleSeek(currentTime - seconds);
    };

    const handleAdvance = (seconds: number = 5) => {
        handleSeek(currentTime + seconds);
    };

    // ── Workspace-scoped transport hotkeys ──────────────────────────────────
    // While the workspace is mounted it OWNS Space/K/J/L/Home/M: they drive the
    // session multitrack transport (never the global player — that would run
    // two audio streams at once). The global engine defers via hotkeyScope.
    const togglePlayRef = useRef(togglePlay);
    const seekRef = useRef(handleSeek);
    const rewindRef = useRef(handleRewind);
    const advanceRef = useRef(handleAdvance);
    togglePlayRef.current = togglePlay;
    seekRef.current = handleSeek;
    rewindRef.current = handleRewind;
    advanceRef.current = handleAdvance;

    useEffect(() => {
        return pushHotkeyScope((e: KeyboardEvent) => {
            if (hasModifier(e)) return false;
            if (isTextEntryTarget(e.target)) return false;
            const t = e.target as HTMLElement | null;
            if ((e.code === 'Space' || e.code === 'Enter') && t?.tagName === 'BUTTON') return false;
            // The piano-roll editor claims editing keys while its grid has
            // focus; the transport keeps Space/K/J/L/Home/M everywhere.
            const EDITOR_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Delete', 'Backspace']);
            if (t?.closest?.('[data-hotkey-local]') && EDITOR_KEYS.has(e.code)) return false;
            switch (e.code) {
                case 'Space':
                case 'KeyK':
                    e.preventDefault();
                    togglePlayRef.current();
                    return true;
                case 'KeyJ':
                case 'ArrowLeft':
                    e.preventDefault();
                    rewindRef.current(e.shiftKey ? 10 : 5);
                    return true;
                case 'KeyL':
                case 'ArrowRight':
                    e.preventDefault();
                    advanceRef.current(e.shiftKey ? 10 : 5);
                    return true;
                case 'Home':
                case 'Digit0':
                    e.preventDefault();
                    seekRef.current(0);
                    return true;
                case 'KeyM':
                    e.preventDefault();
                    setIsMasterMuted(v => !v);
                    return true;
                default:
                    return false;
            }
        });
    }, []);

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
                                {Math.round(bpm)} BPM · {Math.round(beatsPerBar || 4)}/4
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
                    <div className="relative" ref={exportMenuRef}>
                        <button
                            onClick={() => setIsExportOpen(v => !v)}
                            title="Export Multi-Track MIDI, MusicXML Score, or Timed Lyrics"
                            aria-label="Export DAW Assets"
                            aria-expanded={isExportOpen}
                            aria-haspopup="menu"
                            className={`px-3 sm:px-3.5 py-1.5 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-all shadow-md shadow-teal-500/20 active:scale-95 ${isExportOpen ? 'ring-2 ring-teal-500/50' : ''}`}
                        >
                            <Download size={13} />
                            <span className="hidden sm:inline">Export DAW Assets</span>
                            <span className="sm:hidden">Export</span>
                        </button>
                        {isExportOpen && (
                            <div
                                role="menu"
                                className="absolute right-0 mt-1.5 w-52 bg-white dark:bg-[#181a24] border border-black/[0.08] dark:border-white/10 rounded-2xl shadow-apple-lg p-1.5 z-50 animate-fade-in"
                            >
                                <button
                                    role="menuitem"
                                    onClick={() => { handleExport('midi'); setIsExportOpen(false); }}
                                    title="Download .mid MIDI file"
                                    className="w-full text-left px-3 py-2 text-xs font-semibold rounded-xl hover:bg-teal-500/10 text-slate-800 dark:text-slate-200 hover:text-teal-600 dark:hover:text-teal-300 flex items-center justify-between"
                                >
                                    <span>Multi-Track MIDI (.mid)</span>
                                    <Music size={12} className="text-teal-500" />
                                </button>
                                <button
                                    role="menuitem"
                                    onClick={() => { handleExport('musicxml'); setIsExportOpen(false); }}
                                    title="Download W3C MusicXML Sheet Music Score"
                                    className="w-full text-left px-3 py-2 text-xs font-semibold rounded-xl hover:bg-teal-500/10 text-slate-800 dark:text-slate-200 hover:text-teal-600 dark:hover:text-teal-300 flex items-center justify-between"
                                >
                                    <span>MusicXML Sheet (.musicxml)</span>
                                    <FileText size={12} className="text-cyan-500" />
                                </button>
                                <button
                                    role="menuitem"
                                    onClick={() => { handleExport('lrc'); setIsExportOpen(false); }}
                                    title="Download Karaoke Synchronized Lyrics"
                                    className="w-full text-left px-3 py-2 text-xs font-semibold rounded-xl hover:bg-teal-500/10 text-slate-800 dark:text-slate-200 hover:text-teal-600 dark:hover:text-teal-300 flex items-center justify-between"
                                >
                                    <span>Timed Lyrics (.lrc)</span>
                                    <CheckCircle2 size={12} className="text-amber-500" />
                                </button>
                            </div>
                        )}
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
                                    {Math.round(bpm)} BPM
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
                        stemPeaks={stemPeaks}
                        stemDurations={stemDurations}
                    />
                )}

                {mode === 'pianoroll' && (
                    <PianoRoll
                        job={job}
                        currentTime={currentTime}
                        duration={duration}
                        onSeek={handleSeek}
                        isPlaying={isPlaying}
                        getPosition={getPosition}
                        getAudioContext={ensureAudioContext}
                    />
                )}

                {mode === 'notation' && (
                    <NotationViewer
                        job={job}
                        /* Coarse (250ms) time: the notation view re-engraves its
                            full SVG tree per render, and a playhead there only
                            needs bar-level resolution — 4Hz, not 12Hz. */
                        currentTime={Math.round(currentTime * 4) / 4}
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
                        stemAnalysersRef={stemAnalyserRefs}
                        masterAnalyserRef={masterAnalyserRef}
                        masterAuditionMode={masterAuditionMode}
                        onMasterAuditionModeChange={setMasterAuditionMode}
                        hasMasteredTrack={Boolean(masteredPath || job.mastered_path)}
                        onMasteringComplete={async (newPath: string) => {
                            setMasteredPath(newPath);
                            setMasterAuditionMode('mastered');
                            const url = newPath.startsWith('http') ? newPath : `${API_BASE_URL}${newPath}`;
                            await decodeToBuffer('__master_post__', url);
                            if (isPlayingRef.current) {
                                scheduleAll();
                            } else {
                                applyMixParams();
                            }
                        }}
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

                    {/* A-B Loop: set region markers at the playhead, then arm */}
                    <div className="flex items-center rounded-xl bg-black/[0.04] dark:bg-white/5 ml-1 overflow-hidden">
                        <button
                            onClick={() => setLoopStart(currentTime)}
                            title={`Set loop start (A) at ${formatTime(currentTime)}`}
                            aria-label="Set loop start point"
                            className={`px-2 py-1.5 text-[10px] font-mono font-bold transition-colors ${
                                loopStart !== null ? 'text-teal-600 dark:text-teal-400' : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                            }`}
                        >
                            A{loopStart !== null ? '·' : ''}
                        </button>
                        <button
                            onClick={() => setLoopEnd(currentTime)}
                            title={`Set loop end (B) at ${formatTime(currentTime)}`}
                            aria-label="Set loop end point"
                            className={`px-2 py-1.5 text-[10px] font-mono font-bold border-x border-black/[0.06] dark:border-white/10 transition-colors ${
                                loopEnd !== null ? 'text-teal-600 dark:text-teal-400' : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                            }`}
                        >
                            B{loopEnd !== null ? '·' : ''}
                        </button>
                        <button
                            onClick={() => setIsLooping(!isLooping)}
                            aria-pressed={isLooping}
                            aria-label="Toggle loop"
                            title={loopStart !== null && loopEnd !== null ? `Loop A–B (${formatTime(loopStart)}–${formatTime(loopEnd)})` : 'Loop entire track'}
                            className={`p-1.5 transition-colors ${
                                isLooping
                                    ? 'text-teal-600 dark:text-teal-400 bg-teal-500/10'
                                    : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                            }`}
                        >
                            <Repeat size={14} />
                        </button>
                        {(loopStart !== null || loopEnd !== null) && (
                            <button
                                onClick={() => { setLoopStart(null); setLoopEnd(null); }}
                                title="Clear loop region"
                                aria-label="Clear loop region"
                                className="px-1.5 py-1.5 text-[10px] font-bold text-slate-400 hover:text-rose-500 transition-colors"
                            >
                                ✕
                            </button>
                        )}
                    </div>

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
