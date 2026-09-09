import React, { createContext, useContext, useState, useRef, useEffect, useCallback, useMemo } from 'react';
import type { Job } from '../api';
import { api } from '../api';
import { getAudioContext, unlockAudioContext } from '../utils/audioContext';
import { consumeHotkey, isTextEntryTarget, hasModifier } from '../utils/hotkeyScope';
import { toast } from '../utils/toast';

export interface AudioControlsContextValue {
    currentTrack: Job | null;
    isPlaying: boolean;
    volume: number;
    isMuted: boolean;
    playbackRate: number;
    repeatMode: 'off' | 'all' | 'one';
    isShuffle: boolean;
    playlist: Job[];
    analyserNode: AnalyserNode | null;
    playTrack: (track: Job, customPlaylist?: Job[], startAtSeconds?: number) => Promise<void>;
    pause: () => void;
    resume: () => Promise<void>;
    togglePlay: (track?: Job) => void;
    seek: (timeInSeconds: number) => void;
    returnToStart: () => void;
    prevTrackOrRestart: () => void;
    setVolume: (vol: number) => void;
    toggleMute: () => void;
    setPlaybackRate: (rate: number) => void;
    setRepeatMode: (mode: 'off' | 'all' | 'one') => void;
    toggleShuffle: () => void;
    nextTrack: () => void;
    prevTrack: () => void;
    stop: () => void;
    setPlaylist: (list: Job[]) => void;
    addToQueue: (track: Job) => void;
    removeFromQueue: (trackId: string) => void;
    clearQueue: () => void;
    reorderQueue: (fromIndex: number, toIndex: number) => void;
    playbackError: string | null;
    clearPlaybackError: () => void;
    /** Replace current track + queue metadata without touching the element
     *  (used to hydrate an optimistically started track in place). */
    swapCurrentTrack: (track: Job, queue?: Job[]) => void;
}

export interface AudioTimeContextValue {
    currentTime: number;
    duration: number;
    isBuffering: boolean;
}

export type AudioEngineContextValue = AudioControlsContextValue & AudioTimeContextValue;

const AudioControlsContext = createContext<AudioControlsContextValue | null>(null);
const AudioTimeContext = createContext<AudioTimeContextValue | null>(null);
const AudioEngineContext = createContext<AudioEngineContextValue | null>(null);

export const AudioEngineProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const [currentTrack, setCurrentTrack] = useState<Job | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [volume, setVolumeState] = useState<number>(() => {
        const saved = localStorage.getItem('milimo_volume');
        return saved ? parseFloat(saved) : 0.85;
    });
    const [isMuted, setIsMuted] = useState(false);
    const [playbackRate, setPlaybackRateState] = useState<number>(1.0);
    const [repeatMode, setRepeatMode] = useState<'off' | 'all' | 'one'>('off');
    const [isShuffle, setIsShuffle] = useState(false);
    const [playlist, setPlaylist] = useState<Job[]>([]);
    const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);
    // Last user-facing playback failure (media error, CORS, decode, play() rejection).
    const [playbackError, setPlaybackError] = useState<string | null>(null);
    // True while the element is stalled waiting for data (slow connection / seek).
    const [isBuffering, setIsBuffering] = useState(false);

    // Audio node connectivity tracker
    const isSourceConnected = useRef(false);
    // Latest track for stable media-event callbacks.
    const currentTrackRef = useRef<Job | null>(null);
    // Queued seek for a track whose metadata hasn't loaded yet.
    const pendingSeekRef = useRef<{ trackId: string; time: number } | null>(null);

    // Full absolute URL resolver
    const getAudioUrl = useCallback((path?: string | null): string => {
        return api.getAudioUrl(path);
    }, []);

    // Connect WebAudio graph for AnalyserNode
    const ensureAudioGraph = useCallback(() => {
        if (isSourceConnected.current || !audioRef.current) return;
        try {
            const ctx = getAudioContext();
            if (ctx.state === 'suspended') {
                ctx.resume().catch(() => {});
            }
            const source = ctx.createMediaElementSource(audioRef.current);
            const analyser = ctx.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.82;
            source.connect(analyser);
            analyser.connect(ctx.destination);
            setAnalyserNode(analyser);
            isSourceConnected.current = true;
        } catch (e) {
            console.warn('WebAudio Analyser initialization note:', e);
        }
    }, []);

    // Set Volume (Perceptually mapped via quadratic curve for smooth audio taper)
    const setVolume = useCallback((val: number) => {
        const clamped = Math.max(0, Math.min(1, val));
        setVolumeState(clamped);
        localStorage.setItem('milimo_volume', clamped.toString());
        if (audioRef.current) {
            audioRef.current.volume = isMuted ? 0 : Math.pow(clamped, 2);
        }
    }, [isMuted]);

    // Toggle Mute
    const toggleMute = useCallback(() => {
        setIsMuted((prev) => {
            const next = !prev;
            if (audioRef.current) {
                audioRef.current.volume = next ? 0 : Math.pow(volume, 2);
            }
            return next;
        });
    }, [volume]);

    // Set Playback Speed
    const setPlaybackRate = useCallback((rate: number) => {
        const clamped = Math.max(0.25, Math.min(3.0, rate));
        setPlaybackRateState(clamped);
        if (audioRef.current) {
            audioRef.current.playbackRate = clamped;
        }
    }, []);

    // Seek to specific time
    const seek = useCallback((time: number) => {
        if (audioRef.current) {
            const clamped = Math.max(0, Math.min(duration || 1000, time));
            audioRef.current.currentTime = clamped;
            setCurrentTime(clamped);
        }
    }, [duration]);

    // Return to start (0:00)
    const returnToStart = useCallback(() => {
        seek(0);
    }, [seek]);

    // Play specific track. `startAtSeconds` queues a seek applied as soon as
    // the new element's metadata loads — lets list rows start playback at a
    // clicked waveform position without racing load events.
    const playTrack = useCallback(async (track: Job, customPlaylist?: Job[], startAtSeconds?: number) => {
        if (!track?.audio_path) {
            const msg = `“${track?.title || 'Untitled track'}” has no audio file yet — it may still be rendering or its file is missing.`;
            setPlaybackError(msg);
            toast(msg, 'error');
            return;
        }
        setPlaybackError(null);

        if (customPlaylist && customPlaylist.length > 0) {
            setPlaylist(customPlaylist);
        } else {
            setPlaylist((prev) => {
                if (!prev.some((s) => s.id === track.id)) {
                    return [track, ...prev];
                }
                return prev;
            });
        }

        const prevTrack = currentTrackRef.current;
        setCurrentTrack(track);
        currentTrackRef.current = track;
        setCurrentTime(0);
        pendingSeekRef.current = startAtSeconds && startAtSeconds > 0
            ? { trackId: track.id, time: startAtSeconds }
            : null;

        if (audioRef.current) {
            await unlockAudioContext();
            ensureAudioGraph();
            const fullUrl = getAudioUrl(track.audio_path);
            const currentSrc = audioRef.current.src || '';
            const isReload = !currentSrc || prevTrack?.id !== track.id || (
                currentSrc !== fullUrl && !currentSrc.endsWith(track.audio_path)
            );
            if (isReload) {
                audioRef.current.src = fullUrl;
                audioRef.current.load();
            }

            try {
                await audioRef.current.play();
                setIsPlaying(true);
                // Same-source restart with a queued position: metadata won't
                // fire again, so apply the seek directly.
                if (!isReload && pendingSeekRef.current?.trackId === track.id) {
                    audioRef.current.currentTime = pendingSeekRef.current.time;
                    setCurrentTime(pendingSeekRef.current.time);
                    pendingSeekRef.current = null;
                }
            } catch (err: any) {
                if (err.name !== 'AbortError') {
                    const reason = err?.name === 'NotSupportedError'
                        ? 'This audio format could not be decoded by your browser.'
                        : err?.name === 'NotAllowedError'
                            ? 'Playback was blocked by the browser — press play again.'
                            : `Playback failed (${err?.name || 'media error'}). Check connection or file.`;
                    const msg = `“${track.title || 'Untitled track'}”: ${reason}`;
                    setPlaybackError(msg);
                    toast(msg, 'error');
                }
                setIsPlaying(false);
            }
        }
    }, [ensureAudioGraph, getAudioUrl]);

    // Pause
    const pause = useCallback(() => {
        if (audioRef.current) {
            audioRef.current.pause();
            setIsPlaying(false);
            setIsBuffering(false);
        }
    }, []);

    // Resume
    const resume = useCallback(async () => {
        if (audioRef.current) {
            await unlockAudioContext();
            ensureAudioGraph();
            try {
                await audioRef.current.play();
                setIsPlaying(true);
                setPlaybackError(null);
            } catch (err: any) {
                if (err.name !== 'AbortError') {
                    const msg = `Resume failed (${err?.name || 'media error'}).`;
                    setPlaybackError(msg);
                    toast(msg, 'error');
                }
            }
        }
    }, [ensureAudioGraph]);

    const clearPlaybackError = useCallback(() => setPlaybackError(null), []);

    const swapCurrentTrack = useCallback((track: Job, queue?: Job[]) => {
        currentTrackRef.current = track;
        setCurrentTrack(track);
        if (queue && queue.length > 0) {
            setPlaylist(queue);
        }
    }, []);

    // Media-element failures (404, CORS, decode, network stall) are otherwise silent.
    const handleMediaError = useCallback(() => {
        const el = audioRef.current;
        const code = el?.error?.code;
        const reason = code === 4
            ? 'The audio file could not be loaded (unsupported format or corrupt file).'
            : code === 3
                ? 'Audio decoding failed in your browser.'
                : code === 2
                    ? 'A network error interrupted audio loading — check the server connection.'
                    : 'The audio file could not be found on the server (it may have been moved or deleted).';
        const track = currentTrackRef.current;
        const msg = track ? `“${track.title || 'Untitled track'}”: ${reason}` : reason;
        setPlaybackError(msg);
        setIsPlaying(false);
        setIsBuffering(false);
        toast(msg, 'error');
    }, []);

    const handleStalled = useCallback(() => {
        // Only surface stalls when we expected to be playing.
        if (audioRef.current && !audioRef.current.paused) {
            const msg = 'Audio is stalling — the connection may be slow. It will resume automatically.';
            toast(msg, 'info');
        }
    }, []);

    const handleWaiting = useCallback(() => {
        if (currentTrackRef.current) setIsBuffering(true);
    }, []);

    const handlePlaying = useCallback(() => {
        setIsBuffering(false);
        setPlaybackError(null);
    }, []);

    const handleCanPlay = useCallback(() => {
        setIsBuffering(false);
    }, []);

    // Toggle Play
    const togglePlay = useCallback((trackToPlay?: Job) => {
        if (trackToPlay && trackToPlay.id !== currentTrack?.id) {
            playTrack(trackToPlay);
            return;
        }

        if (isPlaying) {
            pause();
        } else {
            if (currentTrack) {
                resume();
            } else if (playlist.length > 0) {
                playTrack(playlist[0]);
            }
        }
    }, [currentTrack, isPlaying, pause, playTrack, playlist, resume]);

    // Stop playback
    const stop = useCallback(() => {
        if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current.currentTime = 0;
        }
        setIsPlaying(false);
        setCurrentTime(0);
        setCurrentTrack(null);
    }, []);

    // Next Track
    const nextTrack = useCallback(() => {
        if (playlist.length === 0) return;
        let nextIdx = 0;
        if (isShuffle) {
            nextIdx = Math.floor(Math.random() * playlist.length);
        } else {
            const curIdx = playlist.findIndex((s) => s.id === currentTrack?.id);
            nextIdx = (curIdx + 1) % playlist.length;
        }
        playTrack(playlist[nextIdx]);
    }, [playlist, isShuffle, currentTrack?.id, playTrack]);

    // Prev Track
    const prevTrack = useCallback(() => {
        if (playlist.length === 0) return;
        const curIdx = playlist.findIndex((s) => s.id === currentTrack?.id);
        const prevIdx = curIdx <= 0 ? playlist.length - 1 : curIdx - 1;
        playTrack(playlist[prevIdx]);
    }, [playlist, currentTrack?.id, playTrack]);

    // Smart Prev or Restart (if time > 3s restarts current track, else jumps to prev)
    const prevTrackOrRestart = useCallback(() => {
        if (currentTime > 3.0) {
            seek(0);
        } else {
            prevTrack();
        }
    }, [currentTime, seek, prevTrack]);

    // Shuffle Toggle
    const toggleShuffle = useCallback(() => {
        setIsShuffle((prev) => !prev);
    }, []);

    // Queue Management Methods
    const addToQueue = useCallback((track: Job) => {
        setPlaylist((prev) => {
            if (prev.some((p) => p.id === track.id)) return prev;
            return [...prev, track];
        });
    }, []);

    const removeFromQueue = useCallback((trackId: string) => {
        setPlaylist((prev) => prev.filter((p) => p.id !== trackId));
    }, []);

    const clearQueue = useCallback(() => {
        setPlaylist(currentTrack ? [currentTrack] : []);
    }, [currentTrack]);

    const reorderQueue = useCallback((fromIndex: number, toIndex: number) => {
        setPlaylist((prev) => {
            const copy = [...prev];
            const [moved] = copy.splice(fromIndex, 1);
            copy.splice(toIndex, 0, moved);
            return copy;
        });
    }, []);

    // Track ended handler
    const handleEnded = useCallback(() => {
        setIsBuffering(false);
        if (repeatMode === 'one') {
            if (audioRef.current) {
                audioRef.current.currentTime = 0;
                audioRef.current.play().catch(console.warn);
            }
        } else if (repeatMode === 'all') {
            nextTrack();
        } else {
            const curIdx = playlist.findIndex((s) => s.id === currentTrack?.id);
            if (curIdx !== -1 && curIdx < playlist.length - 1) {
                nextTrack();
            } else {
                setIsPlaying(false);
                setCurrentTime(0);
            }
        }
    }, [repeatMode, nextTrack, playlist, currentTrack?.id]);

    // Playhead tracking for karaoke & visualizer sync. Runs at strictly ~30fps (33ms cap):
    // Prevents 120Hz/144Hz high-refresh displays from triggering 120 setState calls/sec,
    // which would re-render every context consumer twice or four times as often.
    useEffect(() => {
        if (!isPlaying) return;

        let animFrameId: number;
        let lastTime = 0;
        const PLAYHEAD_INTERVAL_MS = 33;

        const tick = (now: number) => {
            if (document.hidden) {
                animFrameId = requestAnimationFrame(tick);
                return;
            }
            if (now - lastTime >= PLAYHEAD_INTERVAL_MS) {
                lastTime = now;
                if (audioRef.current && !audioRef.current.paused) {
                    setCurrentTime(audioRef.current.currentTime);
                }
            }
            animFrameId = requestAnimationFrame(tick);
        };

        animFrameId = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(animFrameId);
    }, [isPlaying]);

    // Time update handler (heartbeat fallback)
    const handleTimeUpdate = useCallback(() => {
        if (audioRef.current) {
            setCurrentTime(audioRef.current.currentTime);
        }
    }, []);

    // Metadata loaded handler
    const handleLoadedMetadata = useCallback(() => {
        if (audioRef.current && audioRef.current.duration) {
            setDuration(audioRef.current.duration);
        }
        // Apply a queued start-position seek once the element can accept one.
        const pending = pendingSeekRef.current;
        if (pending && audioRef.current) {
            audioRef.current.currentTime = Math.max(0, Math.min(audioRef.current.duration || 0, pending.time));
            setCurrentTime(audioRef.current.currentTime);
            pendingSeekRef.current = null;
        }
    }, []);

    const currentTimeRef = useRef(currentTime);
    useEffect(() => { currentTimeRef.current = currentTime; }, [currentTime]);

    const durationRef = useRef(duration);
    useEffect(() => { durationRef.current = duration; }, [duration]);

    const volumeRef = useRef(volume);
    useEffect(() => { volumeRef.current = volume; }, [volume]);

    useEffect(() => { currentTrackRef.current = currentTrack; }, [currentTrack]);

    // MediaSession API Integration (Hardware keys, Lockscreen, Control Center)
    useEffect(() => {
        if (!('mediaSession' in navigator) || !currentTrack) return;

        const artworkUrl = currentTrack.cover_image_path
            ? api.getAudioUrl(currentTrack.cover_image_path)
            : `${window.location.origin}/milimo_logo.png`;

        navigator.mediaSession.metadata = new MediaMetadata({
            title: currentTrack.title || 'Untitled Generation',
            artist: currentTrack.tags || 'Milimo Music AI',
            album: 'Milimo Studio Productions',
            artwork: [
                { src: artworkUrl, sizes: '96x96', type: 'image/png' },
                { src: artworkUrl, sizes: '256x256', type: 'image/png' },
                { src: artworkUrl, sizes: '512x512', type: 'image/png' }
            ]
        });

        navigator.mediaSession.playbackState = isPlaying ? 'playing' : 'paused';

        navigator.mediaSession.setActionHandler('play', () => resume());
        navigator.mediaSession.setActionHandler('pause', () => pause());
        navigator.mediaSession.setActionHandler('previoustrack', () => prevTrackOrRestart());
        navigator.mediaSession.setActionHandler('nexttrack', () => nextTrack());
        navigator.mediaSession.setActionHandler('seekto', (details) => {
            if (details.seekTime !== undefined) seek(details.seekTime);
        });
        navigator.mediaSession.setActionHandler('seekbackward', (details) => {
            seek(Math.max(0, currentTimeRef.current - (details.seekOffset || 10)));
        });
        navigator.mediaSession.setActionHandler('seekforward', (details) => {
            seek(Math.min(durationRef.current, currentTimeRef.current + (details.seekOffset || 10)));
        });

        return () => {
            if ('mediaSession' in navigator) {
                navigator.mediaSession.setActionHandler('play', null);
                navigator.mediaSession.setActionHandler('pause', null);
                navigator.mediaSession.setActionHandler('previoustrack', null);
                navigator.mediaSession.setActionHandler('nexttrack', null);
                navigator.mediaSession.setActionHandler('seekto', null);
                navigator.mediaSession.setActionHandler('seekbackward', null);
                navigator.mediaSession.setActionHandler('seekforward', null);
            }
        };
    }, [currentTrack, isPlaying, resume, pause, prevTrackOrRestart, nextTrack, seek]);

    // Global Keyboard Hotkeys Listener
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (hasModifier(e)) return;
            const target = e.target as HTMLElement | null;
            if (isTextEntryTarget(target)) return;
            if ((e.code === 'Space' || e.code === 'Enter') && target?.tagName === 'BUTTON') return;
            if (consumeHotkey(e)) return;
            if (!currentTrackRef.current) return;

            if (e.code === 'Space' || e.code === 'KeyK') {
                e.preventDefault();
                togglePlay();
            } else if (e.code === 'ArrowLeft' || e.code === 'KeyJ') {
                e.preventDefault();
                seek(Math.max(0, currentTimeRef.current - (e.shiftKey ? 10 : 5)));
            } else if (e.code === 'ArrowRight' || e.code === 'KeyL') {
                e.preventDefault();
                seek(Math.min(durationRef.current, currentTimeRef.current + (e.shiftKey ? 10 : 5)));
            } else if (e.code === 'Home' || e.code === 'Digit0') {
                e.preventDefault();
                returnToStart();
            } else if (e.code === 'BracketLeft') {
                e.preventDefault();
                prevTrackOrRestart();
            } else if (e.code === 'BracketRight') {
                e.preventDefault();
                nextTrack();
            } else if (e.code === 'ArrowUp') {
                e.preventDefault();
                setVolume(volumeRef.current + 0.05);
            } else if (e.code === 'ArrowDown') {
                e.preventDefault();
                setVolume(volumeRef.current - 0.05);
            } else if (e.code === 'KeyM') {
                e.preventDefault();
                toggleMute();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [togglePlay, seek, returnToStart, prevTrackOrRestart, nextTrack, setVolume, toggleMute]);

    const controlsValue = useMemo<AudioControlsContextValue>(() => ({
        currentTrack,
        isPlaying,
        volume,
        isMuted,
        playbackRate,
        repeatMode,
        isShuffle,
        playlist,
        analyserNode,
        playTrack,
        resume,
        pause,
        togglePlay,
        seek,
        returnToStart,
        prevTrackOrRestart,
        setVolume,
        toggleMute,
        setPlaybackRate,
        setRepeatMode,
        toggleShuffle,
        nextTrack,
        prevTrack,
        stop,
        setPlaylist,
        addToQueue,
        removeFromQueue,
        clearQueue,
        reorderQueue,
        playbackError,
        clearPlaybackError,
        swapCurrentTrack,
    }), [
        currentTrack,
        isPlaying,
        volume,
        isMuted,
        playbackRate,
        repeatMode,
        isShuffle,
        playlist,
        analyserNode,
        playTrack,
        resume,
        pause,
        togglePlay,
        seek,
        returnToStart,
        prevTrackOrRestart,
        setVolume,
        toggleMute,
        setPlaybackRate,
        setRepeatMode,
        toggleShuffle,
        nextTrack,
        prevTrack,
        stop,
        setPlaylist,
        addToQueue,
        removeFromQueue,
        clearQueue,
        reorderQueue,
        playbackError,
        clearPlaybackError,
        swapCurrentTrack,
    ]);

    const timeValue = useMemo<AudioTimeContextValue>(() => ({
        currentTime,
        duration,
        isBuffering
    }), [currentTime, duration, isBuffering]);

    const engineValue = useMemo<AudioEngineContextValue>(() => ({
        ...controlsValue,
        ...timeValue
    }), [controlsValue, timeValue]);

    return (
        <AudioControlsContext.Provider value={controlsValue}>
            <AudioTimeContext.Provider value={timeValue}>
                <AudioEngineContext.Provider value={engineValue}>
                    {/* Single Root Master <audio> element */}
                    <audio
                        ref={audioRef}
                        crossOrigin="anonymous"
                        onTimeUpdate={handleTimeUpdate}
                        onLoadedMetadata={handleLoadedMetadata}
                        onEnded={handleEnded}
                        onError={handleMediaError}
                        onStalled={handleStalled}
                        onWaiting={handleWaiting}
                        onPlaying={handlePlaying}
                        onCanPlay={handleCanPlay}
                        preload="auto"
                    />
                    {children}
                </AudioEngineContext.Provider>
            </AudioTimeContext.Provider>
        </AudioControlsContext.Provider>
    );
};

export const useAudioControls = () => {
    const ctx = useContext(AudioControlsContext);
    if (!ctx) {
        throw new Error('useAudioControls must be used within an AudioEngineProvider');
    }
    return ctx;
};

export const useAudioTime = () => {
    const ctx = useContext(AudioTimeContext);
    if (!ctx) {
        throw new Error('useAudioTime must be used within an AudioEngineProvider');
    }
    return ctx;
};

export const useAudioEngine = () => {
    const ctx = useContext(AudioEngineContext);
    if (!ctx) {
        throw new Error('useAudioEngine must be used within an AudioEngineProvider');
    }
    return ctx;
};
