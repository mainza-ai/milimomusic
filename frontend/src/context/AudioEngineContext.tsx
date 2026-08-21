import React, { createContext, useContext, useState, useRef, useEffect, useCallback } from 'react';
import type { Job } from '../api';
import { API_BASE_URL } from '../api';
import { getAudioContext } from '../utils/audioContext';

export interface AudioEngineContextValue {
    currentTrack: Job | null;
    isPlaying: boolean;
    currentTime: number;
    duration: number;
    volume: number;
    isMuted: boolean;
    playbackRate: number;
    repeatMode: 'off' | 'all' | 'one';
    isShuffle: boolean;
    playlist: Job[];
    analyserNode: AnalyserNode | null;
    playTrack: (track: Job, customPlaylist?: Job[]) => Promise<void>;
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
}

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

    // Audio node connectivity tracker
    const isSourceConnected = useRef(false);

    // Full absolute URL resolver
    const getAudioUrl = useCallback((path?: string | null): string => {
        if (!path) return '';
        if (path.startsWith('http://') || path.startsWith('https://')) return path;
        return `${API_BASE_URL}${path}`;
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

    // Set Volume
    const setVolume = useCallback((val: number) => {
        const clamped = Math.max(0, Math.min(1, val));
        setVolumeState(clamped);
        localStorage.setItem('milimo_volume', clamped.toString());
        if (audioRef.current) {
            audioRef.current.volume = isMuted ? 0 : clamped;
        }
    }, [isMuted]);

    // Toggle Mute
    const toggleMute = useCallback(() => {
        setIsMuted((prev) => {
            const next = !prev;
            if (audioRef.current) {
                audioRef.current.volume = next ? 0 : volume;
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

    // Play specific track
    const playTrack = useCallback(async (track: Job, customPlaylist?: Job[]) => {
        if (!track?.audio_path) return;

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

        setCurrentTrack(track);
        setCurrentTime(0);

        if (audioRef.current) {
            ensureAudioGraph();
            const fullUrl = getAudioUrl(track.audio_path);
            if (audioRef.current.src !== fullUrl) {
                audioRef.current.src = fullUrl;
                audioRef.current.load();
            }

            try {
                await audioRef.current.play();
                setIsPlaying(true);
            } catch (err: any) {
                if (err.name !== 'AbortError') {
                    console.warn('Playback start postponed/blocked:', err);
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
        }
    }, []);

    // Resume
    const resume = useCallback(async () => {
        if (audioRef.current) {
            ensureAudioGraph();
            try {
                await audioRef.current.play();
                setIsPlaying(true);
            } catch (err: any) {
                if (err.name !== 'AbortError') {
                    console.warn('Resume error:', err);
                }
            }
        }
    }, [ensureAudioGraph]);

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

    // High-precision 60fps Playhead tracking for smooth karaoke & visualizer sync
    useEffect(() => {
        if (!isPlaying) return;

        let animFrameId: number;
        const tick = () => {
            if (audioRef.current && !audioRef.current.paused) {
                setCurrentTime(audioRef.current.currentTime);
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
    }, []);

    // OS MediaSession API Integration
    useEffect(() => {
        if (!('mediaSession' in navigator) || !currentTrack) return;

        const artworkUrl = currentTrack.cover_image_path
            ? currentTrack.cover_image_path.startsWith('http')
                ? currentTrack.cover_image_path
                : `${API_BASE_URL}${currentTrack.cover_image_path}`
            : `${window.location.origin}/milimo_logo.png`;

        navigator.mediaSession.metadata = new MediaMetadata({
            title: currentTrack.title || 'Milimo Track',
            artist: currentTrack.prompt ? currentTrack.prompt.slice(0, 40) : 'Milimo Music AI',
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
            seek(Math.max(0, currentTime - (details.seekOffset || 10)));
        });
        navigator.mediaSession.setActionHandler('seekforward', (details) => {
            seek(Math.min(duration, currentTime + (details.seekOffset || 10)));
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
    }, [currentTrack, isPlaying, currentTime, duration, resume, pause, prevTrackOrRestart, nextTrack, seek]);

    // Global Keyboard Hotkeys Listener
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Ignore keystrokes when typing into text inputs, textarea, or content editable
            const target = e.target as HTMLElement | null;
            if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable)) {
                return;
            }

            if (e.code === 'Space' || e.code === 'KeyK') {
                e.preventDefault();
                togglePlay();
            } else if (e.code === 'ArrowLeft' || e.code === 'KeyJ') {
                e.preventDefault();
                seek(Math.max(0, currentTime - (e.shiftKey ? 10 : 5)));
            } else if (e.code === 'ArrowRight' || e.code === 'KeyL') {
                e.preventDefault();
                seek(Math.min(duration, currentTime + (e.shiftKey ? 10 : 5)));
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
                setVolume(volume + 0.05);
            } else if (e.code === 'ArrowDown') {
                e.preventDefault();
                setVolume(volume - 0.05);
            } else if (e.code === 'KeyM') {
                e.preventDefault();
                toggleMute();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [togglePlay, seek, currentTime, duration, returnToStart, prevTrackOrRestart, nextTrack, setVolume, volume, toggleMute]);

    return (
        <AudioEngineContext.Provider
            value={{
                currentTrack,
                isPlaying,
                currentTime,
                duration,
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
                reorderQueue
            }}
        >
            {/* Single Root Master <audio> element */}
            <audio
                ref={audioRef}
                crossOrigin="anonymous"
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onEnded={handleEnded}
                preload="auto"
            />
            {children}
        </AudioEngineContext.Provider>
    );
};

export const useAudioEngine = () => {
    const ctx = useContext(AudioEngineContext);
    if (!ctx) {
        throw new Error('useAudioEngine must be used within an AudioEngineProvider');
    }
    return ctx;
};
