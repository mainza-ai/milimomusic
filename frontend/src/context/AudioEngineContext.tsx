import React, { createContext, useContext, useState, useRef, useEffect, useCallback } from 'react';
import type { Job } from '../api';
import { API_BASE_URL } from '../api';

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
    playTrack: (track: Job, customPlaylist?: Job[]) => Promise<void>;
    pause: () => void;
    resume: () => Promise<void>;
    togglePlay: (track?: Job) => void;
    seek: (timeInSeconds: number) => void;
    setVolume: (vol: number) => void;
    toggleMute: () => void;
    setPlaybackRate: (rate: number) => void;
    setRepeatMode: (mode: 'off' | 'all' | 'one') => void;
    toggleShuffle: () => void;
    nextTrack: () => void;
    prevTrack: () => void;
    stop: () => void;
    setPlaylist: (list: Job[]) => void;
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

    // Get full absolute URL for audio
    const getAudioUrl = useCallback((path?: string | null): string => {
        if (!path) return '';
        if (path.startsWith('http://') || path.startsWith('https://')) return path;
        return `${API_BASE_URL}${path}`;
    }, []);

    // Set Volume Helper
    const setVolume = useCallback((val: number) => {
        const clamped = Math.max(0, Math.min(1, val));
        setVolumeState(clamped);
        localStorage.setItem('milimo_volume', clamped.toString());
        if (audioRef.current) {
            audioRef.current.volume = isMuted ? 0 : clamped;
        }
    }, [isMuted]);

    // Toggle Mute Helper
    const toggleMute = useCallback(() => {
        setIsMuted((prev) => {
            const next = !prev;
            if (audioRef.current) {
                audioRef.current.volume = next ? 0 : volume;
            }
            return next;
        });
    }, [volume]);

    // Set Playback Speed Helper
    const setPlaybackRate = useCallback((rate: number) => {
        setPlaybackRateState(rate);
        if (audioRef.current) {
            audioRef.current.playbackRate = rate;
        }
    }, []);

    // Play specific track
    const playTrack = useCallback(async (track: Job, customPlaylist?: Job[]) => {
        if (!track.audio_path) return;

        if (customPlaylist && customPlaylist.length > 0) {
            setPlaylist(customPlaylist);
        }

        const audio = audioRef.current;
        if (!audio) return;

        const targetUrl = getAudioUrl(track.audio_path);

        // If selecting a new track
        if (currentTrack?.id !== track.id || audio.src !== targetUrl) {
            setCurrentTrack(track);
            setCurrentTime(0);
            setDuration(track.duration_ms ? track.duration_ms / 1000 : 0);
            audio.src = targetUrl;
            audio.playbackRate = playbackRate;
            audio.volume = isMuted ? 0 : volume;
        }

        try {
            await audio.play();
            setIsPlaying(true);
        } catch (err: any) {
            if (err.name !== 'AbortError') {
                console.warn('Audio playback error:', err);
            }
        }
    }, [currentTrack?.id, getAudioUrl, playbackRate, isMuted, volume]);

    // Pause
    const pause = useCallback(() => {
        if (audioRef.current) {
            audioRef.current.pause();
        }
        setIsPlaying(false);
    }, []);

    // Resume
    const resume = useCallback(async () => {
        const audio = audioRef.current;
        if (!audio || !currentTrack) return;
        try {
            await audio.play();
            setIsPlaying(true);
        } catch (err: any) {
            if (err.name !== 'AbortError') {
                console.warn('Audio resume error:', err);
            }
        }
    }, [currentTrack]);

    // Toggle Play
    const togglePlay = useCallback((track?: Job) => {
        if (track && track.id !== currentTrack?.id) {
            playTrack(track);
            return;
        }

        if (isPlaying) {
            pause();
        } else {
            resume();
        }
    }, [currentTrack?.id, isPlaying, pause, playTrack, resume]);

    // Seek
    const seek = useCallback((timeInSeconds: number) => {
        if (audioRef.current) {
            audioRef.current.currentTime = timeInSeconds;
            setCurrentTime(timeInSeconds);
        }
    }, []);

    // Stop
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

    // Shuffle Toggle
    const toggleShuffle = useCallback(() => {
        setIsShuffle((prev) => !prev);
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

    // Time update handler
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

    // Sync volume when changed
    useEffect(() => {
        if (audioRef.current) {
            audioRef.current.volume = isMuted ? 0 : volume;
        }
    }, [volume, isMuted]);

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
                playTrack,
                pause,
                resume,
                togglePlay,
                seek,
                setVolume,
                toggleMute,
                setPlaybackRate,
                setRepeatMode,
                toggleShuffle,
                nextTrack,
                prevTrack,
                stop,
                setPlaylist
            }}
        >
            {/* Single Root Master Audio Node */}
            <audio
                ref={audioRef}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onEnded={handleEnded}
                preload="auto"
            />
            {children}
        </AudioEngineContext.Provider>
    );
};

export const useAudioEngine = (): AudioEngineContextValue => {
    const context = useContext(AudioEngineContext);
    if (!context) {
        throw new Error('useAudioEngine must be used within an AudioEngineProvider');
    }
    return context;
};
