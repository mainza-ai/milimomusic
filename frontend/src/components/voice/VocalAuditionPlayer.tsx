import React, { useState, useRef, useEffect } from 'react';
import {
    Play,
    Pause,
    RotateCcw,
    RotateCw,
    Volume2,
    VolumeX,
    Download,
    Sliders,
    CheckCircle2
} from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';
import { api, type Job } from '../../api';

interface VocalAuditionPlayerProps {
    track: Job | null;
    originalVocalUrl?: string;
    convertedVocalUrl?: string;
    remixedMasterUrl?: string;
    onOpenInDAW?: (track: Job) => void;
    onCommitVocal?: () => void;
}

export type AuditionSource = 'converted' | 'original' | 'master';

export const VocalAuditionPlayer: React.FC<VocalAuditionPlayerProps> = ({
    track,
    originalVocalUrl,
    convertedVocalUrl,
    remixedMasterUrl,
    onOpenInDAW,
    onCommitVocal
}) => {
    const [selectedSource, setSelectedSource] = useState<AuditionSource>('converted');
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [volume, setVolume] = useState(0.85);
    const [isMuted, setIsMuted] = useState(false);

    const audioRef = useRef<HTMLAudioElement | null>(null);

    // Resolve current active audio URL
    const getActiveUrl = (): string | null => {
        if (selectedSource === 'converted' && convertedVocalUrl) {
            return api.getAudioUrl(convertedVocalUrl);
        }
        if (selectedSource === 'original' && originalVocalUrl) {
            return api.getAudioUrl(originalVocalUrl);
        }
        if (selectedSource === 'master') {
            const master = remixedMasterUrl || track?.audio_path;
            return master ? api.getAudioUrl(master) : null;
        }
        return convertedVocalUrl
            ? api.getAudioUrl(convertedVocalUrl)
            : originalVocalUrl
            ? api.getAudioUrl(originalVocalUrl)
            : track?.audio_path
            ? api.getAudioUrl(track.audio_path)
            : null;
    };

    const activeUrl = getActiveUrl();

    // Auto-switch to converted when newly available
    useEffect(() => {
        if (convertedVocalUrl) {
            setSelectedSource('converted');
        }
    }, [convertedVocalUrl]);

    // Update audio source when activeUrl changes, preserving playhead position if applicable
    useEffect(() => {
        if (!audioRef.current || !activeUrl) return;
        const wasPlaying = isPlaying;
        const prevTime = audioRef.current.currentTime;

        audioRef.current.src = activeUrl;
        audioRef.current.load();

        if (prevTime > 0) {
            audioRef.current.currentTime = prevTime;
        }

        if (wasPlaying) {
            window.dispatchEvent(new CustomEvent('milimo:audio-play', { detail: { source: 'vocal-audition' } }));
            audioRef.current.play().catch(() => setIsPlaying(false));
        }
    }, [activeUrl]);

    // External audio bus listener
    useEffect(() => {
        const handleExternalAudioPlay = (e: Event) => {
            const customEvent = e as CustomEvent<{ source?: string }>;
            if (customEvent.detail?.source && customEvent.detail.source !== 'vocal-audition') {
                if (audioRef.current && !audioRef.current.paused) {
                    audioRef.current.pause();
                    setIsPlaying(false);
                }
            }
        };

        window.addEventListener('milimo:audio-play', handleExternalAudioPlay);
        return () => {
            window.removeEventListener('milimo:audio-play', handleExternalAudioPlay);
        };
    }, []);

    const togglePlay = () => {
        if (!audioRef.current || !activeUrl) return;
        if (isPlaying) {
            audioRef.current.pause();
            setIsPlaying(false);
        } else {
            window.dispatchEvent(new CustomEvent('milimo:audio-play', { detail: { source: 'vocal-audition' } }));
            audioRef.current.play().then(() => setIsPlaying(true)).catch(() => setIsPlaying(false));
        }
    };

    const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
        const time = parseFloat(e.target.value);
        setCurrentTime(time);
        if (audioRef.current) {
            audioRef.current.currentTime = time;
        }
    };

    const handleSkip = (sec: number) => {
        if (!audioRef.current) return;
        const newTime = Math.max(0, Math.min(duration, audioRef.current.currentTime + sec));
        audioRef.current.currentTime = newTime;
        setCurrentTime(newTime);
    };

    const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const val = parseFloat(e.target.value);
        setVolume(val);
        if (audioRef.current) {
            audioRef.current.volume = val;
            setIsMuted(val === 0);
        }
    };

    const toggleMute = () => {
        if (!audioRef.current) return;
        if (isMuted) {
            audioRef.current.volume = volume || 0.8;
            setIsMuted(false);
        } else {
            audioRef.current.volume = 0;
            setIsMuted(true);
        }
    };

    const formatTime = (timeInSec: number) => {
        if (isNaN(timeInSec)) return '0:00';
        const m = Math.floor(timeInSec / 60);
        const s = Math.floor(timeInSec % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    };

    const handleDownloadCurrent = () => {
        if (!activeUrl) return;
        const suffix = selectedSource === 'converted' ? 'converted_vocal' : selectedSource === 'original' ? 'dry_vocal' : 'master_remix';
        api.downloadUrlAsFile(activeUrl, `${track?.title || 'vocal'}_${suffix}.wav`);
    };

    return (
        <GlassCard className="p-5 border border-black/[0.08] dark:border-white/10 space-y-4">
            <audio
                ref={audioRef}
                onTimeUpdate={() => audioRef.current && setCurrentTime(audioRef.current.currentTime)}
                onLoadedMetadata={() => audioRef.current && setDuration(audioRef.current.duration)}
                onEnded={() => setIsPlaying(false)}
            />

            {/* Top Bar: Tri-State Mode Selector & Action Buttons */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-black/[0.06] dark:border-white/10">
                <div className="flex items-center gap-1.5 p-1 rounded-xl bg-black/[0.04] dark:bg-white/5 border border-black/[0.06] dark:border-white/10">
                    <button
                        type="button"
                        onClick={() => setSelectedSource('converted')}
                        disabled={!convertedVocalUrl}
                        className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                            selectedSource === 'converted'
                                ? 'bg-teal-500 text-slate-950 shadow-sm'
                                : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white disabled:opacity-40'
                        }`}
                    >
                        <span>Converted Vocal</span>
                        {convertedVocalUrl && (
                            <span className="w-1.5 h-1.5 rounded-full bg-teal-950 animate-pulse" />
                        )}
                    </button>

                    <button
                        type="button"
                        onClick={() => setSelectedSource('original')}
                        disabled={!originalVocalUrl}
                        className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                            selectedSource === 'original'
                                ? 'bg-teal-500 text-slate-950 shadow-sm'
                                : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white disabled:opacity-40'
                        }`}
                    >
                        <span>Original Stem</span>
                    </button>

                    <button
                        type="button"
                        onClick={() => setSelectedSource('master')}
                        disabled={!remixedMasterUrl && !track?.audio_path}
                        className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                            selectedSource === 'master'
                                ? 'bg-teal-500 text-slate-950 shadow-sm'
                                : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white disabled:opacity-40'
                        }`}
                    >
                        <span>Full Master Mix</span>
                    </button>
                </div>

                <div className="flex items-center gap-2">
                    <button
                        type="button"
                        onClick={handleDownloadCurrent}
                        disabled={!activeUrl}
                        className="px-3 py-1.5 rounded-lg bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 text-xs font-bold flex items-center gap-1.5 transition-colors disabled:opacity-40"
                    >
                        <Download size={13} />
                        <span>Export WAV</span>
                    </button>

                    {onCommitVocal && convertedVocalUrl && (
                        <button
                            type="button"
                            onClick={onCommitVocal}
                            className="px-3.5 py-1.5 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 text-xs font-bold flex items-center gap-1.5 transition-colors shadow-sm"
                            title="Replace the vocal stem on the source track with this converted take"
                        >
                            <CheckCircle2 size={13} />
                            <span>Commit to Track</span>
                        </button>
                    )}

                    {onOpenInDAW && track && (
                        <button
                            type="button"
                            onClick={() => onOpenInDAW(track)}
                            className="px-3.5 py-1.5 rounded-lg bg-teal-500/15 text-teal-700 dark:text-teal-400 hover:bg-teal-500/25 text-xs font-bold flex items-center gap-1.5 transition-colors border border-teal-500/20"
                        >
                            <Sliders size={13} />
                            <span>Open in DAW Workspace</span>
                        </button>
                    )}
                </div>
            </div>

            {/* Scrub Bar & Waveform Simulation */}
            <div className="space-y-1.5">
                <input
                    type="range"
                    min="0"
                    max={duration || 100}
                    step="0.1"
                    value={currentTime}
                    onChange={handleSeek}
                    className="w-full accent-teal-500 cursor-pointer h-2 rounded-lg bg-black/10 dark:bg-white/10"
                />
                <div className="flex justify-between text-[11px] font-mono font-medium text-slate-500 dark:text-slate-400">
                    <span>{formatTime(currentTime)}</span>
                    <span className="text-slate-700 dark:text-slate-300 font-bold uppercase tracking-wider">
                        Auditioning: {selectedSource.toUpperCase()}
                    </span>
                    <span>{formatTime(duration)}</span>
                </div>
            </div>

            {/* Transport & Volume Controls */}
            <div className="flex items-center justify-between pt-1">
                <div className="flex items-center space-x-2">
                    <button
                        type="button"
                        onClick={() => handleSkip(-10)}
                        className="p-2 rounded-lg text-slate-500 hover:text-slate-900 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
                        title="Rewind 10s"
                    >
                        <RotateCcw size={16} />
                    </button>

                    <button
                        type="button"
                        onClick={togglePlay}
                        disabled={!activeUrl}
                        className="w-11 h-11 rounded-2xl bg-gradient-to-r from-teal-500 to-cyan-500 text-slate-950 flex items-center justify-center shadow-md shadow-teal-500/25 active:scale-95 transition-all disabled:opacity-40"
                    >
                        {isPlaying ? <Pause size={20} /> : <Play size={20} className="ml-0.5" />}
                    </button>

                    <button
                        type="button"
                        onClick={() => handleSkip(10)}
                        className="p-2 rounded-lg text-slate-500 hover:text-slate-900 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
                        title="Fast Forward 10s"
                    >
                        <RotateCw size={16} />
                    </button>
                </div>

                {/* Volume Slider */}
                <div className="flex items-center space-x-2 w-36">
                    <button
                        type="button"
                        onClick={toggleMute}
                        className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
                    >
                        {isMuted || volume === 0 ? <VolumeX size={15} /> : <Volume2 size={15} />}
                    </button>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={isMuted ? 0 : volume}
                        onChange={handleVolumeChange}
                        className="w-full accent-teal-500 cursor-pointer h-1.5 rounded-lg bg-black/10 dark:bg-white/10"
                    />
                </div>
            </div>
        </GlassCard>
    );
};
