import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { Play, Pause, Download, SkipBack, SkipForward, Volume2, VolumeX } from 'lucide-react';
import { GlassCard } from './ui/GlassCard';
import { api } from '../api';
import { AudioVisualizer } from './ui/AudioVisualizer';

interface AudioPlayerProps {
    audioUrl: string;
    jobId: string; // Added for download link
    className?: string;
    onNext?: () => void;
    onPrev?: () => void;
    title?: string;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({
    audioUrl,
    jobId,
    className,
    onNext,
    onPrev,
    title
}) => {
    // ... refs and state ...
    const containerRef = useRef<HTMLDivElement>(null);
    const wavesurfer = useRef<WaveSurfer | null>(null);
    const [mediaEl, setMediaEl] = useState<HTMLMediaElement | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [duration, setDuration] = useState('0:00');
    const [currentTime, setCurrentTime] = useState('0:00');

    // ... volume state ...
    // Volume Persistence
    const [volume, setVolume] = useState(() => {
        const saved = localStorage.getItem('milimo_volume');
        return saved ? parseFloat(saved) : 0.7;
    });
    const [isMuted, setIsMuted] = useState(false);

    // ... formatTime ...
    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    // ... useEffects ...
    useEffect(() => {
        if (!containerRef.current) return;

        wavesurfer.current = WaveSurfer.create({
            container: containerRef.current,
            waveColor: 'rgba(99, 102, 241, 0.4)',
            progressColor: '#6366f1',
            cursorColor: '#ff0000',
            barWidth: 3,
            barGap: 2,
            barRadius: 3,
            height: 60,
            normalize: true,
            backend: 'MediaElement',
        });

        wavesurfer.current.on('ready', () => {
            setDuration(formatTime(wavesurfer.current?.getDuration() || 0));
            wavesurfer.current?.setVolume(isMuted ? 0 : volume);

            // Extract media element for visualizer
            const media = wavesurfer.current?.getMediaElement();
            if (media) {
                // Use a proxy needed? Wavesurfer 7 exposes it directly.
                // But we need to make sure we don't break WS's internal connections.
                // We'll pass it to visualizer but let visualizer try-catch the createSource.
                setMediaEl(media);
            }
        });

        wavesurfer.current.on('audioprocess', () => {
            setCurrentTime(formatTime(wavesurfer.current?.getCurrentTime() || 0));
        });

        wavesurfer.current.on('finish', () => {
            setIsPlaying(false);
        });

        return () => {
            try {
                wavesurfer.current?.destroy();
            } catch (e) {
                // Ignore AbortError during cleanup
            }
        };
    }, []);

    useEffect(() => {
        if (wavesurfer.current && audioUrl) {
            const fullUrl = api.getAudioUrl(audioUrl);
            wavesurfer.current.load(fullUrl);
            setIsPlaying(false);
        }
    }, [audioUrl]);

    // Handle Global Pause Event
    useEffect(() => {
        const handleGlobalPlay = (e: CustomEvent) => {
            // If another player starts (id !== my id), pause myself
            if (e.detail.id !== jobId && isPlaying) {
                if (wavesurfer.current) {
                    wavesurfer.current.pause();
                    setIsPlaying(false);
                }
            }
        };

        window.addEventListener('milimo_play_start' as any, handleGlobalPlay as any);
        return () => {
            window.removeEventListener('milimo_play_start' as any, handleGlobalPlay as any);
        };
    }, [jobId, isPlaying]);

    const togglePlay = () => {
        if (wavesurfer.current) {
            wavesurfer.current.playPause();
            const newIsPlaying = !isPlaying;
            setIsPlaying(newIsPlaying);

            if (newIsPlaying) {
                // Broadcast I am playing
                const event = new CustomEvent('milimo_play_start', { detail: { id: jobId } });
                window.dispatchEvent(event);
            }
        }
    };

    const downloadAudio = () => {
        // Direct Server Download (Fixes Safari Blob Issues)
        const downloadUrl = api.getDownloadUrl(jobId);
        window.location.href = downloadUrl;
    };

    return (
        <div className={className}>
            <GlassCard className="py-4 px-6 !bg-white/60 !backdrop-blur-2xl">
                <div className="flex flex-col gap-4">
                    {title && <h3 className="text-sm font-medium text-slate-500 uppercase tracking-widest">{title}</h3>}

                    {/* Visualizer Overlay or Parallel */}
                    <div className="relative">
                        <div ref={containerRef} className="w-full cursor-pointer opacity-80 hover:opacity-100 transition-opacity" />
                        {/* Visualizer positioned absolutly or effectively blended? 
                             Let's put it BEHIND or blended. 
                             Actually, let's put it on top as a subtle overlay or below.
                             Putting it below standard waveform for now.
                          */}
                        {isPlaying && (
                            <div className="absolute top-0 left-0 w-full h-full pointer-events-none mix-blend-overlay opacity-50 flex items-center justify-center">
                                <AudioVisualizer
                                    mediaElement={mediaEl}
                                    isPlaying={isPlaying}
                                    className="w-full h-full"
                                    barColor="rgb(255, 255, 255)"
                                />
                            </div>
                        )}
                    </div>

                    <div className="flex items-center justify-between mt-2">
                        <span className="text-xs font-mono text-slate-500 w-12">{currentTime}</span>

                        <div className="flex items-center gap-4">
                            <button onClick={onPrev} className="p-2 rounded-full hover:bg-slate-100/50 transition-colors text-slate-600">
                                <SkipBack className="w-5 h-5" />
                            </button>

                            <button
                                onClick={togglePlay}
                                className="p-4 rounded-full bg-slate-900 text-white shadow-lg hover:scale-105 transition-transform active:scale-95"
                            >
                                {isPlaying ? <Pause className="fill-current w-6 h-6" /> : <Play className="fill-current w-6 h-6 pl-1" />}
                            </button>

                            <button onClick={onNext} className="p-2 rounded-full hover:bg-slate-100/50 transition-colors text-slate-600">
                                <SkipForward className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="flex items-center gap-3">
                            <div className="flex items-center gap-2 group/volume">
                                <button
                                    onClick={() => {
                                        const newMuted = !isMuted;
                                        setIsMuted(newMuted);
                                        if (wavesurfer.current) {
                                            wavesurfer.current.setVolume(newMuted ? 0 : volume);
                                        }
                                    }}
                                    className="text-slate-400 hover:text-slate-600 transition-colors"
                                >
                                    {isMuted || volume === 0 ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                                </button>
                                <div className="w-0 overflow-hidden group-hover/volume:w-20 transition-all duration-300">
                                    <input
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.05"
                                        value={isMuted ? 0 : volume}
                                        onChange={(e) => {
                                            const val = parseFloat(e.target.value);
                                            setVolume(val);
                                            setIsMuted(val === 0);
                                            if (wavesurfer.current) wavesurfer.current.setVolume(val);
                                            localStorage.setItem('milimo_volume', val.toString());
                                        }}
                                        className="w-20 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-slate-500"
                                    />
                                </div>
                            </div>

                            <button onClick={downloadAudio} className="p-2 rounded-full hover:bg-slate-100/50 transition-colors text-slate-600">
                                <Download className="w-4 h-4" />
                            </button>
                            <div className="w-12 text-right text-xs font-mono text-slate-500">{duration}</div>
                        </div>
                    </div>
                </div>
            </GlassCard>
        </div>
    );
};
