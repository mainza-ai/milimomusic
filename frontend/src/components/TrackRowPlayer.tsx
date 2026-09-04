import React, { useState } from 'react';
import { Play, Pause, Download, Wand2 } from 'lucide-react';
import { useAudioEngine } from '../context/AudioEngineContext';
import { API_BASE_URL } from '../api';
import { StaticWaveform } from './ui/StaticWaveform';
import { InpaintModal } from './InpaintModal';
import type { Job } from '../api';

interface TrackRowPlayerProps {
    job: Job;
}

const formatTime = (seconds: number) => {
    if (isNaN(seconds) || seconds < 0) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
};

/**
 * Library-row playback control.
 *
 * Production model: rows own NO audio engine. There is exactly one playback
 * source of truth in the app — the global engine (dock player). A row's play
 * button drives it; its waveform is a server-computed static envelope
 * (a few KB) instead of a full audio download + WaveSurfer decode per card.
 *
 * This removes N simultaneous MediaElement decoders and fixes the old
 * inconsistency where a card could play OVER the dock player.
 */
export const TrackRowPlayer: React.FC<TrackRowPlayerProps> = ({ job }) => {
    const {
        currentTrack,
        isPlaying,
        currentTime: engineTime,
        duration: engineDuration,
        playTrack,
        togglePlay,
        seek
    } = useAudioEngine();
    const [isInpaintOpen, setIsInpaintOpen] = useState(false);

    const isActive = currentTrack?.id === job.id;
    const durationSec = job.duration_ms ? job.duration_ms / 1000 : 0;
    const progressFraction = isActive && engineDuration > 0
        ? Math.min(1, engineTime / engineDuration)
        : null;

    const handleTogglePlay = () => {
        if (isActive) {
            togglePlay();
        } else {
            void playTrack(job);
        }
    };

    const handleSeekFraction = (fraction: number) => {
        const target = fraction * durationSec;
        if (isActive) {
            seek(target);
        } else {
            // Start this track directly at the clicked position.
            void playTrack(job, undefined, target);
        }
    };

    const downloadAudio = () => {
        if (!job.audio_path) return;
        const fullUrl = job.audio_path.startsWith('http') ? job.audio_path : `${API_BASE_URL}${job.audio_path}`;
        const a = document.createElement('a');
        a.href = fullUrl;
        a.download = `milimo-track-${job.id}.wav`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    return (
        <div className="space-y-2">
            {/* Waveform + overlay transport */}
            <div className="flex items-center gap-3">
                <button
                    onClick={handleTogglePlay}
                    className={`w-10 h-10 rounded-2xl flex-shrink-0 flex items-center justify-center font-bold transition-all active:scale-95 shadow-md ${
                        isActive && isPlaying
                            ? 'bg-slate-900 dark:bg-white text-white dark:text-slate-900'
                            : 'bg-gradient-to-tr from-teal-500 via-cyan-400 to-sky-500 hover:from-teal-400 hover:to-cyan-300 text-slate-950 shadow-teal-500/25'
                    }`}
                    title={isActive && isPlaying ? 'Pause' : 'Play'}
                    aria-label={isActive && isPlaying ? `Pause ${job.title || 'track'}` : `Play ${job.title || 'track'}`}
                >
                    {isActive && isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
                </button>

                <StaticWaveform
                    jobId={job.id}
                    progressFraction={progressFraction}
                    onSeekFraction={handleSeekFraction}
                    heightClass="h-12"
                    className="flex-1"
                />

                <span className="text-xs font-mono font-semibold text-slate-500 dark:text-slate-400 w-20 text-right flex-shrink-0 tabular-nums">
                    {isActive ? formatTime(engineTime) : '0:00'} / {formatTime(durationSec)}
                </span>
            </div>

            {/* Row tools */}
            <div className="flex items-center justify-end gap-1.5 pt-0.5 border-t border-black/[0.04] dark:border-white/5">
                <button
                    onClick={() => setIsInpaintOpen(true)}
                    disabled={!job.audio_path}
                    className="p-1.5 rounded-lg bg-teal-500/10 hover:bg-teal-500 text-teal-700 dark:text-teal-300 hover:text-slate-950 transition-colors disabled:opacity-40 disabled:pointer-events-none"
                    title="Repair Audio Segment"
                >
                    <Wand2 className="w-3.5 h-3.5" />
                </button>

                <button
                    onClick={downloadAudio}
                    disabled={!job.audio_path}
                    className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 transition-colors text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 disabled:opacity-40 disabled:pointer-events-none"
                    title="Download Audio"
                >
                    <Download className="w-3.5 h-3.5" />
                </button>
            </div>

            <InpaintModal
                isOpen={isInpaintOpen}
                onClose={() => setIsInpaintOpen(false)}
                jobId={job.id}
                duration={durationSec}
                title={job.title || undefined}
            />
        </div>
    );
};
