import React, { useState } from 'react';
import { Layers, ZoomIn, ZoomOut } from 'lucide-react';
import type { Job } from '../../api';
import type { StemChannel } from './SessionWorkspace';

interface ArrangeTimelineProps {
    job: Job;
    stemChannels: StemChannel[];
    currentTime: number;
    duration: number;
    onSeek: (time: number) => void;
    onToggleMute: (id: string) => void;
    onToggleSolo: (id: string) => void;
}

export const ArrangeTimeline: React.FC<ArrangeTimelineProps> = ({
    job,
    stemChannels,
    currentTime,
    duration,
    onSeek,
    onToggleMute,
    onToggleSolo
}) => {
    const [zoom, setZoom] = useState(1);
    const totalDuration = duration || 60;
    const progressPercent = Math.min(100, Math.max(0, (currentTime / totalDuration) * 100));

    const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const newTime = (clickX / rect.width) * totalDuration;
        onSeek(newTime);
    };

    // Calculate measure markers from the transcription's real beat grid
    // (BPM + beats per bar) so bar markers align with actual note timing.
    const beatGrid = job.beat_grid_json
        ? typeof job.beat_grid_json === 'string'
            ? JSON.parse(job.beat_grid_json)
            : job.beat_grid_json
        : {};
    const bpm = Number(beatGrid.bpm) > 0 ? Number(beatGrid.bpm) : 120;
    const beatsPerBar = Number(beatGrid.beats_per_bar) > 0 ? Number(beatGrid.beats_per_bar) : 4;
    const barDuration = (60 / bpm) * beatsPerBar;
    const totalBars = Math.max(1, Math.ceil(totalDuration / barDuration));
    const barsArray = Array.from({ length: totalBars }, (_, i) => i + 1);

    return (
        <div className="flex flex-col h-full bg-[#f5f5f7] dark:bg-[#10121a] text-slate-800 dark:text-slate-200 select-none overflow-hidden transition-colors duration-200">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-3 border-b border-black/[0.06] dark:border-white/[0.08] bg-white/70 dark:bg-[#141620]/80 backdrop-blur-xl">
                <div className="flex items-center space-x-2">
                    <Layers size={16} className="text-teal-600 dark:text-teal-400" />
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-900 dark:text-slate-100">
                        Stem Arrangement & Song Structure
                    </span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20 font-semibold">
                        {job.title || "Active Master"}
                    </span>
                </div>

                <div className="flex items-center space-x-3 text-xs font-mono text-slate-500 dark:text-slate-400">
                    <div className="flex items-center bg-black/[0.04] dark:bg-[#181a24] border border-black/[0.06] dark:border-white/10 rounded-xl p-1 space-x-1">
                        <button
                            onClick={() => setZoom(prev => Math.max(0.5, prev - 0.25))}
                            className="p-1 text-slate-500 hover:text-slate-900 dark:hover:text-slate-200"
                            title="Zoom Out"
                        >
                            <ZoomOut size={12} />
                        </button>
                        <span className="text-[10px] font-mono px-1.5">{Math.round(zoom * 100)}%</span>
                        <button
                            onClick={() => setZoom(prev => Math.min(3, prev + 0.25))}
                            className="p-1 text-slate-500 hover:text-slate-900 dark:hover:text-slate-200"
                            title="Zoom In"
                        >
                            <ZoomIn size={12} />
                        </button>
                    </div>
                    <span>Playhead: {currentTime.toFixed(1)}s / {totalDuration.toFixed(1)}s</span>
                </div>
            </div>

            {/* Timeline Area */}
            <div className="flex-1 flex overflow-hidden">
                {/* Track Headers (Left Column) */}
                <div className="w-56 bg-white/80 dark:bg-[#12141c] border-r border-black/[0.06] dark:border-white/[0.08] flex flex-col pt-8 flex-shrink-0 z-10 shadow-sm">
                    {stemChannels.map((track) => (
                        <div
                            key={track.id}
                            className="h-20 px-4 border-b border-black/[0.04] dark:border-white/5 flex items-center justify-between bg-white/60 dark:bg-[#151722]/80 backdrop-blur-md"
                        >
                            <div className="min-w-0 pr-2">
                                <span className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate block">
                                    {track.name}
                                </span>
                                <span className="text-[10px] font-mono text-slate-400">
                                    Vol: {track.volume}%
                                </span>
                            </div>

                            {/* Solo & Mute Buttons */}
                            <div className="flex items-center space-x-1.5 flex-shrink-0">
                                <button
                                    onClick={() => onToggleSolo(track.id)}
                                    className={`w-6 h-6 rounded-md text-[10px] font-extrabold transition-all ${
                                        track.isSolo
                                            ? 'bg-amber-500 text-slate-950 shadow-sm font-black'
                                            : 'bg-black/5 dark:bg-white/5 text-slate-400 hover:text-amber-500'
                                    }`}
                                    title="Solo Track"
                                >
                                    S
                                </button>
                                <button
                                    onClick={() => onToggleMute(track.id)}
                                    className={`w-6 h-6 rounded-md text-[10px] font-extrabold transition-all ${
                                        track.isMuted
                                            ? 'bg-rose-500 text-white shadow-sm font-black'
                                            : 'bg-black/5 dark:bg-white/5 text-slate-400 hover:text-rose-500'
                                    }`}
                                    title="Mute Track"
                                >
                                    M
                                </button>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Timeline Tracks Grid (Right Column) */}
                <div className="flex-1 overflow-x-auto relative flex flex-col">
                    {/* Measure Ruler Bar */}
                    <div className="h-8 bg-black/[0.02] dark:bg-black/40 border-b border-black/[0.06] dark:border-white/5 flex items-center px-2 text-[10px] font-mono text-slate-400 select-none">
                        <div className="flex w-full justify-between pr-4">
                            {barsArray.slice(0, 16).map(bar => (
                                <div key={bar} className="flex-1 flex items-center space-x-1 border-l border-slate-300 dark:border-slate-700/50 pl-1">
                                    <span className="font-bold">{bar}</span>
                                    <span className="text-[8px] opacity-40">.1</span>
                                    <span className="text-[8px] opacity-40">.2</span>
                                    <span className="text-[8px] opacity-40">.3</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Tracks Lane & Waveform Blocks */}
                    <div
                        onClick={handleTimelineClick}
                        className="flex-1 relative cursor-pointer"
                        style={{ minWidth: `${100 * zoom}%` }}
                    >
                        {stemChannels.map((track) => (
                            <div
                                key={track.id}
                                className="h-20 border-b border-black/[0.04] dark:border-white/5 p-2 flex items-center relative"
                            >
                                {/* Audio Stem Block with Waveform */}
                                <div
                                    className={`h-16 w-full rounded-xl bg-gradient-to-r ${track.color} p-2 flex items-center justify-between shadow-sm relative overflow-hidden transition-opacity ${
                                        track.isMuted ? 'opacity-30' : 'opacity-90'
                                    }`}
                                >
                                    {/* Waveform Bars Simulation */}
                                    <div className="absolute inset-0 flex items-center justify-around opacity-30 pointer-events-none px-2">
                                        {Array.from({ length: 64 }).map((_, i) => {
                                            const height = 20 + Math.sin(i * 0.4) * 15 + (i % 3) * 10;
                                            return (
                                                <div
                                                    key={i}
                                                    className="w-1 bg-white rounded-full"
                                                    style={{ height: `${height}%` }}
                                                />
                                            );
                                        })}
                                    </div>

                                    <span className="text-xs font-bold text-white relative z-10 drop-shadow-sm">
                                        {track.name} Isolated Stem
                                    </span>
                                    <span className="text-[10px] font-mono text-white/80 relative z-10">
                                        48kHz 24-bit
                                    </span>
                                </div>
                            </div>
                        ))}

                        {/* Interactive Playhead Line */}
                        <div
                            className="absolute top-0 bottom-0 w-0.5 bg-rose-500 z-20 pointer-events-none transition-all duration-75 shadow-lg shadow-rose-500/50"
                            style={{ left: `${progressPercent}%` }}
                        >
                            <div className="w-3.5 h-3.5 bg-rose-500 -ml-1.5 -top-1 absolute rounded-full shadow-md border-2 border-white" />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
