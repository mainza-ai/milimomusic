import React, { useState, useEffect } from 'react';
import { X, RefreshCw, Sparkles, Camera, Sun, Loader2 } from 'lucide-react';
import type { VideoClipSegment } from '../../api';

interface ClipRetakeModalProps {
    isOpen: boolean;
    onClose: () => void;
    clipIndex: number | null;
    clipSegment?: VideoClipSegment;
    onConfirmRetake: (clipIndex: number, newPrompt: string, camera: string, lighting: string) => Promise<void>;
    isRetaking: boolean;
}

export const ClipRetakeModal: React.FC<ClipRetakeModalProps> = ({
    isOpen,
    onClose,
    clipIndex,
    clipSegment,
    onConfirmRetake,
    isRetaking,
}) => {
    const [prompt, setPrompt] = useState('');
    const [camera, setCamera] = useState('Medium cinematic focus');
    const [lighting, setLighting] = useState('Atmospheric rim light');

    useEffect(() => {
        if (clipSegment) {
            setPrompt(clipSegment.prompt || '');
            setCamera(clipSegment.camera || 'Medium cinematic focus');
            setLighting(clipSegment.lighting || 'Atmospheric rim light');
        }
    }, [clipSegment]);

    if (!isOpen || clipIndex === null) return null;

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!prompt.trim()) return;
        await onConfirmRetake(clipIndex, prompt.trim(), camera, lighting);
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md animate-fade-in">
            <div className="w-full max-w-lg bg-white dark:bg-[#12141c] border border-black/[0.08] dark:border-white/10 rounded-3xl shadow-apple-2xl flex flex-col overflow-hidden">
                {/* Header */}
                <div className="px-6 py-4 border-b border-black/[0.06] dark:border-white/10 flex items-center justify-between">
                    <div className="flex items-center space-x-2.5">
                        <div className="w-8 h-8 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400 flex items-center justify-center">
                            <RefreshCw size={16} />
                        </div>
                        <div>
                            <h3 className="text-sm font-bold text-slate-900 dark:text-white">
                                Retake Scene #{clipIndex}
                            </h3>
                            <p className="text-[11px] text-slate-500 font-mono">
                                {clipSegment?.time_str} ({clipSegment?.duration.toFixed(1)}s) · {clipSegment?.scene_type === 'VOCAL_PERFORMANCE' ? 'Vocal' : 'B-Roll'}
                            </p>
                        </div>
                    </div>
                    <button
                        type="button"
                        onClick={onClose}
                        className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition-colors"
                    >
                        <X size={16} />
                    </button>
                </div>

                {/* Body Form */}
                <form onSubmit={handleSubmit} className="p-6 space-y-4">
                    {clipSegment?.lyrics && (
                        <div className="p-3 bg-cyan-500/10 border border-cyan-500/20 rounded-xl text-xs text-cyan-700 dark:text-cyan-300 italic">
                            "{clipSegment.lyrics}"
                        </div>
                    )}

                    <div className="space-y-1.5">
                        <label className="text-xs font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
                            <Sparkles size={13} className="text-teal-500" />
                            <span>Visual Direction Prompt</span>
                        </label>
                        <textarea
                            rows={3}
                            required
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            className="w-full text-xs apple-input rounded-xl p-3 resize-none font-sans"
                            placeholder="Describe visual action, subject, and scene details..."
                        />
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                        <div className="space-y-1.5">
                            <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1">
                                <Camera size={12} />
                                <span>Camera Motion</span>
                            </label>
                            <select
                                value={camera}
                                onChange={(e) => setCamera(e.target.value)}
                                className="w-full apple-input text-xs"
                            >
                                <option value="Tight emotive close-up with soft bokeh">Tight Close-Up</option>
                                <option value="Medium orbital shot focusing on performer">Medium Orbit</option>
                                <option value="Slow cinematic tracking crane down">Tracking Crane</option>
                                <option value="Wide atmospheric environmental sweep">Wide Sweep</option>
                                <option value="Dutch angle low push-in with rim flare">Dutch Low Angle</option>
                            </select>
                        </div>

                        <div className="space-y-1.5">
                            <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1">
                                <Sun size={12} />
                                <span>Lighting & Tone</span>
                            </label>
                            <select
                                value={lighting}
                                onChange={(e) => setLighting(e.target.value)}
                                className="w-full apple-input text-xs"
                            >
                                <option value="Cyan edge luminescence with deep shadows">Neon Cyan Edge</option>
                                <option value="Warm golden hour backlighting with lens flare">Golden Hour</option>
                                <option value="High contrast monochrome spotlight">Monochrome Spot</option>
                                <option value="Soft diffused atmospheric volumetric haze">Volumetric Haze</option>
                            </select>
                        </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex items-center justify-end gap-2 pt-3 border-t border-black/[0.06] dark:border-white/10">
                        <button
                            type="button"
                            onClick={onClose}
                            className="px-3.5 py-1.5 text-xs text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={isRetaking || !prompt.trim()}
                            className="px-4 py-2 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 shadow-md shadow-teal-500/20 disabled:opacity-50"
                        >
                            {isRetaking ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                            <span>{isRetaking ? 'Rendering Retake…' : 'Generate Retake'}</span>
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};
