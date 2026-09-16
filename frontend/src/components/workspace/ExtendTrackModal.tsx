import React, { useState, useEffect, useRef } from 'react';
import {
    X,
    Sparkles,
    FastForward,
    Sliders,
    Loader2,
    Music,
    Clock,
    Lock
} from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';
import { useModalA11y } from '../ui/primitives';
import { useModalStore } from '../../stores/useModalStore';
import { trackApi, api } from '../../api';
import { toast } from '../../utils/toast';

export const ExtendTrackModal: React.FC = () => {
    const { isExtendTrackOpen, extendTrackJob, closeExtendTrack } = useModalStore();
    const modalRef = useRef<HTMLDivElement>(null);
    useModalA11y(isExtendTrackOpen, closeExtendTrack, modalRef);

    // Parent track properties
    const parentDurationSec = extendTrackJob?.duration_ms ? extendTrackJob.duration_ms / 1000 : 60;
    const parentTitle = extendTrackJob?.title || 'Current Track';
    const parentSeed = extendTrackJob?.seed ?? 42;
    const parentTags = extendTrackJob?.tags || '';

    // Extension configuration state
    const [extendFromSec, setExtendFromSec] = useState<number>(parentDurationSec);
    const [targetDurationSec, setTargetDurationSec] = useState<number>(Math.min(300, parentDurationSec + 60));
    const [additionalLyrics, setAdditionalLyrics] = useState<string>('');
    const [autoGenerateLyrics, setAutoGenerateLyrics] = useState<boolean>(false);
    const [isDraftingLyrics, setIsDraftingLyrics] = useState<boolean>(false);
    const [customPrompt, setCustomPrompt] = useState<string>('');
    const [crossfadeSec, setCrossfadeSec] = useState<number>(1.5);
    const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
    const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

    // Sync initial state whenever opened on a job
    useEffect(() => {
        if (extendTrackJob) {
            const dur = extendTrackJob.duration_ms ? extendTrackJob.duration_ms / 1000 : 60;
            setExtendFromSec(dur);
            setTargetDurationSec(Math.min(300, dur + 60));
            setAdditionalLyrics('');
            setAutoGenerateLyrics(false);
            setCustomPrompt('');
        }
    }, [extendTrackJob]);

    const handleDraftLyrics = async () => {
        setIsDraftingLyrics(true);
        try {
            const topic = `Continuation lyrics for track "${parentTitle}", style: ${parentTags || 'pop'}`;
            const drafted = await api.generateLyrics(topic, 'deepseek-v3', additionalLyrics || undefined, parentTags);
            if (drafted) {
                setAdditionalLyrics(prev => prev.trim() ? `${prev.trim()}\n\n${drafted.trim()}` : drafted.trim());
                toast('Drafted continuation lyrics with AI!', 'success');
            }
        } catch (e) {
            console.error('Failed to draft lyrics:', e);
            toast('Failed to draft lyrics with AI', 'error');
        } finally {
            setIsDraftingLyrics(false);
        }
    };

    if (!isExtendTrackOpen || !extendTrackJob) return null;

    const extensionDeltaSec = Math.max(0, targetDurationSec - extendFromSec);

    const formatTime = (secs: number) => {
        const m = Math.floor(secs / 60);
        const s = Math.floor(secs % 60);
        return `${m}:${s < 10 ? '0' : ''}${s}`;
    };

    const handleQuickAddDelta = (delta: number) => {
        const newTarget = Math.min(300, Math.max(extendFromSec + 15, extendFromSec + delta));
        setTargetDurationSec(newTarget);
    };

    const handleInsertTag = (tag: string) => {
        setAdditionalLyrics(prev => {
            const trimmed = prev.trim();
            return trimmed ? `${trimmed}\n\n${tag}\n` : `${tag}\n`;
        });
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (targetDurationSec <= extendFromSec) {
            toast('Target duration must be greater than the extension cut point.', 'error');
            return;
        }

        setIsSubmitting(true);
        try {
            await trackApi.extendTrack(extendTrackJob.id, {
                target_duration_sec: targetDurationSec,
                extend_from_sec: extendFromSec,
                additional_lyrics: additionalLyrics.trim() || undefined,
                auto_generate_lyrics: autoGenerateLyrics,
                prompt: customPrompt.trim() || undefined,
                crossfade_sec: crossfadeSec,
            });

            toast(`Track extension queued! (${formatTime(targetDurationSec)}) Seamless continuation started.`, 'success');
            closeExtendTrack();
        } catch (err: any) {
            console.error('Failed to extend track:', err);
            toast(err?.response?.data?.detail || 'Failed to start track extension', 'error');
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-in fade-in duration-200">
            <div
                ref={modalRef}
                role="dialog"
                aria-modal="true"
                aria-labelledby="extend-modal-title"
                className="w-full max-w-2xl"
            >
                <GlassCard
                    className="w-full bg-zinc-950 border border-zinc-800/80 shadow-2xl flex flex-col max-h-[90vh] overflow-hidden rounded-2xl p-0"
                >
                {/* Header */}
                <div className="p-5 border-b border-zinc-800/60 flex items-center justify-between bg-zinc-900/40">
                    <div className="flex items-center gap-3">
                        <div className="p-2.5 rounded-xl bg-gradient-to-br from-indigo-500/20 to-teal-500/20 border border-indigo-500/30 text-indigo-400">
                            <FastForward className="w-5 h-5" />
                        </div>
                        <div>
                            <h2 id="extend-modal-title" className="text-base font-bold text-white flex items-center gap-2">
                                Extend Track with Musical Continuity
                                <span className="px-2 py-0.5 text-[10px] font-mono bg-teal-500/10 text-teal-300 border border-teal-500/20 rounded-full">
                                    MiniMax Engine
                                </span>
                            </h2>
                            <p className="text-xs text-zinc-400">
                                Seamlessly continue <span className="text-zinc-200 font-semibold">{parentTitle}</span> while preserving acoustic timbre, tempo, and key.
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={closeExtendTrack}
                        className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800/60 transition-colors cursor-pointer"
                        aria-label="Close modal"
                        title="Close modal"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Form Content */}
                <form onSubmit={handleSubmit} className="p-6 overflow-y-auto space-y-6 flex-1 custom-scrollbar">
                    {/* Track Lineage Banner */}
                    <div className="p-3.5 rounded-xl bg-zinc-900/60 border border-zinc-800/80 flex items-center justify-between text-xs">
                        <div className="flex items-center gap-3">
                            <Music className="w-4 h-4 text-teal-400" />
                            <div>
                                <span className="text-zinc-400">Original Duration:</span>{' '}
                                <span className="font-mono text-white font-semibold">{formatTime(parentDurationSec)}</span>
                            </div>
                        </div>
                        <div className="flex items-center gap-3 font-mono text-[11px] text-zinc-400">
                            <span className="flex items-center gap-1 text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded border border-teal-500/20">
                                <Lock className="w-3 h-3" /> Seed #{parentSeed}
                            </span>
                            {parentTags && (
                                <span className="truncate max-w-[180px] bg-zinc-800/60 px-2 py-0.5 rounded text-zinc-300">
                                    {parentTags}
                                </span>
                            )}
                        </div>
                    </div>

                    {/* Timeline & Extension Target */}
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <label className="text-xs font-semibold text-zinc-300 uppercase tracking-wider flex items-center gap-2">
                                <Clock className="w-3.5 h-3.5 text-teal-400" />
                                Target Duration
                            </label>
                            <span className="text-sm font-mono font-bold text-teal-400 bg-teal-500/10 px-2.5 py-0.5 rounded-md border border-teal-500/20">
                                {formatTime(targetDurationSec)} (+{formatTime(extensionDeltaSec)})
                            </span>
                        </div>

                        {/* Quick Presets */}
                        <div className="grid grid-cols-5 gap-2">
                            {[30, 60, 90, 120, 180].map(delta => {
                                const target = parentDurationSec + delta;
                                const isSelected = Math.abs(targetDurationSec - target) < 1;
                                return (
                                    <button
                                        key={delta}
                                        type="button"
                                        onClick={() => handleQuickAddDelta(delta)}
                                        title={`Extend duration by ${delta} seconds (Target: ${formatTime(target)})`}
                                        className={`py-2 text-xs rounded-xl border font-medium transition-all cursor-pointer ${
                                            isSelected
                                                ? 'bg-teal-500/20 border-teal-500/60 text-teal-300 shadow-sm shadow-teal-500/20'
                                                : 'bg-zinc-900/60 border-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-800'
                                        }`}
                                    >
                                        +{delta}s ({formatTime(target)})
                                    </button>
                                );
                            })}
                        </div>

                        {/* Duration Slider */}
                        <div className="space-y-1.5 pt-1">
                            <input
                                type="range"
                                min={extendFromSec + 15}
                                max={300}
                                step={5}
                                value={targetDurationSec}
                                onChange={e => setTargetDurationSec(Number(e.target.value))}
                                className="w-full accent-teal-400 bg-zinc-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="flex justify-between text-[10px] font-mono text-zinc-400">
                                <span>Extend from {formatTime(extendFromSec)}</span>
                                <span>Max 5:00</span>
                            </div>
                        </div>
                    </div>

                    {/* Continuation Cut Point */}
                    <div className="p-4 rounded-xl bg-zinc-900/40 border border-zinc-800/60 space-y-2.5">
                        <div className="flex items-center justify-between">
                            <div>
                                <label className="text-xs font-semibold text-zinc-200 block">
                                    Continuation Seam Point
                                </label>
                                <span className="text-[11px] text-zinc-400">
                                    Where the original audio transitions into the extended section.
                                </span>
                            </div>
                            <span className="font-mono text-xs text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                                {formatTime(extendFromSec)}
                            </span>
                        </div>
                        <input
                            type="range"
                            min={Math.max(5, parentDurationSec - 30)}
                            max={parentDurationSec}
                            step={1}
                            value={extendFromSec}
                            onChange={e => {
                                const val = Number(e.target.value);
                                setExtendFromSec(val);
                                if (targetDurationSec < val + 15) {
                                    setTargetDurationSec(val + 30);
                                }
                            }}
                            className="w-full accent-indigo-400 bg-zinc-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                        />
                        <div className="flex items-center justify-between text-[10px] text-zinc-400">
                            <span>Earlier Section Break</span>
                            <span className="text-emerald-400 font-medium">Song End ({formatTime(parentDurationSec)})</span>
                        </div>
                    </div>

                    {/* Continuation Lyrics Editor */}
                    <div className="space-y-2.5">
                        <div className="flex items-center justify-between">
                            <label className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">
                                Extension Lyrics & Arrangement
                            </label>
                            <div className="flex items-center gap-2">
                                <span className="text-[10px] text-zinc-400">Empty = instrumental continuation</span>
                                <button
                                    type="button"
                                    onClick={handleDraftLyrics}
                                    disabled={isDraftingLyrics}
                                    className="px-2.5 py-1 text-[11px] rounded-lg bg-teal-500/10 hover:bg-teal-500/20 border border-teal-500/30 text-teal-300 font-medium flex items-center gap-1 transition-colors cursor-pointer disabled:opacity-50"
                                    title="Draft continuation lyrics with AI based on song style"
                                >
                                    {isDraftingLyrics ? (
                                        <>
                                            <Loader2 className="w-3 h-3 animate-spin" />
                                            <span>Drafting...</span>
                                        </>
                                    ) : (
                                        <>
                                            <Sparkles className="w-3 h-3 text-teal-400" />
                                            <span>Draft with AI</span>
                                        </>
                                    )}
                                </button>
                            </div>
                        </div>

                        {/* Quick Tag Pills */}
                        <div className="flex flex-wrap gap-1.5">
                            {['[Verse 2]', '[Chorus]', '[Bridge]', '[Guitar Solo]', '[Outro]'].map(tag => (
                                <button
                                    key={tag}
                                    type="button"
                                    onClick={() => handleInsertTag(tag)}
                                    className="px-2.5 py-1 text-[11px] rounded-lg bg-zinc-900/80 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 hover:text-white transition-colors cursor-pointer"
                                    title={`Insert ${tag} marker`}
                                >
                                    + {tag}
                                </button>
                            ))}
                        </div>

                        <textarea
                            value={additionalLyrics}
                            onChange={e => setAdditionalLyrics(e.target.value)}
                            rows={4}
                            placeholder="[Verse 2]&#10;Add lyrics for the continued section...&#10;&#10;[Outro]&#10;Fade into the night..."
                            className="w-full px-3.5 py-2.5 text-xs rounded-xl bg-zinc-900/70 border border-zinc-800 text-white font-mono placeholder:text-zinc-400 focus:outline-none focus:border-teal-500/60 focus:ring-1 focus:ring-teal-500/40 custom-scrollbar resize-none"
                        />

                        {/* Auto-generate lyrics toggle (Default: OFF) */}
                        <div className="flex items-center justify-between p-3 rounded-xl bg-zinc-900/50 border border-zinc-800/60">
                            <div>
                                <label htmlFor="auto-gen-lyrics-toggle" className="text-xs font-medium text-zinc-200 block cursor-pointer">
                                    Auto-generate continuation lyrics with AI
                                </label>
                                <span className="text-[11px] text-zinc-400 block">
                                    Off by default. When disabled, lyrics are never added automatically.
                                </span>
                            </div>
                            <label className="relative inline-flex items-center cursor-pointer">
                                <input
                                    id="auto-gen-lyrics-toggle"
                                    type="checkbox"
                                    checked={autoGenerateLyrics}
                                    onChange={e => setAutoGenerateLyrics(e.target.checked)}
                                    className="sr-only peer"
                                    title="Toggle auto-generation of continuation lyrics"
                                />
                                <div className="w-9 h-5 bg-zinc-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-zinc-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-teal-500"></div>
                            </label>
                        </div>
                    </div>

                    {/* Advanced Settings Toggle */}
                    <div className="border-t border-zinc-800/60 pt-3">
                        <button
                            type="button"
                            onClick={() => setShowAdvanced(!showAdvanced)}
                            title={showAdvanced ? "Hide advanced acoustic settings" : "Show advanced acoustic settings"}
                            className="text-xs text-zinc-400 hover:text-zinc-200 flex items-center gap-1.5 font-medium transition-colors cursor-pointer"
                        >
                            <Sliders className="w-3.5 h-3.5" />
                            <span>{showAdvanced ? 'Hide Advanced Settings' : 'Show Advanced Acoustic Settings'}</span>
                        </button>

                        {showAdvanced && (
                            <div className="mt-3 p-4 rounded-xl bg-zinc-900/40 border border-zinc-800/60 space-y-4 animate-in fade-in duration-150">
                                <div>
                                    <div className="flex justify-between text-xs mb-1">
                                        <span className="text-zinc-300 font-medium">Equal-Power Crossfade Window</span>
                                        <span className="text-teal-400 font-mono">{crossfadeSec.toFixed(1)}s</span>
                                    </div>
                                    <input
                                        type="range"
                                        min={0.5}
                                        max={3.0}
                                        step={0.1}
                                        value={crossfadeSec}
                                        onChange={e => setCrossfadeSec(Number(e.target.value))}
                                        className="w-full accent-teal-400 bg-zinc-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                                    />
                                    <span className="text-[10px] text-zinc-400 block mt-0.5">
                                        Smooth equal-power sine/cosine crossfade eliminates seam clicks and volume dips.
                                    </span>
                                </div>

                                <div>
                                    <label className="text-xs text-zinc-300 font-medium block mb-1">
                                        Continuation Direction Prompt (Optional)
                                    </label>
                                    <input
                                        type="text"
                                        value={customPrompt}
                                        onChange={e => setCustomPrompt(e.target.value)}
                                        placeholder="e.g. Higher energy climax, heavy synthesizer solo, emotional vocal build"
                                        className="w-full px-3 py-1.5 text-xs rounded-lg bg-zinc-800/80 border border-zinc-700 text-white placeholder:text-zinc-400"
                                    />
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Footer Actions */}
                    <div className="flex items-center justify-end gap-3 pt-2">
                        <button
                            type="button"
                            onClick={closeExtendTrack}
                            title="Cancel extension"
                            className="px-4 py-2 text-xs font-semibold text-zinc-300 hover:text-white bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 rounded-xl transition-colors cursor-pointer"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={isSubmitting}
                            title={`Queue extension to ${formatTime(targetDurationSec)}`}
                            className="px-5 py-2 text-xs font-semibold text-white bg-gradient-to-r from-teal-500 to-indigo-600 hover:from-teal-400 hover:to-indigo-500 rounded-xl shadow-lg shadow-teal-500/20 flex items-center gap-2 transition-all disabled:opacity-50 cursor-pointer"
                        >
                            {isSubmitting ? (
                                <>
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    <span>Synthesizing Extension...</span>
                                </>
                            ) : (
                                <>
                                    <Sparkles className="w-4 h-4" />
                                    <span>Extend to {formatTime(targetDurationSec)}</span>
                                </>
                            )}
                        </button>
                    </div>
                </form>
            </GlassCard>
            </div>
        </div>
    );
};
