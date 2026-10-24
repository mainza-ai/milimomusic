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
import { estimateLyricFit, formatFitTime, splitRemainingLyrics, previewRemainder } from '../../utils/lyricFit';

/**
 * Parse a user-entered time string into total seconds.
 * Supports:
 * - "m:ss" or "mm:ss" (e.g. "2:09" -> 129, "3:00" -> 180)
 * - "h:mm:ss" (e.g. "1:02:30" -> 3750)
 * - "2.5m", "2m 30s", "3min"
 * - "90s", "45sec"
 * - pure numbers (e.g. "150" -> 150)
 */
export function parseTimeToSeconds(input: string): number | null {
    const trimmed = input.trim().toLowerCase();
    if (!trimmed) return null;

    if (trimmed.includes(':')) {
        const parts = trimmed.split(':').map(p => parseFloat(p.trim()));
        if (parts.some(isNaN)) return null;
        if (parts.length === 2) {
            const [mins, secs] = parts;
            return Math.round(mins * 60 + secs);
        }
        if (parts.length === 3) {
            const [hours, mins, secs] = parts;
            return Math.round(hours * 3600 + mins * 60 + secs);
        }
    }

    const mMatch = trimmed.match(/^(\d+(?:\.\d+)?)\s*m(?:in)?(?:\s*(\d+(?:\.\d+)?)\s*s(?:ec)?)?$/);
    if (mMatch) {
        const mins = parseFloat(mMatch[1]);
        const secs = mMatch[2] ? parseFloat(mMatch[2]) : 0;
        return Math.round(mins * 60 + secs);
    }

    const sMatch = trimmed.match(/^(\d+(?:\.\d+)?)\s*s(?:ec)?$/);
    if (sMatch) {
        return Math.round(parseFloat(sMatch[1]));
    }

    const num = parseFloat(trimmed);
    if (!isNaN(num) && num > 0) {
        return Math.round(num);
    }

    return null;
}

/**
 * Parse an extension delta time string (e.g. "+30s", "45", "+1:00").
 */
export function parseDeltaToSeconds(input: string): number | null {
    const trimmed = input.trim().replace(/^\+/, '').trim();
    return parseTimeToSeconds(trimmed);
}

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
    const [extendFromInput, setExtendFromInput] = useState<string>('');
    const [targetDurationSec, setTargetDurationSec] = useState<number>(Math.min(600, parentDurationSec + 60));
    const [targetDurationInput, setTargetDurationInput] = useState<string>('');
    const [extensionDeltaInput, setExtensionDeltaInput] = useState<string>('');
    const [additionalLyrics, setAdditionalLyrics] = useState<string>('');
    const [autoGenerateLyrics, setAutoGenerateLyrics] = useState<boolean>(false);
    const [isDraftingLyrics, setIsDraftingLyrics] = useState<boolean>(false);
    const [customPrompt, setCustomPrompt] = useState<string>('');
    const [crossfadeSec, setCrossfadeSec] = useState<number>(1.5);
    const [variationSeed, setVariationSeed] = useState<string>('');
    const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
    const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

    const formatTime = (secs: number) => {
        const m = Math.floor(secs / 60);
        const s = Math.floor(secs % 60);
        return `${m}:${s < 10 ? '0' : ''}${s}`;
    };

    // Sync initial state whenever opened on a job
    useEffect(() => {
        if (extendTrackJob) {
            const dur = extendTrackJob.duration_ms ? extendTrackJob.duration_ms / 1000 : 60;
            const initialTarget = Math.min(600, dur + 60);
            setExtendFromSec(dur);
            setExtendFromInput(formatTime(dur));
            setTargetDurationSec(initialTarget);
            setTargetDurationInput(formatTime(initialTarget));
            setExtensionDeltaInput(`+${formatTime(initialTarget - dur)}`);
            setAdditionalLyrics('');
            setAutoGenerateLyrics(false);
            setCustomPrompt('');
            setVariationSeed('');
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

    // Lyric-fit estimate: do the parent lyrics fit the target duration?
    // (Placed after the guard so extendTrackJob is non-null.)
    const lyricFit = estimateLyricFit(
        extendTrackJob.lyrics || '',
        extendTrackJob.tags,
        targetDurationSec
    );
    const handleFitDuration = () => {
        const fitted = Math.min(600, Math.max(extendFromSec + 5, lyricFit.neededSec));
        if (fitted > extendFromSec) {
            setTargetDurationSec(fitted);
            setTargetDurationInput(formatTime(fitted));
            setExtensionDeltaInput(`+${formatTime(fitted - extendFromSec)}`);
        }
    };

    const handleQuickAddDelta = (delta: number) => {
        const newTarget = Math.min(600, Math.max(extendFromSec + 5, extendFromSec + delta));
        setTargetDurationSec(newTarget);
        setTargetDurationInput(formatTime(newTarget));
        setExtensionDeltaInput(`+${formatTime(newTarget - extendFromSec)}`);
    };

    const handleTargetDurationInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const text = e.target.value;
        setTargetDurationInput(text);
        const parsed = parseTimeToSeconds(text);
        if (parsed !== null && parsed > extendFromSec && parsed <= 600) {
            setTargetDurationSec(parsed);
            setExtensionDeltaInput(`+${formatTime(parsed - extendFromSec)}`);
        }
    };

    const handleTargetDurationInputBlur = () => {
        const parsed = parseTimeToSeconds(targetDurationInput);
        if (parsed !== null && parsed > extendFromSec && parsed <= 600) {
            setTargetDurationSec(parsed);
            setTargetDurationInput(formatTime(parsed));
            setExtensionDeltaInput(`+${formatTime(parsed - extendFromSec)}`);
        } else {
            setTargetDurationInput(formatTime(targetDurationSec));
            setExtensionDeltaInput(`+${formatTime(targetDurationSec - extendFromSec)}`);
        }
    };

    const handleExtensionDeltaInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const text = e.target.value;
        setExtensionDeltaInput(text);
        const delta = parseDeltaToSeconds(text);
        if (delta !== null && delta >= 5 && extendFromSec + delta <= 600) {
            const newTarget = extendFromSec + delta;
            setTargetDurationSec(newTarget);
            setTargetDurationInput(formatTime(newTarget));
        }
    };

    const handleExtensionDeltaInputBlur = () => {
        const delta = parseDeltaToSeconds(extensionDeltaInput);
        if (delta !== null && delta >= 5 && extendFromSec + delta <= 600) {
            const newTarget = extendFromSec + delta;
            setTargetDurationSec(newTarget);
            setTargetDurationInput(formatTime(newTarget));
            setExtensionDeltaInput(`+${formatTime(newTarget - extendFromSec)}`);
        } else {
            setExtensionDeltaInput(`+${formatTime(targetDurationSec - extendFromSec)}`);
        }
    };

    const handleExtendFromInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const text = e.target.value;
        setExtendFromInput(text);
        const parsed = parseTimeToSeconds(text);
        if (parsed !== null && parsed >= 1 && parsed <= parentDurationSec) {
            setExtendFromSec(parsed);
            if (targetDurationSec < parsed + 5) {
                const newTarget = Math.min(600, parsed + 30);
                setTargetDurationSec(newTarget);
                setTargetDurationInput(formatTime(newTarget));
                setExtensionDeltaInput(`+${formatTime(newTarget - parsed)}`);
            } else {
                setExtensionDeltaInput(`+${formatTime(targetDurationSec - parsed)}`);
            }
        }
    };

    const handleExtendFromInputBlur = () => {
        const parsed = parseTimeToSeconds(extendFromInput);
        if (parsed !== null && parsed >= 1 && parsed <= parentDurationSec) {
            setExtendFromSec(parsed);
            setExtendFromInput(formatTime(parsed));
        } else {
            setExtendFromInput(formatTime(extendFromSec));
        }
    };

    const maxSliderDuration = Math.min(600, Math.max(300, Math.ceil(Math.max(targetDurationSec, parentDurationSec + 180) / 60) * 60));

    const handleInsertTag = (tag: string) => {
        setAdditionalLyrics(prev => {
            const trimmed = prev.trim();
            return trimmed ? `${trimmed}\n\n${tag}\n` : `${tag}\n`;
        });
    };

    const handleRedoLastSection = () => {
        const cut = Math.max(5, Math.floor(parentDurationSec - 15));
        const newTarget = Math.min(600, cut + 30);
        setExtendFromSec(cut);
        setExtendFromInput(formatTime(cut));
        setTargetDurationSec(newTarget);
        setTargetDurationInput(formatTime(newTarget));
        setExtensionDeltaInput(`+${formatTime(newTarget - cut)}`);
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        // Final sanity sync from input fields in case user pressed enter without blurring
        const parsedTarget = parseTimeToSeconds(targetDurationInput);
        let finalTargetSec = targetDurationSec;
        if (parsedTarget !== null && parsedTarget > extendFromSec && parsedTarget <= 600) {
            finalTargetSec = parsedTarget;
        }

        const parsedExtendFrom = parseTimeToSeconds(extendFromInput);
        let finalExtendFromSec = extendFromSec;
        if (parsedExtendFrom !== null && parsedExtendFrom >= 1 && parsedExtendFrom <= parentDurationSec) {
            finalExtendFromSec = parsedExtendFrom;
        }

        if (finalTargetSec <= finalExtendFromSec) {
            toast('Target duration must be greater than the extension cut point.', 'error');
            return;
        }

        setIsSubmitting(true);
        try {
            const seedNum = variationSeed.trim() === '' ? undefined : Math.floor(Number(variationSeed));
            const res = await trackApi.extendTrack(extendTrackJob.id, {
                target_duration_sec: finalTargetSec,
                extend_from_sec: finalExtendFromSec,
                additional_lyrics: additionalLyrics.trim() || undefined,
                auto_generate_lyrics: autoGenerateLyrics,
                prompt: customPrompt.trim() || undefined,
                crossfade_sec: crossfadeSec,
                seed: (seedNum !== undefined && !isNaN(seedNum)) ? seedNum : undefined,
            });

            const modeLabel = res.expected_mode === 'state_restored' ? 'exact continuation (restored)'
                : res.expected_mode === 'kv_rollforward' ? 'deterministic continuation'
                : res.expected_mode === 'no_parent_audio' ? 'fresh audio (no parent found!)'
                : 'fresh segment';
            toast(`Extension queued! Cut at ${formatTime(res.extend_from_sec)} → ${formatTime(res.target_duration_sec)} · ${modeLabel}.`, 'success');
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
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                            <label className="text-xs font-semibold text-zinc-300 uppercase tracking-wider flex items-center gap-2">
                                <Clock className="w-3.5 h-3.5 text-teal-400" />
                                Target Duration
                            </label>
                            {/* Dual Editable Inputs: Target Duration (mm:ss) + Extension Added Time (+sec) */}
                            <div className="flex items-center gap-1.5 self-end sm:self-auto">
                                <div className="flex items-center bg-zinc-900/90 border border-teal-500/30 hover:border-teal-500/50 focus-within:border-teal-400 focus-within:ring-1 focus-within:ring-teal-500/40 rounded-xl px-2.5 py-1 transition-all shadow-inner">
                                    <span className="text-[10px] uppercase font-mono font-semibold text-zinc-400 mr-1.5 select-none" title="Total target song duration">Target:</span>
                                    <input
                                        type="text"
                                        value={targetDurationInput}
                                        onChange={handleTargetDurationInputChange}
                                        onBlur={handleTargetDurationInputBlur}
                                        onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); (e.target as HTMLInputElement).blur(); } }}
                                        placeholder="m:ss"
                                        aria-label="Target song duration (e.g. 2:30 or 150)"
                                        title="Type exact target duration (e.g. 2:30, 3:00, or seconds like 150)"
                                        className="w-14 bg-transparent font-mono text-sm font-bold text-teal-300 text-center focus:outline-none placeholder:text-zinc-600 cursor-text"
                                    />
                                    <span className="text-zinc-700 font-mono text-xs mx-1 select-none">|</span>
                                    <span className="text-[10px] uppercase font-mono font-semibold text-zinc-400 mr-1 select-none" title="Additional extension time added">Add:</span>
                                    <input
                                        type="text"
                                        value={extensionDeltaInput}
                                        onChange={handleExtensionDeltaInputChange}
                                        onBlur={handleExtensionDeltaInputBlur}
                                        onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); (e.target as HTMLInputElement).blur(); } }}
                                        placeholder="+s"
                                        aria-label="Additional time to add (e.g. +30s or 45)"
                                        title="Type extension duration to add (e.g. +30s, 45, or 1:00)"
                                        className="w-14 bg-transparent font-mono text-xs font-bold text-teal-400/90 text-center focus:outline-none placeholder:text-zinc-600 cursor-text"
                                    />
                                </div>
                            </div>
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
                                min={extendFromSec + 5}
                                max={maxSliderDuration}
                                step={1}
                                value={targetDurationSec}
                                onChange={e => {
                                    const val = Number(e.target.value);
                                    setTargetDurationSec(val);
                                    setTargetDurationInput(formatTime(val));
                                    setExtensionDeltaInput(`+${formatTime(val - extendFromSec)}`);
                                }}
                                className="w-full accent-teal-400 bg-zinc-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="flex justify-between text-[10px] font-mono text-zinc-400">
                                <span>Extend from {formatTime(extendFromSec)}</span>
                                <span>Max {formatTime(maxSliderDuration)}</span>
                            </div>
                        </div>
                        {/* Lyric-fit hint: warn when the words can't fit the target */}
                        {lyricFit.words > 0 && !lyricFit.fits && (
                            <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-between gap-2">
                                <span className="text-[11px] text-amber-300">
                                    {lyricFit.words} words need ~{formatFitTime(lyricFit.neededSec)} at ~{lyricFit.paceWpm} wpm pace — target {formatTime(targetDurationSec)} may cut the ending.
                                </span>
                                <button
                                    type="button"
                                    onClick={handleFitDuration}
                                    title={`Set target duration to fit the lyrics (${formatFitTime(Math.min(600, Math.max(extendFromSec + 5, lyricFit.neededSec)))})`}
                                    className="shrink-0 px-2.5 py-1 text-[11px] font-semibold rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-200 transition-colors cursor-pointer"
                                >
                                    Fit to {formatFitTime(Math.min(600, Math.max(extendFromSec + 5, lyricFit.neededSec)))}
                                </button>
                            </div>
                        )}
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
                            <div className="flex items-center bg-zinc-900/90 border border-indigo-500/30 hover:border-indigo-500/50 focus-within:border-indigo-400 focus-within:ring-1 focus-within:ring-indigo-500/40 rounded-xl px-2.5 py-1 transition-all shadow-inner">
                                <span className="text-[10px] uppercase font-mono font-semibold text-zinc-400 mr-1.5 select-none" title="Audio transition seam point">Cut:</span>
                                <input
                                    type="text"
                                    value={extendFromInput}
                                    onChange={handleExtendFromInputChange}
                                    onBlur={handleExtendFromInputBlur}
                                    onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); (e.target as HTMLInputElement).blur(); } }}
                                    placeholder="m:ss"
                                    aria-label="Continuation seam cut point (e.g. 1:49, 1:30)"
                                    title="Type continuation seam cut point (e.g. 1:49, 1:30)"
                                    className="w-14 bg-transparent font-mono text-xs font-bold text-indigo-300 text-center focus:outline-none placeholder:text-zinc-600 cursor-text"
                                />
                            </div>
                        </div>
                        <input
                            type="range"
                            min={Math.max(1, parentDurationSec - 60)}
                            max={parentDurationSec}
                            step={1}
                            value={extendFromSec}
                            onChange={e => {
                                const val = Number(e.target.value);
                                setExtendFromSec(val);
                                setExtendFromInput(formatTime(val));
                                if (targetDurationSec < val + 5) {
                                    const newTarget = Math.min(600, val + 30);
                                    setTargetDurationSec(newTarget);
                                    setTargetDurationInput(formatTime(newTarget));
                                    setExtensionDeltaInput(`+${formatTime(newTarget - val)}`);
                                } else {
                                    setExtensionDeltaInput(`+${formatTime(targetDurationSec - val)}`);
                                }
                            }}
                            className="w-full accent-indigo-400 bg-zinc-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                        />
                        <div className="flex items-center justify-between text-[10px] text-zinc-400">
                            <span>Earlier Section Break</span>
                            <div className="flex items-center gap-2">
                                <button
                                    type="button"
                                    onClick={handleRedoLastSection}
                                    title="Rewind: cut 15s before the end and extend 30s (redo the tail)"
                                    className="px-2 py-0.5 rounded-md bg-indigo-500/10 hover:bg-indigo-500/20 border border-indigo-500/30 text-indigo-300 text-[10px] font-semibold transition-colors cursor-pointer"
                                >
                                    ↩ Redo last 15s
                                </button>
                                <span className="text-emerald-400 font-medium">Song End ({formatTime(parentDurationSec)})</span>
                            </div>
                        </div>
                    </div>

                        {/* Tail preview: what the extension will sing from the cut point */}
                        {(() => {
                            let lines: any[] = [];
                            try {
                                const parsed = extendTrackJob.timed_lyrics_json ? JSON.parse(extendTrackJob.timed_lyrics_json) : [];
                                if (Array.isArray(parsed)) lines = parsed;
                            } catch { /* unaligned lyrics — no preview */ }
                            const remainder = splitRemainingLyrics(lines, extendFromSec);
                            const preview = previewRemainder(remainder, 2);
                            if (!remainder) return null;
                            return (
                                <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/30">
                                    <div className="text-[10px] font-bold text-indigo-300 uppercase tracking-wider mb-1">
                                        Tail will continue with ({remainder.split('\n').filter(Boolean).length} lines from {formatTime(extendFromSec)})
                                    </div>
                                    {preview.map((line, i) => (
                                        <div key={i} className="text-[11px] font-mono text-indigo-200/90 truncate">…{line}</div>
                                    ))}
                                </div>
                            );
                        })()}

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
                                Variation Seed (optional)
                            </label>
                            <input
                                type="number"
                                value={variationSeed}
                                onChange={e => setVariationSeed(e.target.value)}
                                placeholder={`Parent seed ${parentSeed}`}
                                title="Leave empty to inherit the parent seed (maximum continuity). Enter a number to reroll just the new section."
                                className="w-full px-3 py-1.5 text-xs rounded-lg bg-zinc-800/80 border border-zinc-700 text-white font-mono placeholder:text-zinc-400"
                            />
                            <span className="text-[10px] text-zinc-400 block mt-0.5">
                                Empty = inherit parent seed (seamless). Set a number to reroll only the new section.
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
