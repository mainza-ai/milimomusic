import React, { useEffect, useState } from 'react';
import { X, Cpu, Zap, Trash2, Layers, Music, Video, Mic } from 'lucide-react';
import { systemApi, type SystemTelemetry } from '../../api';
import { useModalStore } from '../../stores/useModalStore';

interface EngineCard {
    id: string;
    name: string;
    family: string;
    category: 'generation' | 'remix' | 'separation' | 'transcription' | 'voice' | 'video';
    approxVramGb: number;
    icon: React.ComponentType<{ size?: number; className?: string }>;
    description: string;
}

const ENGINES: EngineCard[] = [
    {
        id: 'minimax',
        name: 'MiniMax Music 3',
        family: 'DiT / LLM Transformer',
        category: 'generation',
        approxVramGb: 8.5,
        icon: Music,
        description: 'Default text-to-music foundation model with structured lyric conditioning.',
    },
    {
        id: 'mulacover',
        name: 'MuLaCover 3B',
        family: 'LLaMA-3 Autoregressive',
        category: 'remix',
        approxVramGb: 8.3,
        icon: Layers,
        description: 'Audio-to-audio cover song generator with melody, chords, and drum conditioning.',
    },
    {
        id: 'roformer',
        name: 'BS-Roformer / Demucs',
        family: 'Band-Split Roformer',
        category: 'separation',
        approxVramGb: 3.2,
        icon: Layers,
        description: 'High-precision 6-stem neural source separator (vocals, drums, bass, guitar, piano, other).',
    },
    {
        id: 'muscriptor',
        name: 'MuScriptor & ChordNet',
        family: 'Ensemble Neural Transcriptor',
        category: 'transcription',
        approxVramGb: 2.1,
        icon: Music,
        description: 'Note-level MIDI transcription, beat tracking, and harmonic progression recognition.',
    },
    {
        id: 'neural_svc',
        name: 'Neural Singing Voice Conversion',
        family: 'ContentVec + Formant Morphing',
        category: 'voice',
        approxVramGb: 1.8,
        icon: Mic,
        description: 'Zero-shot singing voice timbre transfer, formant control, and singer replacement.',
    },
    {
        id: 'wan_video',
        name: 'Wan 2.1 Video & LivePortrait',
        family: 'Diffusers Video Diffusion',
        category: 'video',
        approxVramGb: 14.0,
        icon: Video,
        description: 'Multi-modal beat-reactive music video synthesis and avatar lip synchronization.',
    },
];

let cachedEngineTelemetry: SystemTelemetry | null = null;

export const EngineSwitcherModal: React.FC = () => {
    const { isEngineSwitcherOpen, closeEngineSwitcher } = useModalStore();
    const [telemetry, setTelemetry] = useState<SystemTelemetry | null>(cachedEngineTelemetry);
    const [isFlushing, setIsFlushing] = useState(false);
    const [statusMessage, setStatusMessage] = useState<string | null>(null);

    const loadTelemetry = async () => {
        try {
            const data = await systemApi.getTelemetry();
            if (data && typeof data === 'object' && typeof data.device_type === 'string') {
                cachedEngineTelemetry = data;
                setTelemetry(prev => {
                    if (prev && prev.device_type === data.device_type &&
                        prev.vram_allocated_mb === data.vram_allocated_mb &&
                        prev.usage_percent === data.usage_percent &&
                        prev.active_consumer === data.active_consumer) {
                        return prev;
                    }
                    return data;
                });
            }
        } catch {
            // ignore
        }
    };

    useEffect(() => {
        if (isEngineSwitcherOpen) {
            loadTelemetry();
            const timer = setInterval(loadTelemetry, 3000);
            return () => clearInterval(timer);
        }
    }, [isEngineSwitcherOpen]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape' && isEngineSwitcherOpen) {
                closeEngineSwitcher();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isEngineSwitcherOpen, closeEngineSwitcher]);

    if (!isEngineSwitcherOpen) return null;

    const handleFlush = async () => {
        setIsFlushing(true);
        try {
            const res = await systemApi.flushMemory();
            setStatusMessage(`Successfully flushed cache! Freed ${res.reclaimed_mb} MB.`);
            setTimeout(() => setStatusMessage(null), 3000);
            await loadTelemetry();
        } catch {
            setStatusMessage('Failed to flush GPU cache.');
            setTimeout(() => setStatusMessage(null), 2500);
        } finally {
            setIsFlushing(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75">
            <div
                className="w-full max-w-2xl bg-white dark:bg-[#141622] rounded-2xl border border-black/10 dark:border-white/10 shadow-2xl overflow-hidden flex flex-col max-h-[90vh] transform-gpu"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="px-6 py-4 border-b border-black/[0.08] dark:border-white/[0.08] flex items-center justify-between bg-black/[0.02] dark:bg-white/[0.02]">
                    <div className="flex items-center space-x-3">
                        <div className="p-2 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20">
                            <Cpu size={20} />
                        </div>
                        <div>
                            <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                                Neural Engine & Hardware Switcher
                                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-100 dark:bg-white/10 text-slate-600 dark:text-slate-300">
                                    Ctrl+E
                                </span>
                            </h2>
                            <p className="text-xs text-slate-500 dark:text-slate-400">
                                Monitor VRAM allocation, evict dormant weights, and coordinate accelerators.
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={closeEngineSwitcher}
                        className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
                     title="Close">
                        <X size={18} />
                    </button>
                </div>

                {/* Hardware Telemetry Card */}
                {telemetry && typeof telemetry === 'object' && telemetry.device_type && (
                    <div className="p-5 border-b border-black/[0.06] dark:border-white/[0.06] bg-slate-50 dark:bg-[#10121a]">
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center space-x-2">
                                <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                                    {telemetry.device_name || 'System Accelerator'}
                                </span>
                                <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-teal-500/15 text-teal-700 dark:text-teal-300 font-semibold uppercase">
                                    {telemetry.device_type || 'cpu'}
                                </span>
                            </div>

                            <button
                                onClick={handleFlush}
                                disabled={isFlushing}
                                className="flex items-center space-x-1.5 px-3 py-1 rounded-lg text-xs font-semibold bg-rose-500/10 hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 border border-rose-500/20 transition-all cursor-pointer"
                             title="Delete">
                                <Trash2 size={12} className={isFlushing ? 'animate-spin' : ''} />
                                <span>{isFlushing ? 'Flushing...' : 'Flush VRAM Cache'}</span>
                            </button>
                        </div>

                        {/* Progress bar */}
                        <div className="space-y-1.5">
                            <div className="flex justify-between text-xs font-mono text-slate-500">
                                <span>Allocated: {((telemetry.vram_allocated_mb || 0) / 1024).toFixed(2)} GB</span>
                                <span>Total: {((telemetry.vram_total_mb || 0) / 1024).toFixed(1)} GB ({telemetry.usage_percent || 0}%)</span>
                            </div>
                            <div className="w-full h-2.5 bg-black/10 dark:bg-white/10 rounded-full overflow-hidden">
                                <div
                                    className={`h-full rounded-full transition-all duration-300 ${
                                        (telemetry.usage_percent || 0) >= 90
                                            ? 'bg-rose-500'
                                            : (telemetry.usage_percent || 0) >= 75
                                            ? 'bg-amber-500'
                                            : 'bg-teal-500'
                                    }`}
                                    style={{ width: `${Math.min(100, telemetry.usage_percent || 0)}%` }}
                                />
                            </div>
                        </div>

                        {telemetry.active_consumer && telemetry.active_consumer !== 'idle' && (
                            <div className="mt-3 flex items-center space-x-2 text-xs text-amber-600 dark:text-amber-400 font-medium">
                                <Zap size={13} className="animate-pulse" />
                                <span>Hardware lock currently held by: <strong>{telemetry.active_consumer}</strong></span>
                            </div>
                        )}

                        {statusMessage && (
                            <div className="mt-2 text-xs text-teal-600 dark:text-teal-400 font-semibold animate-fade-in">
                                {statusMessage}
                            </div>
                        )}
                    </div>
                )}

                {/* Engine Matrix */}
                <div className="p-5 overflow-y-auto space-y-3 flex-1">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                        Platform Neural Backbones
                    </h3>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {ENGINES.map((eng) => {
                            const Icon = eng.icon;
                            const isActive = Boolean(telemetry?.active_consumer?.toLowerCase().includes(eng.id));

                            return (
                                <div
                                    key={eng.id}
                                    className={`p-3.5 rounded-xl border transition-all ${
                                        isActive
                                            ? 'bg-teal-500/10 border-teal-500/40 shadow-sm'
                                            : 'bg-white dark:bg-white/[0.02] border-black/[0.06] dark:border-white/[0.06] hover:border-black/15 dark:hover:border-white/15'
                                    }`}
                                >
                                    <div className="flex items-start justify-between mb-1.5">
                                        <div className="flex items-center space-x-2">
                                            <div className={`p-1.5 rounded-lg ${isActive ? 'bg-teal-500 text-white' : 'bg-black/5 dark:bg-white/5 text-slate-600 dark:text-slate-300'}`}>
                                                <Icon size={14} />
                                            </div>
                                            <div>
                                                <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100">
                                                    {eng.name}
                                                </h4>
                                                <span className="text-[10px] font-mono text-slate-400 block">
                                                    {eng.family}
                                                </span>
                                            </div>
                                        </div>

                                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/5 dark:bg-white/5 text-slate-500">
                                            ~{eng.approxVramGb} GB
                                        </span>
                                    </div>

                                    <p className="text-[11px] text-slate-500 dark:text-slate-400 line-clamp-2 leading-relaxed">
                                        {eng.description}
                                    </p>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Footer */}
                <div className="px-6 py-3 border-t border-black/[0.06] dark:border-white/[0.06] bg-black/[0.02] dark:bg-white/[0.02] flex items-center justify-between text-xs">
                    <span className="text-slate-400 text-[11px]">
                        Milimo Music Resource Orchestrator • Auto-eviction on context shift
                    </span>
                    <button
                        onClick={closeEngineSwitcher}
                        className="px-4 py-1.5 rounded-xl font-semibold bg-slate-200 dark:bg-white/10 hover:bg-slate-300 dark:hover:bg-white/15 text-slate-800 dark:text-slate-200 transition-colors"
                     title="Close">
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
};
