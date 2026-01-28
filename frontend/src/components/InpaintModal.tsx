
import React, { useState } from 'react';
import { GlassCard } from './ui/GlassCard';
import { X, Wand2, Loader2 } from 'lucide-react';
import { api } from '../api';
import { Toast } from './ui/Toast';

interface InpaintModalProps {
    isOpen: boolean;
    onClose: () => void;
    jobId: string;
    duration: number; // in seconds
    title?: string;
}

export const InpaintModal: React.FC<InpaintModalProps> = ({
    isOpen,
    onClose,
    jobId,
    duration,
    title
}) => {
    const [startTime, setStartTime] = useState(0);
    const [endTime, setEndTime] = useState(Math.min(5, duration));
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [toast, setToast] = useState<{ msg: string, type: 'success' | 'error' } | null>(null);

    if (!isOpen) return null;

    const handleSubmit = async () => {
        setIsSubmitting(true);
        try {
            await api.inpaintTrack(jobId, startTime, endTime);
            setToast({ msg: "Repair started! Watch the feed.", type: 'success' });
            setTimeout(() => {
                onClose();
                setToast(null);
            }, 2000);
        } catch (e) {
            console.error(e);
            setToast({ msg: "Failed to start repair.", type: 'error' });
        } finally {
            setIsSubmitting(false);
        }
    };

    // Helper to format s -> mm:ss
    const fmt = (s: number) => {
        const m = Math.floor(s / 60);
        const sec = Math.floor(s % 60);
        return `${m}:${sec.toString().padStart(2, '0')}`;
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm">
            {toast && <Toast message={toast.msg} type={toast.type} onClose={() => setToast(null)} />}

            <GlassCard className="w-full max-w-md p-6 bg-white/90">
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                        <Wand2 className="w-5 h-5 text-indigo-500" />
                        {title ? `Repair: ${title}` : 'Repair Audio'}
                    </h3>
                    <button onClick={onClose} className="p-1 hover:bg-slate-100 rounded-full">
                        <X className="w-5 h-5 text-slate-500" />
                    </button>
                </div>

                <div className="space-y-6">
                    <p className="text-sm text-slate-600">
                        Select a segment to regenerate. The audio before and after this range will remain unchanged.
                    </p>

                    {/* Range Inputs */}
                    <div className="space-y-4">
                        <div>
                            <label className="text-xs font-semibold uppercase text-slate-500">Start Time</label>
                            <div className="flex items-center gap-4">
                                <input
                                    type="range"
                                    min="0"
                                    max={duration}
                                    step="0.1"
                                    value={startTime}
                                    onChange={(e) => {
                                        const v = parseFloat(e.target.value);
                                        setStartTime(v);
                                        if (v >= endTime) setEndTime(Math.min(v + 1, duration));
                                    }}
                                    className="w-full accent-indigo-500"
                                />
                                <span className="font-mono text-sm w-12">{fmt(startTime)}</span>
                            </div>
                        </div>

                        <div>
                            <label className="text-xs font-semibold uppercase text-slate-500">End Time</label>
                            <div className="flex items-center gap-4">
                                <input
                                    type="range"
                                    min="0"
                                    max={duration}
                                    step="0.1"
                                    value={endTime}
                                    onChange={(e) => {
                                        const v = parseFloat(e.target.value);
                                        setEndTime(v);
                                        if (v <= startTime) setStartTime(Math.max(0, v - 1));
                                    }}
                                    className="w-full accent-indigo-500"
                                />
                                <span className="font-mono text-sm w-12">{fmt(endTime)}</span>
                            </div>
                        </div>
                    </div>

                    {/* Visualization Bar */}
                    <div className="h-4 bg-slate-200 rounded-full overflow-hidden relative">
                        {/* Start Segment (Safe) */}
                        <div
                            className="absolute top-0 left-0 h-full bg-slate-300"
                            style={{ width: `${(startTime / duration) * 100}%` }}
                        />
                        {/* Repair Segment (Danger/Active) */}
                        <div
                            className="absolute top-0 h-full bg-indigo-500/50 backdrop-blur-sm border-x border-indigo-500 flex items-center justify-center"
                            style={{
                                left: `${(startTime / duration) * 100}%`,
                                width: `${((endTime - startTime) / duration) * 100}%`
                            }}
                        >
                            <span className="text-[10px] text-indigo-900 font-bold tracking-tighter">REPAIR</span>
                        </div>
                        {/* End Segment (Safe) */}
                        {/* Implicitly the rest */}
                    </div>

                    <div className="flex justify-end pt-4">
                        <button
                            onClick={handleSubmit}
                            disabled={isSubmitting}
                            className="px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 active:scale-95 transition-all flex items-center gap-2"
                        >
                            {isSubmitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
                            Regenerate Segment
                        </button>
                    </div>
                </div>
            </GlassCard>
        </div>
    );
};
