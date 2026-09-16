import React, { useEffect, useState, useCallback } from 'react';
import { Cpu, Zap, Trash2, SlidersHorizontal } from 'lucide-react';
import { systemApi, type SystemTelemetry } from '../../api';
import { useModalStore } from '../../stores/useModalStore';

export const HardwareTelemetryBar: React.FC = () => {
    const [telemetry, setTelemetry] = useState<SystemTelemetry | null>(null);
    const [isFlushing, setIsFlushing] = useState(false);
    const [flushMessage, setFlushMessage] = useState<string | null>(null);
    const { openEngineSwitcher } = useModalStore();

    const fetchTelemetry = useCallback(async () => {
        try {
            const data = await systemApi.getTelemetry();
            if (data && typeof data === 'object' && typeof data.device_type === 'string') {
                setTelemetry(data);
            }
        } catch {
            // Silently swallow background polling errors
        }
    }, []);

    useEffect(() => {
        fetchTelemetry();
        const interval = setInterval(fetchTelemetry, 4000);
        return () => clearInterval(interval);
    }, [fetchTelemetry]);

    // Global keyboard shortcut: Ctrl+E or Cmd+E to toggle Engine Switcher
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'e') {
                e.preventDefault();
                openEngineSwitcher();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [openEngineSwitcher]);

    const handleFlush = async (e: React.MouseEvent) => {
        e.stopPropagation();
        setIsFlushing(true);
        try {
            const res = await systemApi.flushMemory();
            setFlushMessage(`Freed ${res.reclaimed_mb}MB`);
            setTimeout(() => setFlushMessage(null), 2500);
            await fetchTelemetry();
        } catch {
            setFlushMessage('Flush failed');
            setTimeout(() => setFlushMessage(null), 2000);
        } finally {
            setIsFlushing(false);
        }
    };

    if (!telemetry || typeof telemetry !== 'object' || !telemetry.device_type) return null;

    const usagePercent = typeof telemetry.usage_percent === 'number' ? telemetry.usage_percent : 0;
    const isHigh = usagePercent >= 85;
    const isCritical = usagePercent >= 95;
    const deviceType = (telemetry.device_type || 'cpu').toUpperCase();
    const allocatedGb = ((telemetry.vram_allocated_mb || 0) / 1024).toFixed(1);
    const totalGb = ((telemetry.vram_total_mb || 0) / 1024).toFixed(0);

    return (
        <div className="flex items-center space-x-2 text-xs font-mono select-none">
            {/* VRAM / Accelerator Status Pill */}
            <div
                onClick={openEngineSwitcher}
                title={`Device: ${telemetry.device_name || 'System Accelerator'}\nActive: ${telemetry.active_consumer || 'idle'}\nClick or press Ctrl+E for Engine Switcher`}
                className="flex items-center space-x-2 px-2.5 py-1 rounded-lg bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 border border-black/[0.06] dark:border-white/10 cursor-pointer transition-all duration-150"
            >
                <div className="flex items-center space-x-1.5">
                    <Cpu size={13} className={isCritical ? 'text-rose-500 animate-pulse' : isHigh ? 'text-amber-500' : 'text-teal-500'} />
                    <span className="font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider text-[11px]">
                        {deviceType}
                    </span>
                </div>

                {/* Progress bar */}
                <div className="w-14 h-1.5 bg-black/10 dark:bg-white/10 rounded-full overflow-hidden flex items-center">
                    <div
                        className={`h-full rounded-full transition-all duration-300 ${
                            isCritical ? 'bg-rose-500' : isHigh ? 'bg-amber-500' : 'bg-teal-500'
                        }`}
                        style={{ width: `${Math.min(100, usagePercent)}%` }}
                    />
                </div>

                <span className="text-[10px] tabular-nums text-slate-500 dark:text-slate-400">
                    {allocatedGb}/{totalGb}GB
                </span>

                {telemetry.active_consumer && telemetry.active_consumer !== 'idle' && (
                    <span className="hidden md:inline-flex items-center space-x-1 px-1.5 py-0.5 rounded bg-teal-500/10 text-teal-600 dark:text-teal-400 text-[10px] font-medium border border-teal-500/20">
                        <Zap size={9} />
                        <span className="truncate max-w-[90px]">{telemetry.active_consumer}</span>
                    </span>
                )}
            </div>

            {/* Quick Flush VRAM Button */}
            <button
                type="button"
                onClick={handleFlush}
                disabled={isFlushing}
                title="Flush dormant neural models & clear VRAM cache"
                aria-label="Flush VRAM cache"
                className="p-1 rounded-lg bg-black/[0.04] dark:bg-white/5 hover:bg-rose-500/10 hover:text-rose-500 text-slate-500 dark:text-slate-400 border border-black/[0.06] dark:border-white/10 transition-colors"
            >
                <Trash2 size={12} className={isFlushing ? 'animate-spin text-rose-500' : ''} />
            </button>

            {flushMessage && (
                <span className="text-[10px] font-sans text-teal-600 dark:text-teal-400 font-semibold animate-fade-in">
                    {flushMessage}
                </span>
            )}

            {/* Keyboard shortcut hint chip */}
            <button
                type="button"
                onClick={openEngineSwitcher}
                title="Open Neural Engine Switcher (Ctrl+E)"
                className="hidden xl:inline-flex items-center space-x-1 px-1.5 py-0.5 rounded bg-black/[0.03] dark:bg-white/[0.04] text-[10px] text-slate-400 border border-black/[0.04] dark:border-white/5 hover:text-slate-700 dark:hover:text-slate-200"
            >
                <SlidersHorizontal size={10} />
                <span>Ctrl+E</span>
            </button>
        </div>
    );
};
