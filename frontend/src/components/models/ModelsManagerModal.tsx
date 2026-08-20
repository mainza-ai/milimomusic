import React, { useState, useEffect } from 'react';
import { Cpu, Download, CheckCircle2, X, Activity } from 'lucide-react';
import { modelsApi, type ModelVariant, type HardwareProfile } from '../../api';

interface ModelsManagerModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export const ModelsManagerModal: React.FC<ModelsManagerModalProps> = ({ isOpen, onClose }) => {
    const [models, setModels] = useState<ModelVariant[]>([]);
    const [hardware, setHardware] = useState<HardwareProfile | null>(null);
    const [downloadingId, setDownloadingId] = useState<string | null>(null);

    const loadData = async () => {
        try {
            const [treeData, hwData] = await Promise.all([
                modelsApi.getModelTree(),
                modelsApi.getHardwareProfile()
            ]);
            setModels(treeData);
            setHardware(hwData);
        } catch (e) {
            console.error('Failed to load model manager data', e);
        }
    };

    useEffect(() => {
        if (isOpen) {
            loadData();
        }
    }, [isOpen]);

    const handleDownload = async (modelId: string) => {
        setDownloadingId(modelId);
        setTimeout(() => {
            setDownloadingId(null);
            loadData();
        }, 2000);
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 dark:bg-black/80 backdrop-blur-md animate-fade-in">
            <div className="bg-white/90 dark:bg-[#14161f]/95 border border-black/[0.08] dark:border-white/10 rounded-3xl w-full max-w-4xl overflow-hidden shadow-apple-lg flex flex-col max-h-[85vh] backdrop-blur-2xl">
                {/* Modal Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-black/[0.06] dark:border-white/10 bg-black/[0.02] dark:bg-[#181a24]">
                    <div className="flex items-center space-x-3">
                        <div className="w-9 h-9 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 text-teal-700 dark:text-teal-400 border border-teal-500/20 flex items-center justify-center">
                            <Cpu size={18} />
                        </div>
                        <div>
                            <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
                                Generation Models & Hardware Tiers
                            </h2>
                            <p className="text-xs text-slate-500 dark:text-slate-400">
                                Manage model tree checkpoints, quantization variants, and GPU acceleration.
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
                    >
                        <X size={18} />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {/* Hardware Profiler Card */}
                    {hardware && (
                        <div className="p-4 bg-gradient-to-r from-teal-500/10 via-cyan-500/5 to-sky-500/5 rounded-2xl border border-teal-500/20 flex items-center justify-between shadow-apple-sm">
                            <div className="flex items-center space-x-3">
                                <Activity className="text-teal-600 dark:text-teal-400 flex-shrink-0" size={20} />
                                <div>
                                    <div className="flex items-center space-x-2">
                                        <span className="text-xs font-bold text-slate-900 dark:text-slate-100">Hardware Detection:</span>
                                        <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-teal-500/20 text-teal-800 dark:text-teal-300 font-semibold uppercase">
                                            {hardware.hardware_tier.replace('_', ' ')}
                                        </span>
                                    </div>
                                    <p className="text-xs text-slate-600 dark:text-slate-300 mt-0.5">
                                        {hardware.processor} ({hardware.os_name} {hardware.architecture}) · {hardware.tier_description}
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Model Tree List */}
                    <div className="space-y-4">
                        <h3 className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
                            Model Tree Catalog & Variants
                        </h3>

                        <div className="space-y-3">
                            {models.map(m => (
                                <div
                                    key={m.id}
                                    className={`p-5 rounded-2xl border transition-all ${
                                        m.is_installed
                                            ? 'bg-white dark:bg-[#181a24] border-black/[0.06] dark:border-white/10 shadow-apple-sm'
                                            : 'bg-black/[0.02] dark:bg-[#151720]/60 border-black/[0.04] dark:border-white/5 opacity-80'
                                    }`}
                                >
                                    <div className="flex items-start justify-between">
                                        <div>
                                            <div className="flex items-center space-x-2">
                                                <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">{m.name}</h4>
                                                {m.is_default && (
                                                    <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-teal-500 text-slate-950 shadow-sm">
                                                        Default Engine
                                                    </span>
                                                )}
                                                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-black/[0.04] dark:bg-white/5 text-slate-600 dark:text-slate-400 border border-black/[0.06] dark:border-white/5">
                                                    {m.quantization}
                                                </span>
                                            </div>
                                            <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                                                Architecture: {m.architecture} · License: {m.license}
                                            </p>
                                            <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
                                                Hardware: {m.recommended_hardware}
                                            </p>
                                            {m.local_path && (
                                                <div className="text-[10px] text-slate-500 dark:text-slate-400 font-mono mt-2 truncate max-w-lg">
                                                    Path: {m.local_path}
                                                </div>
                                            )}
                                        </div>

                                        <div className="flex flex-col items-end space-y-2">
                                            <span className="text-xs font-mono font-semibold text-slate-700 dark:text-slate-300">
                                                {m.size_gb} GB
                                            </span>

                                            {m.is_installed ? (
                                                <span className="flex items-center space-x-1 text-xs font-semibold text-teal-700 dark:text-teal-400 bg-teal-500/10 px-3 py-1 rounded-xl border border-teal-500/20">
                                                    <CheckCircle2 size={14} />
                                                    <span>Installed & Ready</span>
                                                </span>
                                            ) : (
                                                <button
                                                    onClick={() => handleDownload(m.id)}
                                                    disabled={downloadingId === m.id}
                                                    className="px-3.5 py-1.5 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs flex items-center space-x-1.5 transition-all shadow-sm active:scale-95"
                                                >
                                                    <Download size={13} />
                                                    <span>{downloadingId === m.id ? 'Downloading...' : 'Download'}</span>
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
