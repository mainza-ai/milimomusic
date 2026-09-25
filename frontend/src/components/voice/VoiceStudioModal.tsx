import React, { useState, useEffect, useRef } from 'react';
import {
    Mic,
    X,
    Sparkles,
    Loader2,
    CheckCircle2,
    Sliders,
    Play,
    Pause
} from 'lucide-react';
import { trainingApi, type YuE2TrainingJob, type Job } from '../../api';
import { toast } from '../../utils/toast';
import { VocalStudioView } from '../views/VocalStudioView';

interface VoiceStudioModalProps {
    isOpen: boolean;
    onClose: () => void;
    initialTrack?: Job | null;
    initialStemPath?: string;
    onOpenWorkspace?: (job: Job) => void;
}

export const VoiceStudioModal: React.FC<VoiceStudioModalProps> = ({
    isOpen,
    onClose,
    initialTrack,
    initialStemPath,
    onOpenWorkspace
}) => {
    const [activeTab, setActiveTab] = useState<'voice_studio' | 'yue2_studio'>('voice_studio');

    // YuE2 Training Studio State
    const [yue2Mode, setYue2Mode] = useState<'auto' | 'guided'>('auto');
    const [datasetName, setDatasetName] = useState('my_custom_music');
    const [loraRank, setLoraRank] = useState<number>(32);
    const [activeYue2Job, setActiveYue2Job] = useState<YuE2TrainingJob | null>(null);
    const [isStartingYue2, setIsStartingYue2] = useState(false);
    const [auditionPlayingKey, setAuditionPlayingKey] = useState<string | null>(null);
    const auditionAudioRef = useRef<HTMLAudioElement | null>(null);

    // Unmount cleanup
    useEffect(() => {
        return () => {
            if (auditionAudioRef.current) {
                auditionAudioRef.current.pause();
                auditionAudioRef.current = null;
            }
        };
    }, []);

    // YuE2 Job Polling
    useEffect(() => {
        if (!activeYue2Job || activeYue2Job.status === 'completed' || activeYue2Job.status === 'failed') return;
        const timer = window.setInterval(async () => {
            try {
                const updated = await trainingApi.getJob(activeYue2Job.job_id);
                setActiveYue2Job(updated);
            } catch {
                // Ignore transient polling error
            }
        }, 2000);
        return () => window.clearInterval(timer);
    }, [activeYue2Job]);

    const handleStartYue2Training = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!datasetName.trim()) return;

        setIsStartingYue2(true);
        try {
            const job = await trainingApi.createJob(datasetName.trim(), yue2Mode, loraRank);
            setActiveYue2Job(job);
            toast(`YuE2 Training Job started in ${yue2Mode.toUpperCase()} mode!`, 'success');
        } catch (err: any) {
            const msg = err.response?.data?.detail?.error?.message || err.response?.data?.detail || err.message;
            toast('Failed to start training: ' + msg, 'error');
        } finally {
            setIsStartingYue2(false);
        }
    };

    const toggleAuditionPreview = (key: string, relPath: string) => {
        if (!relPath) return;
        if (auditionPlayingKey === key) {
            auditionAudioRef.current?.pause();
            setAuditionPlayingKey(null);
        } else {
            if (auditionAudioRef.current) {
                auditionAudioRef.current.pause();
            }
            const fullUrl = trainingApi.getAuditionAudioUrl(relPath);
            const audio = new Audio(fullUrl);
            auditionAudioRef.current = audio;
            setAuditionPlayingKey(key);
            audio.play().catch((err) => {
                console.warn('Audition playback error:', err);
                toast('Audition audio unavailable', 'error');
                setAuditionPlayingKey(null);
            });
            audio.onended = () => setAuditionPlayingKey(null);
            audio.onerror = () => {
                toast('Failed to load audition artifact', 'error');
                setAuditionPlayingKey(null);
            };
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-black/70 backdrop-blur-md">
            <div className="w-full max-w-6xl bg-white dark:bg-[#12141c] border border-black/[0.08] dark:border-white/10 rounded-3xl shadow-apple-2xl flex flex-col max-h-[92vh] overflow-hidden animate-fade-in">
                {/* Header */}
                <div className="px-6 py-4 border-b border-black/[0.06] dark:border-white/10 flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                        <div className="w-9 h-9 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400 flex items-center justify-center">
                            <Mic size={20} />
                        </div>
                        <div>
                            <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                                <span>Voice & Vocal Production Studio</span>
                            </h2>
                            <p className="text-xs text-slate-500 dark:text-slate-400">
                                Clone singer identities, live vocal booth recording, neural SVC, and YuE2 foundation model adapters
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        aria-label="Close modal"
                        title="Close"
                        className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
                    >
                        <X size={18} />
                    </button>
                </div>

                {/* Mode Tabs */}
                <div className="px-6 pt-2 pb-0 border-b border-black/[0.06] dark:border-white/10 flex items-center space-x-2 bg-black/[0.01] dark:bg-white/[0.02]">
                    <button
                        type="button"
                        onClick={() => setActiveTab('voice_studio')}
                        className={`pb-2.5 px-3 text-xs font-bold border-b-2 transition-all flex items-center gap-1.5 ${
                            activeTab === 'voice_studio'
                                ? 'border-teal-500 text-teal-600 dark:text-teal-400'
                                : 'border-transparent text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                        }`}
                    >
                        <Mic size={14} />
                        <span>Vocal Studio & SVC (Singing Voice Conversion)</span>
                    </button>
                    <button
                        type="button"
                        onClick={() => setActiveTab('yue2_studio')}
                        className={`pb-2.5 px-3 text-xs font-bold border-b-2 transition-all flex items-center gap-1.5 ${
                            activeTab === 'yue2_studio'
                                ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400'
                                : 'border-transparent text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                        }`}
                    >
                        <Sparkles size={14} className="text-indigo-400" />
                        <span>YuE2 "My Music" Studio</span>
                        <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 font-bold border border-indigo-500/20">
                            48kHz Stereo LoRA
                        </span>
                    </button>
                </div>

                {/* Body */}
                <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
                    {activeTab === 'voice_studio' ? (
                        <VocalStudioView
                            isModal={true}
                            initialTrack={initialTrack}
                            initialStemPath={initialStemPath}
                            onOpenWorkspace={onOpenWorkspace}
                            onCloseModal={onClose}
                        />
                    ) : (
                        <div className="space-y-6">
                            {/* Mode Selector & Configuration */}
                            <div className="p-5 bg-black/[0.02] dark:bg-[#181a24] border border-black/[0.06] dark:border-white/10 rounded-2xl space-y-4">
                                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                    <div>
                                        <h3 className="text-xs font-bold text-slate-900 dark:text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
                                            <Sparkles size={14} className="text-indigo-400" />
                                            <span>YuE2 Foundation Training Mode</span>
                                        </h3>
                                        <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                                            Train custom 48 kHz stereo style & instrumental LoRA adapters
                                        </p>
                                    </div>
                                    <div className="flex bg-black/[0.04] dark:bg-white/5 p-1 rounded-xl border border-black/[0.06] dark:border-white/10">
                                        <button
                                            type="button"
                                            onClick={() => setYue2Mode('auto')}
                                            className={`px-3 py-1 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 ${
                                                yue2Mode === 'auto'
                                                    ? 'bg-indigo-500 text-white shadow-sm'
                                                    : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                                            }`}
                                        >
                                            <span>⚡ Auto Mode</span>
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => setYue2Mode('guided')}
                                            className={`px-3 py-1 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 ${
                                                yue2Mode === 'guided'
                                                    ? 'bg-indigo-500 text-white shadow-sm'
                                                    : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                                            }`}
                                        >
                                            <span>🛠️ Guided Mode (4 Stages)</span>
                                        </button>
                                    </div>
                                </div>

                                <form onSubmit={handleStartYue2Training} className="space-y-4 pt-2 border-t border-black/[0.04] dark:border-white/5">
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                                                Dataset Identifier *
                                            </label>
                                            <input
                                                type="text"
                                                required
                                                placeholder="e.g., ambient_guitar_48k"
                                                value={datasetName}
                                                onChange={(e) => setDatasetName(e.target.value)}
                                                className="w-full apple-input"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                                                LoRA Rank / Capacity
                                            </label>
                                            <select
                                                value={loraRank}
                                                onChange={(e) => setLoraRank(parseInt(e.target.value, 10))}
                                                className="w-full apple-input"
                                            >
                                                <option value={16}>Rank 16 (Fast, Lightweight)</option>
                                                <option value={32}>Rank 32 (Standard Recommended)</option>
                                                <option value={64}>Rank 64 (Deep Expressive)</option>
                                            </select>
                                        </div>
                                    </div>

                                    <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-xs text-indigo-300 space-y-1">
                                        <div className="font-bold flex items-center gap-1.5 text-indigo-200">
                                            <Sliders size={13} />
                                            {yue2Mode === 'auto' ? 'Auto Mode: Hands-Off 1-Click Pipeline' : 'Guided Mode: 4-Stage Audition Studio'}
                                        </div>
                                        <p className="text-[11px] text-slate-400">
                                            {yue2Mode === 'auto'
                                                ? 'Automatically executes tokenizer adaptation (100 steps), AR style training (200 steps), generates auditory audition checks, and saves the final LoRA in the background.'
                                                : 'Step-by-step 4-stage pipeline with auditory before/after reconstruction checks (original ground truth vs baseline vs adapted decoder) and test song auditions at checkpoints.'}
                                        </p>
                                    </div>

                                    <div className="flex justify-end">
                                        <button
                                            type="submit"
                                            disabled={isStartingYue2 || !datasetName.trim()}
                                            className="px-4 py-2 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500 disabled:opacity-50 text-white font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-all shadow-md shadow-indigo-500/20"
                                        >
                                            {isStartingYue2 ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
                                            <span>{isStartingYue2 ? "Launching YuE2 Job..." : "Start YuE2 Training"}</span>
                                        </button>
                                    </div>
                                </form>
                            </div>

                            {/* Active or Completed YuE2 Job HUD */}
                            {activeYue2Job && (
                                <div className="p-5 bg-white dark:bg-[#181a24] border border-black/[0.06] dark:border-white/10 rounded-2xl space-y-4 shadow-apple-sm">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center space-x-2">
                                            <span className="font-mono text-xs font-bold text-slate-800 dark:text-slate-200">
                                                Job: {activeYue2Job.job_id}
                                            </span>
                                            <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                                                activeYue2Job.status === 'completed'
                                                    ? 'bg-teal-500/20 text-teal-300 border border-teal-500/30'
                                                    : activeYue2Job.status === 'failed'
                                                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                                                    : 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                                            }`}>
                                                {activeYue2Job.status}
                                            </span>
                                        </div>
                                        <span className="text-[11px] font-mono text-slate-400">
                                            Rank {activeYue2Job.rank || 32} · {activeYue2Job.mode.toUpperCase()}
                                        </span>
                                    </div>

                                    {/* Stage Progress */}
                                    <div className="space-y-1.5">
                                        <div className="flex justify-between text-xs">
                                            <span className="font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                                                {activeYue2Job.status === 'processing' && <Loader2 size={12} className="animate-spin text-indigo-400" />}
                                                {activeYue2Job.status === 'completed' && <CheckCircle2 size={12} className="text-teal-400" />}
                                                <span>Stage {activeYue2Job.current_stage} of {activeYue2Job.total_stages}:</span>
                                                <span className="text-indigo-400 font-bold">
                                                    {activeYue2Job.current_stage === 1 && 'Dataset Review & Excerpts'}
                                                    {activeYue2Job.current_stage === 2 && 'Real-Audio Tokenizer/Decoder Adaptation'}
                                                    {activeYue2Job.current_stage === 3 && 'AR Song Style LoRA Training'}
                                                    {activeYue2Job.current_stage === 4 && 'Auditions & Output Finalization'}
                                                </span>
                                            </span>
                                            <span className="font-mono text-slate-400">
                                                Step {activeYue2Job.current_step}/{activeYue2Job.total_steps}
                                            </span>
                                        </div>
                                        <div className="w-full h-1.5 bg-black/[0.06] dark:bg-white/10 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500"
                                                style={{ width: `${Math.round((activeYue2Job.current_step / activeYue2Job.total_steps) * 100)}%` }}
                                            />
                                        </div>
                                    </div>

                                    {/* Audition Checkpoints in Guided Mode */}
                                    {activeYue2Job.reconstruction_auditions && Object.keys(activeYue2Job.reconstruction_auditions).length > 0 && (
                                        <div className="pt-2 border-t border-black/[0.06] dark:border-white/10 space-y-2">
                                            <h4 className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
                                                Decoder Reconstruction Auditions (Stage 2)
                                            </h4>
                                            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                                                {Object.entries(activeYue2Job.reconstruction_auditions).map(([key, path]) => (
                                                    <button
                                                        key={key}
                                                        type="button"
                                                        onClick={() => toggleAuditionPreview(key, path)}
                                                        className={`p-2.5 rounded-xl border text-left flex items-center justify-between text-xs transition-colors ${
                                                            auditionPlayingKey === key
                                                                ? 'bg-indigo-500 text-white border-indigo-400'
                                                                : 'bg-black/[0.02] dark:bg-white/5 border-black/[0.06] dark:border-white/10 hover:border-indigo-500/50'
                                                        }`}
                                                    >
                                                        <span className="capitalize font-medium">{key.replace('_', ' ')}</span>
                                                        {auditionPlayingKey === key ? <Pause size={14} /> : <Play size={14} />}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
