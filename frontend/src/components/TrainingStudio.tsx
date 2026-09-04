import React, { useState, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
    X, Upload, Play, Square, Trash2, FolderPlus, Edit2, HelpCircle,
    Settings2, Loader2, CheckCircle2, AlertCircle,
    Music, Database, Cpu, Package, Sparkles
} from 'lucide-react';
import { trainingApi, type Dataset, type TrainingJob, type Checkpoint } from '../api';

interface TrainingStudioProps {
    isOpen: boolean;
    onClose: () => void;
    onCheckpointsChange?: () => void;
}

type Tab = 'dataset' | 'training' | 'jobs' | 'models';

// Help Tooltip Component
const HelpTooltip: React.FC<{ text: string }> = ({ text }) => {
    const [isVisible, setIsVisible] = useState(false);

    return (
        <div className="relative inline-block ml-1.5 align-middle">
            <button
                type="button"
                onMouseEnter={() => setIsVisible(true)}
                onMouseLeave={() => setIsVisible(false)}
                onClick={(e) => { e.preventDefault(); setIsVisible(!isVisible); }}
                className="text-slate-400 hover:text-teal-500 dark:hover:text-teal-400 transition-colors"
            >
                <HelpCircle className="w-3.5 h-3.5" />
            </button>
            <AnimatePresence>
                {isVisible && (
                    <motion.div
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 5 }}
                        className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-slate-900/95 text-white text-xs rounded-2xl shadow-apple-lg border border-white/10 backdrop-blur-xl"
                    >
                        <div className="relative leading-relaxed">
                            {text}
                            <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-900/95" />
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

// Time formatting helpers
const formatElapsedTime = (startedAt: string | undefined): string => {
    if (!startedAt) return '--:--';
    const start = new Date(startedAt).getTime();
    const now = Date.now();
    const elapsed = Math.floor((now - start) / 1000); // seconds

    const hours = Math.floor(elapsed / 3600);
    const minutes = Math.floor((elapsed % 3600) / 60);
    const seconds = elapsed % 60;

    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    }
    return `${minutes}m ${seconds}s`;
};

const formatETA = (startedAt: string | undefined, progress: number): string => {
    if (!startedAt || progress <= 0) return '--';
    if (progress >= 100) return 'Done';

    const start = new Date(startedAt).getTime();
    const now = Date.now();
    const elapsed = now - start; // ms

    // Estimate total time based on current progress
    const estimatedTotal = elapsed / (progress / 100);
    const remaining = estimatedTotal - elapsed;

    if (remaining <= 0) return 'Soon';

    const remainingSeconds = Math.floor(remaining / 1000);
    const hours = Math.floor(remainingSeconds / 3600);
    const minutes = Math.floor((remainingSeconds % 3600) / 60);

    if (hours > 0) {
        return `~${hours}h ${minutes}m`;
    }
    if (minutes > 0) {
        return `~${minutes}m`;
    }
    return '<1m';
};

// Format ISO timestamp to relative or absolute time
const formatTimestamp = (isoString: string | undefined): string => {
    if (!isoString) return '';
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = diffMs / (1000 * 60 * 60);

    if (diffHours < 1) {
        const mins = Math.floor(diffMs / 60000);
        return mins <= 1 ? 'Just now' : `${mins}m ago`;
    }
    if (diffHours < 24) {
        return `${Math.floor(diffHours)}h ago`;
    }
    if (diffHours < 48) {
        return 'Yesterday';
    }
    // Show date for older items
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
};

export const TrainingStudio: React.FC<TrainingStudioProps> = ({ isOpen, onClose, onCheckpointsChange }) => {
    const [activeTab, setActiveTab] = useState<Tab>('dataset');

    // Dataset state
    const [datasets, setDatasets] = useState<Dataset[]>([]);
    const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
    const [newDatasetName, setNewDatasetName] = useState('');
    const [newDatasetStyles, setNewDatasetStyles] = useState('');
    const [isCreatingDataset, setIsCreatingDataset] = useState(false);
    const [uploadCaption, setUploadCaption] = useState('');
    const [uploadProgress, setUploadProgress] = useState<{ [key: string]: boolean }>({});
    const [editingDataset, setEditingDataset] = useState<Dataset | null>(null);
    const [editName, setEditName] = useState('');
    const [editStyles, setEditStyles] = useState('');

    // Lyrics editing state
    const [editingLyrics, setEditingLyrics] = useState<{ filename: string; caption: string } | null>(null);

    // Training state - defaults match backend lora_trainer.py
    const [trainingMethod, setTrainingMethod] = useState<'lora' | 'full'>('lora');
    const [epochs, setEpochs] = useState(10);
    const [learningRate, setLearningRate] = useState(0.0003);
    const [loraRank, setLoraRank] = useState(32);

    // Jobs state
    const [jobs, setJobs] = useState<TrainingJob[]>([]);
    const [isLoadingJobs, setIsLoadingJobs] = useState(false);

    // Checkpoints state
    const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);

    // Preprocessing state
    const [isPreprocessing, setIsPreprocessing] = useState(false);

    // Audio player state
    const [playingFile, setPlayingFile] = useState<string | null>(null);
    const audioRef = React.useRef<HTMLAudioElement | null>(null);

    // Error state
    const [error, setError] = useState<string | null>(null);

    // Load initial data
    const loadDatasets = useCallback(async () => {
        try {
            const list = await trainingApi.listDatasets();
            setDatasets(list);
            if (list.length > 0 && !selectedDataset) {
                setSelectedDataset(list[0]);
            }
        } catch (e) {
            console.error('Failed to load datasets', e);
        }
    }, [selectedDataset]);

    const loadJobs = useCallback(async () => {
        setIsLoadingJobs(true);
        try {
            const list = await trainingApi.listJobs();
            setJobs(list);
        } catch (e) {
            console.error('Failed to load jobs', e);
        } finally {
            setIsLoadingJobs(false);
        }
    }, []);

    const loadCheckpoints = useCallback(async () => {
        try {
            const list = await trainingApi.listCheckpoints();
            setCheckpoints(list);
        } catch (e) {
            console.error('Failed to load checkpoints', e);
        }
    }, []);

    useEffect(() => {
        if (isOpen) {
            loadDatasets();
            loadJobs();
            loadCheckpoints();
        }
    }, [isOpen, loadDatasets, loadJobs, loadCheckpoints]);

    // Poll jobs while open
    useEffect(() => {
        if (!isOpen || activeTab !== 'jobs') return;

        const hasActiveJobs = jobs.some(j => j.status === 'running' || j.status === 'preprocessing');
        if (!hasActiveJobs && jobs.length > 0) return;

        const interval = setInterval(loadJobs, 2000);
        return () => clearInterval(interval);
    }, [isOpen, activeTab, jobs, loadJobs]);

    // Handlers
    const handleCreateDataset = async () => {
        if (!newDatasetName.trim()) return;
        setIsCreatingDataset(true);
        setError(null);

        try {
            const styles = newDatasetStyles.split(',').map(s => s.trim()).filter(Boolean);
            const dataset = await trainingApi.createDataset(newDatasetName.trim(), styles);
            setDatasets(prev => [...prev, dataset]);
            setSelectedDataset(dataset);
            setNewDatasetName('');
            setNewDatasetStyles('');
        } catch (e: any) {
            setError(e.response?.data?.detail || 'Failed to create dataset');
        } finally {
            setIsCreatingDataset(false);
        }
    };

    const handlePreprocessDataset = async () => {
        if (!selectedDataset) return;
        setIsPreprocessing(true);
        setError(null);

        try {
            const result = await trainingApi.preprocessDataset(selectedDataset.id);
            if (result.success) {
                await loadDatasets();
                setError(null);
            } else {
                setError(result.message || 'Preprocessing completed with warnings');
            }
        } catch (e: any) {
            setError(e.response?.data?.detail || 'Failed to preprocess dataset');
        } finally {
            setIsPreprocessing(false);
        }
    };

    const handleFileUpload = async (files: FileList | null) => {
        if (!files || !selectedDataset) return;

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const isAudio = file.type.startsWith('audio/') || /\.(mp3|wav|flac|ogg|m4a)$/i.test(file.name);
            const isText = file.type === 'text/plain' || file.name.endsWith('.txt');

            if (!isAudio && !isText) continue;

            setUploadProgress(prev => ({ ...prev, [file.name]: true }));

            try {
                await trainingApi.uploadAudio(
                    selectedDataset.id,
                    file,
                    uploadCaption || file.name.replace(/\.[^/.]+$/, '')
                );
                const updated = await trainingApi.getDataset(selectedDataset.id);
                setSelectedDataset(updated);
                setDatasets(prev => prev.map(d => d.id === updated.id ? updated : d));
            } catch (e) {
                console.error('Failed to upload', file.name, e);
            } finally {
                setUploadProgress(prev => {
                    const next = { ...prev };
                    delete next[file.name];
                    return next;
                });
            }
        }
        setUploadCaption('');
    };

    const handlePlayAudio = (filename: string) => {
        if (!selectedDataset) return;

        const audioUrl = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/training/datasets/${selectedDataset.id}/audio/${encodeURIComponent(filename)}`;

        if (playingFile === filename && audioRef.current) {
            audioRef.current.pause();
            audioRef.current = null;
            setPlayingFile(null);
        } else {
            if (audioRef.current) {
                audioRef.current.pause();
            }
            const audio = new Audio(audioUrl);
            audio.onended = () => setPlayingFile(null);
            audio.play();
            audioRef.current = audio;
            setPlayingFile(filename);
        }
    };

    const handleDeleteDataset = async (datasetId: string) => {
        if (!confirm('Delete this dataset and all its audio files?')) return;
        try {
            await trainingApi.deleteDataset(datasetId);
            setDatasets(prev => prev.filter(d => d.id !== datasetId));
            if (selectedDataset?.id === datasetId) {
                setSelectedDataset(null);
            }
        } catch (e) {
            console.error('Failed to delete dataset', e);
        }
    };

    const handleEditDataset = (ds: Dataset) => {
        setEditingDataset(ds);
        setEditName(ds.name);
        setEditStyles(ds.styles.join(', '));
    };

    const handleSaveEdit = async () => {
        if (!editingDataset || !editName.trim()) return;
        try {
            const styles = editStyles.split(',').map(s => s.trim()).filter(Boolean);
            const updated = await trainingApi.updateDataset(editingDataset.id, editName.trim(), styles);
            setDatasets(prev => prev.map(d => d.id === updated.id ? updated : d));
            if (selectedDataset?.id === updated.id) {
                setSelectedDataset(updated);
            }
            setEditingDataset(null);
        } catch (e) {
            console.error('Failed to update dataset', e);
        }
    };

    const handleDeleteAudio = async (filename: string) => {
        if (!selectedDataset) return;
        try {
            await trainingApi.deleteAudio(selectedDataset.id, filename);
            const updatedDataset = {
                ...selectedDataset,
                audio_files: selectedDataset.audio_files.filter(af => af.filename !== filename)
            };
            setSelectedDataset(updatedDataset);
            setDatasets(prev => prev.map(d => d.id === updatedDataset.id ? updatedDataset : d));
        } catch (e) {
            console.error('Failed to delete audio file', e);
        }
    };

    const handleSaveLyrics = async () => {
        if (!selectedDataset || !editingLyrics) return;
        try {
            await trainingApi.updateAudioCaption(
                selectedDataset.id,
                editingLyrics.filename,
                editingLyrics.caption
            );
            const updatedDataset = {
                ...selectedDataset,
                audio_files: selectedDataset.audio_files.map(af =>
                    af.filename === editingLyrics.filename
                        ? { ...af, caption: editingLyrics.caption }
                        : af
                )
            };
            setSelectedDataset(updatedDataset);
            setDatasets(prev => prev.map(d => d.id === updatedDataset.id ? updatedDataset : d));
            setEditingLyrics(null);
        } catch (e) {
            console.error('Failed to update lyrics', e);
        }
    };

    const handleStartTraining = async () => {
        if (!selectedDataset) return;
        setError(null);

        try {
            const job = await trainingApi.startJob({
                dataset_id: selectedDataset.id,
                method: trainingMethod,
                epochs,
                learning_rate: learningRate,
                lora_rank: loraRank
            });
            setJobs(prev => [job, ...prev]);
            setActiveTab('jobs');
        } catch (e: any) {
            setError(e.response?.data?.detail || 'Failed to start training');
        }
    };

    const handleDeleteJob = async (jobId: string) => {
        if (!confirm('Delete this training job?')) return;
        try {
            await trainingApi.deleteJob(jobId);
            setJobs(prev => prev.filter(j => j.id !== jobId));
        } catch (e) {
            console.error('Failed to delete job', e);
        }
    };

    const handleCancelJob = async (jobId: string) => {
        if (!confirm('Cancel this training job?')) return;
        try {
            await trainingApi.cancelJob(jobId);
            await loadJobs();
        } catch (e) {
            console.error('Failed to cancel job', e);
        }
    };

    const handleActivateCheckpoint = async (id: string) => {
        try {
            await trainingApi.activateCheckpoint(id);
            await loadCheckpoints();
            onCheckpointsChange?.();
        } catch (e) {
            console.error('Failed to activate checkpoint', e);
        }
    };

    const handleDeleteCheckpoint = async (id: string) => {
        if (!confirm('Delete this checkpoint?')) return;
        try {
            await trainingApi.deleteCheckpoint(id);
            await loadCheckpoints();
            onCheckpointsChange?.();
        } catch (e) {
            console.error('Failed to delete checkpoint', e);
        }
    };

    const handleDeactivateCheckpoint = async () => {
        try {
            await trainingApi.deactivateCheckpoint();
            await loadCheckpoints();
            onCheckpointsChange?.();
        } catch (e) {
            console.error('Failed to deactivate checkpoint', e);
        }
    };

    const tabs = [
        { id: 'dataset' as Tab, label: 'Dataset Prep', icon: <FolderPlus className="w-4 h-4" /> },
        { id: 'training' as Tab, label: 'Training Config', icon: <Cpu className="w-4 h-4" /> },
        { id: 'jobs' as Tab, label: 'Jobs Monitor', icon: <Settings2 className="w-4 h-4" /> },
        { id: 'models' as Tab, label: 'Checkpoints', icon: <Package className="w-4 h-4" /> },
    ];

    return createPortal(
        <AnimatePresence>
            {isOpen && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 dark:bg-black/80 backdrop-blur-md p-4 animate-fade-in">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.96, y: 15 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.96, y: 15 }}
                        className="bg-white/95 dark:bg-[#12141c]/95 backdrop-blur-2xl rounded-3xl border border-black/[0.08] dark:border-white/10 shadow-apple-2xl w-full max-w-5xl overflow-hidden flex flex-col min-h-[85vh] max-h-[95vh]"
                    >
                        {/* Header */}
                        <div className="flex items-center justify-between p-5 border-b border-black/[0.06] dark:border-white/10 bg-black/[0.01] dark:bg-white/[0.02]">
                            <div className="flex items-center gap-3">
                                <span className="p-2 rounded-2xl bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20 text-base font-bold">
                                    🎓
                                </span>
                                <div>
                                    <div className="flex items-center gap-2">
                                        <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 tracking-tight">
                                            LoRA & Foundation Training Studio
                                        </h2>
                                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-600 dark:text-amber-400 font-bold border border-amber-500/30">
                                            In Development
                                        </span>
                                    </div>
                                    <p className="text-[11px] font-mono text-slate-400">
                                        Experimental LoRA fine-tuning & dataset tokenization (Feature preview)
                                    </p>
                                </div>
                            </div>
                            <button
                                onClick={onClose}
                                className="p-2 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Segmented Tabs Bar */}
                        <div className="flex border-b border-black/[0.06] dark:border-white/10 bg-black/[0.01] dark:bg-[#0f1118] p-2 gap-2">
                            {tabs.map(tab => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`flex-1 py-2 text-xs font-semibold rounded-xl transition-all flex items-center justify-center gap-2 ${activeTab === tab.id
                                        ? 'bg-white dark:bg-white/15 shadow-apple-sm text-teal-700 dark:text-teal-300 font-bold border border-black/[0.04] dark:border-white/10'
                                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 hover:bg-black/[0.03] dark:hover:bg-white/5'
                                        }`}
                                >
                                    {tab.icon}
                                    <span>{tab.label}</span>
                                </button>
                            ))}
                        </div>

                        {/* Content Body */}
                        <div className="flex-1 overflow-y-auto p-6 bg-slate-50/50 dark:bg-[#141620] text-slate-800 dark:text-slate-200">
                            {error && (
                                <div className="mb-4 p-3.5 bg-rose-500/10 border border-rose-500/20 rounded-2xl text-rose-600 dark:text-rose-400 text-xs flex items-center gap-2.5 font-medium">
                                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                                    <span>{error}</span>
                                </div>
                            )}

                            {/* Dataset Tab */}
                            {activeTab === 'dataset' && (
                                <div className="space-y-6">
                                    {/* Create Dataset Card */}
                                    <div className="bg-white/80 dark:bg-[#181a24]/90 rounded-2xl p-5 border border-black/[0.06] dark:border-white/10 shadow-apple-sm space-y-4 backdrop-blur-xl">
                                        <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2 uppercase tracking-wider">
                                            <FolderPlus className="w-4 h-4 text-teal-500" />
                                            <span>Create New Dataset</span>
                                        </h4>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                            <input
                                                type="text"
                                                value={newDatasetName}
                                                onChange={(e) => setNewDatasetName(e.target.value)}
                                                placeholder="Dataset name (e.g., Afrobeat Master Series)"
                                                className="w-full apple-input text-xs"
                                            />
                                            <input
                                                type="text"
                                                value={newDatasetStyles}
                                                onChange={(e) => setNewDatasetStyles(e.target.value)}
                                                placeholder="Target style tags (e.g., Afrobeat, Highlife, Horns)"
                                                className="w-full apple-input text-xs"
                                            />
                                        </div>
                                        <button
                                            onClick={handleCreateDataset}
                                            disabled={isCreatingDataset || !newDatasetName.trim()}
                                            className="px-4 py-2 bg-teal-500 hover:bg-teal-400 disabled:opacity-40 text-slate-950 font-bold text-xs rounded-xl transition-all shadow-sm flex items-center gap-2 active:scale-95"
                                        >
                                            {isCreatingDataset ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <FolderPlus className="w-3.5 h-3.5" />}
                                            <span>Create Dataset</span>
                                        </button>
                                    </div>

                                    {/* Edit Dataset Modal */}
                                    {editingDataset && (
                                        <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/60 backdrop-blur-md p-4 animate-fade-in">
                                            <div className="bg-white/95 dark:bg-[#141620]/95 backdrop-blur-2xl border border-black/[0.08] dark:border-white/10 rounded-3xl p-6 w-full max-w-md shadow-apple-2xl space-y-4">
                                                <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">Edit Dataset</h4>
                                                <div className="space-y-3">
                                                    <input
                                                        type="text"
                                                        value={editName}
                                                        onChange={(e) => setEditName(e.target.value)}
                                                        placeholder="Dataset name"
                                                        className="w-full apple-input text-xs"
                                                    />
                                                    <input
                                                        type="text"
                                                        value={editStyles}
                                                        onChange={(e) => setEditStyles(e.target.value)}
                                                        placeholder="Styles (comma-separated)"
                                                        className="w-full apple-input text-xs"
                                                    />
                                                </div>
                                                <div className="flex justify-end gap-2 pt-2">
                                                    <button
                                                        onClick={() => setEditingDataset(null)}
                                                        className="px-3.5 py-1.5 rounded-xl bg-black/[0.04] dark:bg-white/5 text-slate-700 dark:text-slate-300 font-bold text-xs hover:bg-black/[0.08] dark:hover:bg-white/10"
                                                    >
                                                        Cancel
                                                    </button>
                                                    <button
                                                        onClick={handleSaveEdit}
                                                        className="px-4 py-1.5 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold rounded-xl text-xs transition-all shadow-sm"
                                                    >
                                                        Save Changes
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* Lyrics Editor Modal */}
                                    {editingLyrics && (
                                        <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/60 backdrop-blur-md p-4 animate-fade-in">
                                            <div className="bg-white/95 dark:bg-[#141620]/95 backdrop-blur-2xl border border-black/[0.08] dark:border-white/10 rounded-3xl p-6 w-full max-w-lg shadow-apple-2xl space-y-4">
                                                <div>
                                                    <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">Edit Audio Lyrics & Caption</h4>
                                                    <p className="text-[11px] font-mono text-slate-400 mt-0.5">{editingLyrics.filename}</p>
                                                </div>
                                                <textarea
                                                    value={editingLyrics.caption}
                                                    onChange={(e) => setEditingLyrics({ ...editingLyrics, caption: e.target.value })}
                                                    placeholder="[Verse]&#10;Add synchronized lyrics or acoustic prompt descriptors..."
                                                    rows={8}
                                                    className="w-full apple-input text-xs font-mono p-3 leading-relaxed"
                                                />
                                                <div className="flex justify-end gap-2 pt-2">
                                                    <button
                                                        onClick={() => setEditingLyrics(null)}
                                                        className="px-3.5 py-1.5 rounded-xl bg-black/[0.04] dark:bg-white/5 text-slate-700 dark:text-slate-300 font-bold text-xs"
                                                    >
                                                        Cancel
                                                    </button>
                                                    <button
                                                        onClick={handleSaveLyrics}
                                                        className="px-4 py-1.5 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold rounded-xl text-xs shadow-sm"
                                                    >
                                                        Save Stanzas
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* Dataset Selector Grid */}
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                        {datasets.map(ds => (
                                            <div
                                                key={ds.id}
                                                onClick={() => setSelectedDataset(ds)}
                                                className={`p-4 rounded-2xl border transition-all relative group cursor-pointer ${selectedDataset?.id === ds.id
                                                    ? 'border-teal-500/60 bg-teal-500/10 dark:bg-teal-500/10 shadow-apple-sm'
                                                    : 'border-black/[0.06] dark:border-white/10 bg-white/70 dark:bg-[#181a24]/70 hover:border-black/[0.12] dark:hover:border-white/20'
                                                    }`}
                                            >
                                                <div className="pr-12">
                                                    <h5 className="font-bold text-xs text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
                                                        <Database size={13} className="text-teal-500" />
                                                        <span>{ds.name}</span>
                                                    </h5>
                                                    <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1 font-mono">
                                                        {ds.audio_files.length} audio tracks • {ds.styles.join(', ') || 'No style tags'}
                                                    </p>
                                                </div>
                                                <div className="absolute top-3 right-3 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                                    <button
                                                        onClick={(e) => { e.stopPropagation(); handleEditDataset(ds); }}
                                                        className="p-1.5 text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 hover:bg-black/5 dark:hover:bg-white/10 rounded-lg transition-colors"
                                                        title="Edit Dataset"
                                                    >
                                                        <Edit2 className="w-3.5 h-3.5" />
                                                    </button>
                                                    <button
                                                        onClick={(e) => { e.stopPropagation(); handleDeleteDataset(ds.id); }}
                                                        className="p-1.5 text-slate-400 hover:text-rose-500 hover:bg-rose-500/10 rounded-lg transition-colors"
                                                        title="Delete Dataset"
                                                    >
                                                        <Trash2 className="w-3.5 h-3.5" />
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>

                                    {/* Selected Dataset Upload & Files */}
                                    {selectedDataset && (
                                        <div className="border border-black/[0.06] dark:border-white/10 rounded-2xl p-5 bg-white/80 dark:bg-[#181a24]/90 shadow-apple-sm space-y-4 backdrop-blur-xl">
                                            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-black/[0.04] dark:border-white/5 pb-3">
                                                <div>
                                                    <h4 className="font-bold text-xs text-slate-900 dark:text-slate-100 flex items-center gap-2">
                                                        <Music className="w-4 h-4 text-teal-500" />
                                                        <span>{selectedDataset.name}</span>
                                                    </h4>
                                                    <div className="flex flex-wrap gap-1 mt-1.5">
                                                        {selectedDataset.styles.map((style, i) => (
                                                            <span key={i} className="bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20 px-2 py-0.5 rounded-md text-[10px] font-mono font-bold">
                                                                {style}
                                                            </span>
                                                        ))}
                                                    </div>
                                                </div>
                                                <div className={`px-2.5 py-1 rounded-xl text-xs font-mono font-bold flex items-center gap-1.5 ${selectedDataset.audio_files.length >= 5
                                                    ? 'bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20'
                                                    : 'bg-amber-500/10 text-amber-700 dark:text-amber-300 border border-amber-500/20'
                                                    }`}>
                                                    {selectedDataset.audio_files.length >= 5 ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
                                                    <span>{selectedDataset.audio_files.length}/5 files required</span>
                                                </div>
                                            </div>

                                            {/* Upload Dropzone */}
                                            <div
                                                className="border-2 border-dashed border-black/[0.08] dark:border-white/10 rounded-2xl p-6 text-center hover:border-teal-500/60 transition-colors cursor-pointer bg-black/[0.01] dark:bg-white/[0.01] group"
                                                onDragOver={(e) => e.preventDefault()}
                                                onDrop={(e) => {
                                                    e.preventDefault();
                                                    handleFileUpload(e.dataTransfer.files);
                                                }}
                                                onClick={() => document.getElementById('audio-upload')?.click()}
                                            >
                                                <Upload className="w-7 h-7 text-slate-400 group-hover:text-teal-500 mx-auto mb-2 transition-colors" />
                                                <p className="text-xs font-bold text-slate-800 dark:text-slate-200">Drag & drop training audio files here or click to browse</p>
                                                <p className="text-[11px] text-slate-400 mt-1 font-mono">WAV, MP3, FLAC (48kHz recommended) + optional .txt lyrics files</p>
                                                <input
                                                    id="audio-upload"
                                                    type="file"
                                                    multiple
                                                    accept="audio/*,.txt"
                                                    className="hidden"
                                                    onChange={(e) => handleFileUpload(e.target.files)}
                                                />
                                                {Object.keys(uploadProgress).length > 0 && (
                                                    <div className="mt-3 flex items-center justify-center gap-2 text-teal-600 dark:text-teal-400 text-xs font-mono font-bold">
                                                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                        <span>Uploading {Object.keys(uploadProgress).length} file(s)...</span>
                                                    </div>
                                                )}
                                            </div>

                                            {/* File List */}
                                            {selectedDataset.audio_files.length > 0 && (
                                                <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                                                    {selectedDataset.audio_files.map((af, i) => (
                                                        <div key={i} className="flex items-center justify-between px-3 py-2 bg-black/[0.02] dark:bg-white/[0.03] rounded-xl border border-black/[0.04] dark:border-white/5 text-xs group">
                                                            <div className="flex items-center gap-2 truncate flex-1 pr-2">
                                                                <Music size={13} className="text-teal-500 flex-shrink-0" />
                                                                <span className="text-slate-800 dark:text-slate-200 truncate font-mono text-[11px]">{af.filename}</span>
                                                            </div>
                                                            <span
                                                                className="text-[10px] font-mono text-slate-400 truncate max-w-[180px] cursor-pointer hover:text-teal-500 mr-2"
                                                                onClick={() => setEditingLyrics({ filename: af.filename, caption: af.caption })}
                                                                title="Click to edit lyrics caption"
                                                            >
                                                                {af.caption || '(click to add lyrics)'}
                                                            </span>
                                                            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                                                <button
                                                                    onClick={() => handlePlayAudio(af.filename)}
                                                                    className={`p-1.5 rounded-lg ${playingFile === af.filename ? 'text-teal-500 bg-teal-500/10' : 'text-slate-400 hover:text-teal-500 hover:bg-black/5 dark:hover:bg-white/10'}`}
                                                                    title={playingFile === af.filename ? 'Stop' : 'Play'}
                                                                >
                                                                    {playingFile === af.filename ? <Square className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                                                                </button>
                                                                <button
                                                                    onClick={() => setEditingLyrics({ filename: af.filename, caption: af.caption })}
                                                                    className="p-1.5 text-slate-400 hover:text-teal-500 hover:bg-black/5 dark:hover:bg-white/10 rounded-lg"
                                                                    title="Edit Lyrics"
                                                                >
                                                                    <Edit2 className="w-3 h-3" />
                                                                </button>
                                                                <button
                                                                    onClick={() => handleDeleteAudio(af.filename)}
                                                                    className="p-1.5 text-slate-400 hover:text-rose-500 hover:bg-rose-500/10 rounded-lg"
                                                                    title="Remove File"
                                                                >
                                                                    <Trash2 className="w-3 h-3" />
                                                                </button>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            )}

                                            {/* Preprocess Tokenizer Button */}
                                            {selectedDataset.audio_files.length > 0 && (
                                                <button
                                                    onClick={handlePreprocessDataset}
                                                    disabled={isPreprocessing}
                                                    className="w-full py-2.5 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 border border-black/[0.06] dark:border-white/10 text-slate-800 dark:text-slate-200 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-2 disabled:opacity-50"
                                                >
                                                    {isPreprocessing ? (
                                                        <>
                                                            <Loader2 className="w-3.5 h-3.5 animate-spin text-teal-500" />
                                                            <span>Tokenizing RVQ Discrete Audio...</span>
                                                        </>
                                                    ) : (
                                                        <>
                                                            <Cpu className="w-3.5 h-3.5 text-teal-500" />
                                                            <span>Tokenize RVQ Discrete Audio</span>
                                                        </>
                                                    )}
                                                </button>
                                            )}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Training Config Tab */}
                            {activeTab === 'training' && (
                                <div className="space-y-6">
                                    {!selectedDataset ? (
                                        <div className="text-center py-16 text-slate-400">
                                            <Database className="w-12 h-12 mx-auto mb-3 opacity-40 text-teal-500" />
                                            <p className="text-xs font-medium">Select or create a dataset first in the Dataset Prep tab</p>
                                        </div>
                                    ) : (
                                        <>
                                            {/* Method Selection */}
                                            <div className="space-y-3">
                                                <label className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider flex items-center">
                                                    Training Architecture
                                                    <HelpTooltip text="LoRA: Fast, memory-efficient adapter training. Creates a ~100MB rank adapter. Full: Trains entire 3B model weights." />
                                                </label>
                                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                                    <button
                                                        onClick={() => setTrainingMethod('lora')}
                                                        className={`p-4 rounded-2xl border text-left transition-all ${trainingMethod === 'lora'
                                                            ? 'border-teal-500/60 bg-teal-500/10 shadow-apple-sm'
                                                            : 'border-black/[0.06] dark:border-white/10 bg-white/70 dark:bg-[#181a24]/70'
                                                            }`}
                                                    >
                                                        <h5 className="font-bold text-xs text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
                                                            <Sparkles size={14} className="text-teal-500" />
                                                            <span>LoRA Adapter (Recommended)</span>
                                                        </h5>
                                                        <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">Fast • Low VRAM • ~100MB checkpoint</p>
                                                    </button>
                                                    <button
                                                        onClick={() => setTrainingMethod('full')}
                                                        className={`p-4 rounded-2xl border text-left transition-all ${trainingMethod === 'full'
                                                            ? 'border-teal-500/60 bg-teal-500/10 shadow-apple-sm'
                                                            : 'border-black/[0.06] dark:border-white/10 bg-white/70 dark:bg-[#181a24]/70'
                                                            }`}
                                                    >
                                                        <h5 className="font-bold text-xs text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
                                                            <Cpu size={14} className="text-teal-500" />
                                                            <span>Full Parameter Fine-Tune</span>
                                                        </h5>
                                                        <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">Comprehensive weights • ~24GB+ VRAM required</p>
                                                    </button>
                                                </div>
                                            </div>

                                            {/* Hyperparameters Grid */}
                                            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                                                <div>
                                                    <label className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider flex items-center">
                                                        Epochs
                                                        <HelpTooltip text="Number of passes through dataset. Recommended: 5-10." />
                                                    </label>
                                                    <input
                                                        type="number"
                                                        value={epochs}
                                                        onChange={(e) => setEpochs(Number(e.target.value))}
                                                        min={1}
                                                        max={30}
                                                        className="w-full mt-2 apple-input text-xs"
                                                    />
                                                </div>
                                                <div>
                                                    <label className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider flex items-center">
                                                        Learning Rate
                                                        <HelpTooltip text="Step size for gradient descent. Recommended: 0.0003 for LoRA." />
                                                    </label>
                                                    <input
                                                        type="number"
                                                        value={learningRate}
                                                        onChange={(e) => setLearningRate(Number(e.target.value))}
                                                        step={0.00001}
                                                        className="w-full mt-2 apple-input text-xs"
                                                    />
                                                </div>
                                                {trainingMethod === 'lora' && (
                                                    <div>
                                                        <label className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider flex items-center">
                                                            LoRA Rank
                                                            <HelpTooltip text="Dimensional rank of LoRA weights (8, 16, 32)." />
                                                        </label>
                                                        <input
                                                            type="number"
                                                            value={loraRank}
                                                            onChange={(e) => setLoraRank(Number(e.target.value))}
                                                            min={4}
                                                            max={64}
                                                            className="w-full mt-2 apple-input text-xs"
                                                        />
                                                    </div>
                                                )}
                                            </div>

                                            {/* Start Training CTA */}
                                            <button
                                                onClick={handleStartTraining}
                                                disabled={selectedDataset.audio_files.length < 5}
                                                className="w-full py-3 bg-teal-500 hover:bg-teal-400 disabled:opacity-40 text-slate-950 font-bold rounded-2xl text-xs transition-all shadow-md shadow-teal-500/20 flex items-center justify-center gap-2 active:scale-95"
                                            >
                                                <Play className="w-4 h-4 fill-current" />
                                                <span>Start Training on "{selectedDataset.name}"</span>
                                            </button>
                                        </>
                                    )}
                                </div>
                            )}

                            {/* Jobs Monitor Tab */}
                            {activeTab === 'jobs' && (
                                <div className="space-y-4">
                                    {isLoadingJobs ? (
                                        <div className="flex items-center justify-center py-16">
                                            <Loader2 className="w-6 h-6 animate-spin text-teal-500" />
                                        </div>
                                    ) : jobs.length === 0 ? (
                                        <div className="text-center py-16 text-slate-400">
                                            <Settings2 className="w-12 h-12 mx-auto mb-3 opacity-40" />
                                            <p className="text-xs font-medium">No training jobs active or recorded</p>
                                        </div>
                                    ) : (
                                        jobs.map(job => {
                                            const dataset = datasets.find(d => d.id === job.dataset_id);
                                            const displayName = job.dataset_name || dataset?.name || 'Dataset';
                                            return (
                                                <div key={job.id} className="p-4 bg-white/80 dark:bg-[#181a24]/90 border border-black/[0.06] dark:border-white/10 rounded-2xl shadow-apple-sm space-y-3 backdrop-blur-xl">
                                                    <div className="flex items-center justify-between">
                                                        <div>
                                                            <h5 className="font-bold text-xs text-slate-900 dark:text-slate-100 flex items-center gap-2">
                                                                <span>{job.config.method === 'lora' ? '⚡' : '🔥'}</span>
                                                                <span>{displayName}</span>
                                                            </h5>
                                                            <p className="text-[11px] font-mono text-slate-500 dark:text-slate-400 mt-0.5">
                                                                {job.config.method.toUpperCase()} • {job.config.epochs} epochs • LR: {job.config.learning_rate}
                                                            </p>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold ${job.status === 'completed' ? 'bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20' :
                                                                job.status === 'running' ? 'bg-sky-500/10 text-sky-700 dark:text-sky-300 border border-sky-500/20 animate-pulse' :
                                                                    job.status === 'preprocessing' ? 'bg-cyan-500/10 text-cyan-700 dark:text-cyan-300 border border-cyan-500/20' :
                                                                        job.status === 'failed' ? 'bg-rose-500/10 text-rose-700 dark:text-rose-300 border border-rose-500/20' :
                                                                            'bg-black/5 dark:bg-white/5 text-slate-500'
                                                                }`}>
                                                                {job.status.toUpperCase()}
                                                            </span>
                                                            {(job.status === 'running' || job.status === 'preprocessing') && (
                                                                <button
                                                                    onClick={() => handleCancelJob(job.id)}
                                                                    className="px-2.5 py-1 text-[11px] font-bold bg-amber-500/10 text-amber-700 dark:text-amber-400 hover:bg-amber-500/20 rounded-xl transition-colors"
                                                                >
                                                                    Cancel
                                                                </button>
                                                            )}
                                                            <button
                                                                onClick={() => handleDeleteJob(job.id)}
                                                                className="p-1.5 text-slate-400 hover:text-rose-500 hover:bg-rose-500/10 rounded-lg transition-colors"
                                                            >
                                                                <Trash2 className="w-3.5 h-3.5" />
                                                            </button>
                                                        </div>
                                                    </div>

                                                    {(job.status === 'running' || job.status === 'preprocessing') && (
                                                        <div className="space-y-1.5">
                                                            <div className="h-2 bg-black/[0.04] dark:bg-white/5 rounded-full overflow-hidden">
                                                                <div
                                                                    className="h-full bg-gradient-to-r from-teal-500 to-cyan-500 transition-all duration-300"
                                                                    style={{ width: `${job.progress}%` }}
                                                                />
                                                            </div>
                                                            <div className="flex justify-between items-center text-[10px] font-mono text-slate-400">
                                                                <span>{job.message || `Epoch ${job.current_epoch}/${job.total_epochs} • ${job.progress}%`}</span>
                                                                <div className="flex items-center gap-2">
                                                                    <span>⏱ {formatElapsedTime(job.started_at)}</span>
                                                                    {job.started_at && job.progress > 0 && job.progress < 100 && (
                                                                        <span className="text-teal-600 dark:text-teal-400 font-bold">
                                                                            ETA {formatETA(job.started_at, job.progress)}
                                                                        </span>
                                                                    )}
                                                                </div>
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            );
                                        })
                                    )}
                                </div>
                            )}

                            {/* Checkpoints Tab */}
                            {activeTab === 'models' && (
                                <div className="space-y-3">
                                    {checkpoints.length === 0 ? (
                                        <div className="text-center py-16 text-slate-400">
                                            <Package className="w-12 h-12 mx-auto mb-3 opacity-40 text-teal-500" />
                                            <p className="text-xs font-medium">No LoRA checkpoints trained yet</p>
                                        </div>
                                    ) : (
                                        checkpoints.map(ckpt => (
                                            <div key={ckpt.id} className={`p-4 bg-white/80 dark:bg-[#181a24]/90 border rounded-2xl flex items-center justify-between backdrop-blur-xl shadow-apple-sm ${ckpt.is_active ? 'border-teal-500/60 ring-2 ring-teal-500/20' : 'border-black/[0.06] dark:border-white/10'}`}>
                                                <div>
                                                    <h5 className="font-bold text-xs text-slate-900 dark:text-slate-100 flex items-center gap-2">
                                                        <span>{ckpt.name}</span>
                                                        {ckpt.is_active && (
                                                            <span className="px-2 py-0.5 bg-teal-500/10 text-teal-700 dark:text-teal-300 text-[10px] rounded-full font-bold font-mono border border-teal-500/20 flex items-center gap-1">
                                                                <span className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-pulse" />
                                                                ACTIVE
                                                            </span>
                                                        )}
                                                    </h5>
                                                    <p className="text-[11px] font-mono text-slate-500 dark:text-slate-400 mt-0.5">
                                                        {ckpt.method.toUpperCase()} • {ckpt.styles.join(', ')} • {(ckpt.size_bytes / 1024 / 1024).toFixed(1)} MB {ckpt.created_at && `• ${formatTimestamp(ckpt.created_at)}`}
                                                    </p>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    {ckpt.is_active ? (
                                                        <button
                                                            onClick={handleDeactivateCheckpoint}
                                                            className="px-3 py-1.5 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 rounded-xl text-xs font-bold transition-all"
                                                        >
                                                            Deactivate
                                                        </button>
                                                    ) : (
                                                        <button
                                                            onClick={() => handleActivateCheckpoint(ckpt.id)}
                                                            className="px-3.5 py-1.5 bg-teal-500 hover:bg-teal-400 text-slate-950 rounded-xl text-xs font-bold transition-all shadow-sm active:scale-95"
                                                        >
                                                            Activate Adapter
                                                        </button>
                                                    )}
                                                    <button
                                                        onClick={() => handleDeleteCheckpoint(ckpt.id)}
                                                        className="p-1.5 text-slate-400 hover:text-rose-500 hover:bg-rose-500/10 rounded-lg transition-colors"
                                                    >
                                                        <Trash2 className="w-4 h-4" />
                                                    </button>
                                                </div>
                                            </div>
                                        ))
                                    )}
                                </div>
                            )}
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>,
        document.body
    );
};
