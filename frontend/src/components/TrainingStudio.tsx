import React, { useState, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
    X, Upload, Play, Square, Trash2, FolderPlus, Edit2, HelpCircle,
    Settings2, Loader2, CheckCircle2, AlertCircle,
    Music, Database, Cpu, Package
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
        <div className="relative inline-block ml-1">
            <button
                type="button"
                onMouseEnter={() => setIsVisible(true)}
                onMouseLeave={() => setIsVisible(false)}
                onClick={(e) => { e.preventDefault(); setIsVisible(!isVisible); }}
                className="text-slate-400 hover:text-purple-500 transition-colors"
            >
                <HelpCircle className="w-3.5 h-3.5" />
            </button>
            <AnimatePresence>
                {isVisible && (
                    <motion.div
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 5 }}
                        className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-slate-800 text-white text-xs rounded-lg shadow-xl"
                    >
                        <div className="relative">
                            {text}
                            <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800" />
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

    // Audio preview state
    const [playingFile, setPlayingFile] = useState<string | null>(null);
    const audioRef = React.useRef<HTMLAudioElement | null>(null);

    const [error, setError] = useState<string | null>(null);

    // Load data when modal opens
    useEffect(() => {
        if (isOpen) {
            loadDatasets();
            loadJobs();
            loadCheckpoints();
        }
    }, [isOpen]);

    // Poll jobs every 3 seconds while modal is open for real-time updates
    useEffect(() => {
        if (!isOpen) return;

        const interval = setInterval(() => {
            loadJobs();
            // Also refresh checkpoints in case training completes
            loadCheckpoints();
        }, 3000);

        return () => clearInterval(interval);
    }, [isOpen]);

    const loadDatasets = async () => {
        try {
            const data = await trainingApi.listDatasets();
            setDatasets(data);
        } catch (e) {
            console.error('Failed to load datasets', e);
        }
    };

    const loadJobs = async () => {
        setIsLoadingJobs(true);
        try {
            const data = await trainingApi.listJobs();
            setJobs(data);
        } catch (e) {
            console.error('Failed to load jobs', e);
        } finally {
            setIsLoadingJobs(false);
        }
    };

    const loadCheckpoints = async () => {
        try {
            const data = await trainingApi.listCheckpoints();
            setCheckpoints(data);
        } catch (e) {
            console.error('Failed to load checkpoints', e);
        }
    };

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

        if (!confirm('This will re-process all audio files with the correct tag format. Continue?')) {
            return;
        }

        setIsPreprocessing(true);
        setError(null);

        try {
            const result = await trainingApi.preprocessDataset(selectedDataset.id, true);
            // Refresh dataset to update status
            const updated = await trainingApi.getDataset(selectedDataset.id);
            setSelectedDataset(updated);
            setDatasets(prev => prev.map(d => d.id === updated.id ? updated : d));
            alert(`Successfully processed ${result.processed_count} files!`);
        } catch (e: any) {
            setError(e.response?.data?.detail || 'Preprocessing failed');
        } finally {
            setIsPreprocessing(false);
        }
    };

    const handlePlayAudio = (filename: string) => {
        if (!selectedDataset) return;

        const audioUrl = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/training/datasets/${selectedDataset.id}/audio/${encodeURIComponent(filename)}`;

        if (playingFile === filename && audioRef.current) {
            // Stop playing
            audioRef.current.pause();
            audioRef.current = null;
            setPlayingFile(null);
        } else {
            // Start playing
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
        if (!confirm('Delete this dataset and all its files?')) return;
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
            // Update local state
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
            // Update local state
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

    const handleFileUpload = useCallback(async (files: FileList | null) => {
        if (!files || !selectedDataset) return;

        // Collect .txt files for lyrics
        const txtFiles = new Map<string, string>();
        for (const file of Array.from(files)) {
            if (file.name.endsWith('.txt')) {
                const text = await file.text();
                // Associate by basename (e.g., song.txt -> song.mp3)
                const baseName = file.name.replace(/\.txt$/, '');
                txtFiles.set(baseName, text);
            }
        }

        for (const file of Array.from(files)) {
            if (!file.type.startsWith('audio/')) continue;

            // Check if there's a matching .txt lyrics file
            const baseName = file.name.replace(/\.[^/.]+$/, '');
            const lyricsFromTxt = txtFiles.get(baseName);
            const caption = lyricsFromTxt || uploadCaption || baseName;

            setUploadProgress(prev => ({ ...prev, [file.name]: true }));

            try {
                await trainingApi.uploadAudio(selectedDataset.id, file, caption);
                // Refresh dataset
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
    }, [selectedDataset, uploadCaption]);

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

    const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
        { id: 'dataset', label: 'Dataset', icon: <Database className="w-4 h-4" /> },
        { id: 'training', label: 'Training', icon: <Cpu className="w-4 h-4" /> },
        { id: 'jobs', label: 'Jobs', icon: <Settings2 className="w-4 h-4" /> },
        { id: 'models', label: 'Models', icon: <Package className="w-4 h-4" /> },
    ];

    return createPortal(
        <AnimatePresence>
            {isOpen && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        className="bg-white/80 backdrop-blur-2xl rounded-xl border border-white/50 shadow-2xl w-full max-w-5xl overflow-hidden flex flex-col min-h-[85vh] max-h-[98vh] glass-panel"
                    >
                        {/* Header */}
                        <div className="flex items-center justify-between p-6 border-b border-white/20 bg-gradient-to-r from-cyan-50/80 to-fuchsia-50/80">
                            <h2 className="text-xl font-bold text-slate-800 flex items-center gap-3 tracking-tight">
                                <span className="p-2 bg-white/50 rounded-lg shadow-sm text-lg">🎓</span>
                                <span className="bg-clip-text text-transparent bg-gradient-to-r from-slate-800 to-slate-600">
                                    Training Studio
                                </span>
                            </h2>
                            <button
                                onClick={onClose}
                                className="p-2 rounded-full hover:bg-white/50 text-slate-400 hover:text-red-500 transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Tabs */}
                        <div className="flex border-b border-white/20 bg-white/30 p-2 gap-2">
                            {tabs.map(tab => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`flex-1 py-2.5 text-sm font-bold rounded-lg transition-all flex items-center justify-center gap-2 ${activeTab === tab.id
                                        ? 'bg-white shadow-sm text-cyan-700 ring-1 ring-black/5'
                                        : 'text-slate-500 hover:text-slate-700 hover:bg-white/40'
                                        } `}
                                >
                                    {tab.icon}
                                    {tab.label}
                                </button>
                            ))}
                        </div>

                        {/* Content */}
                        <div className="flex-1 overflow-y-auto p-6">
                            {error && (
                                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm flex items-center gap-2">
                                    <AlertCircle className="w-4 h-4" />
                                    {error}
                                </div>
                            )}

                            {/* Dataset Tab */}
                            {activeTab === 'dataset' && (
                                <div className="space-y-6">
                                    {/* Create Dataset */}
                                    <div className="bg-gradient-to-r from-purple-50 to-cyan-50 rounded-lg p-4 border border-purple-100">
                                        <h4 className="text-sm font-bold text-purple-800 mb-3 flex items-center gap-2">
                                            <FolderPlus className="w-4 h-4" />
                                            Create New Dataset
                                        </h4>
                                        <div className="grid grid-cols-2 gap-3">
                                            <input
                                                type="text"
                                                value={newDatasetName}
                                                onChange={(e) => setNewDatasetName(e.target.value)}
                                                placeholder="Dataset name (e.g., Afrobeat Collection)"
                                                className="px-3 py-2 border border-purple-200 rounded-md focus:ring-2 focus:ring-purple-400 focus:outline-none text-sm bg-white"
                                            />
                                            <input
                                                type="text"
                                                value={newDatasetStyles}
                                                onChange={(e) => setNewDatasetStyles(e.target.value)}
                                                placeholder="Target styles (comma-separated)"
                                                className="px-3 py-2 border border-purple-200 rounded-md focus:ring-2 focus:ring-purple-400 focus:outline-none text-sm bg-white"
                                            />
                                        </div>
                                        <button
                                            onClick={handleCreateDataset}
                                            disabled={isCreatingDataset || !newDatasetName.trim()}
                                            className="mt-3 px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-md font-bold text-sm transition-all disabled:opacity-50 flex items-center gap-2"
                                        >
                                            {isCreatingDataset ? <Loader2 className="w-4 h-4 animate-spin" /> : <FolderPlus className="w-4 h-4" />}
                                            Create Dataset
                                        </button>
                                    </div>

                                    {/* Edit Dataset Modal */}
                                    {editingDataset && (
                                        <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
                                            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-md">
                                                <h4 className="text-lg font-bold text-slate-800 mb-4">Edit Dataset</h4>
                                                <div className="space-y-3">
                                                    <input
                                                        type="text"
                                                        value={editName}
                                                        onChange={(e) => setEditName(e.target.value)}
                                                        placeholder="Dataset name"
                                                        className="w-full px-3 py-2 border border-slate-200 rounded-md focus:ring-2 focus:ring-purple-400 focus:outline-none"
                                                    />
                                                    <input
                                                        type="text"
                                                        value={editStyles}
                                                        onChange={(e) => setEditStyles(e.target.value)}
                                                        placeholder="Styles (comma-separated)"
                                                        className="w-full px-3 py-2 border border-slate-200 rounded-md focus:ring-2 focus:ring-purple-400 focus:outline-none"
                                                    />
                                                </div>
                                                <div className="flex justify-end gap-2 mt-4">
                                                    <button
                                                        onClick={() => setEditingDataset(null)}
                                                        className="px-4 py-2 text-slate-600 hover:bg-slate-100 rounded-md"
                                                    >
                                                        Cancel
                                                    </button>
                                                    <button
                                                        onClick={handleSaveEdit}
                                                        className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white rounded-md font-bold transition-all shadow-md"
                                                    >
                                                        Save
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* Lyrics Editor Modal */}
                                    {editingLyrics && (
                                        <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
                                            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-lg">
                                                <h4 className="text-lg font-bold text-slate-800 mb-2">Edit Lyrics / Caption</h4>
                                                <p className="text-sm text-slate-500 mb-4">File: {editingLyrics.filename}</p>
                                                <textarea
                                                    value={editingLyrics.caption}
                                                    onChange={(e) => setEditingLyrics({ ...editingLyrics, caption: e.target.value })}
                                                    placeholder="Enter lyrics or caption...

[Verse]
Example lyrics here...

[Chorus]
More lyrics..."
                                                    rows={10}
                                                    className="w-full px-3 py-2 border border-slate-200 rounded-md focus:ring-2 focus:ring-purple-400 focus:outline-none font-mono text-sm"
                                                />
                                                <p className="text-xs text-slate-400 mt-2">
                                                    Tip: Use [Verse], [Chorus], [Bridge] sections for best results
                                                </p>
                                                <div className="flex justify-end gap-2 mt-4">
                                                    <button
                                                        onClick={() => setEditingLyrics(null)}
                                                        className="px-4 py-2 text-slate-600 hover:bg-slate-100 rounded-md"
                                                    >
                                                        Cancel
                                                    </button>
                                                    <button
                                                        onClick={handleSaveLyrics}
                                                        className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white rounded-md font-bold transition-all shadow-md"
                                                    >
                                                        Save
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* Dataset List */}
                                    <div className="grid grid-cols-2 gap-3">
                                        {datasets.map(ds => (
                                            <div
                                                key={ds.id}
                                                className={`p-4 rounded-lg border transition-all relative group shadow-sm ${selectedDataset?.id === ds.id
                                                    ? 'border-cyan-400 bg-cyan-50/50 ring-2 ring-cyan-200'
                                                    : 'border-white/50 bg-white/40 hover:border-cyan-200 hover:bg-white/60'
                                                    }`}
                                            >
                                                <button
                                                    onClick={() => setSelectedDataset(ds)}
                                                    className="w-full text-left"
                                                >
                                                    <h5 className="font-medium text-slate-800">{ds.name}</h5>
                                                    <p className="text-xs text-slate-500 mt-1">
                                                        {ds.audio_files.length} files • {ds.styles.join(', ') || 'No styles'}
                                                    </p>
                                                </button>
                                                <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                                    <button
                                                        onClick={(e) => { e.stopPropagation(); handleEditDataset(ds); }}
                                                        className="p-1.5 text-slate-400 hover:text-cyan-600 hover:bg-cyan-50 rounded"
                                                        title="Edit"
                                                    >
                                                        <Edit2 className="w-3.5 h-3.5" />
                                                    </button>
                                                    <button
                                                        onClick={(e) => { e.stopPropagation(); handleDeleteDataset(ds.id); }}
                                                        className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded"
                                                        title="Delete"
                                                    >
                                                        <Trash2 className="w-3.5 h-3.5" />
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>

                                    {/* Selected Dataset Upload */}
                                    {selectedDataset && (
                                        <div className="border border-slate-200 rounded-lg p-4 bg-white">
                                            <h4 className="font-medium text-slate-800 mb-1 flex items-center gap-2">
                                                <Music className="w-4 h-4 text-cyan-600" />
                                                {selectedDataset.name}
                                            </h4>
                                            <div className="text-sm text-slate-500 mb-3 ml-6 flex gap-2">
                                                {selectedDataset.styles.map((style, i) => (
                                                    <span key={i} className="bg-gradient-to-r from-fuchsia-50 to-cyan-50 border border-cyan-100 text-slate-600 px-2 py-0.5 rounded-full text-xs font-mono">
                                                        {style}
                                                    </span>
                                                ))}
                                                {selectedDataset.styles.length === 0 && <span className="italic opacity-50">No target styles set</span>}
                                            </div>

                                            {/* Upload Zone */}
                                            <div
                                                className="border-2 border-dashed border-slate-200 rounded-lg p-6 text-center hover:border-purple-300 transition-colors cursor-pointer bg-slate-50"
                                                onDragOver={(e) => e.preventDefault()}
                                                onDrop={(e) => {
                                                    e.preventDefault();
                                                    handleFileUpload(e.dataTransfer.files);
                                                }}
                                                onClick={() => document.getElementById('audio-upload')?.click()}
                                            >
                                                <Upload className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                                                <p className="text-sm text-slate-600">Drag & drop audio files or click to browse</p>
                                                <p className="text-xs text-slate-400 mt-1">MP3, WAV, FLAC + matching .txt lyrics files</p>
                                                <input
                                                    id="audio-upload"
                                                    type="file"
                                                    multiple
                                                    accept="audio/*,.txt"
                                                    className="hidden"
                                                    onChange={(e) => handleFileUpload(e.target.files)}
                                                />
                                                {Object.keys(uploadProgress).length > 0 && (
                                                    <div className="mt-2 flex items-center gap-2 text-purple-600 text-xs">
                                                        <Loader2 className="w-3 h-3 animate-spin" />
                                                        Uploading {Object.keys(uploadProgress).length} file(s)...
                                                    </div>
                                                )}
                                            </div>

                                            {/* File List */}
                                            <div className="mt-4 space-y-2 max-h-40 overflow-y-auto">
                                                {selectedDataset.audio_files.map((af, i) => (
                                                    <div key={i} className="flex items-center justify-between px-3 py-2 bg-slate-50 rounded text-sm group">
                                                        <span className="text-slate-700 truncate flex-1">{af.filename}</span>
                                                        <span
                                                            className="text-xs text-slate-400 mr-2 truncate max-w-[150px] cursor-pointer hover:text-purple-500"
                                                            onClick={() => setEditingLyrics({ filename: af.filename, caption: af.caption })}
                                                            title="Click to edit lyrics"
                                                        >
                                                            {af.caption || '(no lyrics)'}
                                                        </span>
                                                        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                                            <button
                                                                onClick={() => handlePlayAudio(af.filename)}
                                                                className={`p-1 rounded ${playingFile === af.filename ? 'text-green-600 bg-green-50' : 'text-slate-300 hover:text-green-500 hover:bg-green-50'}`}
                                                                title={playingFile === af.filename ? 'Stop' : 'Play'}
                                                            >
                                                                {playingFile === af.filename ? <Square className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                                                            </button>
                                                            <button
                                                                onClick={() => setEditingLyrics({ filename: af.filename, caption: af.caption })}
                                                                className="p-1 text-slate-300 hover:text-purple-500 hover:bg-purple-50 rounded"
                                                                title="Edit lyrics"
                                                            >
                                                                <Edit2 className="w-3.5 h-3.5" />
                                                            </button>
                                                            <button
                                                                onClick={() => handleDeleteAudio(af.filename)}
                                                                className="p-1 text-slate-300 hover:text-red-500 hover:bg-red-50 rounded"
                                                                title="Remove file"
                                                            >
                                                                <Trash2 className="w-3.5 h-3.5" />
                                                            </button>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>

                                            {/* Validation */}
                                            <div className={`mt - 4 p - 3 rounded - lg text - sm flex items - center gap - 2 ${selectedDataset.audio_files.length >= 5
                                                ? 'bg-green-50 text-green-700 border border-green-200'
                                                : 'bg-amber-50 text-amber-700 border border-amber-200'
                                                } `}>
                                                {selectedDataset.audio_files.length >= 5
                                                    ? <CheckCircle2 className="w-4 h-4" />
                                                    : <AlertCircle className="w-4 h-4" />
                                                }
                                                {selectedDataset.audio_files.length}/5 files (minimum required)
                                            </div>

                                            {/* Re-process Button */}
                                            {selectedDataset.audio_files.length > 0 && (
                                                <button
                                                    onClick={handlePreprocessDataset}
                                                    disabled={isPreprocessing}
                                                    className="mt-3 w-full py-2.5 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-wait"
                                                >
                                                    {isPreprocessing ? (
                                                        <>
                                                            <Loader2 className="w-4 h-4 animate-spin" />
                                                            Processing...
                                                        </>
                                                    ) : (
                                                        <>
                                                            <Cpu className="w-4 h-4" />
                                                            Re-process Dataset
                                                        </>
                                                    )}
                                                </button>
                                            )}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Training Tab */}
                            {activeTab === 'training' && (
                                <div className="space-y-6">
                                    {!selectedDataset ? (
                                        <div className="text-center py-12 text-slate-400">
                                            <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
                                            <p>Select a dataset first in the Dataset tab</p>
                                        </div>
                                    ) : (
                                        <>
                                            {/* Method Selection */}
                                            <div className="space-y-3">
                                                <label className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center">
                                                    Training Method
                                                    <HelpTooltip text="LoRA: Fast, memory-efficient adapter training. Creates a small file that modifies the base model. Full: Trains entire model weights for best quality but requires more VRAM and produces larger files." />
                                                </label>
                                                <div className="grid grid-cols-2 gap-3">
                                                    <button
                                                        onClick={() => setTrainingMethod('lora')}
                                                        className={`p - 4 rounded - lg border text - left transition - all shadow-sm ${trainingMethod === 'lora'
                                                            ? 'border-cyan-400 bg-cyan-50 ring-2 ring-cyan-200'
                                                            : 'border-slate-200 bg-white hover:border-cyan-300'
                                                            } `}
                                                    >
                                                        <h5 className="font-medium text-slate-800">⚡ LoRA</h5>
                                                        <p className="text-xs text-slate-500 mt-1">Fast • ~16GB VRAM • ~100MB output</p>
                                                    </button>
                                                    <button
                                                        onClick={() => setTrainingMethod('full')}
                                                        className={`p - 4 rounded - lg border text - left transition - all shadow-sm ${trainingMethod === 'full'
                                                            ? 'border-cyan-400 bg-cyan-50 ring-2 ring-cyan-200'
                                                            : 'border-slate-200 bg-white hover:border-cyan-300'
                                                            } `}
                                                    >
                                                        <h5 className="font-medium text-slate-800">🔥 Full Fine-Tune</h5>
                                                        <p className="text-xs text-slate-500 mt-1">Best quality • ~24GB+ VRAM • ~6GB output</p>
                                                    </button>
                                                </div>
                                            </div>

                                            {/* Parameters */}
                                            <div className="grid grid-cols-3 gap-4">
                                                <div>
                                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center">
                                                        Epochs
                                                        <HelpTooltip text="Number of times the model sees the entire dataset. More epochs = better learning but risk of overfitting. Start with 3-5, increase if results aren't capturing your style." />
                                                    </label>
                                                    <input
                                                        type="number"
                                                        value={epochs}
                                                        onChange={(e) => setEpochs(Number(e.target.value))}
                                                        min={1}
                                                        max={10}
                                                        className="w-full mt-2 px-3 py-2 border border-slate-200 rounded-md focus:ring-2 focus:ring-purple-400 focus:outline-none"
                                                    />
                                                </div>
                                                <div>
                                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center">
                                                        Learning Rate
                                                        <HelpTooltip text="How fast the model adapts to your data. Lower = stable but slow. Higher = fast but unstable. Recommended: 0.0001 for LoRA, 0.00005 for Full." />
                                                    </label>
                                                    <input
                                                        type="number"
                                                        value={learningRate}
                                                        onChange={(e) => setLearningRate(Number(e.target.value))}
                                                        step={0.00001}
                                                        className="w-full mt-2 px-3 py-2 border border-slate-200 rounded-md focus:ring-2 focus:ring-purple-400 focus:outline-none"
                                                    />
                                                </div>
                                                {trainingMethod === 'lora' && (
                                                    <div>
                                                        <label className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center">
                                                            LoRA Rank
                                                            <HelpTooltip text="Complexity of the LoRA adapter. Higher rank = more expressive but larger file. 8 is good for most styles. Use 16-32 for complex multi-style training." />
                                                        </label>
                                                        <input
                                                            type="number"
                                                            value={loraRank}
                                                            onChange={(e) => setLoraRank(Number(e.target.value))}
                                                            min={4}
                                                            max={32}
                                                            className="w-full mt-2 px-3 py-2 border border-slate-200 rounded-md focus:ring-2 focus:ring-purple-400 focus:outline-none"
                                                        />
                                                    </div>
                                                )}
                                            </div>

                                            {/* Start Training */}
                                            <button
                                                onClick={handleStartTraining}
                                                disabled={selectedDataset.audio_files.length < 5}
                                                className="w-full py-3 bg-gradient-to-r from-purple-500 to-cyan-500 hover:from-purple-600 hover:to-cyan-600 text-white rounded-lg font-bold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                                            >
                                                <Play className="w-4 h-4" />
                                                Start Training on "{selectedDataset.name}"
                                            </button>
                                        </>
                                    )}
                                </div>
                            )}

                            {/* Jobs Tab */}
                            {activeTab === 'jobs' && (
                                <div className="space-y-4">
                                    {isLoadingJobs ? (
                                        <div className="flex items-center justify-center py-12">
                                            <Loader2 className="w-6 h-6 animate-spin text-purple-500" />
                                        </div>
                                    ) : jobs.length === 0 ? (
                                        <div className="text-center py-12 text-slate-400">
                                            <Settings2 className="w-12 h-12 mx-auto mb-3 opacity-50" />
                                            <p>No training jobs yet</p>
                                        </div>
                                    ) : (
                                        jobs.map(job => {
                                            const dataset = datasets.find(d => d.id === job.dataset_id);
                                            const displayName = job.dataset_name || dataset?.name || 'Unknown Dataset';
                                            return (
                                                <div key={job.id} className="p-4 bg-white border border-slate-200 rounded-lg">
                                                    <div className="flex items-center justify-between">
                                                        <div>
                                                            <h5 className="font-medium text-slate-800 flex items-center gap-2">
                                                                <span className="text-lg">{job.config.method === 'lora' ? '⚡' : '🔥'}</span>
                                                                {displayName}
                                                            </h5>
                                                            <p className="text-xs text-slate-500 mt-0.5">
                                                                {job.config.method.toUpperCase()} • {job.config.epochs} epochs • LR: {job.config.learning_rate}
                                                            </p>
                                                            {dataset?.styles && dataset.styles.length > 0 && (
                                                                <div className="flex flex-wrap gap-1 mt-1">
                                                                    {dataset.styles.slice(0, 3).map((style, i) => (
                                                                        <span key={i} className="text-[10px] bg-purple-50 text-purple-600 px-1.5 py-0.5 rounded-full border border-purple-100">
                                                                            {style}
                                                                        </span>
                                                                    ))}
                                                                    {dataset.styles.length > 3 && (
                                                                        <span className="text-[10px] text-slate-400">+{dataset.styles.length - 3} more</span>
                                                                    )}
                                                                </div>
                                                            )}
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <span className={`px-2 py-1 rounded-full text-xs font-bold ${job.status === 'completed' ? 'bg-green-100 text-green-700' :
                                                                job.status === 'running' ? 'bg-blue-100 text-blue-700' :
                                                                    job.status === 'preprocessing' ? 'bg-cyan-100 text-cyan-700' :
                                                                        job.status === 'failed' ? 'bg-red-100 text-red-700' :
                                                                            'bg-slate-100 text-slate-700'
                                                                }`}>
                                                                {job.status}
                                                            </span>
                                                            {/* Timestamp */}
                                                            {(job.created_at || job.started_at) && (
                                                                <span className="text-[10px] text-slate-400">
                                                                    {formatTimestamp(job.started_at || job.created_at)}
                                                                </span>
                                                            )}
                                                            {(job.status === 'running' || job.status === 'preprocessing') && (
                                                                <button
                                                                    onClick={() => handleCancelJob(job.id)}
                                                                    className="px-2 py-1 text-xs font-bold bg-amber-100 text-amber-700 hover:bg-amber-200 rounded transition-colors"
                                                                    title="Cancel training"
                                                                >
                                                                    Cancel
                                                                </button>
                                                            )}
                                                            <button
                                                                onClick={() => handleDeleteJob(job.id)}
                                                                className="p-1.5 text-slate-300 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                                                                title="Delete job"
                                                            >
                                                                <Trash2 className="w-4 h-4" />
                                                            </button>
                                                        </div>
                                                    </div>
                                                    {(job.status === 'running' || job.status === 'preprocessing') && (
                                                        <div className="mt-3">
                                                            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                                                                <div
                                                                    className={`h - full transition - all ${job.status === 'preprocessing'
                                                                        ? 'bg-gradient-to-r from-cyan-400 to-blue-400'
                                                                        : 'bg-gradient-to-r from-purple-500 to-cyan-500'
                                                                        } `}
                                                                    style={{ width: `${job.progress}% ` }}
                                                                />
                                                            </div>
                                                            <div className="flex justify-between items-center mt-1">
                                                                <p className="text-xs text-slate-500">
                                                                    {job.message || (job.status === 'preprocessing' ? 'Preparing data...' : `Epoch ${job.current_epoch}/${job.total_epochs} • ${job.progress}%`)}
                                                                </p>
                                                                <div className="flex items-center gap-3">
                                                                    {/* Elapsed Time */}
                                                                    {job.started_at && (
                                                                        <p className="text-[10px] text-slate-400">
                                                                            ⏱ {formatElapsedTime(job.started_at)}
                                                                        </p>
                                                                    )}
                                                                    {/* ETA */}
                                                                    {job.started_at && job.progress > 0 && job.progress < 100 && (
                                                                        <p className="text-[10px] text-cyan-600 font-medium">
                                                                            ETA: {formatETA(job.started_at, job.progress)}
                                                                        </p>
                                                                    )}
                                                                    {/* Loss */}
                                                                    {job.status === 'running' && job.current_loss != null && (
                                                                        <p className="text-xs font-mono text-slate-400">
                                                                            Loss: <span className="text-slate-600 font-bold">{job.current_loss.toFixed(4)}</span>
                                                                        </p>
                                                                    )}
                                                                </div>
                                                            </div>
                                                        </div>
                                                    )}
                                                    {/* Error Display for Failed Jobs */}
                                                    {job.status === 'failed' && (job.error || job.message) && (
                                                        <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                                                            <div className="flex items-start gap-2">
                                                                <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
                                                                <div className="flex-1 min-w-0">
                                                                    <p className="text-xs font-bold text-red-700">Training Failed</p>
                                                                    <p className="text-xs text-red-600 mt-1 break-words whitespace-pre-wrap font-mono">
                                                                        {job.error || job.message || 'Unknown error occurred'}
                                                                    </p>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    )}
                                                    {/* Completed Job Summary */}
                                                    {job.status === 'completed' && (
                                                        <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg">
                                                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                                                                {/* Duration */}
                                                                <div>
                                                                    <span className="text-slate-500">Duration</span>
                                                                    <p className="font-bold text-slate-700">
                                                                        {job.started_at && job.completed_at ? (() => {
                                                                            const start = new Date(job.started_at).getTime();
                                                                            const end = new Date(job.completed_at).getTime();
                                                                            const mins = Math.floor((end - start) / 60000);
                                                                            const secs = Math.floor(((end - start) % 60000) / 1000);
                                                                            return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
                                                                        })() : '--'}
                                                                    </p>
                                                                </div>
                                                                {/* Loss */}
                                                                <div>
                                                                    <span className="text-slate-500">Loss</span>
                                                                    <p className="font-bold text-slate-700">
                                                                        {job.initial_loss != null && job.final_loss != null ? (
                                                                            <span>
                                                                                {job.initial_loss.toFixed(3)} → {job.final_loss.toFixed(3)}
                                                                                <span className="text-green-600 ml-1">
                                                                                    ({((1 - job.final_loss / job.initial_loss) * 100).toFixed(0)}% ↓)
                                                                                </span>
                                                                            </span>
                                                                        ) : job.final_loss != null ? job.final_loss.toFixed(4) : '--'}
                                                                    </p>
                                                                </div>
                                                                {/* Epochs */}
                                                                <div>
                                                                    <span className="text-slate-500">Epochs</span>
                                                                    <p className="font-bold text-slate-700">{job.config?.epochs || job.total_epochs}</p>
                                                                </div>
                                                                {/* LoRA Rank */}
                                                                <div>
                                                                    <span className="text-slate-500">LoRA Rank</span>
                                                                    <p className="font-bold text-slate-700">{job.config?.lora_rank || '--'}</p>
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

                            {/* Models Tab */}
                            {activeTab === 'models' && (
                                <div className="space-y-4">
                                    {checkpoints.length === 0 ? (
                                        <div className="text-center py-12 text-slate-400">
                                            <Package className="w-12 h-12 mx-auto mb-3 opacity-50" />
                                            <p>No trained models yet</p>
                                        </div>
                                    ) : (
                                        checkpoints.map(ckpt => (
                                            <div key={ckpt.id} className={`p-4 bg-white border rounded-lg flex items-center justify-between ${ckpt.is_active ? 'border-purple-300 ring-2 ring-purple-100' : 'border-slate-200'}`}>
                                                <div>
                                                    <h5 className="font-medium text-slate-800 flex items-center gap-2">
                                                        {ckpt.name}
                                                        {ckpt.is_active && (
                                                            <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full font-bold flex items-center gap-1">
                                                                <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
                                                                ACTIVE
                                                            </span>
                                                        )}
                                                    </h5>
                                                    <p className="text-xs text-slate-500">
                                                        {ckpt.method} • {ckpt.styles.join(', ')} • {(ckpt.size_bytes / 1024 / 1024).toFixed(1)}MB
                                                        {ckpt.created_at && <span className="ml-1">• {formatTimestamp(ckpt.created_at)}</span>}
                                                    </p>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    {ckpt.is_active ? (
                                                        <button
                                                            onClick={handleDeactivateCheckpoint}
                                                            className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded text-xs font-bold transition-all shadow-sm"
                                                            title="Switch back to base model"
                                                        >
                                                            Deactivate
                                                        </button>
                                                    ) : (
                                                        <button
                                                            onClick={() => handleActivateCheckpoint(ckpt.id)}
                                                            className="px-3 py-1.5 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white rounded text-xs font-bold transition-all shadow-sm"
                                                        >
                                                            Activate
                                                        </button>
                                                    )}
                                                    <button
                                                        onClick={() => handleDeleteCheckpoint(ckpt.id)}
                                                        className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded"
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
