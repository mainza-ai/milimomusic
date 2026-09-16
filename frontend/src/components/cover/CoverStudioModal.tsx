import React, { useState, useRef } from 'react';
import axios from 'axios';
import {
    X,
    Sparkles,
    Upload,
    Sliders,
    Disc,
    FileAudio,
    Layers,
    Loader2,
    CheckCircle,
    Play,
    Pause,
    AlertCircle,
    Download,
    Music
} from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';
import { useModalA11y } from '../ui/primitives';
import { remixApi, modelsApi, api, API_BASE_URL, type Job, type LeadSheetResult, type ModelDownloadStatus } from '../../api';
import { toast } from '../../utils/toast';

interface CoverStudioModalProps {
    isOpen: boolean;
    onClose: () => void;
    initialTrack?: Job | null;
    initialMode?: 'audio' | 'midi';
    onCoverStarted?: (jobId: string) => void;
    onOpenPianoRoll?: (midiUrl: string) => void;
}

export const CoverStudioModal: React.FC<CoverStudioModalProps> = ({
    isOpen,
    onClose,
    initialTrack,
    initialMode = 'audio',
    onCoverStarted,
    onOpenPianoRoll
}) => {
    const modalRef = useRef<HTMLDivElement | null>(null);
    useModalA11y(isOpen, onClose, modalRef);

    const [mode, setMode] = useState<'audio' | 'midi'>(initialMode);
    const [title, setTitle] = useState(initialTrack?.title ? `${initialTrack.title} (Remix)` : 'New Remix');
    
    // Audio Reference
    const [refAudioPath, setRefAudioPath] = useState<string>(initialTrack?.audio_path || '');
    const [refAudioFile, setRefAudioFile] = useState<File | null>(null);
    const [isUploadingAudio, setIsUploadingAudio] = useState(false);
    
    // Direct MIDI
    const [melodyMidiPath, setMelodyMidiPath] = useState<string>(initialTrack?.melody_midi_path || '');
    const [chordMidiPath, setChordMidiPath] = useState<string>(initialTrack?.chord_midi_path || '');
    const [drumMidiPath, setDrumMidiPath] = useState<string>(initialTrack?.drum_midi_path || '');

    // MuLaCover Checkpoint State
    const [isModelInstalled, setIsModelInstalled] = useState<boolean | null>(null);
    const [isDownloadingModel, setIsDownloadingModel] = useState<boolean>(false);
    const [downloadStatus, setDownloadStatus] = useState<ModelDownloadStatus | null>(null);

    const checkModelInstallation = async () => {
        try {
            const dep = await modelsApi.checkDependencies('mulacover');
            setIsModelInstalled(!dep.missing);
        } catch (err) {
            console.error('Failed to verify MuLaCover dependencies:', err);
            try {
                const tree = await modelsApi.getModelTree();
                const m = tree.find(item => item.id === 'mulacover');
                setIsModelInstalled(m ? !!m.is_installed : false);
            } catch {
                setIsModelInstalled(null);
            }
        }
    };

    const handleDownloadBundle = async () => {
        setIsDownloadingModel(true);
        try {
            const res = await modelsApi.startModelDownload('HeartMuLa/MuLaCover', 'audio');
            const downloadId = res.id;
            toast('Starting MuLaCover bundle download (8.3 GB)...', 'info');

            const pollTimer = setInterval(async () => {
                try {
                    const status = await modelsApi.getModelDownload(downloadId);
                    setDownloadStatus(status);
                    if (status.status === 'completed') {
                        clearInterval(pollTimer);
                        setIsDownloadingModel(false);
                        setIsModelInstalled(true);
                        toast('MuLaCover bundle downloaded and ready!', 'success');
                    } else if (status.status === 'error') {
                        clearInterval(pollTimer);
                        setIsDownloadingModel(false);
                        toast(`Download failed: ${status.error || 'Network error'}`, 'error');
                    }
                } catch {
                    clearInterval(pollTimer);
                    setIsDownloadingModel(false);
                }
            }, 800);
        } catch (err: any) {
            setIsDownloadingModel(false);
            toast(err?.response?.data?.detail || 'Failed to start download', 'error');
        }
    };

    React.useEffect(() => {
        if (isOpen) {
            checkModelInstallation();
            if (initialMode) setMode(initialMode);
            if (initialTrack) {
                setTitle(initialTrack.title ? `${initialTrack.title} (Remix)` : 'New Remix');
                setRefAudioPath(initialTrack.audio_path || '');
                if (initialTrack.melody_midi_path) setMelodyMidiPath(initialTrack.melody_midi_path);
                if (initialTrack.chord_midi_path) setChordMidiPath(initialTrack.chord_midi_path);
                if (initialTrack.drum_midi_path) setDrumMidiPath(initialTrack.drum_midi_path);
                if (initialTrack.lyrics) setLyrics(initialTrack.lyrics);
            }
        }
    }, [isOpen, initialTrack, initialMode]);

    // Lead sheet extraction state
    const [isTranscribing, setIsTranscribing] = useState(false);
    const [leadSheet, setLeadSheet] = useState<LeadSheetResult | null>(null);
    const [transcriptionEngine, setTranscriptionEngine] = useState<'milimo_neural' | 'upstream'>('milimo_neural');

    // Musical Style & Direction
    const [genre, setGenre] = useState('Synthwave');
    const [mood, setMood] = useState('Energetic');
    const [instrument, setInstrument] = useState('Synthesizer, 808 Drums, Bass');
    const [topic, setTopic] = useState('Midnight highway, neon signs');
    const [bpm, setBpm] = useState<number>(120);

    // Lyrics
    const [lyrics, setLyrics] = useState<string>(initialTrack?.lyrics || '[Intro]\n\n[Verse]\nNeon lights in the rain\nWashing all the doubts away\n\n[Chorus]\nWe drive into the night\nEverything will be alright\n\n[Outro]');

    // Advanced Hyperparameters
    const [durationMs, setDurationMs] = useState<number>(120000);
    const [temperature, setTemperature] = useState<number>(1.0);
    const [cfgScale, setCfgScale] = useState<number>(1.5);
    const [topk, setTopk] = useState<number>(250);
    const [showAdvanced, setShowAdvanced] = useState(false);

    // Generation state
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Audio preview
    const [isPlayingPreview, setIsPlayingPreview] = useState(false);
    const audioPlayerRef = useRef<HTMLAudioElement | null>(null);

    if (!isOpen) return null;

    const handleUploadAudio = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setRefAudioFile(file);
        setIsUploadingAudio(true);
        try {
            const formData = new FormData();
            formData.append('file', file);
            const res = await axios.post<{ url?: string; path?: string; filename?: string }>(`${API_BASE_URL}/upload/audio`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            const data = res.data;
            const uploadedPath = data.path || data.url || `/audio/${data.filename}`;
            setRefAudioPath(uploadedPath);
            toast('Reference audio uploaded successfully!', 'success');
        } catch (err) {
            console.error(err);
            toast('Failed to upload reference audio', 'error');
        } finally {
            setIsUploadingAudio(false);
        }
    };

    const handleTranscribeLeadSheet = async (e?: React.MouseEvent) => {
        if (e) {
            e.preventDefault();
            e.stopPropagation();
        }
        if (!refAudioPath) {
            toast('Please upload or select reference audio first', 'error');
            return;
        }
        setIsTranscribing(true);
        try {
            const res = await remixApi.transcribeLeadSheet(refAudioPath, bpm, transcriptionEngine);
            setLeadSheet(res);
            if (res.bpm) setBpm(Math.round(res.bpm));
            toast(`Lead sheet transcribed! Tempo: ${Math.round(res.bpm)} BPM`, 'success');
        } catch (err: any) {
            console.error(err);
            toast(err.response?.data?.detail || 'Lead sheet transcription failed', 'error');
        } finally {
            setIsTranscribing(false);
        }
    };

    const handleSynthesize = async () => {
        if (isModelInstalled === false) {
            toast('MuLaCover checkpoints must be downloaded before synthesizing.', 'error');
            return;
        }
        if (mode === 'audio' && !refAudioPath) {
            toast('Reference audio is required for audio cover mode', 'error');
            return;
        }
        if (mode === 'midi' && (!melodyMidiPath || !chordMidiPath)) {
            toast('Both Melody MIDI and Chord MIDI are required for MIDI mode', 'error');
            return;
        }

        const tagsString = `topic:[${topic}]; genre:[${genre}]; instrument:[${instrument}]; mood:[${mood}]`;

        setIsSubmitting(true);
        try {
            const res = await remixApi.generateCover({
                title,
                ref_audio_path: mode === 'audio' ? refAudioPath : undefined,
                melody_midi_path: mode === 'midi' ? melodyMidiPath : undefined,
                chord_midi_path: mode === 'midi' ? chordMidiPath : undefined,
                drum_midi_path: mode === 'midi' ? (drumMidiPath || undefined) : undefined,
                bpm,
                lyrics,
                tags: tagsString,
                prompt: `${genre} remix, ${mood}`,
                duration_ms: durationMs,
                temperature,
                cfg_scale: cfgScale,
                topk,
                model_provider: 'mulacover',
                transcription_engine: transcriptionEngine
            });

            toast('MuLaCover remix generation queued! Follow progress in timeline.', 'success');
            if (onCoverStarted) {
                onCoverStarted(res.job_id);
            }
            onClose();
        } catch (err: any) {
            console.error(err);
            toast(err.response?.data?.detail || 'Failed to start cover generation', 'error');
        } finally {
            setIsSubmitting(false);
        }
    };

    const addSectionToLyrics = (sectionHeader: string) => {
        setLyrics(prev => `${prev.trim()}\n\n[${sectionHeader}]\n`);
    };

    const togglePreview = () => {
        if (!audioPlayerRef.current) return;
        if (isPlayingPreview) {
            audioPlayerRef.current.pause();
            setIsPlayingPreview(false);
        } else {
            audioPlayerRef.current.play();
            setIsPlayingPreview(true);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-md overflow-y-auto">
            <div ref={modalRef} className="relative w-full max-w-4xl my-8">
                <GlassCard className="p-6 border border-teal-500/30 shadow-2xl shadow-teal-500/10 rounded-2xl bg-zinc-950/90">
                    {/* Header */}
                    <div className="flex items-center justify-between pb-4 border-b border-zinc-800">
                        <div className="flex items-center gap-3">
                            <div className="p-2.5 rounded-xl bg-gradient-to-br from-teal-500/20 to-cyan-500/20 border border-teal-500/30 text-teal-400">
                                <Disc className="w-6 h-6 animate-spin-slow" />
                            </div>
                            <div>
                                <div className="flex items-center gap-2">
                                    <h2 className="text-xl font-bold text-white tracking-tight">MuLaCover Remix Studio</h2>
                                    <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/30">
                                        Production
                                    </span>
                                </div>
                                <p className="text-xs text-zinc-400">Controllable cover-song & music remixing via symbolic cross-attention</p>
                            </div>
                        </div>
                        <button
                            onClick={onClose}
                            className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
                        >
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    {/* Mode Selector Tabs */}
                    <div className="flex gap-2 mt-4 p-1 rounded-xl bg-zinc-900/80 border border-zinc-800">
                        <button
                            type="button"
                            onClick={() => setMode('audio')}
                            className={`flex-1 flex items-center justify-center gap-2 py-2 text-sm font-medium rounded-lg transition-all ${
                                mode === 'audio'
                                    ? 'bg-teal-600 text-white shadow-lg shadow-teal-600/30'
                                    : 'text-zinc-400 hover:text-zinc-200'
                            }`}
                        >
                            <FileAudio className="w-4 h-4" />
                            Reference Audio Mode
                        </button>
                        <button
                            type="button"
                            onClick={() => setMode('midi')}
                            className={`flex-1 flex items-center justify-center gap-2 py-2 text-sm font-medium rounded-lg transition-all ${
                                mode === 'midi'
                                    ? 'bg-teal-600 text-white shadow-lg shadow-teal-600/30'
                                    : 'text-zinc-400 hover:text-zinc-200'
                            }`}
                        >
                            <Layers className="w-4 h-4" />
                            Symbolic MIDI Lead Sheet Mode
                        </button>
                    </div>

                    {/* Checkpoints Missing Warning Card */}
                    {isModelInstalled === false && (
                        <div className="mt-4 p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-200">
                            <div className="flex items-start gap-3">
                                <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between">
                                        <h4 className="text-sm font-semibold text-amber-300">
                                            MuLaCover Checkpoints Not Installed
                                        </h4>
                                        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                                            Total: ~8.3 GB
                                        </span>
                                    </div>
                                    <p className="text-xs text-amber-200/80 mt-1">
                                        MuLaCover requires four unified neural checkpoints: the 3B Autoregressive LM, Qwen3-Embedding-0.6B, SymbolicTranscriptor (ChordNet + YourMT3), and HeartCodec-oss.
                                    </p>
                                    
                                    <div className="grid grid-cols-3 gap-2 mt-3 text-[11px]">
                                        <div className="p-2 rounded bg-black/30 border border-amber-500/20">
                                            <div className="font-semibold text-white">MuLaCover 3B</div>
                                            <div className="text-amber-300/70">~6.2 GB • PyTorch FP16</div>
                                        </div>
                                        <div className="p-2 rounded bg-black/30 border border-amber-500/20">
                                            <div className="font-semibold text-white">Qwen3 Embedding</div>
                                            <div className="text-amber-300/70">~1.2 GB • Text Guidance</div>
                                        </div>
                                        <div className="p-2 rounded bg-black/30 border border-amber-500/20">
                                            <div className="font-semibold text-white">Transcriptor & Codec</div>
                                            <div className="text-amber-300/70">~0.9 GB • Symbolic Ensemble</div>
                                        </div>
                                    </div>

                                    {/* Progress Bar if downloading */}
                                    {isDownloadingModel && downloadStatus && (
                                        <div className="mt-3 space-y-1.5">
                                            <div className="flex justify-between text-xs font-mono text-amber-300">
                                                <span className="truncate max-w-[280px]">
                                                    {downloadStatus.current_file ? `Downloading ${downloadStatus.current_file} (${downloadStatus.files_done}/${downloadStatus.total_files})` : 'Downloading weights...'}
                                                </span>
                                                <span>{Math.round(downloadStatus.progress_percent || 0)}%</span>
                                            </div>
                                            <div className="w-full h-2 rounded-full bg-amber-950/60 overflow-hidden">
                                                <div
                                                    className="h-full bg-gradient-to-r from-amber-500 to-teal-400 transition-all duration-300"
                                                    style={{ width: `${Math.max(3, downloadStatus.progress_percent || 0)}%` }}
                                                />
                                            </div>
                                        </div>
                                    )}

                                    {/* Download Button */}
                                    <div className="mt-3 flex items-center gap-3">
                                        <button
                                            type="button"
                                            onClick={handleDownloadBundle}
                                            disabled={isDownloadingModel}
                                            className="px-4 py-1.5 text-xs font-semibold rounded-lg bg-amber-500 hover:bg-amber-400 text-black shadow-md flex items-center gap-1.5 disabled:opacity-50 cursor-pointer transition-colors"
                                        >
                                            {isDownloadingModel ? (
                                                <>
                                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                    Downloading Checkpoints...
                                                </>
                                            ) : (
                                                <>
                                                    <Download className="w-3.5 h-3.5" />
                                                    Download MuLaCover Checkpoints (8.3 GB)
                                                </>
                                            )}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Main Content Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                        {/* Left Column: Reference & Symbolic Controls */}
                        <div className="space-y-4">
                            <div>
                                <label className="block text-xs font-medium text-zinc-300 mb-1">Remix Title</label>
                                <input
                                    type="text"
                                    value={title}
                                    onChange={e => setTitle(e.target.value)}
                                    placeholder="e.g. Blinding Lights (Synthwave Cover)"
                                    className="w-full px-3 py-2 text-sm rounded-lg bg-zinc-900 border border-zinc-700 text-white focus:outline-none focus:border-teal-500"
                                />
                            </div>

                            {mode === 'audio' ? (
                                <div className="p-4 rounded-xl bg-zinc-900/50 border border-zinc-800 space-y-3">
                                    <div className="flex items-center justify-between">
                                        <span className="text-xs font-semibold text-teal-400 uppercase tracking-wider">Source Audio</span>
                                        {refAudioPath && (
                                            <span className="flex items-center gap-1 text-xs text-emerald-400">
                                                <CheckCircle className="w-3.5 h-3.5" /> Ready
                                            </span>
                                        )}
                                    </div>

                                    {refAudioPath ? (
                                        <div className="p-3 rounded-lg bg-zinc-800/60 border border-zinc-700 space-y-2">
                                            <div className="flex items-center justify-between text-xs text-zinc-300">
                                                <span className="truncate max-w-[220px] font-mono">{refAudioFile ? refAudioFile.name : refAudioPath.split('/').pop()}</span>
                                                <button
                                                    type="button"
                                                    onClick={togglePreview}
                                                    className="flex items-center gap-1 px-2 py-1 rounded bg-zinc-700 hover:bg-zinc-600 text-white text-xs"
                                                >
                                                    {isPlayingPreview ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                                                    {isPlayingPreview ? 'Pause' : 'Play'}
                                                </button>
                                            </div>
                                            <audio
                                                ref={audioPlayerRef}
                                                src={refAudioPath.startsWith('http') ? refAudioPath : `${API_BASE_URL}${refAudioPath}`}
                                                onEnded={() => setIsPlayingPreview(false)}
                                                className="hidden"
                                            />
                                        </div>
                                    ) : (
                                        <label className="flex flex-col items-center justify-center p-5 border-2 border-dashed border-zinc-700 rounded-lg hover:border-teal-500/50 cursor-pointer transition-colors">
                                            <Upload className="w-6 h-6 text-zinc-400 mb-1" />
                                            <span className="text-xs font-medium text-zinc-300">Drop song audio here or click to browse</span>
                                            <span className="text-[10px] text-zinc-500 mt-0.5">MP3, WAV, FLAC supported</span>
                                            <input
                                                type="file"
                                                accept="audio/*"
                                                onChange={handleUploadAudio}
                                                className="hidden"
                                            />
                                        </label>
                                    )}

                                    {/* Transcription Engine Selector */}
                                    <div className="space-y-1">
                                        <label className="text-xs font-medium text-zinc-300">Transcription Engine</label>
                                        <div className="grid grid-cols-2 gap-2">
                                            <button
                                                type="button"
                                                onClick={() => setTranscriptionEngine('milimo_neural')}
                                                className={`p-2 rounded-lg text-left border text-xs transition-all ${
                                                    transcriptionEngine === 'milimo_neural'
                                                        ? 'bg-teal-500/20 border-teal-500/50 text-white'
                                                        : 'bg-zinc-900 border-zinc-800 text-zinc-400 hover:border-zinc-700'
                                                }`}
                                            >
                                                <div className="font-semibold text-teal-300">Milimo Neural (SOTA)</div>
                                                <div className="text-[10px] text-zinc-400">BS-Roformer 6-stem + Pitch tracking</div>
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => setTranscriptionEngine('upstream')}
                                                className={`p-2 rounded-lg text-left border text-xs transition-all ${
                                                    transcriptionEngine === 'upstream'
                                                        ? 'bg-teal-500/20 border-teal-500/50 text-white'
                                                        : 'bg-zinc-900 border-zinc-800 text-zinc-400 hover:border-zinc-700'
                                                }`}
                                            >
                                                <div className="font-semibold text-teal-300">Upstream Classic</div>
                                                <div className="text-[10px] text-zinc-400">YourMT3 + ChordNet ensemble</div>
                                            </button>
                                        </div>
                                    </div>

                                    {/* Tempo / BPM and Transcribe Action */}
                                    <div className="flex items-center gap-3 pt-1">
                                        <div className="flex-1">
                                            <label className="text-[11px] text-zinc-400 block mb-0.5">Tempo (BPM)</label>
                                            <input
                                                type="number"
                                                value={bpm}
                                                onChange={e => setBpm(Math.max(40, Math.min(240, Number(e.target.value))))}
                                                className="w-full px-2.5 py-1.5 text-xs rounded bg-zinc-800 border border-zinc-700 text-white"
                                            />
                                        </div>
                                        <button
                                            type="button"
                                            onClick={handleTranscribeLeadSheet}
                                            disabled={!refAudioPath || isTranscribing}
                                            className="mt-4 px-3 py-1.5 text-xs font-semibold rounded-lg bg-zinc-800 hover:bg-zinc-700 text-teal-400 border border-teal-500/30 disabled:opacity-50 flex items-center gap-1.5"
                                        >
                                            {isTranscribing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
                                            Extract Lead Sheet
                                        </button>
                                    </div>

                                    {leadSheet && (
                                        <div className="p-3 rounded-lg bg-teal-950/40 border border-teal-600/40 text-xs text-teal-300 space-y-2">
                                            <div className="flex items-center justify-between">
                                                <span className="font-semibold text-white">Lead Sheet Extracted:</span>
                                                <span className="font-mono text-teal-300">
                                                    {leadSheet.symbolic_length_16th} sixteenths @ {Math.round(leadSheet.bpm)} BPM
                                                </span>
                                            </div>
                                            
                                             <div className="flex flex-wrap gap-1.5 pt-1">
                                                 {leadSheet.paths?.melody && (
                                                     <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-zinc-800/80 border border-teal-500/30 text-[11px] text-zinc-200">
                                                         <Music className="w-3 h-3 text-teal-400" />
                                                         <span>Melody</span>
                                                         <a
                                                             href={api.getAudioUrl(leadSheet.paths.melody)}
                                                             download="melody.mid"
                                                             className="text-teal-400 hover:text-teal-300 ml-1 p-0.5 rounded hover:bg-zinc-700"
                                                             title="Download Melody MIDI"
                                                         >
                                                             <Download className="w-3 h-3" />
                                                         </a>
                                                         {onOpenPianoRoll && (
                                                             <button
                                                                 type="button"
                                                                 onClick={() => onOpenPianoRoll(leadSheet.paths.melody)}
                                                                 className="text-teal-400 hover:text-white ml-0.5 underline font-medium"
                                                             >
                                                                 Roll
                                                             </button>
                                                         )}
                                                     </div>
                                                 )}
                                                 {leadSheet.paths?.chord && (
                                                     <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-zinc-800/80 border border-teal-500/30 text-[11px] text-zinc-200">
                                                         <Music className="w-3 h-3 text-teal-400" />
                                                         <span>Chords</span>
                                                         <a
                                                             href={api.getAudioUrl(leadSheet.paths.chord)}
                                                             download="chords.mid"
                                                             className="text-teal-400 hover:text-teal-300 ml-1 p-0.5 rounded hover:bg-zinc-700"
                                                             title="Download Chords MIDI"
                                                         >
                                                             <Download className="w-3 h-3" />
                                                         </a>
                                                     </div>
                                                 )}
                                                 {(leadSheet.paths?.drum || leadSheet.paths?.drums) && (
                                                     <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-zinc-800/80 border border-teal-500/30 text-[11px] text-zinc-200">
                                                         <Music className="w-3 h-3 text-teal-400" />
                                                         <span>Drums</span>
                                                         <a
                                                             href={api.getAudioUrl(leadSheet.paths.drum || leadSheet.paths.drums)}
                                                             download="drums.mid"
                                                             className="text-teal-400 hover:text-teal-300 ml-1 p-0.5 rounded hover:bg-zinc-700"
                                                             title="Download Drums MIDI"
                                                         >
                                                             <Download className="w-3 h-3" />
                                                         </a>
                                                     </div>
                                                 )}
                                                 {(leadSheet.paths?.leadsheet_summary_midi || leadSheet.paths?.leadsheet_midi) && (
                                                     <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-zinc-800/80 border border-cyan-500/30 text-[11px] text-zinc-200">
                                                         <Music className="w-3 h-3 text-cyan-400" />
                                                         <span>Full Lead Sheet</span>
                                                         <a
                                                             href={api.getAudioUrl(leadSheet.paths.leadsheet_summary_midi || leadSheet.paths.leadsheet_midi)}
                                                             download="lead_sheet.mid"
                                                             className="text-cyan-400 hover:text-cyan-300 ml-1 p-0.5 rounded hover:bg-zinc-700"
                                                             title="Download Full Lead Sheet MIDI"
                                                         >
                                                             <Download className="w-3 h-3" />
                                                         </a>
                                                     </div>
                                                 )}
                                             </div>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="p-4 rounded-xl bg-zinc-900/50 border border-zinc-800 space-y-3">
                                    <span className="text-xs font-semibold text-teal-400 uppercase tracking-wider block">Symbolic MIDI Inputs</span>
                                    <div>
                                        <label className="text-[11px] text-zinc-400 block mb-1">Melody MIDI (Vocal or Lead)</label>
                                        <input
                                            type="text"
                                            value={melodyMidiPath}
                                            onChange={e => setMelodyMidiPath(e.target.value)}
                                            placeholder="/path/to/melody.mid"
                                            className="w-full px-3 py-1.5 text-xs rounded bg-zinc-800 border border-zinc-700 text-white font-mono"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-[11px] text-zinc-400 block mb-1">Chord MIDI (Harmonic Progression)</label>
                                        <input
                                            type="text"
                                            value={chordMidiPath}
                                            onChange={e => setChordMidiPath(e.target.value)}
                                            placeholder="/path/to/chord.mid"
                                            className="w-full px-3 py-1.5 text-xs rounded bg-zinc-800 border border-zinc-700 text-white font-mono"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-[11px] text-zinc-400 block mb-1">Drum MIDI (Optional)</label>
                                        <input
                                            type="text"
                                            value={drumMidiPath}
                                            onChange={e => setDrumMidiPath(e.target.value)}
                                            placeholder="/path/to/drums.mid (optional)"
                                            className="w-full px-3 py-1.5 text-xs rounded bg-zinc-800 border border-zinc-700 text-white font-mono"
                                        />
                                    </div>
                                </div>
                            )}

                            {/* Style Description Builder */}
                            <div className="p-4 rounded-xl bg-zinc-900/50 border border-zinc-800 space-y-3">
                                <span className="text-xs font-semibold text-teal-400 uppercase tracking-wider block">Style & Arrangement DNA</span>
                                <div className="grid grid-cols-2 gap-3">
                                    <div>
                                        <label className="text-[11px] text-zinc-400 block mb-0.5">Target Genre</label>
                                        <input
                                            type="text"
                                            value={genre}
                                            onChange={e => setGenre(e.target.value)}
                                            className="w-full px-2.5 py-1.5 text-xs rounded bg-zinc-800 border border-zinc-700 text-white"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-[11px] text-zinc-400 block mb-0.5">Mood</label>
                                        <input
                                            type="text"
                                            value={mood}
                                            onChange={e => setMood(e.target.value)}
                                            className="w-full px-2.5 py-1.5 text-xs rounded bg-zinc-800 border border-zinc-700 text-white"
                                        />
                                    </div>
                                </div>
                                <div>
                                    <label className="text-[11px] text-zinc-400 block mb-0.5">Instrumentation</label>
                                    <input
                                        type="text"
                                        value={instrument}
                                        onChange={e => setInstrument(e.target.value)}
                                        className="w-full px-2.5 py-1.5 text-xs rounded bg-zinc-800 border border-zinc-700 text-white"
                                    />
                                </div>
                                <div>
                                    <label className="text-[11px] text-zinc-400 block mb-0.5">Topic / Narrative</label>
                                    <input
                                        type="text"
                                        value={topic}
                                        onChange={e => setTopic(e.target.value)}
                                        className="w-full px-2.5 py-1.5 text-xs rounded bg-zinc-800 border border-zinc-700 text-white"
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Right Column: Structured Lyrics & Advanced Parameters */}
                        <div className="space-y-4">
                            <div className="p-4 rounded-xl bg-zinc-900/50 border border-zinc-800 space-y-2">
                                <div className="flex items-center justify-between">
                                    <label className="text-xs font-semibold text-teal-400 uppercase tracking-wider">Multiline Structured Lyrics</label>
                                    <div className="flex gap-1">
                                        {['Verse', 'Chorus', 'Bridge', 'Outro'].map(sec => (
                                            <button
                                                key={sec}
                                                type="button"
                                                onClick={() => addSectionToLyrics(sec)}
                                                className="px-1.5 py-0.5 text-[10px] rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border border-zinc-700"
                                            >
                                                +{sec}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                <textarea
                                    rows={10}
                                    value={lyrics}
                                    onChange={e => setLyrics(e.target.value)}
                                    placeholder="Use [Intro], [Verse], [Chorus] section headers..."
                                    className="w-full p-3 text-xs rounded-lg bg-zinc-950 border border-zinc-800 text-zinc-200 font-mono focus:outline-none focus:border-teal-500 leading-relaxed"
                                />
                                <span className="text-[10px] text-zinc-500 block">
                                    MuLaCover synchronizes lyrics with the symbolic vocal melody. Leave blank lines between sections.
                                </span>
                            </div>

                            {/* Advanced Controls Accordion */}
                            <div className="p-3 rounded-xl bg-zinc-900/30 border border-zinc-800/80">
                                <button
                                    type="button"
                                    onClick={() => setShowAdvanced(!showAdvanced)}
                                    className="w-full flex items-center justify-between text-xs font-medium text-zinc-400 hover:text-zinc-200"
                                >
                                    <span className="flex items-center gap-1.5">
                                        <Sliders className="w-3.5 h-3.5" />
                                        Advanced Hyperparameters (CFG, Temperature, Top-K)
                                    </span>
                                    <span>{showAdvanced ? '−' : '+'}</span>
                                </button>

                                {showAdvanced && (
                                    <div className="grid grid-cols-2 gap-3 mt-3 pt-3 border-t border-zinc-800/60 text-xs">
                                        <div>
                                            <label className="text-[11px] text-zinc-400 block mb-1">CFG Scale ({cfgScale})</label>
                                            <input
                                                type="range"
                                                min="1.0"
                                                max="3.0"
                                                step="0.1"
                                                value={cfgScale}
                                                onChange={e => setCfgScale(parseFloat(e.target.value))}
                                                className="w-full"
                                            />
                                        </div>
                                        <div>
                                            <label className="text-[11px] text-zinc-400 block mb-1">Temperature ({temperature})</label>
                                            <input
                                                type="range"
                                                min="0.5"
                                                max="1.5"
                                                step="0.05"
                                                value={temperature}
                                                onChange={e => setTemperature(parseFloat(e.target.value))}
                                                className="w-full"
                                            />
                                        </div>
                                        <div>
                                            <label className="text-[11px] text-zinc-400 block mb-1">Top-K ({topk})</label>
                                            <input
                                                type="number"
                                                value={topk}
                                                onChange={e => setTopk(parseInt(e.target.value) || 250)}
                                                className="w-full px-2 py-1 rounded bg-zinc-800 border border-zinc-700 text-white text-xs"
                                            />
                                        </div>
                                        <div>
                                            <label className="text-[11px] text-zinc-400 block mb-1">Duration</label>
                                            <select
                                                value={durationMs}
                                                onChange={e => setDurationMs(parseInt(e.target.value))}
                                                className="w-full px-2 py-1 rounded bg-zinc-800 border border-zinc-700 text-white text-xs"
                                            >
                                                <option value={30000}>30 seconds</option>
                                                <option value={60000}>1 minute</option>
                                                <option value={120000}>2 minutes</option>
                                                <option value={180000}>3 minutes</option>
                                                <option value={240000}>4 minutes</option>
                                            </select>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Footer Actions */}
                    <div className="flex items-center justify-between pt-5 mt-6 border-t border-zinc-800">
                        <div className="text-xs text-zinc-500">
                            Powered by <span className="text-teal-400 font-medium">MuLaCover 3B</span> + <span className="text-zinc-300">HeartCodec</span>
                        </div>
                        <div className="flex items-center gap-3">
                            <button
                                type="button"
                                onClick={onClose}
                                className="px-4 py-2 text-xs font-semibold rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                type="button"
                                onClick={handleSynthesize}
                                disabled={isSubmitting || isUploadingAudio || isTranscribing || isModelInstalled === false}
                                title={isModelInstalled === false ? "MuLaCover checkpoints are missing - download above" : undefined}
                                className="px-5 py-2 text-xs font-semibold rounded-lg bg-gradient-to-r from-teal-500 to-cyan-600 hover:from-teal-400 hover:to-cyan-500 text-white shadow-lg shadow-teal-500/25 flex items-center gap-2 disabled:opacity-50 transition-all cursor-pointer disabled:cursor-not-allowed"
                            >
                                {isSubmitting ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Synthesizing Cover...
                                    </>
                                ) : isModelInstalled === false ? (
                                    <>
                                        <AlertCircle className="w-4 h-4" />
                                        Checkpoints Required
                                    </>
                                ) : (
                                    <>
                                        <Sparkles className="w-4 h-4" />
                                        Synthesize MuLaCover Remix
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </GlassCard>
            </div>
        </div>
    );
};
