import React, { useState } from 'react';
import {
    X,
    Download,
    FileAudio,
    FileCode,
    FileText,
    Wand2,
    Layers,
    CheckCircle2,
    Sliders,
    Disc,
    Sparkles,
    Loader2
} from 'lucide-react';
import { api, API_BASE_URL, workspaceApi, type Job } from '../../api';
import { toast } from '../../utils/toast';

interface MasteringExportModalProps {
    isOpen: boolean;
    onClose: () => void;
    job: Job;
    hasMasteredTrack?: boolean;
    onMasteringComplete?: (masteredPath: string, lufs: number) => void;
}

export const MasteringExportModal: React.FC<MasteringExportModalProps> = ({
    isOpen,
    onClose,
    job,
    hasMasteredTrack = false,
    onMasteringComplete,
}) => {
    const [audioFormat, setAudioFormat] = useState<'wav' | 'flac' | 'mp3'>('wav');
    const [sampleRate, setSampleRate] = useState<'44100' | '48000' | '96000'>('48000');
    const [targetLufs, setTargetLufs] = useState<number>(-14);
    const [isMastering, setIsMastering] = useState(false);
    const [isDownloading, setIsDownloading] = useState<string | null>(null);

    if (!isOpen) return null;

    const handleDownloadAsset = (type: string, url: string, filename: string) => {
        setIsDownloading(type);
        try {
            const a = document.createElement('a');
            a.href = url.startsWith('http') ? url : `${API_BASE_URL}${url}`;
            a.download = filename;
            a.target = '_blank';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            toast(`Downloading ${filename}`, 'success');
        } catch (e) {
            console.error('Download error:', e);
            toast(`Failed to download ${filename}`, 'error');
        } finally {
            setTimeout(() => setIsDownloading(null), 1000);
        }
    };

    const handleRunMastering = async () => {
        if (!job.id || isMastering) return;
        setIsMastering(true);
        try {
            const res = await workspaceApi.applyMastering(job.id, targetLufs);
            if (res.status === 'completed' && res.audio_path) {
                toast(`Track mastered cleanly to ${targetLufs} LUFS!`, 'success');
                onMasteringComplete?.(res.audio_path, res.lufs || targetLufs);
            } else {
                toast('Mastering engine finished without file', 'info');
            }
        } catch (e) {
            console.error('Mastering failed', e);
            toast('Mastering failed. Check server status.', 'error');
        } finally {
            setIsMastering(false);
        }
    };

    const masterAudioUrl = job.audio_path
        ? api.getAudioUrl(job.audio_path)
        : '';

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md animate-fade-in"
            onClick={onClose}
        >
            <div
                className="relative w-full max-w-2xl bg-white/95 dark:bg-[#12141c]/95 backdrop-blur-2xl rounded-3xl shadow-2xl border border-black/10 dark:border-white/10 overflow-hidden text-slate-900 dark:text-slate-100 animate-scale-up flex flex-col max-h-[90vh]"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Modal Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-black/5 dark:border-white/10">
                    <div className="flex items-center gap-3">
                        <div className="p-2.5 rounded-2xl bg-gradient-to-tr from-teal-500/20 to-cyan-500/20 text-teal-600 dark:text-teal-400 border border-teal-500/30">
                            <Sliders size={20} />
                        </div>
                        <div>
                            <h2 className="text-base sm:text-lg font-bold">
                                Mastering & DAW Export Studio
                            </h2>
                            <p className="text-xs text-slate-500 dark:text-slate-400">
                                Export broadcast-ready stereo masters, isolated stems, and note-level notation assets.
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-xl text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
                        aria-label="Close export modal"
                    >
                        <X size={18} />
                    </button>
                </div>

                {/* Modal Body */}
                <div className="p-6 overflow-y-auto space-y-6">
                    {/* Section 1: AI Matchering Mastering Engine */}
                    <div className="p-4 rounded-2xl bg-black/[0.02] dark:bg-white/[0.03] border border-black/5 dark:border-white/5 space-y-4">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Wand2 size={16} className="text-teal-500" />
                                <span className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                                    Reference Mastering Target
                                </span>
                            </div>
                            {hasMasteredTrack && (
                                <span className="flex items-center gap-1 text-[11px] font-mono font-bold text-teal-600 dark:text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded-full border border-teal-500/20">
                                    <CheckCircle2 size={12} /> Mastered Track Ready
                                </span>
                            )}
                        </div>

                        {/* LUFS Preset Selectors */}
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                            {[
                                { lufs: -14, label: 'Streaming Master', desc: 'Spotify & YouTube (-14 LUFS)' },
                                { lufs: -16, label: 'Apple Music / Hi-Res', desc: 'Spatial & Dynamic (-16 LUFS)' },
                                { lufs: -9, label: 'Club / Loudness War', desc: 'Peak Energy & Dance (-9 LUFS)' },
                            ].map((preset) => (
                                <button
                                    key={preset.lufs}
                                    onClick={() => setTargetLufs(preset.lufs)}
                                    className={`p-3 rounded-xl border text-left transition-all flex flex-col justify-between ${
                                        targetLufs === preset.lufs
                                            ? 'bg-teal-500/10 border-teal-500 text-teal-700 dark:text-teal-300 shadow-sm'
                                            : 'bg-black/[0.02] dark:bg-white/[0.02] border-black/5 dark:border-white/5 hover:border-black/20 dark:hover:border-white/20 text-slate-600 dark:text-slate-400'
                                    }`}
                                >
                                    <div className="font-bold text-xs">{preset.label}</div>
                                    <div className="text-[10px] opacity-75 mt-1 font-mono">{preset.desc}</div>
                                </button>
                            ))}
                        </div>

                        <div className="flex items-center justify-between pt-1">
                            <span className="text-xs text-slate-400">
                                Automated spectral balancing and RMS normalization via Matchering 2.0.
                            </span>
                            <button
                                onClick={handleRunMastering}
                                disabled={isMastering}
                                className="px-4 py-2 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 text-slate-950 font-bold text-xs flex items-center gap-2 active:scale-95 transition-transform shadow-md shadow-teal-500/20 disabled:opacity-50"
                            >
                                {isMastering ? (
                                    <>
                                        <Loader2 size={14} className="animate-spin" />
                                        <span>Mastering...</span>
                                    </>
                                ) : (
                                    <>
                                        <Sparkles size={14} />
                                        <span>Master Track Now</span>
                                    </>
                                )}
                            </button>
                        </div>
                    </div>

                    {/* Section 2: Audio Format & Stem Downloads */}
                    <div className="space-y-3">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                            <span className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                                Audio Masters & Stems
                            </span>

                            {/* Format & Sample Rate Pills */}
                            <div className="flex items-center gap-2">
                                <div className="flex items-center p-0.5 rounded-lg bg-black/5 dark:bg-white/5 text-[10px] font-mono font-bold">
                                    {(['wav', 'flac', 'mp3'] as const).map((fmt) => (
                                        <button
                                            key={fmt}
                                            onClick={() => setAudioFormat(fmt)}
                                            className={`px-2 py-0.5 rounded uppercase ${
                                                audioFormat === fmt
                                                    ? 'bg-teal-500 text-slate-950 font-black'
                                                    : 'text-slate-400 hover:text-slate-200'
                                            }`}
                                        >
                                            {fmt}
                                        </button>
                                    ))}
                                </div>

                                <div className="flex items-center p-0.5 rounded-lg bg-black/5 dark:bg-white/5 text-[10px] font-mono font-bold">
                                    {(['44100', '48000', '96000'] as const).map((sr) => (
                                        <button
                                            key={sr}
                                            onClick={() => setSampleRate(sr)}
                                            className={`px-1.5 py-0.5 rounded ${
                                                sampleRate === sr
                                                    ? 'bg-cyan-500 text-slate-950 font-black'
                                                    : 'text-slate-400 hover:text-slate-200'
                                            }`}
                                        >
                                            {sr === '44100' ? '44.1k' : sr === '48000' ? '48k' : '96k'}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            {/* Full Stereo Master Mix */}
                            <div className="p-3.5 rounded-2xl bg-black/[0.02] dark:bg-white/[0.03] border border-black/5 dark:border-white/5 flex items-center justify-between">
                                <div className="flex items-center gap-3 min-w-0">
                                    <div className="p-2 rounded-xl bg-teal-500/10 text-teal-500 shrink-0">
                                        <Disc size={18} />
                                    </div>
                                    <div className="min-w-0">
                                        <div className="text-xs font-bold truncate">Stereo Master Mix</div>
                                        <div className="text-[10px] text-slate-400 font-mono">
                                            24-bit {sampleRate === '44100' ? '44.1kHz' : sampleRate === '48000' ? '48kHz' : '96kHz'} Stereo .{audioFormat.toUpperCase()}
                                        </div>
                                    </div>
                                </div>
                                <button
                                    onClick={() => handleDownloadAsset('master', masterAudioUrl, `${job.title || 'master'}.${audioFormat}`)}
                                    disabled={!masterAudioUrl || isDownloading === 'master'}
                                    className="p-2 rounded-xl bg-black/5 dark:bg-white/10 hover:bg-teal-500 hover:text-slate-950 transition-colors shrink-0 disabled:opacity-50"
                                    title="Download Master Stereo Mix"
                                >
                                    {isDownloading === 'master' ? (
                                        <Loader2 size={15} className="animate-spin text-teal-500" />
                                    ) : (
                                        <Download size={15} />
                                    )}
                                </button>
                            </div>

                            {/* Separated Stems Package */}
                            <div className="p-3.5 rounded-2xl bg-black/[0.02] dark:bg-white/[0.03] border border-black/5 dark:border-white/5 flex items-center justify-between">
                                <div className="flex items-center gap-3 min-w-0">
                                    <div className="p-2 rounded-xl bg-cyan-500/10 text-cyan-500 shrink-0">
                                        <Layers size={18} />
                                    </div>
                                    <div className="min-w-0">
                                        <div className="text-xs font-bold truncate">All Stem Channels</div>
                                        <div className="text-[10px] text-slate-400 font-mono">Vocals, Drums, Bass, Other (.ZIP)</div>
                                    </div>
                                </div>
                                <button
                                    onClick={() => handleDownloadAsset('stems', `${API_BASE_URL}/workspace/${job.id}/stems/export`, `${job.title || 'stems'}.zip`)}
                                    disabled={isDownloading === 'stems'}
                                    className="p-2 rounded-xl bg-black/5 dark:bg-white/10 hover:bg-cyan-500 hover:text-slate-950 transition-colors shrink-0 disabled:opacity-50"
                                    title="Download Isolated Stems (.ZIP)"
                                >
                                    {isDownloading === 'stems' ? (
                                        <Loader2 size={15} className="animate-spin text-cyan-500" />
                                    ) : (
                                        <Download size={15} />
                                    )}
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Section 3: Notation & Composition Assets */}
                    <div className="space-y-3">
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 block">
                            Composition & Transcription Assets
                        </span>

                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                            {/* Standard MIDI */}
                            <div className="p-3 rounded-2xl bg-black/[0.02] dark:bg-white/[0.03] border border-black/5 dark:border-white/5 flex flex-col justify-between gap-3">
                                <div className="flex items-center gap-2">
                                    <FileAudio size={16} className="text-amber-500" />
                                    <span className="text-xs font-bold">Multi-Track MIDI</span>
                                </div>
                                <p className="text-[10px] text-slate-400">
                                    Standard .MID file for Logic Pro, Ableton, FL Studio, and Reaper.
                                </p>
                                <button
                                    onClick={() => handleDownloadAsset('midi', `${API_BASE_URL}/transcribe/export/${job.id}/midi`, `${job.title || 'composition'}.mid`)}
                                    className="w-full py-1.5 px-3 rounded-xl bg-black/5 dark:bg-white/10 hover:bg-amber-500 hover:text-slate-950 text-xs font-bold flex items-center justify-center gap-1.5 transition-colors"
                                >
                                    <Download size={13} />
                                    <span>Export .MID</span>
                                </button>
                            </div>

                            {/* MusicXML Score */}
                            <div className="p-3 rounded-2xl bg-black/[0.02] dark:bg-white/[0.03] border border-black/5 dark:border-white/5 flex flex-col justify-between gap-3">
                                <div className="flex items-center gap-2">
                                    <FileCode size={16} className="text-teal-500" />
                                    <span className="text-xs font-bold">MusicXML Score</span>
                                </div>
                                <p className="text-[10px] text-slate-400">
                                    Universal sheet music score for Sibelius, Dorico, and MuseScore.
                                </p>
                                <button
                                    onClick={() => handleDownloadAsset('musicxml', `${API_BASE_URL}/transcribe/export/${job.id}/musicxml`, `${job.title || 'score'}.musicxml`)}
                                    className="w-full py-1.5 px-3 rounded-xl bg-black/5 dark:bg-white/10 hover:bg-teal-500 hover:text-slate-950 text-xs font-bold flex items-center justify-center gap-1.5 transition-colors"
                                >
                                    <Download size={13} />
                                    <span>Export Score</span>
                                </button>
                            </div>

                            {/* Timed Lyrics (.LRC) */}
                            <div className="p-3 rounded-2xl bg-black/[0.02] dark:bg-white/[0.03] border border-black/5 dark:border-white/5 flex flex-col justify-between gap-3">
                                <div className="flex items-center gap-2">
                                    <FileText size={16} className="text-purple-500" />
                                    <span className="text-xs font-bold">Timed Lyrics</span>
                                </div>
                                <p className="text-[10px] text-slate-400">
                                    Millisecond-synchronized karaoke .LRC file with line tags.
                                </p>
                                <button
                                    onClick={() => handleDownloadAsset('lrc', `${API_BASE_URL}/transcribe/export/${job.id}/lrc`, `${job.title || 'lyrics'}.lrc`)}
                                    className="w-full py-1.5 px-3 rounded-xl bg-black/5 dark:bg-white/10 hover:bg-purple-500 hover:text-slate-950 text-xs font-bold flex items-center justify-center gap-1.5 transition-colors"
                                >
                                    <Download size={13} />
                                    <span>Export .LRC</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Modal Footer */}
                <div className="px-6 py-3.5 border-t border-black/5 dark:border-white/10 bg-black/[0.02] dark:bg-white/[0.02] flex items-center justify-between text-xs text-slate-400">
                    <span className="font-mono text-[11px]">
                        Track ID: {job.id.slice(0, 8)} · {job.tags || 'Pop'}
                    </span>
                    <button
                        onClick={onClose}
                        className="px-4 py-1.5 rounded-xl bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 text-slate-700 dark:text-slate-200 font-bold transition-colors"
                    >
                        Done
                    </button>
                </div>
            </div>
        </div>
    );
};
