import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
    Mic,
    Square,
    Play,
    Pause,
    RotateCcw,
    Check,
    Radio,
    Clock,
    AlertCircle
} from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';

interface VocalBoothRecorderProps {
    onAudioCaptured: (audioFile: File, durationSec: number) => void;
    onUseAsVocalTrack?: (audioFile: File, durationSec: number) => void;
    onCancel?: () => void;
    maxDurationSec?: number;
}

export const VocalBoothRecorder: React.FC<VocalBoothRecorderProps> = ({
    onAudioCaptured,
    onUseAsVocalTrack,
    onCancel,
    maxDurationSec = 60
}) => {
    const [state, setState] = useState<'idle' | 'countdown' | 'recording' | 'review'>('idle');
    const [countdown, setCountdown] = useState<number>(3);
    const [elapsedSec, setElapsedSec] = useState<number>(0);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [isPlaying, setIsPlaying] = useState<boolean>(false);
    const [meterLevels, setMeterLevels] = useState<number[]>(Array(16).fill(0));
    const [errorMsg, setErrorMsg] = useState<string | null>(null);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const animFrameRef = useRef<number | null>(null);
    const timerIntervalRef = useRef<number | null>(null);
    const countdownIntervalRef = useRef<number | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const previewAudioRef = useRef<HTMLAudioElement | null>(null);

    // Clean up all resources on unmount
    const cleanupAudio = useCallback(() => {
        if (countdownIntervalRef.current) {
            window.clearInterval(countdownIntervalRef.current);
            countdownIntervalRef.current = null;
        }
        if (animFrameRef.current) {
            cancelAnimationFrame(animFrameRef.current);
            animFrameRef.current = null;
        }
        if (timerIntervalRef.current) {
            window.clearInterval(timerIntervalRef.current);
            timerIntervalRef.current = null;
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
        }
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close().catch(() => {});
            audioContextRef.current = null;
        }
        if (previewAudioRef.current) {
            previewAudioRef.current.pause();
            previewAudioRef.current = null;
        }
    }, []);

    useEffect(() => {
        return () => {
            cleanupAudio();
            if (audioUrl) {
                URL.revokeObjectURL(audioUrl);
            }
        };
    }, [cleanupAudio, audioUrl]);

    // External audio bus listener
    useEffect(() => {
        const handleExternalAudioPlay = (e: Event) => {
            const customEvent = e as CustomEvent<{ source?: string }>;
            if (customEvent.detail?.source && customEvent.detail.source !== 'vocal-booth-preview') {
                if (previewAudioRef.current && !previewAudioRef.current.paused) {
                    previewAudioRef.current.pause();
                    setIsPlaying(false);
                }
            }
        };

        window.addEventListener('milimo:audio-play', handleExternalAudioPlay);
        return () => {
            window.removeEventListener('milimo:audio-play', handleExternalAudioPlay);
        };
    }, []);

    // Live level visualizer loop
    const updateMeter = useCallback(() => {
        if (!analyserRef.current) return;
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);

        // Sample 16 discrete frequency buckets
        const step = Math.floor(dataArray.length / 16);
        const levels = Array.from({ length: 16 }, (_, i) => {
            const val = dataArray[i * step] || 0;
            return Math.min(100, Math.round((val / 255) * 100));
        });
        setMeterLevels(levels);

        animFrameRef.current = requestAnimationFrame(updateMeter);
    }, []);

    // Start recording after countdown
    const startRecordingSession = async () => {
        setErrorMsg(null);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                },
            });
            streamRef.current = stream;

            // Audio Analyser Setup
            const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
            const ctx = new AudioContextClass();
            audioContextRef.current = ctx;
            const source = ctx.createMediaStreamSource(stream);
            const analyser = ctx.createAnalyser();
            analyser.fftSize = 64;
            source.connect(analyser);
            analyserRef.current = analyser;

            // Determine mime type
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')
                ? 'audio/ogg;codecs=opus'
                : 'audio/mp4';

            const recorder = new MediaRecorder(stream, { mimeType });
            mediaRecorderRef.current = recorder;
            audioChunksRef.current = [];

            recorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) {
                    audioChunksRef.current.push(e.data);
                }
            };

            recorder.onstop = () => {
                const blob = new Blob(audioChunksRef.current, { type: mimeType });
                setAudioBlob(blob);
                if (audioUrl) URL.revokeObjectURL(audioUrl);
                const url = URL.createObjectURL(blob);
                setAudioUrl(url);
                setState('review');
                cleanupAudio();
            };

            // Start countdown
            setState('countdown');
            setCountdown(3);

            let count = 3;
            countdownIntervalRef.current = window.setInterval(() => {
                count -= 1;
                if (count > 0) {
                    setCountdown(count);
                } else {
                    if (countdownIntervalRef.current) {
                        window.clearInterval(countdownIntervalRef.current);
                        countdownIntervalRef.current = null;
                    }
                    setState('recording');
                    setElapsedSec(0);
                    recorder.start(250);
                    updateMeter();

                    // Track elapsed time
                    const startTime = Date.now();
                    timerIntervalRef.current = window.setInterval(() => {
                        const elapsed = Math.floor((Date.now() - startTime) / 1000);
                        setElapsedSec(elapsed);
                        if (elapsed >= maxDurationSec) {
                            stopRecording();
                        }
                    }, 250);
                }
            }, 1000);
        } catch (err: any) {
            console.error('Microphone access failed:', err);
            setErrorMsg(err.name === 'NotAllowedError' ? 'Microphone permission denied. Please allow microphone access in your browser settings.' : 'Failed to access microphone.');
            setState('idle');
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
    };

    const handleTogglePlayback = () => {
        if (!audioUrl) return;
        if (!previewAudioRef.current) {
            const audio = new Audio(audioUrl);
            previewAudioRef.current = audio;
            audio.onended = () => setIsPlaying(false);
            audio.onerror = () => setIsPlaying(false);
        }

        if (isPlaying) {
            previewAudioRef.current.pause();
            setIsPlaying(false);
        } else {
            window.dispatchEvent(new CustomEvent('milimo:audio-play', { detail: { source: 'vocal-booth-preview' } }));
            previewAudioRef.current.play().then(() => setIsPlaying(true)).catch(() => setIsPlaying(false));
        }
    };

    const handleConfirmRecording = () => {
        if (!audioBlob) return;
        const ext = audioBlob.type.includes('webm') ? 'webm' : audioBlob.type.includes('ogg') ? 'ogg' : 'wav';
        const file = new File([audioBlob], `vocal_sample_${Date.now()}.${ext}`, { type: audioBlob.type });
        onAudioCaptured(file, elapsedSec);
    };

    const handleReset = () => {
        cleanupAudio();
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        setAudioBlob(null);
        setAudioUrl(null);
        setIsPlaying(false);
        setElapsedSec(0);
        setState('idle');
    };

    const formatTime = (sec: number) => {
        const m = Math.floor(sec / 60);
        const s = sec % 60;
        return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    };

    return (
        <GlassCard className="p-5 border border-teal-500/20 bg-gradient-to-b from-teal-500/[0.03] to-cyan-500/[0.02]">
            <div className="flex items-center justify-between pb-3 border-b border-black/[0.06] dark:border-white/10">
                <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400 flex items-center justify-center">
                        <Radio size={16} className={state === 'recording' ? 'animate-pulse text-rose-500' : ''} />
                    </div>
                    <div>
                        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-900 dark:text-white flex items-center gap-1.5">
                            <span>Vocal Booth (Live Mic Recording)</span>
                            {state === 'recording' && (
                                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9px] font-bold bg-rose-500/15 text-rose-500 border border-rose-500/30 animate-pulse">
                                    REC
                                </span>
                            )}
                        </h4>
                        <p className="text-[11px] text-slate-500 dark:text-slate-400">
                            Record 10–30s of clean singing or speech to train your custom vocal profile
                        </p>
                    </div>
                </div>

                {onCancel && state !== 'recording' && (
                    <button
                        onClick={onCancel}
                        className="text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
                    >
                        Cancel
                    </button>
                )}
            </div>

            {errorMsg && (
                <div className="mt-3 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-500 flex items-start gap-2">
                    <AlertCircle size={15} className="flex-shrink-0 mt-0.5" />
                    <span>{errorMsg}</span>
                </div>
            )}

            {/* State 1: IDLE */}
            {state === 'idle' && (
                <div className="py-6 flex flex-col items-center justify-center space-y-4 text-center">
                    <div className="w-16 h-16 rounded-3xl bg-teal-500/10 dark:bg-teal-500/20 text-teal-600 dark:text-teal-400 flex items-center justify-center shadow-inner">
                        <Mic size={28} />
                    </div>
                    <div className="max-w-sm">
                        <p className="text-xs font-medium text-slate-700 dark:text-slate-300">
                            Click below to record your voice sample using your built-in or USB microphone.
                        </p>
                        <p className="text-[11px] text-slate-400 mt-1">
                            Sing cleanly in a quiet room. 15–30 seconds is optimal for F0 pitch extraction.
                        </p>
                    </div>
                    <button
                        type="button"
                        onClick={startRecordingSession}
                        className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs flex items-center space-x-2 shadow-md shadow-teal-500/20 active:scale-95 transition-all"
                    >
                        <Radio size={14} />
                        <span>Start Vocal Recording</span>
                    </button>
                </div>
            )}

            {/* State 2: COUNTDOWN */}
            {state === 'countdown' && (
                <div className="py-10 flex flex-col items-center justify-center space-y-3 text-center">
                    <div className="text-5xl font-black text-teal-500 animate-bounce">
                        {countdown}
                    </div>
                    <p className="text-xs font-semibold text-slate-600 dark:text-slate-300">
                        Get ready to sing...
                    </p>
                </div>
            )}

            {/* State 3: RECORDING */}
            {state === 'recording' && (
                <div className="py-5 space-y-4">
                    {/* Live Peak Meter Bars */}
                    <div className="flex items-end justify-center gap-1.5 h-16 px-4 bg-black/10 dark:bg-black/40 rounded-xl p-2">
                        {meterLevels.map((lvl, idx) => (
                            <div
                                key={idx}
                                style={{ height: `${Math.max(8, lvl)}%` }}
                                className={`w-3.5 rounded-t-sm transition-all duration-75 ${
                                    lvl > 80
                                        ? 'bg-rose-500'
                                        : lvl > 50
                                        ? 'bg-amber-400'
                                        : 'bg-teal-400'
                                }`}
                            />
                        ))}
                    </div>

                    <div className="flex items-center justify-between px-2">
                        <div className="flex items-center space-x-2">
                            <Clock size={14} className="text-slate-400" />
                            <span className="text-sm font-mono font-bold text-slate-900 dark:text-white">
                                {formatTime(elapsedSec)} / {formatTime(maxDurationSec)}
                            </span>
                        </div>
                        <button
                            type="button"
                            onClick={stopRecording}
                            className="px-4 py-2 rounded-xl bg-rose-500 hover:bg-rose-600 text-white font-bold text-xs flex items-center space-x-2 shadow-md shadow-rose-500/20 active:scale-95 transition-all"
                        >
                            <Square size={13} fill="currentColor" />
                            <span>Stop Recording</span>
                        </button>
                    </div>
                </div>
            )}

            {/* State 4: REVIEW & CONFIRM */}
            {state === 'review' && (
                <div className="py-4 space-y-4">
                    <div className="p-4 rounded-xl bg-black/[0.03] dark:bg-white/[0.04] border border-black/[0.05] dark:border-white/5 flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                            <button
                                type="button"
                                onClick={handleTogglePlayback}
                                className="w-10 h-10 rounded-xl bg-teal-500 text-slate-950 flex items-center justify-center shadow-md shadow-teal-500/20 active:scale-95 transition-all"
                            >
                                {isPlaying ? <Pause size={18} /> : <Play size={18} className="ml-0.5" />}
                            </button>
                            <div>
                                <h5 className="text-xs font-bold text-slate-900 dark:text-white">
                                    Vocal Take Recorded
                                </h5>
                                <p className="text-[11px] text-slate-500 dark:text-slate-400">
                                    Duration: {formatTime(elapsedSec)} · Clean Vocal Sample
                                </p>
                            </div>
                        </div>

                        <button
                            type="button"
                            onClick={handleReset}
                            className="p-2 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
                            title="Record Again"
                        >
                            <RotateCcw size={16} />
                        </button>
                    </div>

                    <div className="flex items-center justify-end space-x-3 pt-2">
                        <button
                            type="button"
                            onClick={handleReset}
                            className="px-3.5 py-2 text-xs font-bold text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
                        >
                            Discard & Re-Record
                        </button>
                        {onUseAsVocalTrack && (
                            <button
                                type="button"
                                onClick={() => {
                                    if (!audioBlob) return;
                                    const file = new File([audioBlob], `vocal_mic_take_${Date.now()}.wav`, { type: 'audio/wav' });
                                    onUseAsVocalTrack(file, elapsedSec);
                                }}
                                className="px-3.5 py-2 rounded-xl bg-teal-500/20 text-teal-700 dark:text-teal-300 hover:bg-teal-500/30 font-bold text-xs flex items-center space-x-1.5 border border-teal-500/30 transition-all"
                            >
                                <Radio size={13} />
                                <span>Use as Vocal Track</span>
                            </button>
                        )}
                        <button
                            type="button"
                            onClick={handleConfirmRecording}
                            className="px-4 py-2 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs flex items-center space-x-1.5 shadow-md shadow-teal-500/20 active:scale-95 transition-all"
                        >
                            <Check size={14} />
                            <span>Save as Voice Identity</span>
                        </button>
                    </div>
                </div>
            )}
        </GlassCard>
    );
};
