import React, { useState, useEffect } from 'react';
import { Sliders, RefreshCw, Wand2, CheckCircle2, AlertTriangle } from 'lucide-react';
import { workspaceApi, type Job } from '../../api';
import type { StemChannel } from './SessionWorkspace';

interface MultitrackMixerProps {
    job: Job;
    stemChannels: StemChannel[];
    onVolumeChange: (id: string, volume: number) => void;
    onPanChange: (id: string, pan: number) => void;
    onToggleMute: (id: string) => void;
    onToggleSolo: (id: string) => void;
    masterVolume: number;
    onMasterVolumeChange: (volume: number) => void;
    isPlaying: boolean;
    /** Live AnalyserNode taps (post-gain) per channel id — real signal levels. */
    stemAnalysersRef: React.MutableRefObject<Record<string, AnalyserNode>>;
    /** AnalyserNode tapped off the master bus. */
    masterAnalyserRef: React.MutableRefObject<AnalyserNode | null>;
}

export const MultitrackMixer: React.FC<MultitrackMixerProps> = ({
    job,
    stemChannels,
    onVolumeChange,
    onPanChange,
    onToggleMute,
    onToggleSolo,
    masterVolume,
    onMasterVolumeChange,
    isPlaying,
    stemAnalysersRef,
    masterAnalyserRef
}) => {
    const [isMastering, setIsMastering] = useState(false);
    const [masteringStatus, setMasteringStatus] = useState<{ ok: boolean; text: string } | null>(null);
    const [masteredLufs, setMasteredLufs] = useState<number | null>(null);
    const [meterLevels, setMeterLevels] = useState<Record<string, number>>({});

    // ── REAL peak meters ────────────────────────────────────────────────────
    // Levels are read from AnalyserNodes tapped AFTER each channel's gain node
    // (and after the master fader), so mute/solo/volume/pan are all reflected
    // in what the meter shows. No simulated values anywhere.
    useEffect(() => {
        if (!isPlaying) {
            setMeterLevels({});
            return;
        }

        const buf = new Float32Array(512);
        let raf = 0;
        let frame = 0;

        const readPeak = (analyser: AnalyserNode): number => {
            analyser.getFloatTimeDomainData(buf);
            let peak = 0;
            for (let i = 0; i < buf.length; i++) {
                const v = Math.abs(buf[i]);
                if (v > peak) peak = v;
            }
            // Scale for headroom: full-scale sine ≈ 100%, typical program
            // material lands in a usable meter range.
            return Math.min(100, peak * 140);
        };

        const tick = () => {
            frame++;
            if (frame % 2 === 0) { // ~30Hz is plenty for LED meters
                const next: Record<string, number> = {};
                stemChannels.forEach(stem => {
                    const an = stemAnalysersRef.current[stem.id];
                    if (an) next[stem.id] = readPeak(an);
                });
                const man = masterAnalyserRef.current;
                if (man) next['master'] = readPeak(man);
                setMeterLevels(next);
            }
            raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [isPlaying, stemChannels, stemAnalysersRef, masterAnalyserRef]);

    // Gain-staging vocabulary: faders stay 0-100 internally but READ OUT in
    // dB (20·log10), which is how engineers actually mix.
    const toDb = (pct: number): string => {
        if (pct <= 0) return '-∞';
        const db = 20 * Math.log10(pct / 100);
        return `${db > 0 ? '+' : ''}${db.toFixed(1)}`;
    };

    const handleMastering = async () => {
        setIsMastering(true);
        setMasteringStatus({ ok: true, text: 'Analyzing frequency spectrum and matching LUFS target...' });
        try {
            const res = await workspaceApi.applyMastering(job.id);
            setMasteredLufs(typeof res.lufs === 'number' ? res.lufs : null);
            setMasteringStatus({
                ok: true,
                text: `Reference Mastering Complete · Measured ${typeof res.lufs === 'number' ? res.lufs.toFixed(1) : '—'} LUFS`
            });
            setTimeout(() => setMasteringStatus(null), 5000);
        } catch (e) {
            console.error('Mastering failed', e);
            // Honesty rule: a failed render NEVER reports success.
            setMasteringStatus({
                ok: false,
                text: 'Mastering failed — the track was not modified. Check backend logs and retry.'
            });
            setTimeout(() => setMasteringStatus(null), 6000);
        } finally {
            setIsMastering(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-[#f5f5f7] dark:bg-[#0d0f15] text-slate-900 dark:text-slate-200 select-none overflow-hidden transition-colors duration-200">
            {/* Mixer Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-black/[0.06] dark:border-white/[0.08] bg-white/70 dark:bg-[#12141c]/80 backdrop-blur-xl">
                <div className="flex items-center space-x-3">
                    <Sliders size={18} className="text-teal-600 dark:text-teal-400" />
                    <div>
                        <h2 className="text-xs font-bold text-slate-900 dark:text-slate-100 uppercase tracking-wider">
                            DAW Console Mixer & Matchering DSP
                        </h2>
                        <p className="text-[11px] text-slate-500 dark:text-slate-400">
                            Multitrack gain staging, stereo panning, and reference mastering.
                        </p>
                    </div>
                </div>

                <div className="flex items-center space-x-3">
                    {masteringStatus && (
                        <span
                            className={`text-xs font-mono font-bold flex items-center gap-1.5 px-3 py-1.5 rounded-xl border ${
                                masteringStatus.ok
                                    ? 'text-teal-600 dark:text-teal-400 animate-pulse bg-teal-500/10 border-teal-500/20'
                                    : 'text-rose-600 dark:text-rose-400 bg-rose-500/10 border-rose-500/30'
                            }`}
                        >
                            {masteringStatus.ok ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
                            <span>{masteringStatus.text}</span>
                        </span>
                    )}

                    <button
                        onClick={handleMastering}
                        disabled={isMastering}
                        title="Apply Matchering DSP Reference Mastering (-14 LUFS broadcast standard)"
                        aria-label="Matchering Reference Master"
                        className="px-4 py-2 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-2 transition-all shadow-md shadow-teal-500/20 disabled:opacity-50 active:scale-95"
                    >
                        {isMastering ? <RefreshCw size={14} className="animate-spin" /> : <Wand2 size={14} />}
                        <span>{isMastering ? 'Mastering Track (-14 LUFS)...' : 'Matchering Reference Master'}</span>
                    </button>
                </div>
            </div>

            {/* Fader Console Strips */}
            <div className="flex-1 flex items-center justify-center p-8 space-x-4 md:space-x-8 overflow-x-auto">
                {/* 4 Stem Channels */}
                {stemChannels.map((channel) => {
                    const meter = meterLevels[channel.id] || 0;

                    return (
                        <div
                            key={channel.id}
                            className="w-32 bg-white/80 dark:bg-[#161824]/90 border border-black/[0.06] dark:border-white/10 rounded-3xl p-4 flex flex-col items-center justify-between h-[420px] shadow-apple-md backdrop-blur-xl relative group transition-all"
                        >
                            {/* Channel Title & Color Indicator */}
                            <div className="w-full text-center space-y-1">
                                <div className={`h-1.5 w-8 mx-auto rounded-full bg-gradient-to-r ${channel.color}`} />
                                <span className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate block">
                                    {channel.name}
                                </span>
                                {typeof channel.midiProgram === 'number' && (
                                    <span
                                        className="inline-flex text-[10px] font-mono text-teal-600 dark:text-teal-300 bg-teal-500/10 border border-teal-500/20 px-1.5 py-0.5 rounded-md"
                                        title={`General MIDI program ${channel.midiProgram}`}
                                    >
                                        GM {channel.midiProgram}
                                    </span>
                                )}
                            </div>

                            {/* Stereo Pan Control */}
                            <div className="w-full space-y-1 text-center">
                                <span className="text-[10px] font-mono text-slate-400 block font-bold">
                                    PAN: {channel.pan === 0 ? 'C' : channel.pan < 0 ? `L${Math.abs(channel.pan)}` : `R${channel.pan}`}
                                </span>
                                <input
                                    type="range"
                                    min="-50"
                                    max="50"
                                    value={channel.pan}
                                    onChange={(e) => onPanChange(channel.id, parseInt(e.target.value))}
                                    title={`${channel.name} Stereo Pan: ${channel.pan === 0 ? 'Center' : channel.pan < 0 ? `Left ${Math.abs(channel.pan)}%` : `Right ${channel.pan}%`}`}
                                    aria-label={`${channel.name} Pan Slider`}
                                    className="w-full h-1 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
                                />
                            </div>

                            {/* Solo & Mute Segmented Buttons */}
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={() => onToggleSolo(channel.id)}
                                    title={`Solo ${channel.name} stem`}
                                    aria-label={`Solo ${channel.name}`}
                                    className={`w-7 h-7 rounded-lg text-xs font-black transition-all ${
                                        channel.isSolo
                                            ? 'bg-amber-500 text-slate-950 shadow-sm font-black'
                                            : 'bg-black/5 dark:bg-white/5 text-slate-400 hover:text-amber-500'
                                    }`}
                                >
                                    S
                                </button>
                                <button
                                    onClick={() => onToggleMute(channel.id)}
                                    title={`Mute ${channel.name} stem`}
                                    aria-label={`Mute ${channel.name}`}
                                    className={`w-7 h-7 rounded-lg text-xs font-black transition-all ${
                                        channel.isMuted
                                            ? 'bg-rose-500 text-white shadow-sm font-black'
                                            : 'bg-black/5 dark:bg-white/5 text-slate-400 hover:text-rose-500'
                                    }`}
                                >
                                    M
                                </button>
                            </div>

                            {/* Fader & Peak Meter Container */}
                            <div className="flex items-center space-x-3 h-44 py-2">
                                {/* Vertical Fader Slider — double-click resets to unity (0.0 dB) */}
                                <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    value={channel.volume}
                                    onChange={(e) => onVolumeChange(channel.id, parseInt(e.target.value))}
                                    onDoubleClick={() => onVolumeChange(channel.id, 100)}
                                    title={`${channel.name} Volume: ${toDb(channel.volume)} dB (double-click = unity)`}
                                    aria-label={`${channel.name} Volume Fader (dB)`}
                                    aria-valuetext={`${toDb(channel.volume)} dB`}
                                    className="h-36 w-2 appearance-none bg-slate-200 dark:bg-slate-700 rounded-lg cursor-pointer accent-teal-500"
                                    style={{ writingMode: 'vertical-lr', direction: 'rtl' }}
                                />

                                {/* Real-time LED Peak Level Meter */}
                                <div
                                    title={`${channel.name} Live Peak Level: ${Math.round(meter)}%`}
                                    className="w-2.5 h-36 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden flex flex-col justify-end p-0.5 border border-black/10 dark:border-white/5"
                                >
                                    <div
                                        className="w-full rounded-full transition-all duration-75 bg-gradient-to-t from-emerald-500 via-teal-400 to-amber-500"
                                        style={{ height: `${meter}%` }}
                                    />
                                </div>
                            </div>

                            {/* Fader Decibel Value (unity highlighted) */}
                            <div className={`text-center font-mono text-[11px] font-bold tabular-nums ${
                                channel.volume === 100
                                    ? 'text-teal-600 dark:text-teal-400'
                                    : 'text-slate-700 dark:text-slate-300'
                            }`}>
                                {toDb(channel.volume)} dB
                            </div>
                        </div>
                    );
                })}

                {/* Master Bus Channel Strip */}
                <div className="w-36 bg-white dark:bg-[#1a1c2a] border-2 border-teal-500/40 rounded-3xl p-4 flex flex-col items-center justify-between h-[420px] shadow-apple-lg backdrop-blur-xl relative">
                    <div className="w-full text-center space-y-1">
                        <div className="h-1.5 w-12 mx-auto rounded-full bg-gradient-to-r from-teal-500 to-cyan-500" />
                        <span className="text-xs font-black text-teal-600 dark:text-teal-400 uppercase tracking-wider block">
                            MASTER BUS
                        </span>
                    </div>

                    <div className="w-full text-center">
                        <span
                            className="text-[10px] font-mono font-bold text-teal-600 dark:text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded-md border border-teal-500/20"
                            title={masteredLufs !== null ? `Measured integrated loudness after reference mastering` : 'Run Matchering Reference Master to measure loudness'}
                        >
                            {masteredLufs !== null ? `${masteredLufs.toFixed(1)} LUFS` : 'LUFS —'}
                        </span>
                    </div>

                    {/* Master Fader & Meter */}
                    <div className="flex items-center space-x-3 h-48 py-2">
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.01"
                            value={masterVolume}
                            onChange={(e) => onMasterVolumeChange(parseFloat(e.target.value))}
                            onDoubleClick={() => onMasterVolumeChange(1)}
                            title={`Master Bus Volume: ${toDb(Math.round(masterVolume * 100))} dB (double-click = unity)`}
                            aria-label="Master Bus Volume Fader (dB)"
                            className="h-40 w-3 appearance-none bg-slate-200 dark:bg-slate-700 rounded-lg cursor-pointer accent-teal-500"
                            style={{ writingMode: 'vertical-lr', direction: 'rtl' }}
                        />

                        <div
                            title={`Master Bus Peak: ${Math.round(meterLevels['master'] || 0)}%`}
                            className="w-3.5 h-40 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden flex flex-col justify-end p-0.5 border border-black/10 dark:border-white/5"
                        >
                            <div
                                className="w-full rounded-full transition-all duration-75 bg-gradient-to-t from-teal-500 via-cyan-400 to-amber-500"
                                style={{ height: `${meterLevels['master'] || 0}%` }}
                            />
                        </div>
                    </div>

                    <div className="text-center font-mono text-xs font-black text-teal-600 dark:text-teal-400 tabular-nums">
                        {toDb(Math.round(masterVolume * 100))} dB
                    </div>
                </div>
            </div>
        </div>
    );
};
