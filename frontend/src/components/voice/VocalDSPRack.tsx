import React from 'react';
import {
    Sliders,
    Zap,
    Activity,
    Loader2
} from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';
import type { VoiceProfile, Job } from '../../api';

interface VocalDSPRackProps {
    selectedTrack: Job | null;
    selectedStemPath?: string;
    hasVocalStem: boolean;
    voiceProfiles: VoiceProfile[];
    selectedProfileId: string;
    onSelectProfileId: (id: string) => void;
    pitchShift: number;
    onChangePitchShift: (val: number) => void;
    dryWet: number;
    onChangeDryWet: (val: number) => void;
    formantPreserve: boolean;
    onChangeFormantPreserve: (val: boolean) => void;
    f0Method: string;
    onChangeF0Method: (val: string) => void;
    isConverting: boolean;
    onConvertVocals: () => void;
}

export const VocalDSPRack: React.FC<VocalDSPRackProps> = ({
    selectedTrack,
    hasVocalStem,
    voiceProfiles,
    selectedProfileId,
    onSelectProfileId,
    pitchShift,
    onChangePitchShift,
    dryWet,
    onChangeDryWet,
    formantPreserve,
    onChangeFormantPreserve,
    f0Method,
    onChangeF0Method,
    isConverting,
    onConvertVocals
}) => {
    const selectedProfile = voiceProfiles.find((p) => p.id === selectedProfileId);

    const semitonePresets = [
        { label: 'Original (0)', value: 0 },
        { label: '+12 Octave Up (M→F)', value: 12 },
        { label: '-12 Octave Down (F→M)', value: -12 },
        { label: '+3 Minor 3rd', value: 3 },
        { label: '+7 Perfect 5th', value: 7 },
    ];

    return (
        <GlassCard className="p-5 space-y-5 border border-black/[0.08] dark:border-white/10">
            {/* Header */}
            <div className="flex items-center justify-between pb-3 border-b border-black/[0.06] dark:border-white/10">
                <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 rounded-xl bg-amber-500/10 text-amber-600 dark:text-amber-400 flex items-center justify-center">
                        <Sliders size={16} />
                    </div>
                    <div>
                        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-900 dark:text-white flex items-center gap-1.5">
                            <span>Singing Voice Conversion (SVC) Rack</span>
                            <span className="text-[10px] px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-600 dark:text-teal-400 font-bold border border-teal-500/20">
                                {selectedProfile ? `Target: ${selectedProfile.name}` : 'Neural Timbre'}
                            </span>
                        </h4>
                        <p className="text-[11px] text-slate-500 dark:text-slate-400">
                            Morph vocal tract timbre and harmonics with phase vocoder pitch shifting
                        </p>
                    </div>
                </div>
            </div>

            {/* Target Voice Profile Selector */}
            <div className="space-y-2">
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                    Target Voice Identity
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {voiceProfiles.map((p) => {
                        const isSelected = p.id === selectedProfileId;
                        return (
                            <button
                                key={p.id}
                                type="button"
                                onClick={() => onSelectProfileId(p.id)}
                                className={`p-3 rounded-xl text-left border transition-all flex flex-col justify-between ${
                                    isSelected
                                        ? 'bg-teal-500/10 border-teal-500/50 shadow-sm'
                                        : 'bg-black/[0.02] dark:bg-white/[0.02] border-black/[0.06] dark:border-white/10 hover:border-black/20 dark:hover:border-white/20'
                                }`}
                            >
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-bold text-slate-900 dark:text-white truncate">
                                        {p.name}
                                    </span>
                                    {p.is_default && (
                                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-teal-500/20 text-teal-600 dark:text-teal-400 font-semibold">
                                            Default
                                        </span>
                                    )}
                                </div>
                                <p className="text-[10px] text-slate-500 dark:text-slate-400 line-clamp-1 mt-0.5">
                                    {p.description || 'Custom Voice Model'}
                                </p>
                                {p.acoustic_features?.median_f0_hz && (
                                    <div className="flex items-center gap-1.5 mt-2 text-[9px] text-slate-400">
                                        <Activity size={10} className="text-teal-500" />
                                        <span>Median F0: {p.acoustic_features.median_f0_hz} Hz</span>
                                    </div>
                                )}
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* Musical Pitch Transposition Slider & Presets */}
            <div className="space-y-2.5 p-4 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.06] dark:border-white/10">
                <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider">
                        Musical Pitch Transposition
                    </span>
                    <span className="text-xs font-mono font-bold text-amber-500 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
                        {pitchShift > 0 ? `+${pitchShift}` : pitchShift} Semitones
                    </span>
                </div>

                <input
                    type="range"
                    min="-12"
                    max="12"
                    step="1"
                    value={pitchShift}
                    onChange={(e) => onChangePitchShift(parseInt(e.target.value, 10))}
                    className="w-full accent-amber-500 cursor-pointer h-2 rounded-lg bg-black/10 dark:bg-white/10"
                />

                {/* Quick Presets */}
                <div className="flex flex-wrap gap-1.5 pt-1">
                    {semitonePresets.map((preset) => (
                        <button
                            key={preset.value}
                            type="button"
                            onClick={() => onChangePitchShift(preset.value)}
                            className={`px-2 py-1 rounded-md text-[10px] font-bold transition-all ${
                                pitchShift === preset.value
                                    ? 'bg-amber-500 text-slate-950 shadow-sm'
                                    : 'bg-black/[0.04] dark:bg-white/5 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                            }`}
                        >
                            {preset.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Formant Preservation & Wet/Dry Blend */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Formant Preservation */}
                <div className="p-4 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.06] dark:border-white/10 flex flex-col justify-between">
                    <div>
                        <div className="flex items-center justify-between">
                            <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                                Formant Preservation
                            </span>
                            <button
                                type="button"
                                onClick={() => onChangeFormantPreserve(!formantPreserve)}
                                className={`w-9 h-5 rounded-full transition-colors relative ${
                                    formantPreserve ? 'bg-teal-500' : 'bg-black/20 dark:bg-white/20'
                                }`}
                            >
                                <div
                                    className={`w-3.5 h-3.5 rounded-full bg-white transition-transform absolute top-0.75 left-0.75 ${
                                        formantPreserve ? 'translate-x-4' : 'translate-x-0'
                                    }`}
                                />
                            </button>
                        </div>
                        <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">
                            Phase-locked formant tracking prevents unnatural chipmunk timbre on pitch shifts.
                        </p>
                    </div>
                </div>

                {/* Wet / Dry Ratio */}
                <div className="p-4 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.06] dark:border-white/10 space-y-2">
                    <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                            Wet / Dry Mix
                        </span>
                        <span className="text-xs font-mono font-bold text-teal-600 dark:text-teal-400">
                            {dryWet}% Wet
                        </span>
                    </div>
                    <input
                        type="range"
                        min="0"
                        max="100"
                        step="5"
                        value={dryWet}
                        onChange={(e) => onChangeDryWet(parseInt(e.target.value, 10))}
                        className="w-full accent-teal-500 cursor-pointer h-2 rounded-lg bg-black/10 dark:bg-white/10"
                    />
                    <div className="flex justify-between text-[9px] text-slate-400 font-mono">
                        <span>Original Vocal (0%)</span>
                        <span>100% Converted</span>
                    </div>
                </div>
            </div>

            {/* F0 Pitch Detection Algorithm */}
            <div className="p-3.5 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.06] dark:border-white/10 flex items-center justify-between">
                <div>
                    <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                        Pitch Extraction Method (F0)
                    </span>
                    <p className="text-[10px] text-slate-500 dark:text-slate-400">
                        Algorithm used to isolate source vocal pitch contours
                    </p>
                </div>
                <select
                    value={f0Method}
                    onChange={(e) => onChangeF0Method(e.target.value)}
                    className="apple-input text-xs py-1 px-2.5 font-medium max-w-[140px]"
                >
                    <option value="rmvpe">RMVPE (High Quality)</option>
                    <option value="crepe">CREPE (Harmonic)</option>
                    <option value="harvest">Harvest (Robust)</option>
                    <option value="pm">PM (Fast)</option>
                </select>
            </div>

            {/* Conversion Trigger Button */}
            <div className="pt-2">
                <button
                    type="button"
                    onClick={onConvertVocals}
                    disabled={!selectedTrack || !hasVocalStem || isConverting}
                    className="w-full py-3.5 rounded-xl bg-gradient-to-r from-teal-500 via-cyan-500 to-blue-500 hover:from-teal-400 hover:to-blue-400 disabled:opacity-40 text-slate-950 font-bold text-sm flex items-center justify-center gap-2 shadow-lg shadow-teal-500/20 active:scale-[0.99] transition-all"
                >
                    {isConverting ? (
                        <>
                            <Loader2 size={16} className="animate-spin" />
                            <span>Synthesizing Target Voice Timbre...</span>
                        </>
                    ) : (
                        <>
                            <Zap size={16} />
                            <span>Transform Vocal Stem & Remix Master ⚡</span>
                        </>
                    )}
                </button>

                {!hasVocalStem && selectedTrack && (
                    <p className="text-center text-[11px] text-amber-500 mt-2">
                        ⚠️ No isolated vocal stem found for this track. Stem separation must complete first.
                    </p>
                )}
            </div>
        </GlassCard>
    );
};
