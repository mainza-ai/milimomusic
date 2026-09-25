import React, { useState } from 'react';
import {
    Sparkles,
    Cpu,
    Mic,
    Users,
    Type,
    Film,
    Wand2,
    Check
} from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';
import type { VideoModelKey } from '../views/MusicVideosView';

export interface VideoInspectorProps {
    // Engine & Hardware
    videoModel: VideoModelKey;
    onSelectModel: (m: VideoModelKey) => void;
    modelConstraints: Record<string, { label: string; minSec: number; maxSec: number; defaultSec: number; desc: string }>;
    modelRegistry: Record<string, any>;
    activeVideoEngine: VideoModelKey | null;
    videoProvider: 'local' | 'cloud_fal' | 'cloud_replicate';
    onSelectProvider: (p: 'local' | 'cloud_fal' | 'cloud_replicate') => void;
    clipDuration: number;
    onChangeClipDuration: (dur: number) => void;
    onResetClipDuration: () => void;
    resolution: '720p' | '1080p';
    onChangeResolution: (res: '720p' | '1080p') => void;

    // Directing & Pacing
    videoStyle: string;
    onSelectStyle: (style: any) => void;
    pacingBias: number;
    onChangePacingBias: (val: number) => void;
    vocalBypass: boolean;
    onChangeVocalBypass: (val: boolean) => void;
    fidelityRetries: number;
    onChangeFidelityRetries: (val: number) => void;
    autoContinue: boolean;
    onChangeAutoContinue: (val: boolean) => void;
    onGenerateStoryboard?: () => void;
    isGeneratingStory?: boolean;

    // Lip-Sync & FX
    enableLipSync: boolean;
    onChangeEnableLipSync: (val: boolean) => void;
    lipSyncEngine: 'live_portrait' | 'fallback';
    onChangeLipSyncEngine: (e: 'live_portrait' | 'fallback') => void;
    burnSubtitles: boolean;
    onChangeBurnSubtitles: (val: boolean) => void;
    subtitleStyle: 'neon' | 'cinematic' | 'karaoke';
    onChangeSubtitleStyle: (style: 'neon' | 'cinematic' | 'karaoke') => void;
    transitionStyle: 'beat_cut' | 'crossfade' | 'flash' | 'whip_pan' | 'glitch';
    onChangeTransitionStyle: (trans: 'beat_cut' | 'crossfade' | 'flash' | 'whip_pan' | 'glitch') => void;

    // Cast & Character Seeds
    visibleCast: string[];
    onToggleCastMember: (cast: string) => void;
    characterPromptNote?: string;
    onChangeCharacterPromptNote?: (note: string) => void;
}

export const VideoInspectorDock: React.FC<VideoInspectorProps> = ({
    videoModel,
    onSelectModel,
    modelConstraints,
    modelRegistry,
    activeVideoEngine,
    videoProvider,
    onSelectProvider,
    clipDuration,
    onChangeClipDuration,
    onResetClipDuration,
    resolution,
    onChangeResolution,
    videoStyle,
    onSelectStyle,
    pacingBias,
    onChangePacingBias,
    vocalBypass,
    onChangeVocalBypass,
    fidelityRetries,
    onChangeFidelityRetries,
    autoContinue,
    onChangeAutoContinue,
    onGenerateStoryboard,
    isGeneratingStory = false,
    enableLipSync,
    onChangeEnableLipSync,
    lipSyncEngine,
    onChangeLipSyncEngine,
    burnSubtitles,
    onChangeBurnSubtitles,
    subtitleStyle,
    onChangeSubtitleStyle,
    transitionStyle,
    onChangeTransitionStyle,
    visibleCast,
    onToggleCastMember,
    characterPromptNote = '',
    onChangeCharacterPromptNote,
}) => {
    const [activeTab, setActiveTab] = useState<'directing' | 'engine' | 'lipsync' | 'cast'>('directing');

    const pacingLabels = {
        '-2': '🌊 Sweeping (-2)',
        '-1': '🎬 Cinematic (-1)',
        '0': '⚖️ Balanced (0)',
        '1': '⚡ Rhythmic (+1)',
        '2': '🔥 Montage (+2)'
    };

    return (
        <GlassCard className="p-4 flex flex-col h-full rounded-2xl border border-black/[0.08] dark:border-white/10 shadow-apple-lg">
            {/* Dock Header Tabs */}
            <div className="flex items-center space-x-1 p-1 bg-black/[0.04] dark:bg-white/5 rounded-xl border border-black/[0.06] dark:border-white/10 mb-4">
                <button
                    type="button"
                    onClick={() => setActiveTab('directing')}
                    className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-1.5 ${
                        activeTab === 'directing'
                            ? 'bg-white dark:bg-white/15 text-indigo-600 dark:text-indigo-400 shadow-sm'
                            : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-300'
                    }`}
                >
                    <Sparkles size={13} />
                    <span>Directing</span>
                </button>

                <button
                    type="button"
                    onClick={() => setActiveTab('engine')}
                    className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-1.5 ${
                        activeTab === 'engine'
                            ? 'bg-white dark:bg-white/15 text-teal-600 dark:text-teal-400 shadow-sm'
                            : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-300'
                    }`}
                >
                    <Cpu size={13} />
                    <span>Engine & HW</span>
                </button>

                <button
                    type="button"
                    onClick={() => setActiveTab('lipsync')}
                    className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-1.5 ${
                        activeTab === 'lipsync'
                            ? 'bg-white dark:bg-white/15 text-cyan-600 dark:text-cyan-400 shadow-sm'
                            : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-300'
                    }`}
                >
                    <Mic size={13} />
                    <span>Lip-Sync & FX</span>
                </button>

                <button
                    type="button"
                    onClick={() => setActiveTab('cast')}
                    className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-1.5 ${
                        activeTab === 'cast'
                            ? 'bg-white dark:bg-white/15 text-purple-600 dark:text-purple-400 shadow-sm'
                            : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-300'
                    }`}
                >
                    <Users size={13} />
                    <span>Cast & Seeds</span>
                </button>
            </div>

            {/* Tab 1: Directing & Musical Pacing */}
            {activeTab === 'directing' && (
                <div className="space-y-4 overflow-y-auto pr-1">
                    {/* Visual Aesthetic Preset Grid */}
                    <div className="space-y-2">
                        <label className="text-xs font-bold uppercase tracking-wider text-slate-400 block">
                            Visual Aesthetic Palette
                        </label>
                        <div className="grid grid-cols-2 gap-2">
                            {[
                                { id: 'neon-cyberpunk', name: 'Cyberpunk', desc: 'Rain-slicked neon & cyan lens flares' },
                                { id: 'anime-cinematic', name: 'Anime Cinematic', desc: 'Hand-drawn Makoto Shinkai aesthetic' },
                                { id: 'retro-vhs', name: '80s Retro VHS', desc: 'Analog tape saturation & scanlines' },
                                { id: 'minimal-lyrics', name: 'Minimal Stage', desc: 'High-contrast monochrome & spotlight' }
                            ].map((style) => (
                                <button
                                    key={style.id}
                                    type="button"
                                    onClick={() => onSelectStyle(style.id)}
                                    className={`p-2.5 rounded-xl border text-left transition-all ${
                                        videoStyle === style.id
                                            ? 'bg-indigo-500/10 border-indigo-500/40 text-indigo-700 dark:text-indigo-300 font-bold shadow-sm'
                                            : 'bg-black/[0.02] dark:bg-white/[0.02] border-transparent text-slate-600 dark:text-slate-400 hover:bg-black/[0.04]'
                                    }`}
                                >
                                    <div className="text-xs font-bold">{style.name}</div>
                                    <div className="text-[10px] text-slate-400 font-normal mt-0.5 line-clamp-1">{style.desc}</div>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Cut Speed / Pacing Bias Slider */}
                    <div className="p-3.5 bg-black/[0.02] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 rounded-2xl space-y-2">
                        <div className="flex items-center justify-between text-xs">
                            <span className="font-semibold text-slate-700 dark:text-slate-300">
                                Cut Speed / Pacing Bias
                            </span>
                            <span className="font-mono font-bold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded text-[11px]">
                                {pacingLabels[String(pacingBias) as keyof typeof pacingLabels] || `${pacingBias}`}
                            </span>
                        </div>
                        <input
                            type="range"
                            min={-2}
                            max={2}
                            step={1}
                            value={pacingBias}
                            onChange={(e) => onChangePacingBias(parseInt(e.target.value, 10))}
                            className="w-full accent-indigo-500 h-1.5 bg-black/[0.06] dark:bg-white/10 rounded-lg cursor-pointer"
                        />
                        <div className="flex justify-between text-[9px] font-mono text-slate-400">
                            <span>-2 Sweeping Takes</span>
                            <span>0 Downbeats</span>
                            <span>+2 Fast Cuts</span>
                        </div>
                    </div>

                    {/* Two-Tier Vocal Bypass Toggle */}
                    <div className="p-3.5 bg-black/[0.02] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 rounded-2xl flex items-center justify-between">
                        <div className="pr-2">
                            <div className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                                Music Timeline Vocal Bypass
                            </div>
                            <div className="text-[10px] text-slate-400 mt-0.5">
                                Strictly locks singer mouth to vocal stem; suppresses dialogue hallucination
                            </div>
                        </div>
                        <input
                            type="checkbox"
                            checked={vocalBypass}
                            onChange={(e) => onChangeVocalBypass(e.target.checked)}
                            className="w-4 h-4 rounded border-slate-700 text-indigo-500 focus:ring-indigo-500 cursor-pointer accent-indigo-500"
                        />
                    </div>

                    {/* Fidelity Repair Retries */}
                    <div className="p-3.5 bg-black/[0.02] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 rounded-2xl space-y-2">
                        <div className="flex items-center justify-between text-xs">
                            <span className="font-semibold text-slate-700 dark:text-slate-300">
                                Fidelity Repair Retries
                            </span>
                            <span className="font-mono text-indigo-400 text-xs font-bold">
                                {fidelityRetries} {fidelityRetries === 1 ? 'attempt' : 'attempts'}
                            </span>
                        </div>
                        <input
                            type="range"
                            min={0}
                            max={5}
                            step={1}
                            value={fidelityRetries}
                            onChange={(e) => onChangeFidelityRetries(parseInt(e.target.value, 10))}
                            className="w-full accent-indigo-500 h-1.5 bg-black/[0.06] dark:bg-white/10 rounded-lg cursor-pointer"
                        />
                        <div className="flex items-center justify-between pt-1">
                            <span className="text-[10px] text-slate-400">Auto-continue if checks fail</span>
                            <input
                                type="checkbox"
                                checked={autoContinue}
                                onChange={(e) => onChangeAutoContinue(e.target.checked)}
                                className="w-3.5 h-3.5 rounded border-slate-700 text-indigo-500 cursor-pointer accent-indigo-500"
                            />
                        </div>
                    </div>

                    {/* Storyboard Notes Trigger */}
                    {onGenerateStoryboard && (
                        <div className="flex justify-end pt-1">
                            <button
                                type="button"
                                onClick={onGenerateStoryboard}
                                disabled={isGeneratingStory}
                                className="text-[11px] text-indigo-600 dark:text-indigo-400 hover:underline flex items-center gap-1 font-semibold"
                            >
                                <Wand2 size={12} />
                                <span>{isGeneratingStory ? 'Writing Notes…' : 'Generate Director Notes'}</span>
                            </button>
                        </div>
                    )}
                </div>
            )}

            {/* Tab 2: Engine, Duration & Hardware */}
            {activeTab === 'engine' && (
                <div className="space-y-4 overflow-y-auto pr-1">
                    {/* Execution Provider Picker */}
                    <div className="space-y-2">
                        <label className="text-xs font-bold uppercase tracking-wider text-slate-400 block">
                            Execution Provider
                        </label>
                        <div className="grid grid-cols-2 gap-2">
                            <button
                                type="button"
                                onClick={() => onSelectProvider('local')}
                                className={`p-2.5 rounded-xl border text-left transition-all ${
                                    videoProvider === 'local'
                                        ? 'bg-teal-500/10 border-teal-500/40 text-teal-700 dark:text-teal-300 font-bold shadow-sm'
                                        : 'bg-black/[0.02] dark:bg-white/[0.02] border-transparent text-slate-600 dark:text-slate-400'
                                }`}
                            >
                                <div className="text-xs font-bold flex items-center justify-between">
                                    <span>🖥️ Local M3 Max</span>
                                    <span className="text-[9px] font-mono px-1 py-0.5 rounded bg-teal-500/20 text-teal-700 dark:text-teal-300 font-bold">Free</span>
                                </div>
                                <div className="text-[10px] text-slate-400 font-normal mt-0.5">PyTorch MPS Engine</div>
                            </button>

                            <button
                                type="button"
                                onClick={() => onSelectProvider('cloud_fal')}
                                className={`p-2.5 rounded-xl border text-left transition-all ${
                                    videoProvider === 'cloud_fal'
                                        ? 'bg-cyan-500/10 border-cyan-500/40 text-cyan-700 dark:text-cyan-300 font-bold shadow-sm'
                                        : 'bg-black/[0.02] dark:bg-white/[0.02] border-transparent text-slate-600 dark:text-slate-400'
                                }`}
                            >
                                <div className="text-xs font-bold flex items-center justify-between">
                                    <span>⚡ Cloud GPU</span>
                                    <span className="text-[9px] font-mono px-1 py-0.5 rounded bg-cyan-500/20 text-cyan-700 dark:text-cyan-300 font-bold">Fast</span>
                                </div>
                                <div className="text-[10px] text-slate-400 font-normal mt-0.5">Parallel H100 Cluster</div>
                            </button>
                        </div>
                    </div>

                    {/* Deduplicated Video Engines List */}
                    <div className="space-y-2">
                        <label className="text-xs font-bold uppercase tracking-wider text-slate-400 block">
                            Diffusion Engine
                        </label>
                        <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
                            {Object.entries(modelConstraints)
                                .filter(([k]) => k !== 'wan2.1') // Deduplicate duplicate Wan key
                                .map(([key, conf]) => {
                                    const isSel = videoModel === key;
                                    const isAct = activeVideoEngine === key;
                                    const isLocal = modelRegistry[key]?.local_weights_present;
                                    return (
                                        <button
                                            key={key}
                                            type="button"
                                            onClick={() => onSelectModel(key as VideoModelKey)}
                                            className={`w-full p-2.5 rounded-xl border text-left transition-all ${
                                                isSel
                                                    ? 'bg-teal-500/10 border-teal-500/40 text-teal-700 dark:text-teal-300 font-bold shadow-sm'
                                                    : 'bg-black/[0.02] dark:bg-white/[0.02] border-transparent text-slate-600 dark:text-slate-400 hover:bg-black/[0.04]'
                                            }`}
                                        >
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center gap-1.5">
                                                    <span className="text-xs font-bold">{conf.label}</span>
                                                    {isAct && (
                                                        <span className="text-[9px] font-mono px-1 py-0.5 rounded bg-teal-500/15 text-teal-700 dark:text-teal-300 font-bold border border-teal-500/20">
                                                            ● Active
                                                        </span>
                                                    )}
                                                    {isLocal && (
                                                        <span className="text-[9px] font-mono px-1 py-0.5 rounded bg-amber-500/10 text-amber-600 dark:text-amber-400 font-bold">
                                                            ⚡ Ready
                                                        </span>
                                                    )}
                                                </div>
                                                <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-black/5 dark:bg-white/10 text-slate-500">
                                                    Max {conf.maxSec}s
                                                </span>
                                            </div>
                                            <div className="text-[10px] text-slate-400 font-normal mt-0.5 line-clamp-1">{conf.desc}</div>
                                        </button>
                                    );
                                })}
                        </div>
                    </div>

                    {/* Clip Duration Slider */}
                    <div className="p-3.5 bg-black/[0.02] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 rounded-2xl space-y-2">
                        <div className="flex items-center justify-between text-xs">
                            <span className="font-semibold text-slate-700 dark:text-slate-300">
                                Max Clip Duration
                            </span>
                            <div className="flex items-center gap-2">
                                <span className="font-mono font-bold text-teal-600 dark:text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded-md text-xs">
                                    {clipDuration.toFixed(1)}s
                                </span>
                                {clipDuration !== (modelConstraints[videoModel]?.maxSec || 5.0) && (
                                    <button
                                        type="button"
                                        onClick={onResetClipDuration}
                                        className="text-[10px] text-teal-600 dark:text-teal-400 hover:underline font-mono"
                                    >
                                        Reset Max
                                    </button>
                                )}
                            </div>
                        </div>
                        <input
                            type="range"
                            min={modelConstraints[videoModel]?.minSec || 2.0}
                            max={modelConstraints[videoModel]?.maxSec || 5.0}
                            step={0.5}
                            value={clipDuration}
                            onChange={(e) => onChangeClipDuration(parseFloat(e.target.value))}
                            className="w-full accent-teal-500 h-1.5 bg-black/[0.06] dark:bg-white/10 rounded-lg cursor-pointer"
                        />
                    </div>

                    {/* Output Resolution */}
                    <div className="flex items-center justify-between pt-1">
                        <span className="text-xs text-slate-400 font-bold uppercase tracking-wider">Output Resolution</span>
                        <div className="flex gap-1.5">
                            <button
                                type="button"
                                onClick={() => onChangeResolution('720p')}
                                className={`px-2.5 py-1 text-xs rounded-lg font-bold transition-all ${
                                    resolution === '720p'
                                        ? 'bg-teal-500 text-slate-950 shadow-sm'
                                        : 'bg-black/5 dark:bg-white/5 text-slate-400'
                                }`}
                            >
                                720p HD
                            </button>
                            <button
                                type="button"
                                onClick={() => onChangeResolution('1080p')}
                                className={`px-2.5 py-1 text-xs rounded-lg font-bold transition-all ${
                                    resolution === '1080p'
                                        ? 'bg-teal-500 text-slate-950 shadow-sm'
                                        : 'bg-black/5 dark:bg-white/5 text-slate-400'
                                }`}
                            >
                                1080p FHD
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Tab 3: Lip-Syncing, Subtitles & FX */}
            {activeTab === 'lipsync' && (
                <div className="space-y-4 overflow-y-auto pr-1">
                    {/* Vocal Lip-Syncing Card */}
                    <div className="p-3.5 bg-black/[0.02] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 rounded-2xl space-y-3">
                        <div className="flex items-center justify-between">
                            <label className="text-xs font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
                                <Mic size={14} className="text-teal-500" />
                                <span>Vocal Lip-Sync Animation</span>
                            </label>
                            <input
                                type="checkbox"
                                checked={enableLipSync}
                                onChange={(e) => onChangeEnableLipSync(e.target.checked)}
                                className="w-4 h-4 rounded border-slate-700 text-teal-500 cursor-pointer accent-teal-500"
                            />
                        </div>

                        {enableLipSync && (
                            <div className="space-y-2 pt-1 border-t border-black/[0.04] dark:border-white/5">
                                <div className="grid grid-cols-2 gap-2">
                                    <button
                                        type="button"
                                        onClick={() => onChangeLipSyncEngine('live_portrait')}
                                        className={`p-2 rounded-xl text-left border transition-all ${
                                            lipSyncEngine === 'live_portrait'
                                                ? 'bg-teal-500/10 border-teal-500/40 text-teal-700 dark:text-teal-300 font-bold'
                                                : 'border-transparent text-slate-500'
                                        }`}
                                    >
                                        <div className="text-xs font-bold">LivePortrait</div>
                                        <div className="text-[9px] text-slate-400">Neural avatar with natural eye blinks</div>
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => onChangeLipSyncEngine('fallback')}
                                        className={`p-2 rounded-xl text-left border transition-all ${
                                            lipSyncEngine === 'fallback'
                                                ? 'bg-teal-500/10 border-teal-500/40 text-teal-700 dark:text-teal-300 font-bold'
                                                : 'border-transparent text-slate-500'
                                        }`}
                                    >
                                        <div className="text-xs font-bold">Viseme Mesh</div>
                                        <div className="text-[9px] text-slate-400">Fast volume-reactive mouth motion</div>
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Subtitle / Lyric Burning Card */}
                    <div className="p-3.5 bg-black/[0.02] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 rounded-2xl space-y-3">
                        <div className="flex items-center justify-between">
                            <label className="text-xs font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
                                <Type size={14} className="text-cyan-500" />
                                <span>Burn Subtitles / Lyrics</span>
                            </label>
                            <input
                                type="checkbox"
                                checked={burnSubtitles}
                                onChange={(e) => onChangeBurnSubtitles(e.target.checked)}
                                className="w-4 h-4 rounded border-slate-700 text-cyan-500 cursor-pointer accent-cyan-500"
                            />
                        </div>

                        {burnSubtitles && (
                            <div className="flex gap-2 pt-1 border-t border-black/[0.04] dark:border-white/5">
                                {(['neon', 'cinematic', 'karaoke'] as const).map((style) => (
                                    <button
                                        key={style}
                                        type="button"
                                        onClick={() => onChangeSubtitleStyle(style)}
                                        className={`flex-1 py-1.5 text-xs rounded-xl border capitalize font-semibold transition-all ${
                                            subtitleStyle === style
                                                ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-600 dark:text-cyan-400 font-bold'
                                                : 'border-transparent bg-black/[0.02] dark:bg-white/5 text-slate-400'
                                        }`}
                                    >
                                        {style}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Transition Style Selector */}
                    <div className="p-3.5 bg-black/[0.02] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 rounded-2xl space-y-2">
                        <label className="text-xs font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
                            <Film size={14} className="text-purple-400" />
                            <span>Cut Transition Style</span>
                        </label>
                        <div className="grid grid-cols-3 gap-1.5 pt-1">
                            {[
                                { id: 'beat_cut', label: 'Beat Cut' },
                                { id: 'crossfade', label: 'Crossfade' },
                                { id: 'flash', label: 'White Flash' },
                                { id: 'whip_pan', label: 'Whip Pan' },
                                { id: 'glitch', label: 'Glitch' },
                            ].map((tr) => (
                                <button
                                    key={tr.id}
                                    type="button"
                                    onClick={() => onChangeTransitionStyle(tr.id as any)}
                                    className={`py-1.5 px-2 rounded-xl text-xs font-semibold border transition-all text-center ${
                                        transitionStyle === tr.id
                                            ? 'bg-purple-500/15 border-purple-500/40 text-purple-600 dark:text-purple-400 font-bold'
                                            : 'border-transparent bg-black/[0.02] dark:bg-white/5 text-slate-400 hover:text-slate-200'
                                    }`}
                                >
                                    {tr.label}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Tab 4: Cast & Character Seeds */}
            {activeTab === 'cast' && (
                <div className="space-y-4 overflow-y-auto pr-1">
                    {/* Visible Cast Scoping */}
                    <div className="p-3.5 bg-black/[0.02] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 rounded-2xl space-y-2">
                        <label className="text-xs font-bold uppercase tracking-wider text-slate-400 block">
                            Visible Cast Members
                        </label>
                        <div className="flex flex-wrap gap-2">
                            {['Lead Vocalist', 'Guitarist', 'Drummer', 'Atmospheric B-Roll'].map((cast) => {
                                const isSel = visibleCast.includes(cast);
                                return (
                                    <button
                                        key={cast}
                                        type="button"
                                        onClick={() => onToggleCastMember(cast)}
                                        className={`px-3 py-1 text-xs rounded-xl font-semibold transition-all flex items-center gap-1.5 ${
                                            isSel
                                                ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 font-bold'
                                                : 'bg-black/[0.03] dark:bg-white/5 text-slate-400 border border-transparent hover:text-slate-200'
                                        }`}
                                    >
                                        {isSel && <Check size={12} />}
                                        <span>{cast}</span>
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Character Visual Prompt / Continuity Notes */}
                    <div className="p-3.5 bg-black/[0.02] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 rounded-2xl space-y-2">
                        <label className="text-xs font-bold text-slate-800 dark:text-slate-200 flex items-center justify-between">
                            <span>Character Visual Seed & Prompt</span>
                            <span className="text-[10px] text-slate-400 font-mono">Continuity</span>
                        </label>
                        <textarea
                            rows={3}
                            placeholder="e.g. Lead female cyber-singer with silver undercut, glowing teal cybernetic collar, dark leather vest..."
                            value={characterPromptNote}
                            onChange={(e) => onChangeCharacterPromptNote?.(e.target.value)}
                            className="w-full text-xs apple-input rounded-xl p-2.5 font-sans resize-none"
                        />
                        <p className="text-[10px] text-slate-400">
                            Prepends character identity keywords to every vocal shot to preserve persona across cuts.
                        </p>
                    </div>
                </div>
            )}
        </GlassCard>
    );
};
