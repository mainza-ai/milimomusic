import React, { useMemo } from 'react';
import type { Job } from '../../api';
import { coverApi } from '../../api';
import type { AspectRatioType } from './VideoTopBar';

export interface TimedWord {
    word: string;
    start: number;
    end: number;
}

export interface TimedLine {
    text: string;
    start: number;
    end: number;
    words?: TimedWord[];
    section?: string;
}

export interface LyricCanvasOverlayProps {
    activeSong?: Job;
    currentTime: number;
    isPlaying: boolean;
    stylePreset?: string;
    aspectRatio?: AspectRatioType;
    backgroundMode?: 'cover_art' | 'spectrum' | 'procedural' | string;
    fontFamily?: string;
    fontSizeOverride?: number;
    isOverVideo?: boolean;
    onSeek?: (timeSec: number) => void;
}

export const PRESET_CONFIGS: Record<string, {
    sungColor: string;
    unsungColor: string;
    fontFamily: string;
    fontSizeRatio: number;
    alignment: 'center' | 'left' | 'vertical_safe';
    glow: string;
    textTransform?: 'uppercase' | 'none';
    letterSpacing?: string;
}> = {
    neon: {
        sungColor: '#00F0FF',
        unsungColor: '#E2E8F0',
        fontFamily: "'Syne', 'Orbitron', 'Inter', sans-serif",
        fontSizeRatio: 0.052,
        alignment: 'center',
        glow: '0 0 16px rgba(0, 240, 255, 0.75), 0 0 32px rgba(0, 240, 255, 0.4)',
        letterSpacing: '0.02em',
    },
    spotify: {
        sungColor: '#10B981',
        unsungColor: 'rgba(255, 255, 255, 0.45)',
        fontFamily: "'Inter', system-ui, sans-serif",
        fontSizeRatio: 0.046,
        alignment: 'left',
        glow: 'none',
        letterSpacing: '-0.01em',
    },
    kinetic_pop: {
        sungColor: '#FFE600',
        unsungColor: '#FFFFFF',
        fontFamily: "'Montserrat', 'Inter', sans-serif",
        fontSizeRatio: 0.054,
        alignment: 'center',
        glow: '0 2px 8px rgba(0, 0, 0, 0.8), 0 0 14px rgba(255, 230, 0, 0.6)',
        letterSpacing: '0.01em',
    },
    cinematic: {
        sungColor: '#E6C280',
        unsungColor: 'rgba(226, 232, 240, 0.6)',
        fontFamily: "'Playfair Display', 'Cinzel', Georgia, serif",
        fontSizeRatio: 0.044,
        alignment: 'center',
        glow: '0 2px 10px rgba(0, 0, 0, 0.9)',
        textTransform: 'uppercase',
        letterSpacing: '0.14em',
    },
    social_vertical: {
        sungColor: '#FFE600',
        unsungColor: '#FFFFFF',
        fontFamily: "'Impact', 'Bebas Neue', sans-serif",
        fontSizeRatio: 0.058,
        alignment: 'vertical_safe',
        glow: '0 0 0 3px #000, 0 4px 12px rgba(0, 0, 0, 0.9)',
        textTransform: 'uppercase',
        letterSpacing: '0.04em',
    },
    retro_vhs: {
        sungColor: '#FF3366',
        unsungColor: '#33FFCC',
        fontFamily: "'Space Mono', 'Courier New', monospace",
        fontSizeRatio: 0.048,
        alignment: 'center',
        glow: '2px 2px 0px #00ffff, -2px -2px 0px #ff00ff',
        letterSpacing: '0.06em',
    },
};

export const LyricCanvasOverlay: React.FC<LyricCanvasOverlayProps> = ({
    activeSong,
    currentTime,
    isPlaying,
    stylePreset = 'neon',
    aspectRatio = '16:9',
    backgroundMode = 'cover_art',
    fontFamily,
    fontSizeOverride,
    isOverVideo = false,
    onSeek,
}) => {
    // 1. Parse Timed Lines
    const timedLines = useMemo<TimedLine[]>(() => {
        if (!activeSong) return [];
        if (activeSong.timed_lyrics_json) {
            try {
                const parsed = JSON.parse(activeSong.timed_lyrics_json);
                if (Array.isArray(parsed) && parsed.length > 0) {
                    return parsed;
                }
            } catch {
                // Ignore parse errors, fallback to raw lyrics
            }
        }
        if (activeSong.lyrics) {
            // Rough synthetic parsing from raw lines
            const raw = activeSong.lyrics.split('\n').filter(l => l.trim() && !l.trim().startsWith('['));
            const estDur = (activeSong.duration_ms ? activeSong.duration_ms / 1000 : 180) / Math.max(1, raw.length);
            return raw.map((lineText, idx) => {
                const start = idx * estDur;
                const end = start + estDur;
                const words = lineText.trim().split(/\s+/).map((w, wIdx, arr) => {
                    const wDur = (end - start) / arr.length;
                    return {
                        word: w,
                        start: start + wIdx * wDur,
                        end: start + (wIdx + 1) * wDur
                    };
                });
                return { text: lineText.trim(), start, end, words };
            });
        }
        return [];
    }, [activeSong]);

    // 2. Identify Active, Previous, and Next Lines
    const { activeLine, prevLine, nextLine } = useMemo(() => {
        if (timedLines.length === 0) {
            return { activeIndex: -1, activeLine: null, prevLine: null, nextLine: null };
        }
        const idx = timedLines.findIndex(l => currentTime >= l.start && currentTime <= l.end);
        if (idx !== -1) {
            return {
                activeIndex: idx,
                activeLine: timedLines[idx],
                prevLine: idx > 0 ? timedLines[idx - 1] : null,
                nextLine: idx < timedLines.length - 1 ? timedLines[idx + 1] : null
            };
        }
        // Fallback: Find closest upcoming line if between lines
        const upcomingIdx = timedLines.findIndex(l => l.start > currentTime);
        if (upcomingIdx > 0) {
            return {
                activeIndex: -1,
                activeLine: null,
                prevLine: timedLines[upcomingIdx - 1],
                nextLine: timedLines[upcomingIdx]
            };
        }
        return { activeIndex: -1, activeLine: null, prevLine: null, nextLine: null };
    }, [timedLines, currentTime]);

    const preset = PRESET_CONFIGS[stylePreset] || PRESET_CONFIGS.neon;
    const isVertical = aspectRatio === '9:16';
    const activeAlignment = isVertical ? 'vertical_safe' : preset.alignment;

    // 3. Resolve Background Artwork
    const coverUrl = useMemo(() => {
        if (!activeSong?.cover_image_path) return null;
        return coverApi.getCoverUrl(activeSong.cover_image_path);
    }, [activeSong?.cover_image_path]);

    // Alignment Classes
    const containerAlignmentClass = useMemo(() => {
        if (activeAlignment === 'vertical_safe') {
            return 'justify-center items-center text-center px-6';
        }
        if (activeAlignment === 'left') {
            return isOverVideo
                ? 'justify-end items-start text-left px-8 pb-8'
                : 'justify-end items-start text-left px-12 pb-14';
        }
        return isOverVideo
            ? 'justify-end items-center text-center px-8 pb-8'
            : 'justify-end items-center text-center px-8 pb-12';
    }, [activeAlignment, isOverVideo]);

    const activeWords = useMemo(() => {
        if (!activeLine) return [];
        if (activeLine.words && activeLine.words.length > 0) {
            return activeLine.words;
        }
        // Fallback: estimate words across active line
        const toks = activeLine.text.trim().split(/\s+/);
        const wDur = (activeLine.end - activeLine.start) / Math.max(1, toks.length);
        return toks.map((w, idx) => ({
            word: w,
            start: activeLine.start + idx * wDur,
            end: activeLine.start + (idx + 1) * wDur
        }));
    }, [activeLine]);

    return (
        <div className={`absolute inset-0 overflow-hidden select-none rounded-xl ${isOverVideo ? 'pointer-events-none' : 'pointer-events-auto'}`}>
            {/* Background Layer (Only rendered if NOT overlaid on active video) */}
            {!isOverVideo && (
                <div className="absolute inset-0 overflow-hidden bg-slate-950">
                    {coverUrl && backgroundMode !== 'spectrum' ? (
                        <>
                            <img
                                src={coverUrl}
                                alt="Cover Art"
                                className={`w-full h-full object-cover transition-transform duration-1000 ${
                                    isPlaying ? 'scale-110 motion-safe:animate-pulse' : 'scale-100'
                                }`}
                                style={{
                                    filter: 'brightness(0.55) contrast(1.15) saturate(1.2)',
                                    transform: isPlaying ? 'scale(1.08) translate(-1%, -1%)' : 'scale(1.0)',
                                    transition: 'transform 8s ease-out',
                                }}
                            />
                            <div className="absolute inset-0 bg-gradient-to-t from-slate-950/90 via-slate-950/40 to-slate-950/30" />
                        </>
                    ) : (
                        /* Audio Reactive / Procedural Dark Wave Backdrop */
                        <div className="w-full h-full bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950/40 relative flex items-center justify-center">
                            <div className="absolute inset-0 opacity-25 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-teal-500/20 via-transparent to-transparent animate-pulse" />
                            {/* Simulated waveform bars */}
                            <div className="flex items-center gap-1.5 opacity-40">
                                {[...Array(32)].map((_, i) => (
                                    <div
                                        key={i}
                                        className="w-1 bg-gradient-to-t from-teal-400 to-cyan-300 rounded-full transition-all duration-150"
                                        style={{
                                            height: isPlaying
                                                ? `${Math.max(12, Math.sin(currentTime * 4 + i * 0.4) * 48 + 36)}px`
                                                : '16px',
                                        }}
                                    />
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Karaoke Subtitles Overlay Viewport */}
            <div className={`absolute inset-0 flex flex-col ${containerAlignmentClass} ${isOverVideo ? 'pointer-events-none' : 'pointer-events-auto'} z-10 transition-all`}>
                {timedLines.length === 0 ? (
                    !isOverVideo ? (
                        <div className="text-center p-6 bg-black/40 backdrop-blur-md rounded-2xl border border-white/10 max-w-md">
                            <p className="text-xs font-mono text-teal-300">★ WYSIWYG Lyric Canvas Preview ★</p>
                            <p className="text-xs text-slate-300 mt-1">
                                {activeSong?.lyrics
                                    ? 'Parsing synchronized lyric timestamps…'
                                    : 'No lyrics detected for this track. Add lyrics in Composer to enable karaoke videos.'}
                            </p>
                        </div>
                    ) : null
                ) : (
                    <div className="w-full max-w-4xl space-y-2">
                        {/* 1. Previous Line (Faded, clickable to rewind) */}
                        {prevLine && stylePreset === 'spotify' && (
                            <div
                                onClick={() => onSeek?.(prevLine.start)}
                                className="cursor-pointer transition-opacity duration-300 opacity-40 hover:opacity-85 text-sm md:text-base font-medium"
                                style={{
                                    fontFamily: fontFamily || preset.fontFamily,
                                    color: preset.unsungColor,
                                    textShadow: '0 2px 6px rgba(0, 0, 0, 0.9)',
                                }}
                            >
                                {prevLine.text}
                            </div>
                        )}

                        {/* 2. Active Karaoke Line with Progressive Highlight Wipe */}
                        {activeLine ? (
                            <div
                                onClick={() => onSeek?.(activeLine.start)}
                                className="cursor-pointer font-bold leading-tight select-none transition-all py-1"
                                style={{
                                    fontFamily: fontFamily || preset.fontFamily,
                                    fontSize: fontSizeOverride ? `${fontSizeOverride}px` : 'clamp(18px, 3.2vw, 32px)',
                                    textTransform: preset.textTransform || 'none',
                                    letterSpacing: preset.letterSpacing || 'normal',
                                }}
                            >
                                <div className={`flex flex-wrap items-center gap-x-2.5 gap-y-1.5 ${
                                    activeAlignment === 'left' ? 'justify-start text-left' : 'justify-center text-center'
                                }`}>
                                    {activeWords.map((wInfo, wIdx) => {
                                        // Calculate word-level sweep progress [0.0 - 1.0]
                                        const wDur = Math.max(0.05, wInfo.end - wInfo.start);
                                        const progress = Math.max(0, Math.min(1, (currentTime - wInfo.start) / wDur));
                                        const pct = (progress * 100).toFixed(1);
                                        const isSung = currentTime >= wInfo.end;
                                        const isSinging = currentTime >= wInfo.start && currentTime < wInfo.end;

                                        return (
                                            <span
                                                key={wIdx}
                                                className={`relative inline-block align-baseline transition-transform duration-75 ${
                                                    stylePreset === 'kinetic_pop' && isSinging ? 'scale-110' : 'scale-100'
                                                }`}
                                            >
                                                {/* Base Layer: Crisp Unsung Text with Dark Drop Shadow */}
                                                <span
                                                    className="block whitespace-nowrap select-none"
                                                    style={{
                                                        color: preset.unsungColor,
                                                        textShadow: '0 2px 8px rgba(0, 0, 0, 0.95), 0 1px 3px rgba(0, 0, 0, 0.9)',
                                                    }}
                                                >
                                                    {wInfo.word}
                                                </span>

                                                {/* Progressive Karaoke Sweep Layer (Clipped Horizontally from Right) */}
                                                <span
                                                    aria-hidden="true"
                                                    className="absolute inset-0 pointer-events-none select-none block overflow-hidden whitespace-nowrap"
                                                    style={{
                                                        color: preset.sungColor,
                                                        clipPath: `inset(0 ${Math.max(0, 100 - Number(pct)).toFixed(1)}% 0 0)`,
                                                        WebkitClipPath: `inset(0 ${Math.max(0, 100 - Number(pct)).toFixed(1)}% 0 0)`,
                                                        textShadow:
                                                            preset.glow !== 'none' && (isSung || isSinging)
                                                                ? `${preset.glow}, 0 2px 8px rgba(0, 0, 0, 0.95)`
                                                                : '0 2px 8px rgba(0, 0, 0, 0.95)',
                                                        filter:
                                                            isSinging && preset.glow !== 'none'
                                                                ? `drop-shadow(${preset.glow})`
                                                                : undefined,
                                                    }}
                                                >
                                                    {wInfo.word}
                                                </span>
                                            </span>
                                        );
                                    })}
                                </div>
                            </div>
                        ) : (
                            /* Interlude / Silence Indicator */
                            <div className="py-2 text-center opacity-50">
                                <span
                                    className="text-xs font-mono tracking-widest text-slate-300 uppercase"
                                    style={{ textShadow: '0 2px 6px rgba(0, 0, 0, 0.9)' }}
                                >
                                    ♪ ♪ ♪
                                </span>
                            </div>
                        )}

                        {/* 3. Next Line (Preview, clickable to skip ahead) */}
                        {nextLine && stylePreset === 'spotify' && (
                            <div
                                onClick={() => onSeek?.(nextLine.start)}
                                className="cursor-pointer transition-opacity duration-300 opacity-40 hover:opacity-85 text-sm md:text-base font-medium"
                                style={{
                                    fontFamily: fontFamily || preset.fontFamily,
                                    color: preset.unsungColor,
                                    textShadow: '0 2px 6px rgba(0, 0, 0, 0.9)',
                                }}
                            >
                                {nextLine.text}
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Bottom Status HUD Indicator (Preview Canvas mode only) */}
            {!isOverVideo && (
                <div className="absolute top-3 right-3 z-20 flex items-center gap-2 pointer-events-auto">
                    <div className="px-2.5 py-1 bg-black/60 backdrop-blur-md rounded-lg border border-white/10 text-[10px] font-mono text-teal-300 flex items-center gap-1.5 shadow-md">
                        <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse" />
                        <span className="capitalize">{stylePreset} Typography</span>
                        <span className="text-white/40">·</span>
                        <span className="text-white/80">{currentTime.toFixed(1)}s</span>
                    </div>
                </div>
            )}
        </div>
    );
};
