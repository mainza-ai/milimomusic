import React, { useState, useEffect, useRef } from 'react';
import { Download, Trash2, Volume2, Maximize2 } from 'lucide-react';
import { API_BASE_URL, workspaceApi } from '../../api';
import type { Job, NoteEvent } from '../../api';

interface PianoRollProps {
    job: Job;
    currentTime: number;
    duration: number;
    onSeek: (time: number) => void;
    isPlaying?: boolean;
}

// ---------------------------------------------------------------------------
// Dynamic pitch-range model (production-grade: never clamp, never drop notes).
// The visible range is computed from the ACTUAL transcribed notes (extended to
// the enclosing C octaves), so every real MIDI pitch — bass below C2, leads
// above C6 — has its own row. Nothing is squeezed onto an edge.
// ---------------------------------------------------------------------------
const PC_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const SHARP_PCS = new Set([1, 3, 6, 8, 10]);

function pitchName(midi: number): { name: string; isBlack: boolean; isC: boolean } {
    const pc = ((midi % 12) + 12) % 12;
    const oct = Math.floor(midi / 12) - 1;
    return {
        name: `${PC_NAMES[pc]}${oct}`,
        isBlack: SHARP_PCS.has(pc),
        isC: pc === 0,
    };
}

interface RollKey { num: number; name: string; isC: boolean; isBlack: boolean; }

function buildRange(minMidi: number, maxMidi: number): RollKey[] {
    const lo = Math.max(0, minMidi);
    const hi = Math.min(127, maxMidi);
    const keys: RollKey[] = [];
    for (let p = hi; p >= lo; p--) {
        const { name, isBlack, isC } = pitchName(p);
        keys.push({ num: p, name, isC, isBlack });
    }
    return keys;
}

// Default graceful range when a track has no notes yet (C2..C6).
const DEFAULT_LO = 36; // C2
const DEFAULT_HI = 84; // C6

function midiToFreq(midi: number): number {
    return 440 * Math.pow(2, (midi - 69) / 12);
}

export const PianoRoll: React.FC<PianoRollProps> = ({
    job,
    currentTime,
    duration,
    onSeek,
    isPlaying = false
}) => {
    const [selectedTrack, setSelectedTrack] = useState('all');
    const [isMidiSynthEnabled, setIsMidiSynthEnabled] = useState(true);
    const [activePitches, setActivePitches] = useState<Set<number>>(new Set());
    const playedNotesRef = useRef<Set<number>>(new Set());
    const audioCtxRef = useRef<AudioContext | null>(null);

    // Parse transcribed note events
    const initialNotes: NoteEvent[] = job.notes_json
        ? typeof job.notes_json === 'string'
            ? JSON.parse(job.notes_json)
            : job.notes_json
        : [];

    const [notesList, setNotesList] = useState<NoteEvent[]>(initialNotes);
    const [selectedNoteIndex, setSelectedNoteIndex] = useState<number | null>(null);

    useEffect(() => {
        if (job.notes_json) {
            const parsed = typeof job.notes_json === 'string' ? JSON.parse(job.notes_json) : job.notes_json;
            setNotesList(parsed);
        }
    }, [job.notes_json]);

    const filteredNotes = selectedTrack === 'all'
        ? notesList
        : notesList.filter(n => ((n.instrument || '')).toLowerCase().includes(selectedTrack.toLowerCase()));

    const totalDuration = duration || 30;

    // Use the transcription's real beat grid (BPM + beats per bar) so the measure
    // ruler aligns with where notes actually fall, instead of a hardcoded 120 BPM.
    const beatGrid = job.beat_grid_json
        ? typeof job.beat_grid_json === 'string'
            ? JSON.parse(job.beat_grid_json)
            : job.beat_grid_json
        : {};
    const bpm = Number(beatGrid.bpm) > 0 ? Number(beatGrid.bpm) : 120;
    const beatsPerMeasure = Number(beatGrid.beats_per_bar) > 0 ? Number(beatGrid.beats_per_bar) : 4;
    const measureDuration = (60 / bpm) * beatsPerMeasure; // seconds per bar

    // Dynamic pitch range covering ALL transcribed midi notes (no clamping/dropping).
    const [pitchRange, setPitchRange] = useState<RollKey[]>(() => buildRange(DEFAULT_LO, DEFAULT_HI));
    const [rangeHi, setRangeHi] = useState<number>(DEFAULT_HI);
    const [rangeLo, setRangeLo] = useState<number>(DEFAULT_LO);
    const fitToNotes = () => {
        if (notesList.length === 0) { setPitchRange(buildRange(DEFAULT_LO, DEFAULT_HI)); setRangeHi(DEFAULT_HI); setRangeLo(DEFAULT_LO); return; }
        let min = Infinity, max = -Infinity;
        for (const n of notesList) { if (n.pitch > max) max = n.pitch; if (n.pitch < min) min = n.pitch; }
        // Enclose in C octaves with breathing room (one octave each side).
        const lo = Math.max(0, Math.min(DEFAULT_LO, Math.floor(min / 12) * 12 - 12));
        const hi = Math.min(127, Math.max(DEFAULT_HI, Math.ceil(max / 12) * 12 + 12));
        setPitchRange(buildRange(lo, hi));
        setRangeHi(hi); setRangeLo(lo);
    };
    useEffect(() => { fitToNotes(); }, [notesList]); // eslint-disable-line react-hooks/exhaustive-deps

    // (Optional) widen range around the current anchor when a note is added out of view.
    const ensurePitchVisible = (pitch: number) => {
        if (pitch > rangeHi) { const hi = Math.min(127, Math.ceil(pitch / 12) * 12 + 12); setPitchRange(buildRange(rangeLo, hi)); setRangeHi(hi); }
        else if (pitch < rangeLo) { const lo = Math.max(0, Math.floor(pitch / 12) * 12 - 12); setPitchRange(buildRange(lo, rangeHi)); setRangeLo(lo); }
    };

    // Single shared horizontal time basis. It spans the FULL content extent — the
    // declared duration or the actual last note end, whichever is larger — so the
    // ruler covers every note, nothing is clamped/truncated, and ruler, note
    // placement and click-insert all use the same scale (they align by
    // construction). Production-grade: never clamp, never drop content.
    const contentEnd = notesList.length > 0
        ? Math.max(...notesList.map(n => (n.end_time ?? n.start_time ?? 0)))
        : 0;
    const timeScale = Math.max(totalDuration, contentEnd, 0.001);
    const progressPercent = Math.min(100, Math.max(0, (currentTime / timeScale) * 100));
    // Bars extend to cover the full time scale so the ruler’s last bar is never cut.
    const totalMeasures = Math.max(1, Math.ceil(timeScale / measureDuration));
    const measuresArray = Array.from({ length: totalMeasures }, (_, i) => i + 1);
    const ROW_H = 24; // px per pitch row (shared by keyboard + grid)
    const pitchIndexFor = (midi: number): number => {
        // Returns the grid row index (0 = top = highest shown pitch).
        const idx = pitchRange.findIndex(k => k.num === midi);
        return idx === -1 ? Math.max(0, Math.min(pitchRange.length - 1, pitchRange[0] ? pitchRange[0].num - midi : 0)) : idx;
    };

    // Get or initialize Web Audio context
    const getAudioContext = () => {
        if (!audioCtxRef.current) {
            const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
            audioCtxRef.current = new AudioCtx();
        }
        if (audioCtxRef.current.state === 'suspended') {
            audioCtxRef.current.resume();
        }
        return audioCtxRef.current;
    };

    // Rich Polyphonic Multi-Harmonic Synthesizer
    const playSynthesizerTone = (pitch: number, instrument: string = 'Piano') => {
        try {
            const ctx = getAudioContext();
            const now = ctx.currentTime;
            const freq = midiToFreq(pitch);
            const inst = instrument.toLowerCase();

            // Flash active piano key visually
            setActivePitches(prev => new Set(prev).add(pitch));
            setTimeout(() => {
                setActivePitches(prev => {
                    const next = new Set(prev);
                    next.delete(pitch);
                    return next;
                });
            }, 300);

            // Master Gain & Filter
            const masterGain = ctx.createGain();
            const filter = ctx.createBiquadFilter();
            filter.type = 'lowpass';

            if (inst.includes('bass')) {
                filter.frequency.setValueAtTime(450, now);
                filter.frequency.exponentialRampToValueAtTime(120, now + 0.5);

                const osc1 = ctx.createOscillator();
                osc1.type = 'sawtooth';
                osc1.frequency.setValueAtTime(freq, now);

                const osc2 = ctx.createOscillator();
                osc2.type = 'sine';
                osc2.frequency.setValueAtTime(freq * 0.5, now);

                const g1 = ctx.createGain();
                const g2 = ctx.createGain();
                g1.gain.setValueAtTime(0.25, now);
                g2.gain.setValueAtTime(0.35, now);

                osc1.connect(g1);
                osc2.connect(g2);
                g1.connect(filter);
                g2.connect(filter);

                masterGain.gain.setValueAtTime(0.001, now);
                masterGain.gain.exponentialRampToValueAtTime(0.4, now + 0.03);
                masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.7);

                filter.connect(masterGain);
                masterGain.connect(ctx.destination);

                osc1.start(now);
                osc2.start(now);
                osc1.stop(now + 0.75);
                osc2.stop(now + 0.75);

            } else if (inst.includes('drum')) {
                const osc = ctx.createOscillator();
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(150, now);
                osc.frequency.exponentialRampToValueAtTime(40, now + 0.12);

                masterGain.gain.setValueAtTime(0.5, now);
                masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.15);

                osc.connect(masterGain);
                masterGain.connect(ctx.destination);

                osc.start(now);
                osc.stop(now + 0.2);

            } else if (inst.includes('vocal')) {
                const osc = ctx.createOscillator();
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(freq, now);

                const lfo = ctx.createOscillator();
                const lfoGain = ctx.createGain();
                lfo.frequency.setValueAtTime(5.5, now);
                lfoGain.gain.setValueAtTime(4.0, now);
                lfo.connect(lfoGain);
                lfoGain.connect(osc.frequency);

                masterGain.gain.setValueAtTime(0.001, now);
                masterGain.gain.exponentialRampToValueAtTime(0.28, now + 0.08);
                masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.8);

                filter.frequency.setValueAtTime(2200, now);
                osc.connect(filter);
                filter.connect(masterGain);
                masterGain.connect(ctx.destination);

                lfo.start(now);
                osc.start(now);
                lfo.stop(now + 0.85);
                osc.stop(now + 0.85);

            } else {
                // Grand Piano with natural acoustic decay & sparkle
                filter.frequency.setValueAtTime(3200, now);
                filter.frequency.exponentialRampToValueAtTime(600, now + 0.9);

                const osc1 = ctx.createOscillator();
                osc1.type = 'triangle';
                osc1.frequency.setValueAtTime(freq, now);

                const osc2 = ctx.createOscillator();
                osc2.type = 'sine';
                osc2.frequency.setValueAtTime(freq * 2, now);

                const g1 = ctx.createGain();
                const g2 = ctx.createGain();
                g1.gain.setValueAtTime(0.3, now);
                g2.gain.setValueAtTime(0.12, now);

                osc1.connect(g1);
                osc2.connect(g2);
                g1.connect(filter);
                g2.connect(filter);

                masterGain.gain.setValueAtTime(0.001, now);
                masterGain.gain.exponentialRampToValueAtTime(0.38, now + 0.02);
                masterGain.gain.exponentialRampToValueAtTime(0.14, now + 0.2);
                masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.95);

                filter.connect(masterGain);
                masterGain.connect(ctx.destination);

                osc1.start(now);
                osc2.start(now);
                osc1.stop(now + 1.0);
                osc2.stop(now + 1.0);
            }
        } catch (e) {
            console.error('Synthesizer error', e);
        }
    };

    // Live MIDI playback synchronized with DAW playhead
    useEffect(() => {
        if (!isPlaying || !isMidiSynthEnabled) {
            playedNotesRef.current.clear();
            return;
        }

        filteredNotes.forEach((note, idx) => {
            const timeDiff = currentTime - note.start_time;
            if (timeDiff >= 0 && timeDiff <= 0.12 && !playedNotesRef.current.has(idx)) {
                playedNotesRef.current.add(idx);
                playSynthesizerTone(note.pitch, note.instrument);
            }
        });

        if (playedNotesRef.current.size > 300) {
            playedNotesRef.current.clear();
        }
    }, [currentTime, isPlaying, isMidiSynthEnabled, filteredNotes]);

    const handleGridClick = (e: React.MouseEvent<HTMLDivElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const newTime = (clickX / rect.width) * timeScale;
        onSeek(newTime);
    };

    // Persist note changes to the backend so edits survive a reload (not just local state).
    const persistNotes = (next: NoteEvent[]) => {
        setNotesList(next);
        if (job.id) {
            workspaceApi.saveNotes(job.id, next).catch(err => console.error('Failed to save notes', err));
        }
    };

    const handleAddNoteAt = (pitch: number, time: number) => {
        playSynthesizerTone(pitch, selectedTrack === 'all' ? 'Piano' : selectedTrack);
        ensurePitchVisible(pitch);
        const { name } = pitchName(pitch);
        const newNote: NoteEvent = {
            pitch,
            start_time: Math.round(time * 100) / 100,
            end_time: Math.round((time + 0.5) * 100) / 100,
            duration: 0.5,
            velocity: 85,
            instrument: selectedTrack === 'all' ? 'Piano' : selectedTrack,
            channel: 0,
            note_name: name
        };
        persistNotes([...notesList, newNote]);
    };

    const handleDeleteSelectedNote = () => {
        if (selectedNoteIndex !== null) {
            persistNotes(notesList.filter((_, idx) => idx !== selectedNoteIndex));
            setSelectedNoteIndex(null);
        }
    };

    // Color-Coding Design by Instrument Role (Clean Studio Aesthetics, No AI Purples)
    const getNoteStyle = (instrument: string) => {
        const inst = instrument.toLowerCase();
        if (inst.includes('bass')) {
            return {
                bg: 'bg-gradient-to-r from-amber-400 via-amber-500 to-orange-500',
                border: 'border-amber-300 dark:border-amber-400/80',
                text: 'text-slate-950 font-black',
                shadow: 'shadow-md shadow-amber-500/25',
                badge: 'bg-amber-950/20 text-slate-950',
                label: 'Bass'
            };
        }
        if (inst.includes('drum') || inst.includes('percussion')) {
            return {
                bg: 'bg-gradient-to-r from-rose-400 via-rose-500 to-red-500',
                border: 'border-rose-300 dark:border-rose-400/80',
                text: 'text-white font-black',
                shadow: 'shadow-md shadow-rose-500/25',
                badge: 'bg-black/30 text-white',
                label: 'Drums'
            };
        }
        if (inst.includes('vocal') || inst.includes('lead')) {
            return {
                bg: 'bg-gradient-to-r from-sky-400 via-cyan-400 to-teal-400',
                border: 'border-sky-300 dark:border-sky-400/80',
                text: 'text-slate-950 font-black',
                shadow: 'shadow-md shadow-sky-500/25',
                badge: 'bg-sky-950/20 text-slate-950',
                label: 'Vocal'
            };
        }
        if (inst.includes('guitar') || inst.includes('string')) {
            return {
                bg: 'bg-gradient-to-r from-yellow-400 via-amber-400 to-orange-400',
                border: 'border-yellow-300 dark:border-yellow-400/80',
                text: 'text-slate-950 font-black',
                shadow: 'shadow-md shadow-yellow-500/25',
                badge: 'bg-amber-950/20 text-slate-950',
                label: 'Strings'
            };
        }
        // Default Piano / Keys (Studio Teal & Mint)
        return {
            bg: 'bg-gradient-to-r from-teal-400 via-teal-500 to-emerald-400',
            border: 'border-teal-300 dark:border-teal-400/80',
            text: 'text-slate-950 font-black',
            shadow: 'shadow-md shadow-teal-500/25',
            badge: 'bg-teal-950/20 text-slate-950',
            label: 'Piano'
        };
    };

    return (
        <div className="flex flex-col h-full bg-[#f4f4f7] dark:bg-[#0b0d13] text-slate-900 dark:text-slate-200 select-none overflow-hidden transition-colors duration-200">
            {/* Top Toolbar Header */}
            <div className="flex flex-wrap items-center justify-between gap-3 px-6 py-3 border-b border-black/[0.06] dark:border-white/[0.08] bg-white/80 dark:bg-[#12141c]/90 backdrop-blur-2xl z-20 shadow-apple-sm">
                <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 text-teal-700 dark:text-teal-300 border border-teal-500/20 flex items-center justify-center font-bold text-xs">
                        🎹
                    </div>
                    <div>
                        <div className="flex items-center gap-2">
                            <h2 className="text-xs sm:text-sm font-extrabold tracking-tight text-slate-900 dark:text-white uppercase">
                                Grand Piano Roll & MIDI Score
                            </h2>
                            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 font-bold border border-teal-500/20">
                                {filteredNotes.length} Notes Transcribed
                            </span>
                        </div>
                        <p className="text-[10px] text-slate-500 dark:text-slate-400 font-mono">
                            Double-click anywhere on the grid to insert a note · Click to audition · Dynamic range (auto-fits transcribed notes)
                        </p>
                    </div>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                    {/* Live Synth Toggle */}
                    <button
                        onClick={() => setIsMidiSynthEnabled(!isMidiSynthEnabled)}
                        title="Toggle live Web Audio multi-harmonic synth playback"
                        aria-label="Toggle Live Synth"
                        className={`px-3 py-1.5 rounded-xl text-xs font-bold flex items-center space-x-1.5 transition-all border shadow-sm ${
                            isMidiSynthEnabled
                                ? 'bg-gradient-to-r from-teal-500 to-cyan-500 text-slate-950 border-teal-400'
                                : 'bg-black/[0.04] dark:bg-white/5 border-black/[0.06] dark:border-white/10 text-slate-600 dark:text-slate-400 hover:text-slate-900'
                        }`}
                    >
                        <Volume2 size={13} />
                        <span>{isMidiSynthEnabled ? '🎹 Live Synth ON' : '🎹 Live Synth OFF'}</span>
                    </button>

                    {/* Fit range to transcribed notes (dynamic, no clamping) */}
                    <button
                        onClick={fitToNotes}
                        title="Auto-fit the pitch range to all transcribed notes"
                        aria-label="Fit to Notes"
                        className="px-3 py-1.5 rounded-xl text-xs font-bold flex items-center space-x-1.5 transition-all border shadow-sm bg-black/[0.04] dark:bg-white/5 border-black/[0.06] dark:border-white/10 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
                    >
                        <Maximize2 size={13} />
                        <span>Fit</span>
                    </button>

                    {/* Track Filter / Solo Pills */}
                    <div className="flex items-center bg-black/[0.04] dark:bg-[#181a24] border border-black/[0.06] dark:border-white/10 rounded-xl p-1 space-x-1 text-xs">
                        {[
                            { id: 'all', label: 'All', color: 'bg-teal-500' },
                            { id: 'piano', label: 'Piano', color: 'bg-teal-500' },
                            { id: 'bass', label: 'Bass', color: 'bg-amber-500' },
                            { id: 'drums', label: 'Drums', color: 'bg-rose-500' },
                            { id: 'vocal', label: 'Vocal', color: 'bg-cyan-500' }
                        ].map(t => (
                            <button
                                key={t.id}
                                onClick={() => setSelectedTrack(t.id)}
                                title={`Filter piano roll notes to ${t.label}`}
                                aria-label={`Filter notes to ${t.label}`}
                                className={`px-2.5 py-1 rounded-lg font-bold capitalize transition-all ${
                                    selectedTrack === t.id
                                        ? `${t.color} text-slate-950 shadow-sm`
                                        : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
                                }`}
                            >
                                {t.label}
                            </button>
                        ))}
                    </div>

                    {selectedNoteIndex !== null && (
                        <button
                            onClick={handleDeleteSelectedNote}
                            title="Delete currently selected MIDI note"
                            aria-label="Delete selected note"
                            className="p-1.5 rounded-xl bg-rose-500/10 text-rose-600 dark:text-rose-400 text-xs font-bold flex items-center gap-1 hover:bg-rose-500/20 transition-colors"
                        >
                            <Trash2 size={13} />
                            <span>Delete Note</span>
                        </button>
                    )}

                    <button
                        onClick={() => window.open(`${API_BASE_URL}/transcribe/export/${job.id}/midi`, '_blank')}
                        title="Download Multi-Track .mid MIDI file"
                        aria-label="Download Multi-Track MIDI"
                        className="px-3.5 py-1.5 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 shadow-sm active:scale-95 transition-all"
                    >
                        <Download size={13} />
                        <span>Download MIDI</span>
                    </button>
                </div>
            </div>

            {/* Main Piano Roll Canvas: Synchronized Scroll Container */}
            <div className="flex-1 overflow-auto relative flex bg-[#ebebf0] dark:bg-[#0a0c12]">
                {/* 1. Left Piano Keys Keyboard (Sticky on Left X, Scrolling on Y) */}
                <div className="sticky left-0 z-20 flex-shrink-0 w-36 bg-[#1a1c24] border-r-4 border-r-teal-600 shadow-2xl flex flex-col pt-8 select-none">
                    {/* Top Fallboard Label */}
                    <div className="absolute top-0 left-0 right-0 h-8 bg-gradient-to-b from-[#252836] via-[#1c1e28] to-[#12131b] border-b border-black/60 flex items-center justify-between px-3 text-[10px] font-mono text-slate-400 font-bold shadow-sm">
                        <span className="tracking-wider">KEY · MIDI</span>
                        <span className="text-teal-400 font-serif italic text-[11px] font-bold">Studio Grand</span>
                    </div>

                    {pitchRange.map(p => {
                        const isBlack = p.isBlack;
                        const isC = p.isC;
                        const isMiddleC = p.num === 60;
                        const isKeyActive = activePitches.has(p.num);

                        return (
                            <button
                                key={p.num}
                                onClick={() => playSynthesizerTone(p.num, selectedTrack)}
                                title={`Click to play ${p.name} (MIDI Note ${p.num})`}
                                aria-label={`Piano Key ${p.name}`}
                                className={`h-6 text-[10px] font-mono font-bold flex items-center justify-between px-3 transition-all duration-75 relative group ${
                                    isBlack
                                        ? `bg-gradient-to-r from-[#121318] via-[#242633] to-[#15161f] text-slate-200 border-t border-b border-black/90 shadow-[inset_0_1px_1px_rgba(255,255,255,0.12),0_2px_4px_rgba(0,0,0,0.8)] pr-4 ${
                                              isKeyActive
                                                  ? 'brightness-150 translate-x-1 !bg-teal-500 !text-slate-950 font-black shadow-teal-500/50'
                                                  : 'hover:brightness-125'
                                          }`
                                        : `bg-gradient-to-r from-[#ffffff] via-[#f7f5f0] to-[#eae4d8] dark:from-[#2a2d3d] dark:via-[#222533] dark:to-[#1a1c28] text-slate-900 dark:text-slate-100 border-b border-[#cfc7b8] dark:border-[#12141c] shadow-[inset_0_1px_0_rgba(255,255,255,0.8)] ${
                                              isKeyActive
                                                  ? 'brightness-125 translate-x-1 !bg-teal-400 !text-slate-950 font-black shadow-teal-500/50'
                                                  : 'hover:brightness-105'
                                          }`
                                }`
                                }
                            >
                                {/* Key Name with Root C Highlighting */}
                                <div className="flex items-center gap-1.5 min-w-0">
                                    <span
                                        className={`tracking-tight ${
                                            isMiddleC
                                                ? 'text-teal-600 dark:text-teal-400 font-extrabold text-[11px] underline decoration-teal-500 decoration-2'
                                                : isC
                                                ? 'text-teal-600 dark:text-teal-400 font-bold text-[11px]'
                                                : isBlack
                                                ? 'text-slate-300'
                                                : 'text-slate-800 dark:text-slate-200'
                                        }`}
                                    >
                                        {p.name}
                                    </span>
                                    {isMiddleC ? (
                                        <span className="text-[8px] font-mono font-bold px-1 rounded bg-teal-500/20 text-teal-700 dark:text-teal-300 border border-teal-500/30">
                                            MID
                                        </span>
                                    ) : isC ? (
                                        <span className="w-1.5 h-1.5 rounded-full bg-teal-500 shadow-[0_0_6px_rgba(20,184,166,0.8)]" />
                                    ) : null}
                                </div>

                                <span className="text-[9px] font-mono tabular-nums opacity-45 group-hover:opacity-100 text-slate-600 dark:text-slate-400">
                                    {p.num}
                                </span>
                            </button>
                        );
                    })}
                </div>

                {/* 2. Right Note Grid Canvas */}
                <div
                    onClick={handleGridClick}
                    className="flex-1 min-w-[1200px] relative cursor-crosshair flex flex-col"
                >
                    {/* Top Measure Ruler Bar */}
                    <div className="sticky top-0 h-8 bg-white/95 dark:bg-[#141622]/95 border-b border-black/[0.08] dark:border-white/10 flex items-center z-10 backdrop-blur-md shadow-sm">
                        <div className="w-full relative h-full">
                            {measuresArray.map(bar => {
                                const barStart = (bar - 1) * measureDuration;
                                const leftPct = (barStart / timeScale) * 100;
                                const widthPct = Math.max(0.5, (measureDuration / timeScale) * 100);
                                return (
                                    <div
                                        key={bar}
                                        className="absolute top-0 bottom-0 border-l-2 border-slate-400 dark:border-slate-500 pl-1.5 flex items-center font-mono text-[10px] text-slate-500 dark:text-slate-400"
                                        style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                                    >
                                        <span className="font-extrabold text-teal-600 dark:text-teal-400">
                                            {bar}
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Pitch Rows Grid + Vertical Beat Overlay */}
                    <div className="relative">
                        {/* Background Vertical Bar & Beat Division Lines */}
                        <div className="absolute inset-0 pointer-events-none z-0">
                            {measuresArray.map(bar => {
                                const barStart = (bar - 1) * measureDuration;
                                const leftPct = (barStart / timeScale) * 100;
                                return (
                                    <div
                                        key={bar}
                                        className="absolute top-0 bottom-0 border-l border-slate-300/80 dark:border-white/15"
                                        style={{ left: `${leftPct}%` }}
                                    />
                                );
                            })}
                        </div>

                        {/* Horizontal Pitch Rows */}
                        {pitchRange.map(p => {
                            const isBlack = p.isBlack;
                            const isC = p.isC;
                            const isMiddleC = p.num === 60;

                            return (
                                <div
                                    key={p.num}
                                    onDoubleClick={(e) => {
                                        e.stopPropagation();
                                        const rect = e.currentTarget.getBoundingClientRect();
                                        const clickX = e.clientX - rect.left;
                                        const time = (clickX / rect.width) * timeScale;
                                        handleAddNoteAt(p.num, time);
                                    }}
                                    className={`h-6 border-b w-full transition-colors flex items-center relative ${
                                        isMiddleC
                                            ? 'bg-teal-500/[0.12] dark:bg-teal-500/[0.18] border-teal-500/40'
                                            : isC
                                            ? 'bg-teal-500/[0.06] dark:bg-teal-500/[0.10] border-teal-500/30'
                                            : isBlack
                                            ? 'bg-black/[0.04] dark:bg-black/35 border-black/[0.04] dark:border-white/[0.04]'
                                            : 'bg-white/50 dark:bg-[#12141e]/50 border-black/[0.03] dark:border-white/[0.02]'
                                    }`}
                                >
                                    {/* Octave Guide Label */}
                                    {isC && (
                                        <span className="ml-3 text-[9px] font-mono font-bold text-teal-600/70 dark:text-teal-400/70 pointer-events-none select-none">
                                            {p.name} {isMiddleC ? '· Middle C' : '· Root Octave'}
                                        </span>
                                    )}
                                </div>
                            );
                        })}

                        {/* Note Blocks (Color-Coded by Instrument with Apple Gloss Effect) */}
                        {filteredNotes.map((note, idx) => {
                            const pitchIndex = pitchIndexFor(note.pitch);
                            const top = pitchIndex * ROW_H;
                            const noteDur = note.duration || (note.end_time ? note.end_time - note.start_time : 0.5);
                            const noteEnd = note.end_time || (note.start_time + noteDur);
                            const leftPercent = (note.start_time / timeScale) * 100;
                            const widthPercent = Math.max(0.6, (noteDur / timeScale) * 100);
                            const isSelected = selectedNoteIndex === idx;
                            const noteStyle = getNoteStyle(note.instrument);
                            const isDrum = note.instrument?.toLowerCase().includes('drum') || note.instrument?.toLowerCase().includes('percussion');
                            const displayPitch = note.note_name || pitchName(note.pitch).name;

                            // Render Drum triggers as compact punch dots to eliminate horizontal pileups
                            if (isDrum) {
                                return (
                                    <div
                                        key={idx}
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            playSynthesizerTone(note.pitch, note.instrument);
                                            setSelectedNoteIndex(isSelected ? null : idx);
                                        }}
                                        className={`absolute h-[20px] min-w-[20px] max-w-[28px] rounded-full border shadow-sm flex items-center justify-center cursor-pointer transition-all z-10 ${
                                            noteStyle.bg
                                        } ${noteStyle.border} ${noteStyle.shadow} ${
                                            isSelected
                                                ? 'ring-2 ring-white scale-125 z-30 shadow-lg'
                                                : 'hover:scale-110 hover:brightness-110'
                                        }`}
                                        style={{
                                            top: `${top + 2}px`,
                                            left: `${leftPercent}%`,
                                        }}
                                        title={`Drum Hit [${note.start_time.toFixed(2)}s] · Velocity: ${note.velocity || 85}`}
                                    >
                                        <span className="text-[9px] text-white select-none">🥁</span>
                                    </div>
                                );
                            }

                            return (
                                <div
                                    key={idx}
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        playSynthesizerTone(note.pitch, note.instrument);
                                        setSelectedNoteIndex(isSelected ? null : idx);
                                    }}
                                    className={`absolute h-[22px] rounded-lg border shadow-sm flex items-center justify-between px-2 text-[9px] font-mono font-bold cursor-pointer transition-all z-10 overflow-hidden ${
                                        noteStyle.bg
                                    } ${noteStyle.border} ${noteStyle.text} ${noteStyle.shadow} ${
                                        isSelected
                                            ? 'ring-2 ring-white scale-[1.02] z-30 shadow-lg'
                                            : 'hover:scale-[1.01] hover:brightness-110'
                                    }`}
                                    style={{
                                        top: `${top + 1}px`,
                                        left: `${leftPercent}%`,
                                        width: `${widthPercent}%`,
                                        minWidth: '24px'
                                    }}
                                    title={`${displayPitch} (${note.instrument}) [${note.start_time.toFixed(2)}s - ${noteEnd.toFixed(2)}s] · Velocity: ${note.velocity || 85}`}
                                >
                                    {/* Left Note Name Badge */}
                                    <span className="truncate">{displayPitch}</span>

                                    {/* Right Instrument Tag Pill (Visible on wider note events) */}
                                    {widthPercent > 2.5 && (
                                        <span
                                            className={`hidden sm:inline-block text-[8px] font-mono px-1 py-0.2 rounded-md ${noteStyle.badge}`}
                                        >
                                            {noteStyle.label}
                                        </span>
                                    )}
                                </div>
                            );
                        })}

                        {/* Interactive Playhead Line */}
                        <div
                            className="absolute top-0 bottom-0 w-0.5 bg-rose-500 z-30 pointer-events-none transition-all duration-75 shadow-lg shadow-rose-500/50"
                            style={{ left: `${progressPercent}%` }}
                        >
                            <div className="w-3.5 h-3.5 bg-rose-500 -ml-1.5 -top-1 absolute rounded-full shadow-md border-2 border-white" />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
