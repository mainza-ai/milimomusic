import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
    Download, Trash2, Volume2, Maximize2, Check, RefreshCw, Music, Drum,
    Undo2, Redo2, Magnet, Crosshair, ZoomIn, ZoomOut, Save, AlertTriangle
} from 'lucide-react';
import { API_BASE_URL, trackApi } from '../../api';
import type { Job, NoteEvent } from '../../api';
import { pushHotkeyScope, isTextEntryTarget } from '../../utils/hotkeyScope';
import { safeJsonParse } from '../../utils/safeJsonParse';

interface PianoRollProps {
    job: Job;
    currentTime: number;
    duration: number;
    onSeek: (time: number) => void;
    isPlaying?: boolean;
    /** Sample-accurate transport position from the session's audio clock. */
    getPosition: () => number;
    /**
     * The SESSION's shared AudioContext (created/resumed by the workspace).
     * One context per workspace: no duplicate clocks, no extra hardware
     * voices, and synth timing is trivially coherent with the transport.
     */
    getAudioContext: () => AudioContext | null;
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

const clampPitch = (p: number): number => Math.max(0, Math.min(127, Math.round(p)));
const round2 = (v: number): number => Math.round(v * 100) / 100;

type SaveState = 'idle' | 'dirty' | 'saving' | 'saved' | 'error';

interface DragItem { n: NoteEvent; start: number; end: number; pitch: number; }
interface PreviewPos { start: number; end: number; pitch: number; }

export const PianoRoll: React.FC<PianoRollProps> = ({
    job,
    currentTime,
    duration,
    onSeek,
    isPlaying = false,
    getPosition,
    getAudioContext
}) => {
    const [selectedTrack, setSelectedTrack] = useState('all');
    const [isMidiSynthEnabled, setIsMidiSynthEnabled] = useState(true);
    const [activePitches, setActivePitches] = useState<Set<number>>(new Set());
    // Notes already handed to the audio clock this pass (object identity).
    const scheduledRef = useRef<Set<NoteEvent>>(new Set());
    const lastSchedulePosRef = useRef(0);

    // Parse transcribed note events
    const initialNotes: NoteEvent[] = safeJsonParse<NoteEvent[]>(job.notes_json, [], 'notes_json');

    const [notesList, setNotesList] = useState<NoteEvent[]>(initialNotes);

    // ── Editor state ────────────────────────────────────────────────────────
    const [selectedNotes, setSelectedNotes] = useState<Set<NoteEvent>>(new Set());
    const [dragPreview, setDragPreview] = useState<Map<NoteEvent, PreviewPos> | null>(null);
    const [marquee, setMarquee] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
    const [snapDiv, setSnapDiv] = useState<'off' | '1/16' | '1/8' | '1/4'>('1/16');
    const [zoomX, setZoomX] = useState(1);
    const [rowH, setRowH] = useState(24);
    const [followPlayhead, setFollowPlayhead] = useState(true);
    const [saveState, setSaveState] = useState<SaveState>('idle');
    const [histFlags, setHistFlags] = useState({ u: false, r: false });

    const undoRef = useRef<NoteEvent[][]>([]);
    const redoStackRef = useRef<NoteEvent[][]>([]);

    const previewMapRef = useRef<Map<NoteEvent, PreviewPos> | null>(null);
    const marqueeRef = useRef<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
    const saveTimerRef = useRef<number | undefined>(undefined);
    const savedTimerRef = useRef<number | undefined>(undefined);
    const scrollRef = useRef<HTMLDivElement>(null);
    const contentRef = useRef<HTMLDivElement>(null);
    const gridFocusRef = useRef<HTMLDivElement>(null);
    const lastActionRef = useRef<string>('');

    // Prop → state sync: re-parse when the parent swaps the active track.
    // Resetting selection + history here is intentional — a new document
    // invalidates prior undo state. (Pre-existing v1 pattern.)
    useEffect(() => {
        if (job.notes_json) {
            // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional prop→state sync
            setNotesList(safeJsonParse<NoteEvent[]>(job.notes_json, [], 'notes_json'));
            setSelectedNotes(new Set());
            undoRef.current = [];
            redoStackRef.current = [];
            setHistFlags({ u: false, r: false });
        }
    }, [job.notes_json]);

    const filteredNotes = selectedTrack === 'all'
        ? notesList
        : notesList.filter(n => ((n.instrument || '')).toLowerCase().includes(selectedTrack.toLowerCase()));
    const filteredRef = useRef(filteredNotes);
    useEffect(() => { filteredRef.current = filteredNotes; });

    const totalDuration = duration || 30;

    // Real beat grid (BPM + beats per bar) drives the measure ruler AND the
    // musical snap grid.
    const beatGrid = safeJsonParse<Record<string, number>>(job.beat_grid_json, {}, 'beat_grid_json');
    const bpm = Number(beatGrid.bpm) > 0 ? Number(beatGrid.bpm) : 120;
    const beatsPerMeasure = Number(beatGrid.beats_per_bar) > 0 ? Number(beatGrid.beats_per_bar) : 4;
    const measureDuration = (60 / bpm) * beatsPerMeasure; // seconds per bar
    const secPerBeat = 60 / bpm;
    const snapFrac = snapDiv === 'off' ? 0 : snapDiv === '1/16' ? 0.25 : snapDiv === '1/8' ? 0.5 : 1;
    const snapSec = snapFrac * secPerBeat;
    const snapTime = (t: number): number =>
        snapSec > 0 ? Math.max(0, Math.round(t / snapSec) * snapSec) : t;

    // Dynamic pitch range covering ALL transcribed midi notes (no clamping/dropping).
    const [pitchRange, setPitchRange] = useState<RollKey[]>(() => buildRange(DEFAULT_LO, DEFAULT_HI));
    const [rangeHi, setRangeHi] = useState<number>(DEFAULT_HI);
    const [rangeLo, setRangeLo] = useState<number>(DEFAULT_LO);
    const fitToNotes = () => {
        if (notesList.length === 0) { setPitchRange(buildRange(DEFAULT_LO, DEFAULT_HI)); setRangeHi(DEFAULT_HI); setRangeLo(DEFAULT_LO); return; }
        let min = Infinity, max = -Infinity;
        for (const n of notesList) { if (n.pitch > max) max = n.pitch; if (n.pitch < min) min = n.pitch; }
        const lo = Math.max(0, Math.min(DEFAULT_LO, Math.floor(min / 12) * 12 - 12));
        const hi = Math.min(127, Math.max(DEFAULT_HI, Math.ceil(max / 12) * 12 + 12));
        setPitchRange(buildRange(lo, hi));
        setRangeHi(hi); setRangeLo(lo);
    };
    // Derived-range adjust: the visible pitch range follows the note list.
    // (Pre-existing v1 pattern — intentional derived-state sync.)
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional derived sync
    useEffect(() => { fitToNotes(); }, [notesList]); // eslint-disable-line react-hooks/exhaustive-deps

    const ensurePitchVisible = (pitch: number) => {
        if (pitch > rangeHi) { const hi = Math.min(127, Math.ceil(pitch / 12) * 12 + 12); setPitchRange(buildRange(rangeLo, hi)); setRangeHi(hi); }
        else if (pitch < rangeLo) { const lo = Math.max(0, Math.floor(pitch / 12) * 12 - 12); setPitchRange(buildRange(lo, rangeHi)); setRangeLo(lo); }
    };

    // Single shared horizontal time basis spanning the FULL content extent.
    const contentEnd = notesList.length > 0
        ? notesList.reduce((m, n) => Math.max(m, noteEndOf(n)), 0)
        : 0;
    const timeScale = Math.max(totalDuration, contentEnd, 0.001);
    const progressPercent = Math.min(100, Math.max(0, (currentTime / timeScale) * 100));
    const totalMeasures = Math.max(1, Math.ceil(timeScale / measureDuration));
    const measuresArray = Array.from({ length: totalMeasures }, (_, i) => i + 1);
    const pitchIndexFor = (midi: number): number => {
        const idx = pitchRange.findIndex(k => k.num === midi);
        return idx === -1 ? Math.max(0, Math.min(pitchRange.length - 1, pitchRange[0] ? pitchRange[0].num - midi : 0)) : idx;
    };

    function noteEndOf(n: NoteEvent): number {
        return n.end_time ?? (n.start_time + (n.duration || 0.5));
    }

    // ── Persistence: batched, honest about its state ────────────────────────
    const flushPersist = async (notes: NoteEvent[]) => {
        if (!job.id) return;
        window.clearTimeout(saveTimerRef.current);
        setSaveState('saving');
        try {
            await trackApi.updateMidiNotes(job.id, notes);
            setSaveState('saved');
            window.clearTimeout(savedTimerRef.current);
            savedTimerRef.current = window.setTimeout(() => setSaveState(s => (s === 'saved' ? 'idle' : s)), 2500);
        } catch (err) {
            console.error('Failed to sync MIDI/MusicXML notes with backend:', err);
            setSaveState('error'); // surfaced in the toolbar with a Retry action
        }
    };

    const schedulePersist = (next: NoteEvent[]) => {
        setSaveState('dirty');
        window.clearTimeout(saveTimerRef.current);
        saveTimerRef.current = window.setTimeout(() => void flushPersist(next), 700);
    };

    // ── History (undo/redo) ─────────────────────────────────────────────────
    const markHist = () => setHistFlags({ u: undoRef.current.length > 0, r: redoStackRef.current.length > 0 });

    const commit = (next: NoteEvent[], action?: string) => {
        // Coalesce history ONLY for continuous gestures (velocity slider
        // dragging) — every discrete edit must be individually undoable.
        const shouldCoalesce = action === 'velocity' && lastActionRef.current === 'velocity';
        if (!shouldCoalesce) {
            undoRef.current.push(notesList);
            if (undoRef.current.length > 100) undoRef.current.shift();
            redoStackRef.current = [];
        }
        lastActionRef.current = action || '';
        setNotesList(next);
        schedulePersist(next);
        markHist();
    };

    const doUndo = () => {
        const prev = undoRef.current.pop();
        if (!prev) return;
        redoStackRef.current.push(notesList);
        setNotesList(prev);
        setSelectedNotes(new Set());
        schedulePersist(prev);
        lastActionRef.current = '';
        markHist();
    };

    const doRedo = () => {
        const next = redoStackRef.current.pop();
        if (!next) return;
        undoRef.current.push(notesList);
        setNotesList(next);
        setSelectedNotes(new Set());
        schedulePersist(next);
        lastActionRef.current = '';
        markHist();
    };

    // ── Editing commands ────────────────────────────────────────────────────
    const replaceSelected = (mutate: (n: NoteEvent) => NoteEvent, action: string) => {
        if (selectedNotes.size === 0) return;
        const replacements = new Map<NoteEvent, NoteEvent>();
        const next = notesList.map(n => {
            if (!selectedNotes.has(n)) return n;
            const nn = mutate(n);
            replacements.set(n, nn);
            return nn;
        });
        commit(next, action);
        setSelectedNotes(new Set(replacements.values()));
        let lo = Infinity, hi = -Infinity;
        replacements.forEach(nn => { lo = Math.min(lo, nn.pitch); hi = Math.max(hi, nn.pitch); });
        if (lo !== Infinity) { ensurePitchVisible(lo); ensurePitchVisible(hi); }
    };

    const deleteSelection = () => {
        if (selectedNotes.size === 0) return;
        commit(notesList.filter(n => !selectedNotes.has(n)));
        setSelectedNotes(new Set());
    };

    const transposeSelection = (delta: number) => replaceSelected(n => {
        const pitch = clampPitch(n.pitch + delta);
        return { ...n, pitch, note_name: pitchName(pitch).name };
    }, `transpose-${delta}`);

    const duplicateSelection = () => {
        if (selectedNotes.size === 0) return;
        const copies: NoteEvent[] = Array.from(selectedNotes).map(n => {
            const dur = noteEndOf(n) - n.start_time;
            return { ...n, start_time: round2(n.start_time + dur), end_time: round2(noteEndOf(n) + dur) };
        });
        commit([...notesList, ...copies], 'duplicate');
        setSelectedNotes(new Set(copies));
    };

    const quantizeSelection = () => {
        if (snapSec <= 0) return;
        const targets = selectedNotes.size > 0 ? selectedNotes : new Set(filteredNotes);
        const replacements = new Map<NoteEvent, NoteEvent>();
        const next = notesList.map(n => {
            if (!targets.has(n)) return n;
            const nn = { ...n, start_time: round2(snapTime(n.start_time)) };
            replacements.set(n, nn);
            return nn;
        });
        commit(next, 'quantize');
        setSelectedNotes(new Set(replacements.values()));
    };

    const applyVelocity = (vel: number) => {
        replaceSelected(n => ({ ...n, velocity: vel }), 'velocity');
    };

    // ── Pointer interactions: drag-move / resize / marquee ──────────────────
    const beginDrag = (e: React.MouseEvent, note: NoteEvent, mode: 'move' | 'resize') => {
        if (e.button !== 0) return;
        e.stopPropagation();
        e.preventDefault();
        gridFocusRef.current?.focus();

        let sel = new Set(selectedNotes);
        if (!sel.has(note)) {
            sel = e.shiftKey ? new Set([...sel, note]) : new Set([note]);
            setSelectedNotes(sel);
        }
        const items: DragItem[] = Array.from(sel).map(n => ({
            n, start: n.start_time, end: noteEndOf(n), pitch: n.pitch
        }));
        const startX = e.clientX, startY = e.clientY;
        const rect = contentRef.current?.getBoundingClientRect();
        if (!rect) return;
        const secPerPx = timeScale / rect.width;
        let moved = false;

        const onMove = (ev: MouseEvent) => {
            const dx = ev.clientX - startX;
            const dy = ev.clientY - startY;
            if (!moved && Math.abs(dx) < 4 && Math.abs(dy) < 4) return;
            moved = true;
            const rawDt = dx * secPerPx;
            const dt = snapSec > 0 ? Math.round(rawDt / snapSec) * snapSec : rawDt;
            const dp = mode === 'move' ? Math.round(-dy / rowH) : 0;
            const next = new Map<NoteEvent, PreviewPos>();
            for (const it of items) {
                if (mode === 'move') {
                    const s = Math.max(0, it.start + dt);
                    next.set(it.n, { start: s, end: s + (it.end - it.start), pitch: clampPitch(it.pitch + dp) });
                } else {
                    const end = Math.max(it.start + 0.05, it.end + dt);
                    next.set(it.n, { start: it.start, end, pitch: it.pitch });
                }
            }
            previewMapRef.current = next;
            setDragPreview(next);
        };

        const onUp = () => {
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
            if (!moved) {
                // Plain click (no drag): audition the note, matching the
                // piano-key behavior. Selection was already applied at mousedown.
                playSynthesizerTone(note.pitch, note.instrument);
                return;
            }
            const pm = previewMapRef.current;
            previewMapRef.current = null;
            setDragPreview(null);
            if (pm && pm.size > 0) {
                const replacements = new Map<NoteEvent, NoteEvent>();
                const nextList = notesList.map(n => {
                    const p = pm.get(n);
                    if (!p) return n;
                    const nn: NoteEvent = {
                        ...n,
                        start_time: round2(p.start),
                        end_time: round2(p.end),
                        duration: round2(p.end - p.start),
                        pitch: p.pitch,
                        note_name: pitchName(p.pitch).name
                    };
                    replacements.set(n, nn);
                    return nn;
                });
                commit(nextList, mode === 'move' ? 'drag-move' : 'drag-resize');
                setSelectedNotes(new Set(replacements.values()));
            }
        };

        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
    };

    const beginMarqueeOrClick = (e: React.MouseEvent) => {
        if (e.button !== 0) return;
        const rect = contentRef.current?.getBoundingClientRect();
        if (!rect) return;
        gridFocusRef.current?.focus();
        const x0 = e.clientX - rect.left, y0 = e.clientY - rect.top;
        let moved = false;

        const onMove = (ev: MouseEvent) => {
            const x1 = ev.clientX - rect.left, y1 = ev.clientY - rect.top;
            if (!moved && Math.abs(x1 - x0) < 4 && Math.abs(y1 - y0) < 4) return;
            moved = true;
            const mq = { x0, y0, x1, y1 };
            marqueeRef.current = mq;
            setMarquee(mq);
        };

        const onUp = (ev: MouseEvent) => {
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
            const mq = marqueeRef.current;
            marqueeRef.current = null;
            setMarquee(null);
            if (moved && mq) {
                const minX = Math.min(mq.x0, mq.x1), maxX = Math.max(mq.x0, mq.x1);
                const minY = Math.min(mq.y0, mq.y1), maxY = Math.max(mq.y0, mq.y1);
                const t0 = (minX / rect.width) * timeScale;
                const t1 = (maxX / rect.width) * timeScale;
                const idxLo = Math.max(0, Math.floor(maxY / rowH));
                const idxHi = Math.min(pitchRange.length - 1, Math.floor(minY / rowH));
                const pHi = pitchRange[idxLo]?.num ?? 127;
                const pLo = pitchRange[idxHi]?.num ?? 0;
                const hits = filteredRef.current.filter(n =>
                    n.start_time < t1 && noteEndOf(n) > t0 && n.pitch >= pLo && n.pitch <= pHi
                );
                setSelectedNotes(new Set(hits));
            } else {
                // Plain background click: clear selection + seek (transport UX).
                setSelectedNotes(new Set());
                const t = ((ev.clientX - rect.left) / rect.width) * timeScale;
                onSeek(Math.max(0, Math.min(timeScale, t)));
            }
        };

        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
    };

    // Insert on double-click anywhere in the grid (snapped).
    const handleInsertAt = (e: React.MouseEvent) => {
        const rect = contentRef.current?.getBoundingClientRect();
        if (!rect) return;
        const t = ((e.clientX - rect.left) / rect.width) * timeScale;
        const rowIdx = Math.floor((e.clientY - rect.top) / rowH);
        const key = pitchRange[Math.max(0, Math.min(pitchRange.length - 1, rowIdx))];
        if (!key) return;
        const snappedStart = snapTime(t);
        const dur = snapSec > 0 ? snapSec : 0.5;
        playSynthesizerTone(key.num, selectedTrack === 'all' ? 'Piano' : selectedTrack);
        ensurePitchVisible(key.num);
        const newNote: NoteEvent = {
            pitch: key.num,
            start_time: round2(snappedStart),
            end_time: round2(snappedStart + dur),
            duration: round2(dur),
            velocity: 85,
            instrument: selectedTrack === 'all' ? 'Piano' : selectedTrack,
            channel: 0,
            note_name: key.name
        };
        commit([...notesList, newNote], 'insert');
        setSelectedNotes(new Set([newNote]));
    };

    // ── Editor hotkey scope ─────────────────────────────────────────────────
    const editorApiRef = useRef({
        undo: doUndo, redo: doRedo, del: deleteSelection,
        transpose: transposeSelection, dup: duplicateSelection, selSize: 0
    });
    useEffect(() => {
        editorApiRef.current = {
            undo: doUndo, redo: doRedo, del: deleteSelection,
            transpose: transposeSelection, dup: duplicateSelection, selSize: selectedNotes.size
        };
    });

    useEffect(() => {
        return pushHotkeyScope((e) => {
            if (isTextEntryTarget(e.target)) return false;
            const t = e.target as HTMLElement | null;
            const inGrid = !!t?.closest?.('[data-hotkey-local]');
            const api = editorApiRef.current;
            const mod = e.metaKey || e.ctrlKey;
            if (mod && e.code === 'KeyZ') { if (!inGrid) return false; e.preventDefault(); if (e.shiftKey) { api.redo(); } else { api.undo(); } return true; }
            if (mod && e.code === 'KeyY') { if (!inGrid) return false; e.preventDefault(); api.redo(); return true; }
            if (mod && e.code === 'KeyD') { if (!inGrid) return false; e.preventDefault(); api.dup(); return true; }
            if (!inGrid) return false;
            switch (e.code) {
                case 'Delete':
                case 'Backspace':
                    if (api.selSize) { e.preventDefault(); api.del(); return true; }
                    return false;
                case 'ArrowUp':
                case 'ArrowDown':
                    if (api.selSize) {
                        e.preventDefault();
                        api.transpose(e.code === 'ArrowUp' ? (e.shiftKey ? 12 : 1) : (e.shiftKey ? -12 : -1));
                        return true;
                    }
                    return false;
                case 'Escape':
                    if (api.selSize) { setSelectedNotes(new Set()); return true; }
                    return false;
                default:
                    return false;
            }
        });
    }, []);

    // Flush pending saves when unmounting mid-edit.
    useEffect(() => () => {
        window.clearTimeout(saveTimerRef.current);
        window.clearTimeout(savedTimerRef.current);
    }, []);

    // ── Follow playhead ─────────────────────────────────────────────────────
    useEffect(() => {
        if (!isPlaying || !followPlayhead || !scrollRef.current || !contentRef.current) return;
        const el = scrollRef.current;
        const content = contentRef.current;
        const gridW = content.getBoundingClientRect().width;
        const px = content.offsetLeft + (progressPercent / 100) * gridW;
        if (px < el.scrollLeft + 48 || px > el.scrollLeft + el.clientWidth - 96) {
            el.scrollLeft = Math.max(0, px - el.clientWidth * 0.35);
        }
    }, [currentTime, isPlaying, followPlayhead, progressPercent]);

    // Resume helper for the SHARED session context (auditioning paths).
    const getSharedContext = (): AudioContext | null => {
        const ctx = getAudioContext();
        if (ctx && ctx.state === 'suspended') {
            void ctx.resume().catch(() => {});
        }
        return ctx;
    };

    // Rich Polyphonic Multi-Harmonic Synthesizer.
    // `when` schedules the note at an exact AudioContext timestamp (sample-
    // accurate playback sync); omit it to sound immediately (auditioning).
    // `durSec` gates the envelope to the note's WRITTEN length — without it a
    // 16th note rang for a full second and the part smeared into mush.
    const playSynthesizerTone = (pitch: number, instrument: string = 'Piano', when?: number, durSec?: number) => {
        try {
            const ctx = getSharedContext();
            if (!ctx) return;
            const now = when ?? ctx.currentTime;
            const freq = midiToFreq(pitch);
            const inst = instrument.toLowerCase();
            const REL = 0.08; // release tail after the gated length
            const dur = durSec && durSec > 0 ? Math.min(durSec, 8) : null;
            const end = now + (dur ?? 0);

            setActivePitches(prev => new Set(prev).add(pitch));
            setTimeout(() => {
                setActivePitches(prev => {
                    const next = new Set(prev);
                    next.delete(pitch);
                    return next;
                });
            }, 300);

            const masterGain = ctx.createGain();
            const filter = ctx.createBiquadFilter();
            filter.type = 'lowpass';

            // ── Node lifecycle ──────────────────────────────────────────────
            // Every tone used to LEAK its gain/filter chain into the destination
            // graph forever. Thousands of dead nodes later, the audio thread's
            // per-quantum work explodes and playback lags. Now each tone's
            // sources are tracked and the whole subgraph is severed on end.
            const sources: AudioScheduledSourceNode[] = [];
            const teardown = () => {
                for (const s of sources) { try { s.disconnect(); } catch { /* noop */ } }
                try { masterGain.disconnect(); } catch { /* noop */ }
                try { filter.disconnect(); } catch { /* noop */ }
            };

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
                let stopAt: number;
                if (dur) {
                    masterGain.gain.exponentialRampToValueAtTime(0.28, now + Math.max(0.12, dur * 0.5));
                    masterGain.gain.exponentialRampToValueAtTime(0.0001, end);
                    stopAt = end + REL;
                } else {
                    masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.7);
                    stopAt = now + 0.75;
                }

                filter.connect(masterGain);
                masterGain.connect(ctx.destination);

                osc1.start(now);
                osc2.start(now);
                osc1.stop(stopAt);
                osc2.stop(stopAt);
                sources.push(osc1, osc2);
                osc2.onended = teardown;

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
                sources.push(osc);
                osc.onended = teardown;

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
                let stopAt: number;
                if (dur) {
                    masterGain.gain.exponentialRampToValueAtTime(0.22, now + Math.max(0.16, dur * 0.6));
                    masterGain.gain.exponentialRampToValueAtTime(0.0001, end);
                    stopAt = end + REL;
                } else {
                    masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.8);
                    stopAt = now + 0.85;
                }

                filter.frequency.setValueAtTime(2200, now);
                osc.connect(filter);
                filter.connect(masterGain);
                masterGain.connect(ctx.destination);

                lfo.start(now);
                osc.start(now);
                lfo.stop(stopAt);
                osc.stop(stopAt);
                sources.push(osc, lfo);
                osc.onended = teardown;

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
                let stopAt: number;
                if (dur) {
                    masterGain.gain.exponentialRampToValueAtTime(0.09, now + Math.max(0.3, dur * 0.7));
                    masterGain.gain.exponentialRampToValueAtTime(0.0001, end);
                    stopAt = end + REL;
                } else {
                    masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.95);
                    stopAt = now + 1.0;
                }

                filter.connect(masterGain);
                masterGain.connect(ctx.destination);

                osc1.start(now);
                osc2.start(now);
                osc1.stop(stopAt);
                osc2.stop(stopAt);
                sources.push(osc1, osc2);
                osc2.onended = teardown;
            }
        } catch (e) {
            console.error('Synthesizer error', e);
        }
    };

    // ── Live MIDI playback, scheduled ON the audio clock ────────────────────
    // The old implementation chased throttled React state (~12Hz) with a
    // 120ms catch-up window: notes fired up to ~120ms late, were tracked by
    // ARRAY INDEX (breaking when filters/edits reordered notes), never reset
    // on seek (backward seeks skipped notes) and never reset on loop wrap
    // (second pass was silent).
    //
    // Now: a rAF scheduler polls the transport's SAMPLE-ACCURATE position and
    // hands each note to the AudioContext with an exact start timestamp —
    // the same mechanism the stem transport itself uses. Lookahead of 150ms
    // absorbs any jitter; seeks/loop-wraps are detected as backward jumps and
    // resynchronize cleanly.
    // Sorted once per note-set change so the scheduler can binary-search its
    // trigger window instead of scanning every note at 60fps.
    const sortedNotes = useMemo(
        () => [...filteredNotes].sort((a, b) => a.start_time - b.start_time),
        [filteredNotes]
    );
    const sortedSchedRef = useRef(sortedNotes);
    useEffect(() => { sortedSchedRef.current = sortedNotes; });

    useEffect(() => {
        if (!isPlaying || !isMidiSynthEnabled) {
            scheduledRef.current.clear();
            lastSchedulePosRef.current = 0;
            return;
        }

        const ctx = getSharedContext();
        if (!ctx) return;
        const LOOKAHEAD = 0.15;   // seconds scheduled ahead of the playhead
        const MISSED_TOLERANCE = 0.03; // notes older than this are skipped, not machine-gunned
        let raf = 0;

        const tick = () => {
            const pos = getPosition();
            const notes = sortedSchedRef.current;

            // Backward jump = seek or loop wrap → re-arm the whole pass.
            if (pos < lastSchedulePosRef.current - 0.08) {
                scheduledRef.current.clear();
            }

            // Only visit notes inside [pos - tolerance, pos + lookahead]:
            // binary-search the window start, break at the horizon.
            const windowStart = pos - MISSED_TOLERANCE;
            let lo = 0, hi = notes.length;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (notes[mid].start_time < windowStart) lo = mid + 1; else hi = mid;
            }
            const horizon = pos + LOOKAHEAD;
            for (let i = lo; i < notes.length; i++) {
                const note = notes[i];
                const st = note.start_time;
                if (st > horizon) break;
                if (scheduledRef.current.has(note)) continue;
                scheduledRef.current.add(note);
                const noteDur = note.duration
                    || (note.end_time ? note.end_time - note.start_time : 0.5);
                playSynthesizerTone(
                    note.pitch,
                    note.instrument,
                    ctx.currentTime + Math.max(0, st - pos),
                    noteDur
                );
            }

            if (scheduledRef.current.size > 2000) scheduledRef.current.clear();
            lastSchedulePosRef.current = pos;
            raf = requestAnimationFrame(tick);
        };

        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [isPlaying, isMidiSynthEnabled, getPosition]);

    // Color-Coding Design by Instrument Role (Clean Studio Aesthetics)
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
        return {
            bg: 'bg-gradient-to-r from-teal-400 via-teal-500 to-emerald-400',
            border: 'border-teal-300 dark:border-teal-400/80',
            text: 'text-slate-950 font-black',
            shadow: 'shadow-md shadow-teal-500/25',
            badge: 'bg-teal-950/20 text-slate-950',
            label: 'Piano'
        };
    };

    // Effective geometry for a note, honoring any active drag preview.
    const geomFor = (note: NoteEvent): { start: number; end: number; pitch: number } => {
        const p = dragPreview?.get(note);
        if (p) return { start: p.start, end: p.end, pitch: p.pitch };
        return { start: note.start_time, end: noteEndOf(note), pitch: note.pitch };
    };

    const firstSelVel = selectedNotes.size > 0
        ? (Array.from(selectedNotes)[0].velocity || 85)
        : 85;

    // ── Render-layer memoization ────────────────────────────────────────────
    // During playback the workspace re-renders ~12Hz. Only the playhead
    // depends on time — keyboard, rows, bar lines and note blocks are
    // memoized so those ticks stop reconciling hundreds of DOM nodes.
    // Handlers reach the memoized trees through refs (always-fresh closures).
    const onSeekRef = useRef(onSeek);
    const beginDragRef = useRef(beginDrag);
    const playToneRef = useRef(playSynthesizerTone);
    useEffect(() => { onSeekRef.current = onSeek; });
    useEffect(() => { beginDragRef.current = beginDrag; });
    useEffect(() => { playToneRef.current = playSynthesizerTone; });

    const keyboardKeys = useMemo(() => pitchRange.map(p => {
        const isBlack = p.isBlack;
        const isMiddleC = p.num === 60;
        const isKeyActive = activePitches.has(p.num);

        return (
            <button
                key={p.num}
                style={{ height: rowH }}
                onClick={() => playToneRef.current(p.num, selectedTrack)}
                title={`Click to play ${p.name} (MIDI Note ${p.num})`}
                aria-label={`Piano Key ${p.name}`}
                className={`text-[10px] font-mono font-bold flex items-center justify-between px-3 transition-all duration-75 relative group ${
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
                }`}
            >
                <div className="flex items-center gap-1.5 min-w-0">
                    <span
                        className={`tracking-tight ${
                            isMiddleC
                                ? 'text-teal-600 dark:text-teal-400 font-extrabold text-[11px] underline decoration-teal-500 decoration-2'
                                : p.isC
                                ? 'text-teal-600 dark:text-teal-400 font-bold text-[11px]'
                                : isBlack
                                ? 'text-slate-300'
                                : 'text-slate-800 dark:text-slate-200'
                        }`}
                    >
                        {p.name}
                    </span>
                    {isMiddleC ? (
                        <span className="text-[10px] font-mono font-bold px-1 rounded bg-teal-500/20 text-teal-700 dark:text-teal-300 border border-teal-500/30">
                            MID
                        </span>
                    ) : p.isC ? (
                        <span className="w-1.5 h-1.5 rounded-full bg-teal-500 shadow-[0_0_6px_rgba(20,184,166,0.8)]" />
                    ) : null}
                </div>

                <span className="text-[10px] font-mono tabular-nums opacity-45 group-hover:opacity-100 text-slate-600 dark:text-slate-400">
                    {p.num}
                </span>
            </button>
        );
    }), [pitchRange, rowH, activePitches, selectedTrack]);

    const rulerBars = useMemo(() => measuresArray.map(bar => {
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
    }), [measuresArray, measureDuration, timeScale]);

    const gridBarLines = useMemo(() => measuresArray.map(bar => {
        const barStart = (bar - 1) * measureDuration;
        const leftPct = (barStart / timeScale) * 100;
        return (
            <div
                key={bar}
                className="absolute top-0 bottom-0 border-l border-slate-300/80 dark:border-white/15"
                style={{ left: `${leftPct}%` }}
            />
        );
    }), [measuresArray, measureDuration, timeScale]);

    const pitchRows = useMemo(() => pitchRange.map(p => {
        const isBlack = p.isBlack;
        const isMiddleC = p.num === 60;

        return (
            <div
                key={p.num}
                style={{ height: rowH }}
                className={`border-b w-full transition-colors flex items-center relative ${
                    isMiddleC
                        ? 'bg-teal-500/[0.12] dark:bg-teal-500/[0.18] border-teal-500/40'
                        : p.isC
                        ? 'bg-teal-500/[0.06] dark:bg-teal-500/[0.10] border-teal-500/30'
                        : isBlack
                        ? 'bg-black/[0.04] dark:bg-black/35 border-black/[0.04] dark:border-white/[0.04]'
                        : 'bg-white/50 dark:bg-[#12141e]/50 border-black/[0.03] dark:border-white/[0.02]'
                }`}
            >
                {p.isC && (
                    <span className="ml-3 text-[10px] font-mono font-bold text-teal-600/70 dark:text-teal-400/70 pointer-events-none select-none">
                        {p.name} {isMiddleC ? '· Middle C' : '· Root Octave'}
                    </span>
                )}
            </div>
        );
    }), [pitchRange, rowH]);

    const noteBlocks = useMemo(() => filteredNotes.map((note, idx) => {
        const g = geomFor(note);
        const pitchIndex = pitchIndexFor(g.pitch);
        const top = pitchIndex * rowH;
        const noteDur = Math.max(0.05, g.end - g.start);
        const leftPercent = (g.start / timeScale) * 100;
        const widthPercent = Math.max(0.6, (noteDur / timeScale) * 100);
        const isSelected = selectedNotes.has(note);
        const noteStyle = getNoteStyle(note.instrument);
        const isDrum = note.instrument?.toLowerCase().includes('drum') || note.instrument?.toLowerCase().includes('percussion');
        const displayPitch = note.note_name || pitchName(g.pitch).name;

        if (isDrum) {
            return (
                <div
                    key={idx}
                    onMouseDown={(e) => beginDragRef.current(e, note, 'move')}
                    className={`absolute min-w-[20px] max-w-[28px] rounded-full border shadow-sm flex items-center justify-center cursor-grab active:cursor-grabbing transition-shadow z-10 ${
                        noteStyle.bg
                    } ${noteStyle.border} ${noteStyle.shadow} ${
                        isSelected
                            ? 'ring-2 ring-white z-30 shadow-lg'
                            : 'hover:brightness-110'
                    }`}
                    style={{
                        top: `${top + 2}px`,
                        left: `${leftPercent}%`,
                        height: `${Math.max(14, rowH - 4)}px`,
                    }}
                    title={`Drum Hit [${g.start.toFixed(2)}s] · Velocity: ${note.velocity || 85}`}
                >
                    <span className="text-white select-none"><Drum size={10} /></span>
                </div>
            );
        }

        return (
            <div
                key={idx}
                onMouseDown={(e) => {
                    // Right-edge zone initiates resize instead of move.
                    const el = e.currentTarget.getBoundingClientRect();
                    beginDragRef.current(e, note, el.right - e.clientX <= 8 ? 'resize' : 'move');
                }}
                className={`absolute rounded-lg border shadow-sm flex items-center justify-between px-2 text-[10px] font-mono font-bold transition-shadow z-10 overflow-hidden ${
                    noteStyle.bg
                } ${noteStyle.border} ${noteStyle.text} ${noteStyle.shadow} ${
                    isSelected
                        ? 'ring-2 ring-white z-30 shadow-lg'
                        : 'hover:brightness-110'
                }`}
                style={{
                    top: `${top + 1}px`,
                    left: `${leftPercent}%`,
                    width: `${widthPercent}%`,
                    minWidth: '24px',
                    height: `${rowH - 2}px`,
                    cursor: 'grab'
                }}
                title={`${displayPitch} (${note.instrument}) [${g.start.toFixed(2)}s - ${g.end.toFixed(2)}s] · Velocity: ${note.velocity || 85} · drag right edge to resize`}
            >
                <span className="truncate pointer-events-none">{displayPitch}</span>

                {widthPercent > 2.5 && (
                    <span
                        className={`hidden sm:inline-block text-[10px] font-mono px-1 py-px rounded-md pointer-events-none ${noteStyle.badge}`}
                    >
                        {noteStyle.label}
                    </span>
                )}
                {/* Resize affordance strip */}
                <span
                    className="absolute right-0 top-0 bottom-0 w-2 cursor-ew-resize"
                    aria-hidden
                />
            </div>
        );
    }), [filteredNotes, selectedNotes, dragPreview, rowH, timeScale]);

    return (
        <div className="flex flex-col h-full bg-[#f4f4f7] dark:bg-[#0b0d13] text-slate-900 dark:text-slate-200 select-none overflow-hidden transition-colors duration-200">
            {/* Top Toolbar Header */}
            <div className="flex flex-wrap items-center justify-between gap-3 px-6 py-3 border-b border-black/[0.06] dark:border-white/[0.08] bg-white/80 dark:bg-[#12141c]/90 backdrop-blur-2xl z-20 shadow-apple-sm">
                <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 text-teal-700 dark:text-teal-300 border border-teal-500/20 flex items-center justify-center font-bold text-xs">
                        <Music size={15} />
                    </div>
                    <div>
                        <div className="flex items-center gap-2">
                            <h2 className="text-xs sm:text-sm font-extrabold tracking-tight text-slate-900 dark:text-white uppercase">
                                Grand Piano Roll & MIDI Score
                            </h2>
                            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 font-bold border border-teal-500/20">
                                {filteredNotes.length} Notes
                            </span>
                            {selectedNotes.size > 0 && (
                                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-700 dark:text-cyan-300 font-bold border border-cyan-500/30">
                                    {selectedNotes.size} selected
                                </span>
                            )}
                        </div>
                        <p className="text-[10px] text-slate-500 dark:text-slate-400 font-mono">
                            Drag to move · edge-drag to resize · double-click to insert · ⌘Z undo · ⌘D duplicate
                        </p>
                    </div>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                    {/* Undo / Redo */}
                    <div className="flex items-center rounded-xl bg-black/[0.04] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 overflow-hidden">
                        <button
                            onClick={doUndo}
                            disabled={!histFlags.u}
                            title="Undo (⌘Z)"
                            aria-label="Undo"
                            className="p-2 text-slate-600 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400 disabled:opacity-30 disabled:pointer-events-none transition-colors"
                        >
                            <Undo2 size={14} />
                        </button>
                        <button
                            onClick={doRedo}
                            disabled={!histFlags.r}
                            title="Redo (⇧⌘Z)"
                            aria-label="Redo"
                            className="p-2 text-slate-600 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400 disabled:opacity-30 disabled:pointer-events-none border-l border-black/[0.06] dark:border-white/10 transition-colors"
                        >
                            <Redo2 size={14} />
                        </button>
                    </div>

                    {/* Snap selector */}
                    <select
                        value={snapDiv}
                        onChange={(e) => setSnapDiv(e.target.value as typeof snapDiv)}
                        title="Snap grid for edits and inserts"
                        aria-label="Snap grid"
                        className="apple-input !py-1.5 !px-2 text-[11px] font-mono cursor-pointer"
                    >
                        <option value="off">Snap: Off</option>
                        <option value="1/16">Snap: 1/16</option>
                        <option value="1/8">Snap: 1/8</option>
                        <option value="1/4">Snap: 1/4</option>
                    </select>

                    {/* Quantize */}
                    <button
                        onClick={quantizeSelection}
                        disabled={snapSec <= 0}
                        title={selectedNotes.size > 0 ? 'Quantize selected note starts to the snap grid' : 'Quantize all visible notes to the snap grid'}
                        className="px-2.5 py-1.5 rounded-xl text-xs font-bold flex items-center space-x-1.5 transition-all border shadow-sm bg-black/[0.04] dark:bg-white/5 border-black/[0.06] dark:border-white/10 text-slate-600 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400 disabled:opacity-40 disabled:pointer-events-none"
                    >
                        <Magnet size={13} />
                        <span>Quantize</span>
                    </button>

                    {/* Velocity for selection */}
                    {selectedNotes.size > 0 && (
                        <label className="flex items-center gap-1.5 text-[10px] font-mono font-bold text-slate-500 dark:text-slate-400" title="Velocity of selected notes">
                            VEL
                            <input
                                type="range"
                                min={1}
                                max={127}
                                value={firstSelVel}
                                onChange={(e) => applyVelocity(parseInt(e.target.value))}
                                className="w-16 h-1 accent-teal-500 cursor-pointer"
                                aria-label="Velocity of selected notes"
                            />
                            <span className="tabular-nums w-6">{firstSelVel}</span>
                        </label>
                    )}

                    {/* Delete selection */}
                    {selectedNotes.size > 0 && (
                        <button
                            onClick={deleteSelection}
                            title="Delete selected notes (Del)"
                            className="p-1.5 rounded-xl bg-rose-500/10 text-rose-600 dark:text-rose-400 text-xs font-bold flex items-center gap-1 hover:bg-rose-500/20 transition-colors"
                        >
                            <Trash2 size={13} />
                            <span>Delete ({selectedNotes.size})</span>
                        </button>
                    )}

                    {/* Live Synth Toggle */}
                    <button
                        onClick={() => setIsMidiSynthEnabled(!isMidiSynthEnabled)}
                        title="Toggle live Web Audio multi-harmonic synth playback"
                        aria-label="Toggle Live Synth"
                        aria-pressed={isMidiSynthEnabled}
                        className={`px-3 py-1.5 rounded-xl text-xs font-bold flex items-center space-x-1.5 transition-all border shadow-sm ${
                            isMidiSynthEnabled
                                ? 'bg-gradient-to-r from-teal-500 to-cyan-500 text-slate-950 border-teal-400'
                                : 'bg-black/[0.04] dark:bg-white/5 border-black/[0.06] dark:border-white/10 text-slate-600 dark:text-slate-400 hover:text-slate-900'
                        }`}
                    >
                        <Volume2 size={13} />
                        <span>{isMidiSynthEnabled ? 'Live Synth ON' : 'Live Synth OFF'}</span>
                    </button>

                    {/* Fit range */}
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
                                aria-pressed={selectedTrack === t.id}
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

                    {/* Save-state chip: batched persistence with honest states */}
                    {saveState === 'error' ? (
                        <button
                            onClick={() => void flushPersist(notesList)}
                            title="Click to retry saving your edits"
                            className="px-3 py-1.5 rounded-xl text-xs font-bold flex items-center space-x-1.5 bg-rose-500/15 text-rose-600 dark:text-rose-400 border border-rose-500/30 hover:bg-rose-500/25 transition-colors"
                        >
                            <AlertTriangle size={13} />
                            <span>Save failed · Retry</span>
                        </button>
                    ) : (
                        <button
                            onClick={() => void flushPersist(notesList)}
                            disabled={saveState === 'saving'}
                            title="Save edited notes and re-engrave the MusicXML score"
                            className={`px-3 py-1.5 rounded-xl text-xs font-bold flex items-center space-x-1.5 shadow-sm transition-all ${
                                saveState === 'saved'
                                    ? 'bg-emerald-500 text-slate-950'
                                    : saveState === 'dirty'
                                    ? 'bg-amber-500/15 text-amber-700 dark:text-amber-400 border border-amber-500/30'
                                    : 'bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-200'
                            }`}
                        >
                            {saveState === 'saving' ? (
                                <RefreshCw size={13} className="animate-spin text-teal-500" />
                            ) : saveState === 'saved' ? (
                                <Check size={13} />
                            ) : (
                                <Save size={13} className="text-teal-600 dark:text-teal-400" />
                            )}
                            <span>
                                {saveState === 'saving' ? 'Syncing…'
                                    : saveState === 'saved' ? 'Score Synced'
                                    : saveState === 'dirty' ? 'Unsaved edits'
                                    : 'Score Synced'}
                            </span>
                        </button>
                    )}

                    <button
                        onClick={() => window.open(`${API_BASE_URL}/transcribe/export/${job.id}/midi`, '_blank')}
                        title="Download Multi-Track .mid MIDI file"
                        aria-label="Download Multi-Track MIDI"
                        className="px-3.5 py-1.5 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 shadow-sm active:scale-95 transition-all"
                    >
                        <Download size={13} />
                        <span>MIDI</span>
                    </button>
                </div>
            </div>

            {/* Main Piano Roll Canvas: Synchronized Scroll Container */}
            <div ref={scrollRef} className="flex-1 overflow-auto relative flex bg-[#ebebf0] dark:bg-[#0a0c12]">
                {/* 1. Left Piano Keys Keyboard (Sticky on Left X, Scrolling on Y) */}
                <div className="sticky left-0 z-20 flex-shrink-0 w-36 bg-slate-200 dark:bg-[#1a1c24] border-r-4 border-r-teal-600 shadow-2xl flex flex-col pt-8 select-none">
                    <div className="absolute top-0 left-0 right-0 h-8 bg-gradient-to-b from-slate-300 via-slate-200 to-slate-400 dark:from-[#252836] dark:via-[#1c1e28] dark:to-[#12131b] border-b border-black/30 dark:border-black/60 flex items-center justify-between px-3 text-[10px] font-mono text-slate-600 dark:text-slate-400 font-bold shadow-sm">
                        <span className="tracking-wider">KEY · MIDI</span>
                        <span className="text-teal-600 dark:text-teal-400 font-serif italic text-[11px] font-bold">Studio Grand</span>
                    </div>

                    {keyboardKeys}
                </div>

                {/* 2. Right Note Grid Canvas — focusable so editing keys work */}
                <div
                    ref={gridFocusRef}
                    tabIndex={0}
                    data-hotkey-local
                    className="flex-1 relative cursor-crosshair flex flex-col outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-teal-500/40"
                    style={{ minWidth: `${1200 * zoomX}px` }}
                >
                    {/* Top Measure Ruler Bar (click to seek) */}
                    <div
                        onClick={(e) => {
                            const rect = e.currentTarget.getBoundingClientRect();
                            onSeek(((e.clientX - rect.left) / rect.width) * timeScale);
                        }}
                        className="sticky top-0 h-8 bg-white/95 dark:bg-[#141622]/95 border-b border-black/[0.08] dark:border-white/10 flex items-center z-10 backdrop-blur-md shadow-sm cursor-pointer"
                        title="Click to move the playhead"
                    >
                        <div className="w-full relative h-full">
                            {rulerBars}
                        </div>
                    </div>

                    {/* Pitch Rows Grid — marquee/drag surface */}
                    <div
                        ref={contentRef}
                        className="relative"
                        onMouseDown={beginMarqueeOrClick}
                        onDoubleClick={handleInsertAt}
                    >
                        {/* Background Vertical Bar Lines */}
                        <div className="absolute inset-0 pointer-events-none z-0">
                            {gridBarLines}
                        </div>

                        {/* Horizontal Pitch Rows */}
                        {pitchRows}

                        {/* Note Blocks (multi-select, drag-move, edge-resize) — memoized:
                            re-renders only on note/selection/drag changes, never on time. */}
                        {noteBlocks}

                        {/* Marquee rubber-band overlay */}
                        {marquee && (
                            <div
                                className="absolute border-2 border-teal-500/70 bg-teal-500/10 pointer-events-none z-40 rounded-sm"
                                style={{
                                    left: Math.min(marquee.x0, marquee.x1),
                                    top: Math.min(marquee.y0, marquee.y1),
                                    width: Math.abs(marquee.x1 - marquee.x0),
                                    height: Math.abs(marquee.y1 - marquee.y0),
                                }}
                            />
                        )}

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

            {/* Bottom status strip: zoom + follow */}
            <div className="flex items-center justify-between px-4 py-1.5 border-t border-black/[0.06] dark:border-white/[0.08] bg-white/70 dark:bg-[#12141c]/80 backdrop-blur-xl text-[10px] font-mono text-slate-500 dark:text-slate-400 flex-shrink-0">
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1">
                        <button
                            onClick={() => setZoomX(z => Math.max(0.5, +(z / 1.3).toFixed(2)))}
                            className="p-1 hover:text-slate-900 dark:hover:text-slate-200"
                            title="Zoom Out (horizontal)"
                            aria-label="Zoom Out"
                        >
                            <ZoomOut size={12} />
                        </button>
                        <span className="tabular-nums w-9 text-center">{Math.round(zoomX * 100)}%</span>
                        <button
                            onClick={() => setZoomX(z => Math.min(6, +(z * 1.3).toFixed(2)))}
                            className="p-1 hover:text-slate-900 dark:hover:text-slate-200"
                            title="Zoom In (horizontal)"
                            aria-label="Zoom In"
                        >
                            <ZoomIn size={12} />
                        </button>
                    </div>
                    <div className="flex items-center gap-1">
                        <span>Rows:</span>
                        {[18, 24, 32].map(h => (
                            <button
                                key={h}
                                onClick={() => setRowH(h)}
                                aria-pressed={rowH === h}
                                className={`px-1.5 py-0.5 rounded font-bold ${
                                    rowH === h ? 'bg-teal-500/15 text-teal-600 dark:text-teal-400' : 'hover:text-slate-800 dark:hover:text-slate-200'
                                }`}
                            >
                                {h === 18 ? 'S' : h === 24 ? 'M' : 'L'}
                            </button>
                        ))}
                    </div>
                </div>
                <button
                    onClick={() => setFollowPlayhead(v => !v)}
                    aria-pressed={followPlayhead}
                    title="Auto-scroll to keep the playhead visible during playback"
                    className={`flex items-center gap-1 px-2 py-0.5 rounded font-bold ${
                        followPlayhead ? 'text-teal-600 dark:text-teal-400 bg-teal-500/10' : 'hover:text-slate-800 dark:hover:text-slate-200'
                    }`}
                >
                    <Crosshair size={11} />
                    <span>Follow</span>
                </button>
            </div>
        </div>
    );
};
