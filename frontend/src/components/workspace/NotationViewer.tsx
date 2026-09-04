import React, { useMemo, useState } from 'react';
import { FileText, Download, ZoomIn, ZoomOut, Printer, FileDown, X } from 'lucide-react';
import { API_BASE_URL, trackApi, type SheetScoreItem } from '../../api';
import type { Job, NoteEvent } from '../../api';
import { safeJsonParse } from '../../utils/safeJsonParse';

interface NotationViewerProps {
    job: Job;
    currentTime?: number;
    onSeek?: (time: number) => void;
}

// ---------------------------------------------------------------------------
// Engraving Math & Musical Diatonic Coordinate System
// ---------------------------------------------------------------------------
// Universal reference: Diatonic position `pos` relative to Middle C (C4 = 60).
// Each increment of 1 represents a diatonic step (line or space).
//
// Treble Clef:
//   - Top line 5 (F5) = pos 10
//   - Line 4 (D5)     = pos 8
//   - Line 3 (B4)     = pos 6 (Mid-line)
//   - Line 2 (G4)     = pos 4
//   - Bottom line 1 (E4) = pos 2
//   - Middle C (C4)   = pos 0 (Ledger line below)
//
// Bass Clef:
//   - Middle C (C4)   = pos 0 (Ledger line above)
//   - Top line 5 (A3) = pos -2
//   - Line 4 (F3)     = pos -4
//   - Line 3 (D3)     = pos -6 (Mid-line)
//   - Line 2 (B2)     = pos -8
//   - Bottom line 1 (G2) = pos -10
//   - Ledger line 1 below (E2) = pos -12

const PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const LETTERS: Record<string, number> = { C: 0, D: 1, E: 2, F: 3, G: 4, A: 5, B: 6 };
const SHARP_PCS = new Set([1, 3, 6, 8, 10]);

export function midiToDiatonic(midi: number): { pos: number; accidental: '#' | 'b' | ''; name: string } {
    const pc = ((midi % 12) + 12) % 12;
    const octave = Math.floor(midi / 12) - 1; // C4 = 60 => octave 4
    const letter = PITCH_CLASS[pc].replace('#', '');
    const letterIdx = LETTERS[letter];
    // Diatonic position relative to Middle C (C4 = pos 0): 7 degrees per octave
    const pos = (octave - 4) * 7 + letterIdx;
    const accidental: '#' | 'b' | '' = SHARP_PCS.has(pc) ? '#' : '';
    const spell = accidental === '#' ? PITCH_CLASS[pc] : letter;
    return { pos, accidental, name: `${spell}${octave}` };
}

// Rendering Layout Constants (SVG Pixels)
const HALF = 7.5;             // px per diatonic step (half of a staff line space)
const LINE_GAP = HALF * 2;   // 15px between adjacent staff lines
const STAFF_HEIGHT = 8 * HALF; // 60px between top and bottom lines
const MARGIN_TOP = 50;       // Vertical clearance above staff for high ledger notes
const SVG_HEIGHT = 160;      // Total SVG height per staff

// Rhythm glyphs are derived from the track's real tempo, not raw seconds.
// The beat grid defines BPM in quarter-note beats (beat unit 4), so a note's
// musical length in BEATS is duration / seconds-per-beat. Bucketing raw
// seconds was only correct at ~70 BPM and mis-engraved everything else.
const BEAT_UNIT = 4; // beat-grid beat unit (quarter note); one place to change if the backend ever emits real time signatures.

function durClass(dur: number | undefined, bpm: number): { glyph: 'whole' | 'half' | 'quarter' | 'eighth'; dur: number } {
    const d = dur && dur > 0 ? dur : 0.5;
    const secPerBeat = 60 / Math.max(1, bpm);
    const beats = d / secPerBeat;
    // Thresholds allow dotted values to round up to the nearest engraved glyph.
    if (beats >= 3) return { glyph: 'whole', dur: d };
    if (beats >= 1.5) return { glyph: 'half', dur: d };
    if (beats >= 0.75) return { glyph: 'quarter', dur: d };
    return { glyph: 'eighth', dur: d };
}

// Compute standard ledger lines for notes outside the 5 staff lines
function getLedgerPositions(pos: number, isTreble: boolean): number[] {
    const ledgers: number[] = [];
    if (isTreble) {
        if (pos >= 12) {
            // Above treble staff: even pos from 12 up to pos (A5, C6, E6...)
            for (let p = 12; p <= pos; p += 2) ledgers.push(p);
        } else if (pos <= 0) {
            // Below treble staff: even pos from 0 down to pos (C4, A3, F3...)
            for (let p = 0; p >= pos; p -= 2) ledgers.push(p);
        }
    } else {
        if (pos >= 0) {
            // Above bass staff: even pos from 0 up to pos (C4, E4, G4...)
            for (let p = 0; p <= pos; p += 2) ledgers.push(p);
        } else if (pos <= -12) {
            // Below bass staff: even pos from -12 down to pos (E2, C2, A1...)
            for (let p = -12; p >= pos; p -= 2) ledgers.push(p);
        }
    }
    return ledgers;
}

interface Measure {
    index: number;
    start: number;
    end: number;
    treble: NoteEvent[];
    bass: NoteEvent[];
}

export const NotationViewer: React.FC<NotationViewerProps> = ({ job, currentTime = 0, onSeek }) => {
    const [zoom, setZoom] = useState(100);
    const [selectedInstrument, setSelectedInstrument] = useState<string>('all');
    const [isSheetsModalOpen, setIsSheetsModalOpen] = useState(false);
    const [availableSheets, setAvailableSheets] = useState<SheetScoreItem[]>([]);
    const [isLoadingSheets, setIsLoadingSheets] = useState(false);

    const rawNotes: NoteEvent[] = useMemo(
        () => safeJsonParse<NoteEvent[]>(job.notes_json, [], 'notes_json'),
        [job.notes_json]
    );

    const availableInstruments = useMemo(() => {
        const set = new Set<string>();
        rawNotes.forEach(n => {
            if (n.instrument) set.add(n.instrument);
        });
        return Array.from(set);
    }, [rawNotes]);

    const notes = useMemo(() => {
        if (selectedInstrument === 'all') return rawNotes;
        return rawNotes.filter(n => n.instrument === selectedInstrument);
    }, [rawNotes, selectedInstrument]);

    const beatGrid = useMemo(
        () => safeJsonParse<Record<string, number>>(job.beat_grid_json, {}, 'beat_grid_json'),
        [job.beat_grid_json]
    );
    const bpm = Number(beatGrid.bpm) > 0 ? Number(beatGrid.bpm) : 120;
    const beatsPerBar = Number(beatGrid.beats_per_bar) > 0 ? Number(beatGrid.beats_per_bar) : 4;
    const totalDuration = job.duration_ms ? job.duration_ms / 1000 : 30;
    const measureDuration = (60 / bpm) * beatsPerBar;
    const totalMeasures = Math.max(1, Math.ceil(totalDuration / measureDuration));

    const measures: Measure[] = useMemo(() => {
        const out: Measure[] = [];
        for (let m = 1; m <= totalMeasures; m++) {
            const start = (m - 1) * measureDuration;
            const end = m * measureDuration;
            const inm = notes.filter(n => n.start_time >= start && n.start_time < end);
            out.push({
                index: m,
                start,
                end,
                treble: inm.filter(n => n.pitch >= 60),
                bass: inm.filter(n => n.pitch < 60),
            });
        }
        return out;
    }, [notes, totalMeasures, measureDuration]);

    const loadSheets = async () => {
        setIsLoadingSheets(true);
        setIsSheetsModalOpen(true);
        try {
            const res = await trackApi.getSheets(job.id);
            setAvailableSheets(res.sheets || []);
        } catch (e) {
            console.error('Failed to load sheets:', e);
        } finally {
            setIsLoadingSheets(false);
        }
    };

    function xInBar(ms: Measure, t: number): number {
        const dur = Math.max(0.001, ms.end - ms.start);
        return Math.max(0, Math.min(1, (t - ms.start) / dur));
    }

    const handleExport = () => window.open(`${API_BASE_URL}/transcribe/export/${job.id}/musicxml`, '_blank');
    const handlePrint = () => window.print();

    // -----------------------------------------------------------------------
    // Accurate Vertical Coordinate Calculations
    // -----------------------------------------------------------------------
    // Treble: Top line (F5 = 10) is at MARGIN_TOP; Bottom line (E4 = 2) is at MARGIN_TOP + 8 * HALF
    function yOfTreble(pos: number): number {
        return MARGIN_TOP + (10 - pos) * HALF;
    }

    // Bass: Top line (A3 = -2) is at MARGIN_TOP; Bottom line (G2 = -10) is at MARGIN_TOP + 8 * HALF
    function yOfBass(pos: number): number {
        return MARGIN_TOP + (-2 - pos) * HALF;
    }

    interface ChordNote {
        midi: number;
        pos: number;
        accidental: '#' | 'b' | '';
        showAccidental: boolean;
        name: string;
        dur: number;
        glyph: 'whole' | 'half' | 'quarter' | 'eighth';
    }

    interface ChordGroup {
        startTime: number;
        notes: ChordNote[];
    }

    // Group notes occurring at the same beat into chords and deduplicate accidentals within the measure
    function groupNotesIntoChords(notesList: NoteEvent[]): ChordGroup[] {
        const sorted = notesList.slice().sort((a, b) => a.start_time - b.start_time);
        const groups: ChordGroup[] = [];
        const measureAccidentals = new Map<number, '#' | 'b' | ''>(); // pitch -> accidental shown in measure

        for (const n of sorted) {
            const { pos, accidental, name } = midiToDiatonic(n.pitch);
            const { glyph } = durClass(n.duration, bpm);
            let showAccidental = false;
            if (accidental) {
                // Show accidental only if not already shown in this measure for this specific pitch
                if (measureAccidentals.get(n.pitch) !== accidental) {
                    showAccidental = true;
                    measureAccidentals.set(n.pitch, accidental);
                }
            } else {
                if (measureAccidentals.has(n.pitch)) {
                    measureAccidentals.delete(n.pitch);
                }
            }

            const chordNote: ChordNote = {
                midi: n.pitch,
                pos,
                accidental,
                showAccidental,
                name,
                dur: n.duration || 0.5,
                glyph
            };

            const lastGroup = groups[groups.length - 1];
            if (lastGroup && Math.abs(lastGroup.startTime - n.start_time) < 0.05) {
                if (!lastGroup.notes.some(cn => cn.midi === n.pitch)) {
                    lastGroup.notes.push(chordNote);
                    lastGroup.notes.sort((a, b) => a.pos - b.pos);
                }
            } else {
                groups.push({
                    startTime: n.start_time,
                    notes: [chordNote]
                });
            }
        }
        return groups;
    }

    function ChordGlyph({ chord, isTreble, x, color }: {
        chord: ChordGroup; isTreble: boolean; x: number; color: string;
    }) {
        const midPos = isTreble ? 6 : -6; // Treble mid-line is B4 (pos 6), Bass mid-line is D3 (pos -6)
        const lowestNote = chord.notes[0];
        const highestNote = chord.notes[chord.notes.length - 1];
        const avgPos = (lowestNote.pos + highestNote.pos) / 2;
        const stemUp = avgPos < midPos;
        const stemLen = 34;

        const isAllWhole = chord.notes.every(n => n.glyph === 'whole');
        const hasEighth = chord.notes.some(n => n.glyph === 'eighth');

        const headRx = 5.4;
        const headRy = 3.8;
        const headCx = x;
        const stemX = stemUp ? headCx + headRx - 0.6 : headCx - headRx + 0.6;

        // Unique ledger lines needed across all notes in the chord
        const uniqueLedgers = Array.from(
            new Set(chord.notes.flatMap(n => getLedgerPositions(n.pos, isTreble)))
        );

        // Stem vertical span
        const startY = isTreble
            ? (stemUp ? yOfTreble(lowestNote.pos) + 0.5 : yOfTreble(highestNote.pos) - 0.5)
            : (stemUp ? yOfBass(lowestNote.pos) + 0.5 : yOfBass(highestNote.pos) - 0.5);

        const tipY = isTreble
            ? (stemUp ? yOfTreble(highestNote.pos) - stemLen : yOfTreble(lowestNote.pos) + stemLen)
            : (stemUp ? yOfBass(highestNote.pos) - stemLen : yOfBass(lowestNote.pos) + stemLen);

        return (
            <g key={`${chord.startTime}-${lowestNote.midi}`} className="group/chord cursor-pointer">
                {/* Ledger Lines (Deduplicated across the chord) */}
                {uniqueLedgers.map(p => {
                    const ly = isTreble ? yOfTreble(p) : yOfBass(p);
                    return (
                        <line
                            key={p}
                            x1={headCx - headRx - 4}
                            y1={ly}
                            x2={headCx + headRx + 4}
                            y2={ly}
                            stroke="#1e293b"
                            strokeWidth="1.3"
                        />
                    );
                })}

                {/* Single Shared Stem for the Chord */}
                {!isAllWhole && (
                    <line
                        x1={stemX}
                        y1={startY}
                        x2={stemX}
                        y2={tipY}
                        stroke={color}
                        strokeWidth="1.5"
                        strokeLinecap="round"
                    />
                )}

                {/* Single Eighth Note Flag for the Chord */}
                {hasEighth && !isAllWhole && (
                    <path
                        d={stemUp
                            ? `M ${stemX} ${tipY} c 3 6, 8 12, 8 20 c -2 -7, -5 -12, -8 -16 Z`
                            : `M ${stemX} ${tipY} c 3 -6, 8 -12, 8 -20 c -2 7, -5 12, -8 16 Z`}
                        fill={color}
                    />
                )}

                {/* Noteheads & Accidentals in the Chord */}
                {chord.notes.map((cn) => {
                    const y = isTreble ? yOfTreble(cn.pos) : yOfBass(cn.pos);
                    const isWhole = cn.glyph === 'whole';
                    const isHalf = cn.glyph === 'half';
                    const isOpen = isWhole || isHalf;

                    return (
                        <g key={cn.midi}>
                            {/* Accidental (only shown on the first instance in measure) */}
                            {cn.showAccidental && cn.accidental === '#' && (
                                <text
                                    x={headCx - headRx - 10}
                                    y={y + 4.5}
                                    fontSize="14"
                                    fontFamily="serif"
                                    fontWeight="bold"
                                    fill={color}
                                    textAnchor="middle"
                                >
                                    ♯
                                </text>
                            )}
                            {cn.showAccidental && cn.accidental === 'b' && (
                                <text
                                    x={headCx - headRx - 9}
                                    y={y + 3.5}
                                    fontSize="14"
                                    fontFamily="serif"
                                    fontWeight="bold"
                                    fill={color}
                                    textAnchor="middle"
                                >
                                    ♭
                                </text>
                            )}

                            {/* Notehead */}
                            <ellipse
                                cx={headCx}
                                cy={y}
                                rx={isWhole ? 6.2 : headRx}
                                ry={isWhole ? 4.4 : headRy}
                                transform={`rotate(-15 ${headCx} ${y})`}
                                fill={isOpen ? '#ffffff' : color}
                                stroke={color}
                                strokeWidth={isOpen ? (isWhole ? '2.0' : '1.7') : '1.2'}
                                className="group-hover/chord:fill-teal-500 transition-colors"
                            />
                        </g>
                    );
                })}

                <title>{chord.notes.map(n => n.name).join(' + ')} · {chord.notes[0].dur}s</title>
            </g>
        );
    }

    function Stave({ measureA, measureB, isTreble, color }: {
        measureA: Measure; measureB: Measure | undefined; isTreble: boolean; color: string;
    }) {
        const width = 860; // Expanded width for plenty of breathing room
        const leftClefW = 95; // px for Clef and Time Signature
        const usableW = width - leftClefW - 12;
        const halfW = usableW / 2;
        const bar1X = leftClefW;
        const bar2X = leftClefW + halfW;
        const two = Boolean(measureB);
        const partNotes = isTreble ? (m: Measure) => m.treble : (m: Measure) => m.bass;

        // Render notes for a given bar with proportional beat spacing and minimum clearance
        function renderNotesFor(bar: Measure, startX: number, barW: number): React.ReactNode {
            const chordGroups = groupNotesIntoChords(partNotes(bar));
            let lastX = startX;

            return chordGroups.map((chord) => {
                const pct = xInBar(bar, chord.startTime);
                const idealX = startX + 22 + pct * Math.max(0, barW - 44);
                const noteX = Math.max(lastX + (lastX === startX ? 0 : 20), idealX);
                lastX = noteX;

                return (
                    <ChordGlyph
                        key={chord.startTime}
                        chord={chord}
                        isTreble={isTreble}
                        x={noteX}
                        color={isTreble ? '#0f766e' : color}
                    />
                );
            });
        }

        // 5 Staff Lines: Top line (i=0) to bottom line (i=4)
        const staffLines = [0, 1, 2, 3, 4].map(i => MARGIN_TOP + i * LINE_GAP);

        // Playhead indicator for active playback
        const isCurrentInBarA = currentTime >= measureA.start && currentTime <= measureA.end;
        const isCurrentInBarB = measureB && currentTime >= measureB.start && currentTime <= measureB.end;

        return (
            <svg viewBox={`0 0 ${width} ${SVG_HEIGHT}`} className="w-full" style={{ height: SVG_HEIGHT }}>
                {/* 5 Horizontal Staff Lines */}
                {staffLines.map((yLine, i) => (
                    <line
                        key={i}
                        x1={15}
                        y1={yLine}
                        x2={width}
                        y2={yLine}
                        stroke="#334155"
                        strokeWidth="1.1"
                    />
                ))}

                {/* Left Clef Symbol */}
                <text
                    x={22}
                    y={isTreble ? MARGIN_TOP + 46 : MARGIN_TOP + 38}
                    fontSize={isTreble ? 46 : 38}
                    fontFamily="serif"
                    fill="#1e293b"
                    className="select-none"
                >
                    {isTreble ? '𝄞' : '𝄢'}
                </text>

                {/* Stacked Time Signature (e.g. 4/4) in Staff Spaces */}
                <g className="font-serif font-bold select-none" fill="#1e293b" fontSize="20" textAnchor="middle">
                    <text x={76} y={MARGIN_TOP + 23}>{beatsPerBar}</text>
                    <text x={76} y={MARGIN_TOP + 53}>{BEAT_UNIT}</text>
                </g>

                {/* Start Barline after Clef */}
                <line
                    x1={bar1X}
                    y1={MARGIN_TOP}
                    x2={bar1X}
                    y2={MARGIN_TOP + STAFF_HEIGHT}
                    stroke="#1e293b"
                    strokeWidth="1.2"
                />

                {/* Notes in Bar 1 */}
                {renderNotesFor(measureA, bar1X, halfW)}

                {/* Bar 1 Playhead Line */}
                {isCurrentInBarA && (
                    <line
                        x1={bar1X + 18 + xInBar(measureA, currentTime) * Math.max(0, halfW - 36)}
                        y1={MARGIN_TOP - 12}
                        x2={bar1X + 18 + xInBar(measureA, currentTime) * Math.max(0, halfW - 36)}
                        y2={MARGIN_TOP + STAFF_HEIGHT + 12}
                        stroke="#0d9488"
                        strokeWidth="2"
                        strokeDasharray="3 3"
                        className="animate-pulse"
                    />
                )}

                {/* Mid Barline between Bar 1 & Bar 2 */}
                {two && (
                    <line
                        x1={bar2X}
                        y1={MARGIN_TOP}
                        x2={bar2X}
                        y2={MARGIN_TOP + STAFF_HEIGHT}
                        stroke="#1e293b"
                        strokeWidth="1.2"
                    />
                )}

                {/* Notes in Bar 2 */}
                {two && measureB && renderNotesFor(measureB, bar2X, halfW)}

                {/* Bar 2 Playhead Line */}
                {isCurrentInBarB && measureB && (
                    <line
                        x1={bar2X + 18 + xInBar(measureB, currentTime) * Math.max(0, halfW - 36)}
                        y1={MARGIN_TOP - 12}
                        x2={bar2X + 18 + xInBar(measureB, currentTime) * Math.max(0, halfW - 36)}
                        y2={MARGIN_TOP + STAFF_HEIGHT + 12}
                        stroke="#0d9488"
                        strokeWidth="2"
                        strokeDasharray="3 3"
                        className="animate-pulse"
                    />
                )}

                {/* End Barline */}
                <line
                    x1={width - 6}
                    y1={MARGIN_TOP}
                    x2={width - 6}
                    y2={MARGIN_TOP + STAFF_HEIGHT}
                    stroke="#1e293b"
                    strokeWidth="2.5"
                />
                <line
                    x1={width - 12}
                    y1={MARGIN_TOP}
                    x2={width - 12}
                    y2={MARGIN_TOP + STAFF_HEIGHT}
                    stroke="#1e293b"
                    strokeWidth="1.1"
                />
            </svg>
        );
    }

    return (
        <div className="flex flex-col h-full bg-[#f5f5f7] dark:bg-[#0c0e14] text-slate-900 dark:text-slate-200 select-none overflow-hidden transition-colors duration-200">
            {/* Top Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-2 px-6 py-3 border-b border-black/[0.06] dark:border-white/[0.08] bg-white/70 dark:bg-[#12141c]/80 backdrop-blur-xl z-10 shadow-sm">
                <div className="flex items-center space-x-3">
                    <FileText size={16} className="text-teal-600 dark:text-teal-400" />
                    <div>
                        <h2 className="text-xs font-bold text-slate-900 dark:text-slate-100 uppercase tracking-wider">
                            MusicXML Score Notation & Engraving
                        </h2>
                        <p className="text-[10px] text-slate-500 dark:text-slate-400 font-mono">
                            Multi-Instrument Engraving · MuScriptor · {notes.length} notes in {totalMeasures} bars @ ♩={bpm}
                        </p>
                    </div>
                </div>

                {/* Instrument Part Filter Pills */}
                {availableInstruments.length > 0 && (
                    <div className="flex items-center space-x-1 bg-black/[0.04] dark:bg-[#181a24] border border-black/[0.06] dark:border-white/10 rounded-xl p-1">
                        <button
                            onClick={() => setSelectedInstrument('all')}
                            className={`px-2.5 py-1 text-[11px] font-bold rounded-lg transition-colors ${
                                selectedInstrument === 'all'
                                    ? 'bg-teal-500 text-slate-950 shadow-sm'
                                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                            }`}
                        >
                            All Parts
                        </button>
                        {availableInstruments.map(inst => (
                            <button
                                key={inst}
                                onClick={() => setSelectedInstrument(inst)}
                                className={`px-2.5 py-1 text-[11px] font-bold rounded-lg transition-colors ${
                                    selectedInstrument === inst
                                        ? 'bg-teal-500 text-slate-950 shadow-sm'
                                        : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                                }`}
                            >
                                {inst}
                            </button>
                        ))}
                    </div>
                )}

                <div className="flex items-center space-x-2">
                    <div className="flex items-center bg-black/[0.04] dark:bg-[#181a24] border border-black/[0.06] dark:border-white/10 rounded-xl p-1 space-x-1 shadow-sm">
                        <button
                            onClick={() => setZoom(v => Math.max(60, v - 15))}
                            className="p-1 text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 rounded-lg hover:bg-black/5"
                            title="Zoom Out"
                        >
                            <ZoomOut size={13} />
                        </button>
                        <span className="text-[10px] font-mono px-2 text-slate-700 dark:text-slate-300 font-semibold">{zoom}%</span>
                        <button
                            onClick={() => setZoom(v => Math.min(180, v + 15))}
                            className="p-1 text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 rounded-lg hover:bg-black/5"
                            title="Zoom In"
                        >
                            <ZoomIn size={13} />
                        </button>
                    </div>

                    <button
                        onClick={loadSheets}
                        title="Download Engraved PDF Scores and Tablatures"
                        className="px-3 py-1.5 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-200 font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-all shadow-sm"
                    >
                        <FileDown size={13} className="text-teal-600 dark:text-teal-400" />
                        <span>Scores & PDFs</span>
                    </button>

                    <button
                        onClick={handlePrint}
                        title="Print or Save Sheet Music Score as PDF"
                        aria-label="Print Sheet Music Score"
                        className="px-3 py-1.5 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-200 font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-all shadow-sm"
                    >
                        <Printer size={13} />
                        <span>Print</span>
                    </button>

                    <button
                        onClick={handleExport}
                        title="Export and Download W3C MusicXML 3.1 Sheet Music Score"
                        aria-label="Export MusicXML Score"
                        className="px-3.5 py-1.5 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-all shadow-sm active:scale-95"
                    >
                        <Download size={13} />
                        <span>Export MusicXML</span>
                    </button>
                </div>
            </div>

            {/* Engraved Sheets & PDFs Modal */}
            {isSheetsModalOpen && (
                <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 animate-fade-in">
                    <div className="w-full max-w-lg bg-white dark:bg-[#151722] border border-black/10 dark:border-white/10 rounded-3xl shadow-apple-2xl p-6 flex flex-col space-y-4">
                        <div className="flex items-center justify-between border-b border-black/10 dark:border-white/10 pb-3">
                            <div className="flex items-center space-x-2">
                                <FileDown size={18} className="text-teal-500" />
                                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                                    Engraved Scores & Tablatures
                                </h3>
                            </div>
                            <button
                                onClick={() => setIsSheetsModalOpen(false)}
                                className="p-1 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
                            >
                                <X size={16} />
                            </button>
                        </div>

                        <div className="space-y-2 max-h-72 overflow-y-auto custom-scrollbar">
                            {isLoadingSheets ? (
                                <div className="py-8 text-center text-xs text-slate-400">
                                    Scanning engraved sheets...
                                </div>
                            ) : availableSheets.length > 0 ? (
                                availableSheets.map((sheet, idx) => (
                                    <div
                                        key={idx}
                                        className="p-3 rounded-2xl bg-black/[0.03] dark:bg-white/5 border border-black/[0.06] dark:border-white/5 flex items-center justify-between hover:border-teal-500/30 transition-colors"
                                    >
                                        <div className="min-w-0 pr-2">
                                            <div className="text-xs font-bold text-slate-800 dark:text-slate-200 truncate">
                                                {sheet.name}
                                            </div>
                                            <div className="text-[10px] font-mono text-slate-400 uppercase">
                                                {sheet.type} • {sheet.filename}
                                            </div>
                                        </div>
                                        <a
                                            href={sheet.url.startsWith('http') ? sheet.url : `${API_BASE_URL}${sheet.url}`}
                                            download={sheet.filename}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="px-3 py-1 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1 shadow-sm"
                                        >
                                            <Download size={12} />
                                            <span>Download</span>
                                        </a>
                                    </div>
                                ))
                            ) : (
                                <div className="space-y-2 text-center py-4">
                                    <p className="text-xs text-slate-500">
                                        Primary W3C MusicXML 3.1 score is ready for instant download.
                                    </p>
                                    <a
                                        href={`${API_BASE_URL}/audio/${job.id}.musicxml`}
                                        download={`${job.title || 'milimo_score'}.musicxml`}
                                        className="inline-flex items-center space-x-1.5 px-4 py-2 bg-teal-500 text-slate-950 font-bold text-xs rounded-xl shadow-sm hover:bg-teal-400"
                                    >
                                        <Download size={13} />
                                        <span>Download MusicXML Score</span>
                                    </a>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Score Page Canvas */}
            <div className="flex-1 overflow-auto p-6 md:p-10 flex items-start justify-center bg-[#eaeaf0] dark:bg-[#0a0c12]">
                <div
                    className="w-full max-w-5xl bg-white text-slate-900 p-8 sm:p-12 rounded-3xl shadow-apple-lg border border-black/[0.06] transition-transform duration-200"
                    style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'top center' }}
                >
                    {/* Sheet Music Header */}
                    <div className="text-center border-b border-black/10 pb-6 space-y-1">
                        <h1 className="text-2xl font-extrabold font-serif text-slate-900 tracking-tight">
                            {job.title || "Full Grand Arrangement"}
                        </h1>
                        <div className="flex items-center justify-between text-xs text-slate-600 font-serif italic pt-2">
                            <span>Tempo: ♩ = {bpm}</span>
                            <span>{beatsPerBar}/{BEAT_UNIT} · {selectedInstrument === 'all' ? 'Conductor Full Score' : `${selectedInstrument} Part`} · MuScriptor</span>
                        </div>
                    </div>

                    {/* Systems (Grand Staff Pairs) */}
                    <div className="space-y-10 mt-8">
                        {Array.from({ length: Math.max(1, Math.ceil(totalMeasures / 2)) }).map((_, sys) => {
                            const mA = measures[sys * 2];
                            const mB = measures[sys * 2 + 1];
                            if (!mA) return null;

                            return (
                                <div key={sys} className="space-y-1.5">
                                    <div className="flex justify-between items-center text-[11px] font-serif font-bold text-slate-600 px-1">
                                        <span>Measure {mA.index}{mB ? ` – ${mB.index}` : ''}</span>
                                        {onSeek && (
                                            <button
                                                onClick={() => onSeek(mA.start)}
                                                className="text-[10px] text-teal-600 hover:text-teal-800 font-mono font-semibold"
                                                title={`Jump to Measure ${mA.index}`}
                                            >
                                                Jump to Bar ▶
                                            </button>
                                        )}
                                    </div>

                                    {/* Grand Staff System with Left Bracket */}
                                    <div className="relative border border-black/15 rounded-2xl p-2 bg-[#fcfcfd] shadow-sm flex">
                                        {/* Grand Staff Left Vertical Bracket Bar */}
                                        <div className="w-4 flex flex-col items-center justify-between py-8 border-r-2 border-r-slate-800 my-4 select-none">
                                            <div className="w-2 h-2 rounded-full bg-slate-800 -ml-1" />
                                            <div className="w-2 h-2 rounded-full bg-slate-800 -ml-1" />
                                        </div>

                                        <div className="flex-1 overflow-hidden">
                                            {/* Upper Treble Clef Staff */}
                                            <Stave measureA={mA} measureB={mB} isTreble={true} color="#0f766e" />
                                            {/* Lower Bass Clef Staff */}
                                            <Stave measureA={mA} measureB={mB} isTreble={false} color="#334155" />
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>
        </div>
    );
};
