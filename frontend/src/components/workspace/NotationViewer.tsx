import React, { useMemo, useState } from 'react';
import { FileText, Download, ZoomIn, ZoomOut, Printer } from 'lucide-react';
import { API_BASE_URL } from '../../api';
import type { Job, NoteEvent } from '../../api';

interface NotationViewerProps {
    job: Job;
    currentTime?: number;
    onSeek?: (time: number) => void;
}

// ---------------------------------------------------------------------------
// Accurate grand-staff notation model
// ---------------------------------------------------------------------------
// Universal note coordinate: `pos` = diatonic position relative to middle C (C4),
// where each diatonic half-step (line OR space) is 1 unit. This is what musicians
// call an "index on the staff".
//   - Treble clef staff: bottom line E4 = pos 2 ... top line F5 = pos 10
//   - Bass clef staff:   bottom line G2 = pos -10 ... top line A3 = pos -2
//   - C4 = pos 0  (ledger line between the two staves)
// Lines sit at EVEN `pos` values; spaces at ODD `pos` values.

const PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const LETTERS: Record<string, number> = { C: 0, D: 1, E: 2, F: 3, G: 4, A: 5, B: 6 };
const FLAT_NAMES  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
const SHARP_PCS = new Set([1, 3, 6, 8, 10]);

function midiToPos(midi: number): { pos: number; accidental: '#' | 'b' | ''; name: string } {
    const pc = ((midi % 12) + 12) % 12;
    const octave = Math.floor(midi / 12) - 1;              // C4 = 60 => octave 4
    const letter = PITCH_CLASS[pc].replace('#', '');        // C, D, E, F… (accidental stripped)
    const letterIdx = LETTERS[letter];                       // C=0 … B=6
    // Diatonic position relative to middle C (C4): 7 degrees per octave.
    const pos = (octave - 4) * 7 + letterIdx;
    let accidental: '#' | 'b' | '' = '';
    if (SHARP_PCS.has(pc)) accidental = '#';
    const spell = accidental === '#' ? PITCH_CLASS[pc] : PITCH_CLASS[pc].replace('#', '');
    return { pos, accidental, name: `${spell}${octave}` };
}

// Rendering params (px). Each pos unit = half the line gap.
const HALF = 9;          // px per staff unit
const TREBLE_BASE = 2;   // bottom line pos (E4)
const BASS_BASE = -10;   // bottom line pos (G2)
const MARGIN = 7 * HALF; // vertical room above/below staves for ledger notes

function durClass(dur: number | undefined): { glyph: 'whole' | 'half' | 'quarter' | 'eighth'; dur: number } {
    const d = dur && dur > 0 ? dur : 0.5;
    if (d >= 3.6) return { glyph: 'whole', dur: d };
    if (d >= 1.8) return { glyph: 'half', dur: d };
    if (d >= 0.9) return { glyph: 'quarter', dur: d };
    return { glyph: 'eighth', dur: d };
}

// Determine staff units where a note needs a ledger line.
// Lines are at EVEN pos; staff spans base..base+8 (bottom..top line).
function ledgerPoses(pos: number, base: number): number[] {
    const out: number[] = [];
    const lo = base;         // bottom line pos
    const hi = base + 8;     // top line pos
    if (pos < lo) {
        // below staff: ledger lines at even pos from up to `pos` (exclude the real bottom line)
        for (let p = lo - 2; p >= pos; p -= 2) out.push(p);
    } else if (pos > hi) {
        for (let p = hi + 2; p <= pos; p += 2) out.push(p);
    }
    return out;
}

interface Measure {
    index: number;
    start: number;
    end: number;
    treble: NoteEvent[];
    bass: NoteEvent[];
}

export const NotationViewer: React.FC<NotationViewerProps> = ({ job, currentTime: _ct = 0, onSeek }) => {
    const [zoom, setZoom] = useState(100);

    const notes: NoteEvent[] = useMemo(
        () => (job.notes_json ? (typeof job.notes_json === 'string' ? JSON.parse(job.notes_json) : job.notes_json) : []),
        [job.notes_json]
    );

    const beatGrid = useMemo(
        () => (job.beat_grid_json ? (typeof job.beat_grid_json === 'string' ? JSON.parse(job.beat_grid_json) : job.beat_grid_json) : {}),
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
                index: m, start, end,
                treble: inm.filter(n => n.pitch >= 60),
                bass: inm.filter(n => n.pitch < 60),
            });
        }
        return out;
    }, [notes, totalMeasures, measureDuration]);

    function xInBar(ms: Measure, t: number): number {
        const dur = Math.max(0.001, ms.end - ms.start);
        return Math.max(3, Math.min(97, ((t - ms.start) / dur) * 100));
    }

    const handleExport = () => window.open(`${API_BASE_URL}/transcribe/export/${job.id}/musicxml`, '_blank');
    const handlePrint = () => window.print();

    // --- Stave / note rendering ------------------------------------------
    const staffTop = MARGIN;
    const staffHeight = 8 * HALF;

    function yOfPos(pos: number, base: number): number {
        return staffTop + (pos - base) * HALF;
    }

    function NoteGlyph({ midi, base, dur, x, color }: {
        midi: number; base: number; dur: number; x: number; color: string;
    }) {
        const { pos, accidental, name } = midiToPos(midi);
        const y = yOfPos(pos, base);
        const ledgers = ledgerPoses(pos, base);
        const { glyph } = durClass(dur);
        const headW = glyph === 'whole' ? 12 : 11;
        const headR = glyph === 'whole' ? 5.5 : 5;
        const open = glyph === 'whole' || glyph === 'half';
        const stemUp = pos <= (base + 4); // stems up for notes below mid-staff
        const stemLen = 30;
        const headX = x;
        const headCx = headX + headW / 2;
        const stemX = stemUp ? headX + headW : headX;
        return (
            <g key={`${midi}-${x}`}>
                {/* ledger lines */}
                {ledgers.map(p => (
                    <rect key={p} x={headX - 5} y={yOfPos(p, base) - 0.6} width={headW + 10} height={1.2} fill={color} />
                ))}
                {/* accidental */}
                {accidental && (
                    <text x={headX - 11} y={y + 4.5} fontSize="13" fontFamily="serif" fill={color}>{accidental === 'b' ? '♭' : '♯'}</text>
                )}
                {/* note head */}
                <ellipse cx={headCx} cy={y} rx={headR} ry={glyph === 'whole' ? 5.2 : 4.6} fill={open ? '#ffffff' : color} stroke={color} strokeWidth="1.4" />
                {/* stem (whole notes have none) */}
                {glyph !== 'whole' && (
                    <line x1={stemX} y1={y} x2={stemX} y2={stemUp ? y - stemLen : y + stemLen} stroke={color} strokeWidth="1.6" />
                )}
                {/* eight-note flag */}
                {glyph === 'eighth' && (
                    <path d={stemUp
                        ? `M ${stemX} ${y - stemLen} q 6 -2 4 8 q 2 -1 2 -8 Z`
                        : `M ${stemX} ${y + stemLen} q 6 2 4 -8 q 2 1 2 8 Z`}
                        fill={color} />
                )}
                <title>{name} · #{midi}</title>
            </g>
        );
    }

    function Stave({ measureA, measureB, base, isTreble, color }: {
        measureA: Measure; measureB: Measure | undefined; base: number; isTreble: boolean; color: string;
    }) {
        const width = 560;
        const leftOf = 96;          // px left of the first barline (clef + time signature)
        const halfW = (width - leftOf - 8) / 2;  // usable width per bar (in px)
        const bar1X = leftOf;
        const bar2X = leftOf + halfW;
        const svgH = staffHeight + MARGIN * 2;
        const two = Boolean(measureB);
        const partNotes = isTreble ? (m: Measure) => m.treble : (m: Measure) => m.bass;

        function renderNotesFor(bar: Measure, barX: number): React.ReactNode {
            return partNotes(bar).slice().sort((a, b) => a.start_time - b.start_time).map(n => {
                const pct = xInBar(bar, n.start_time) / 100;   // 0..1 within this bar
                const x = barX + 10 + pct * Math.max(0, halfW - 24);
                return <NoteGlyph key={`${n.start_time}-${n.pitch}`} midi={n.pitch} base={base} dur={n.duration || 0.5} x={x} color={isTreble ? '#0f766e' : color} />;
            });
        }

        return (
            <svg viewBox={`0 0 ${width} ${svgH}`} className="w-full" style={{ height: svgH }}>
                {/* staff lines (even pos, bottom..top) */}
                {[0, 1, 2, 3, 4].map(i => {
                    const p = base + i * 2;
                    return <line key={i} x1={0} y1={yOfPos(p, base)} x2={width} y2={yOfPos(p, base)} stroke="#1f2937" strokeWidth={1} />;
                })}
                {/* clef + time signature */}
                <text x={10} y={staffTop + 4.5 * HALF} fontSize={isTreble ? 34 : 30} fontFamily="serif" fill="#1f2937">{isTreble ? '𝄞' : '𝄢'}</text>
                <text x={64} y={staffTop} fontSize="11" fontFamily="serif" fill="#1f2937">
                    <tspan x={64} y={staffTop + 3.4 * HALF}>{beatsPerBar}</tspan>
                    <tspan x={64} y={staffTop + 4.9 * HALF}>4</tspan>
                </text>
                {/* bar 1 + barline */}
                {renderNotesFor(measureA, bar1X)}
                <line x1={bar1X} y1={staffTop - 2 * HALF} x2={bar1X} y2={staffTop + 6 * HALF} stroke="#1f2937" strokeWidth="1.1" />
                {/* bar 2 + barline (if present) */}
                {two && measureB && (
                    <>
                        {renderNotesFor(measureB, bar2X)}
                        <line x1={bar2X} y1={staffTop - 2 * HALF} x2={bar2X} y2={staffTop + 6 * HALF} stroke="#1f2937" strokeWidth="1.1" />
                    </>
                )}
                {/* final double barline */}
                <line x1={width - 5} y1={staffTop - 2 * HALF} x2={width - 5} y2={staffTop + 6 * HALF} stroke="#1f2937" strokeWidth="1.4" />
                <line x1={width - 10} y1={staffTop - 2 * HALF} x2={width - 10} y2={staffTop + 6 * HALF} stroke="#1f2937" strokeWidth="1" />
            </svg>
        );
    }

    return (
        <div className="flex flex-col h-full bg-[#f5f5f7] dark:bg-[#0c0e14] text-slate-900 dark:text-slate-200 select-none overflow-hidden transition-colors duration-200">
            <div className="flex items-center justify-between px-6 py-3 border-b border-black/[0.06] dark:border-white/[0.08] bg-white/70 dark:bg-[#12141c]/80 backdrop-blur-xl">
                <div className="flex items-center space-x-3">
                    <FileText size={16} className="text-teal-600 dark:text-teal-400" />
                    <div>
                        <h2 className="text-xs font-bold text-slate-900 dark:text-slate-100 uppercase tracking-wider">
                            MusicXML Score Notation & Sheet Music
                        </h2>
                        <p className="text-[10px] text-slate-500 dark:text-slate-400">
                            Accurate grand-staff engraving · MuScriptor · {notes.length} notes in {totalMeasures} bars @ ♩={bpm}
                        </p>
                    </div>
                </div>
                <div className="flex items-center space-x-2">
                    <div className="flex items-center bg-black/[0.04] dark:bg-[#181a24] border border-black/[0.06] dark:border-white/10 rounded-xl p-1 space-x-1 shadow-sm">
                        <button onClick={() => setZoom(v => Math.max(60, v - 15))} className="p-1 text-slate-500 hover:text-slate-900 dark:hover:text-slate-200" title="Zoom Out"><ZoomOut size={13} /></button>
                        <span className="text-[10px] font-mono px-2 text-slate-700 dark:text-slate-300 font-semibold">{zoom}%</span>
                        <button onClick={() => setZoom(v => Math.min(180, v + 15))} className="p-1 text-slate-500 hover:text-slate-900 dark:hover:text-slate-200" title="Zoom In"><ZoomIn size={13} /></button>
                    </div>
                    <button onClick={handlePrint} title="Print or Save Sheet Music Score as PDF" aria-label="Print Sheet Music Score"
                        className="px-3 py-1.5 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-200 font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-all">
                        <Printer size={13} /> <span>Print</span>
                    </button>
                    <button onClick={handleExport} title="Export and Download W3C MusicXML 3.1 Sheet Music Score" aria-label="Export MusicXML Score"
                        className="px-3.5 py-1.5 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center space-x-1.5 transition-all shadow-sm active:scale-95">
                        <Download size={13} /> <span>Export MusicXML</span>
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-auto p-6 md:p-10 flex items-start justify-center bg-[#eaeaf0] dark:bg-[#0a0c12]">
                <div className="w-full max-w-5xl bg-white text-slate-900 p-8 sm:p-12 rounded-3xl shadow-apple-lg border border-black/[0.06] transition-transform duration-200"
                    style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'top center' }}>
                    <div className="text-center border-b border-black/10 pb-6 space-y-1">
                        <h1 className="text-2xl font-extrabold font-serif text-slate-900 tracking-tight">{job.title || "Full Grand Arrangement"}</h1>
                        <div className="flex items-center justify-between text-xs text-slate-600 font-serif italic pt-2">
                            <span>Tempo: ♩ = {bpm}</span>
                            <span>{beatsPerBar}/4 · Engraved & Transcribed with MuScriptor</span>
                        </div>
                    </div>

                    <div className="space-y-8 mt-8">
                        {Array.from({ length: Math.max(1, Math.ceil(totalMeasures / 2)) }).map((_, sys) => {
                            const mA = measures[sys * 2];
                            const mB = measures[sys * 2 + 1];
                            if (!mA) return null;
                            return (
                                <div key={sys}>
                                    <div className="flex justify-between items-center mb-1 text-[11px] font-serif font-bold text-slate-500 px-1">
                                        <span>Session {Math.floor(sys / 4) + 1} · Bar {mA.index}{mB ? `–${mB.index}` : ''}</span>
                                        {onSeek && (
                                            <button onClick={() => onSeek(mA.start)} className="text-[10px] text-teal-600 hover:underline font-mono">Jump to Bar ▶</button>
                                        )}
                                    </div>
                                    {/* grand staff: treble on top, bass below */}
                                    <div className="border border-black/10 rounded-lg overflow-hidden">
                                        <Stave measureA={mA} measureB={mB} base={TREBLE_BASE} isTreble color="#475569" />
                                        <Stave measureA={mA} measureB={mB} base={BASS_BASE} isTreble={false} color="#475569" />
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
