"""
MuScriptor Provider — Multi-Instrument Music Transcription Engine.
Uses Kyutai/Mirelo MuScriptor to transcribe multi-instrument audio
into MIDI, dynamic MusicXML 3.1 notation scores, and rich note events.
"""

import io
import os
import sys
import json
import logging
import asyncio
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass, asdict

# Ensure muscriptor repo is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../muscriptor")))

from muscriptor.transcription_model import TranscriptionModel
from muscriptor.events import NoteStartEvent, NoteEndEvent, ProgressEvent
from muscriptor.tokenizer.notes import Note
from muscriptor.utils.midi import notes_to_midi

logger = logging.getLogger(__name__)

# General MIDI & MT3 Instrument Program Map
INSTRUMENT_PROGRAM_MAP: Dict[str, int] = {
    "Piano": 0,
    "Acoustic Piano": 0,
    "Electric Piano": 4,
    "Organ": 16,
    "Acoustic Guitar": 24,
    "Electric Guitar": 27,
    "Clean Electric Guitar": 27,
    "Distorted Electric Guitar": 29,
    "Acoustic Bass": 32,
    "Electric Bass": 33,
    "Contrabass": 43,
    "Strings": 48,
    "String Ensemble": 48,
    "Violin": 40,
    "Viola": 41,
    "Cello": 42,
    "Harp": 46,
    "Trumpet": 56,
    "Trombone": 57,
    "Tuba": 58,
    "French Horn": 60,
    "Brass Section": 61,
    "Saxophone": 65,
    "Oboe": 68,
    "Clarinet": 71,
    "Flute": 73,
    "Synth Lead": 80,
    "Synth Pad": 88,
    "Synth Strings": 50,
    "Drums": 128,
    "Drum Kit": 128,
}


@dataclass
class TranscriptionResult:
    midi_path: str
    musicxml_path: str
    notes: list[dict]
    beat_grid: dict
    notes_json: str
    bpm: float
    key: str
    # Provenance (plan 1.1): what produced this result and whether it is real.
    # "muscriptor" = genuine neural transcription.
    # "unavailable" = transcription failed; nothing was invented.
    source: str = "muscriptor"
    degraded: bool = False
    reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.source != "unavailable" and bool(self.notes)

    def provenance(self) -> dict:
        return {
            "stage": "transcription",
            "source": self.source,
            "degraded": self.degraded or not self.available,
            "reason": self.reason,
            "notes": len(self.notes),
            "bpm": self.bpm,
        }


class MuScriptorProvider:
    _instance = None
    _model: Optional[TranscriptionModel] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MuScriptorProvider, cls).__new__(cls)
        return cls._instance

    def _get_model(self) -> TranscriptionModel:
        if self._model is None:
            logger.info("Loading MuScriptor TranscriptionModel ('small' on Apple Silicon MPS / CPU)...")
            self._model = TranscriptionModel.load_model("small")
            logger.info("MuScriptor TranscriptionModel loaded successfully!")
        return self._model

    def unload(self) -> bool:
        """Release cached transcription model weights from memory."""
        if self._model is not None:
            self._model = None
            import gc
            gc.collect()
            try:
                from app.core.hardware_lock import GlobalHardwareCoordinator
                GlobalHardwareCoordinator.flush_memory()
            except Exception:
                pass
            logger.info("MuScriptorProvider: Model unloaded and VRAM flushed.")
        return True

    async def transcribe(

        self,
        audio_file_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        job_id: str = "",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[Any] = None,
    ) -> TranscriptionResult:
        """
        Transcribes audio into multi-instrument MIDI, MusicXML, and Note Events.

        `cancel_event` is checked between passes: a 4-minute track can take
        minutes on the `small` model and the pipeline's between-step guard never
        fires while this is running (plan 2.5).
        """
        def _check_cancel(stage: str) -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError(f"Transcription cancelled during {stage}")

        target_path = audio_file_path or audio_path or ""
        from app.core.paths import get_generated_audio_dir, get_repo_root
        from app.transcription.karaoke import _resolve_audio_file

        gen_dir = get_generated_audio_dir()
        gen_dir.mkdir(parents=True, exist_ok=True)
        midi_file = str(gen_dir / f"{job_id}.mid")
        musicxml_file = str(gen_dir / f"{job_id}.musicxml")

        local_audio_path = _resolve_audio_file(target_path)
        if not local_audio_path or not os.path.exists(local_audio_path):
            cand = str(gen_dir / os.path.basename(target_path))
            if os.path.exists(cand):
                local_audio_path = cand
            else:
                local_audio_path = target_path.replace("/audio/", "generated_audio/")


        if progress_callback:
            progress_callback(1, 4, "MuScriptor: Initializing transcription engine...")
        await asyncio.sleep(0.02)

        try:
            # 1. Load Model & Run Real Transcription in Worker Thread
            loop = asyncio.get_event_loop()
            
            def run_model_inference():
                model = self._get_model()
                
                # Detect beat grid
                beat_grid = None
                try:
                    beat_grid = model.detect_grid(local_audio_path, mode="best-effort")
                except Exception as e:
                    logger.info(f"Beat grid detection fallback: {e}")

                # Transcribe events
                events = list(model.transcribe(local_audio_path, batch_size=1))
                return model, events, beat_grid

            if progress_callback:
                progress_callback(2, 4, "MuScriptor: Transcribing multi-instrument notes...")

            model, events, beat_grid = await loop.run_in_executor(None, run_model_inference)

            # 2. Extract structured notes with accurate GM programs and channels
            notes: list[dict] = []
            open_events: dict[int, NoteStartEvent] = {}

            PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

            def get_pitch_name(p: int) -> str:
                octave = (p // 12) - 1
                name = PITCH_NAMES[p % 12]
                return f"{name}{octave}"

            for ev in events:
                if isinstance(ev, ProgressEvent):
                    continue
                if isinstance(ev, NoteStartEvent):
                    open_events[ev.index] = ev
                elif isinstance(ev, NoteEndEvent):
                    start_ev = open_events.pop(ev.start_event_index, None)
                    if start_ev:
                        dur = max(0.08, ev.end_time - start_ev.start_time)
                        raw_inst = start_ev.instrument.replace("_", " ").title() if hasattr(start_ev, 'instrument') else "Piano"
                        program = INSTRUMENT_PROGRAM_MAP.get(raw_inst, 0)
                        channel = 9 if program == 128 else 0
                        notes.append({
                            "pitch": start_ev.pitch,
                            "start_time": round(start_ev.start_time, 3),
                            "end_time": round(ev.end_time, 3),
                            "duration": round(dur, 3),
                            "velocity": getattr(start_ev, 'velocity', 85),
                            "instrument": raw_inst,
                            "program": program,
                            "channel": channel,
                            "note_name": get_pitch_name(start_ev.pitch)
                        })

            # Handle any remaining open notes
            for idx, start_ev in open_events.items():
                dur = 0.5
                raw_inst = start_ev.instrument.replace("_", " ").title() if hasattr(start_ev, 'instrument') else "Piano"
                program = INSTRUMENT_PROGRAM_MAP.get(raw_inst, 0)
                channel = 9 if program == 128 else 0
                notes.append({
                    "pitch": start_ev.pitch,
                    "start_time": round(start_ev.start_time, 3),
                    "end_time": round(start_ev.start_time + dur, 3),
                    "duration": dur,
                    "velocity": getattr(start_ev, 'velocity', 80),
                    "instrument": raw_inst,
                    "program": program,
                    "channel": channel,
                    "note_name": get_pitch_name(start_ev.pitch)
                })

            if progress_callback:
                progress_callback(3, 4, f"MuScriptor: Generating MIDI & Multi-Part MusicXML ({len(notes)} notes)...")

            # 3. Generate MIDI bytes
            midi_bytes = model.events_to_midi_bytes(events, beat_grid=beat_grid)
            with open(midi_file, "wb") as f:
                f.write(midi_bytes)

            # 4. Generate Multi-Part W3C MusicXML 3.1 Sheet Score
            detected_bpm = beat_grid.bpm if beat_grid and hasattr(beat_grid, 'bpm') and beat_grid.bpm else 120.0
            xml_content = self._generate_musicxml(notes, bpm=detected_bpm, title=f"Session {job_id[:8]}")
            with open(musicxml_file, "w", encoding="utf-8") as f:
                f.write(xml_content)

            # Mirror MIDI and MusicXML across canonical and relative directories
            mirror_dirs = [
                get_repo_root() / "backend" / "generated_audio",
                get_repo_root() / "generated_audio",
                Path("generated_audio"),
            ]
            for mdir in mirror_dirs:
                try:
                    mdir.resolve().mkdir(parents=True, exist_ok=True)
                    tgt_mid = mdir / f"{job_id}.mid"
                    tgt_xml = mdir / f"{job_id}.musicxml"
                    if not tgt_mid.resolve().exists() or tgt_mid.resolve() != Path(midi_file).resolve():
                        shutil.copy2(midi_file, str(tgt_mid))
                    if not tgt_xml.resolve().exists() or tgt_xml.resolve() != Path(musicxml_file).resolve():
                        shutil.copy2(musicxml_file, str(tgt_xml))
                except Exception:
                    pass

            # 5. Attempt MuseScore 4 PDF & Tab engraving if available
            sheets_dir = gen_dir / "sheets" / job_id
            try:
                from muscriptor.utils.sheets import write_sheets, find_musescore
                find_musescore()
                sheets_dir.mkdir(parents=True, exist_ok=True)
                write_sheets(midi_bytes, sheets_dir)
                logger.info(f"MuseScore 4 sheet engraving complete: {sheets_dir}")
            except Exception as sheet_err:
                logger.debug(f"MuseScore engraving note (fallback to XML): {sheet_err}")

            _check_cancel("transcription")
            if progress_callback:
                progress_callback(4, 4, f"MuScriptor: Complete ({len(notes)} notes transcribed).")

            bg_dict = {
                "bpm": detected_bpm,
                "beats_per_bar": getattr(beat_grid, "beats_per_bar", 4) if beat_grid else 4,
                "first_downbeat": getattr(beat_grid, "first_downbeat", 0.0) if beat_grid else 0.0
            }

            return TranscriptionResult(
                midi_path=f"/audio/{job_id}.mid",
                musicxml_path=f"/audio/{job_id}.musicxml",
                notes=notes,
                beat_grid=bg_dict,
                notes_json=json.dumps(notes),
                bpm=detected_bpm,
                key="C Major",
                source="muscriptor",
                degraded=False,
            )

        except Exception as e:
            logger.error(f"MuScriptor execution error: {e}", exc_info=True)
            return await self._fallback_transcription(
                audio_path or target_path, job_id, progress_callback, reason=str(e)[:200]
            )
        finally:
            try:
                from app.core.hardware_lock import GlobalHardwareCoordinator
                if GlobalHardwareCoordinator.get_memory_policy()["policy"] == "eager":
                    self.unload()
            except Exception:
                pass

    async def update_midi_notes(self, job_id: str, notes: list[dict], bpm: float = 120.0) -> TranscriptionResult:
        """Saves user-edited note events from Piano Roll back to MIDI and MusicXML."""
        os.makedirs("generated_audio", exist_ok=True)
        midi_file = f"generated_audio/{job_id}.mid"
        musicxml_file = f"generated_audio/{job_id}.musicxml"

        # Re-generate MusicXML
        xml_content = self._generate_musicxml(notes, bpm=bpm, title=f"Session {job_id[:8]} (Edited)")
        with open(musicxml_file, "w", encoding="utf-8") as f:
            f.write(xml_content)

        # Build mido MIDI file
        try:
            import mido
            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)

            tempo = round(60_000_000 / max(30.0, bpm))
            track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
            track.append(mido.MetaMessage('track_name', name='Milimo Master Track', time=0))

            # Event-based delta time reconstruction for authentic polyphony
            # Discrete events: (tick, priority, type, pitch, velocity)
            # Priority: note_off (0) before note_on (1) at identical tick to avoid note-stealing
            events = []
            for n in notes:
                p = max(0, min(127, int(n.get('pitch', 60))))
                vel = max(1, min(127, int(n.get('velocity', 85))))
                start_s = max(0.0, float(n.get('start_time', 0.0)))
                dur_s = max(0.01, float(n.get('duration', 0.5)))

                start_tick = int(round(start_s * 480 * (bpm / 60)))
                dur_tick = max(1, int(round(dur_s * 480 * (bpm / 60))))
                end_tick = start_tick + dur_tick

                events.append((start_tick, 1, 'note_on', p, vel))
                events.append((end_tick, 0, 'note_off', p, 0))

            # Sort by tick ascending, then note_off before note_on
            events.sort(key=lambda e: (e[0], e[1]))

            last_tick = 0
            for tick, _, msg_type, pitch, velocity in events:
                delta_tick = max(0, tick - last_tick)
                track.append(mido.Message(msg_type, note=pitch, velocity=velocity, time=delta_tick))
                last_tick = tick

            mid.save(midi_file)
        except Exception as e:
            logger.warn(f"Failed to rebuild MIDI from notes: {e}")

        bg_dict = {"bpm": bpm, "beats_per_bar": 4, "first_downbeat": 0.0}

        return TranscriptionResult(
            midi_path=f"/audio/{job_id}.mid",
            musicxml_path=f"/audio/{job_id}.musicxml",
            notes=notes,
            beat_grid=bg_dict,
            notes_json=json.dumps(notes),
            bpm=bpm,
            key="C Major"
        )

    def get_available_sheets(self, job_id: str) -> List[Dict[str, str]]:
        """Lists all engraved sheet music scores and PDFs available for a track."""
        sheets: List[Dict[str, str]] = []
        musicxml_path = f"generated_audio/{job_id}.musicxml"
        if os.path.exists(musicxml_path):
            sheets.append({
                "name": "Full Multi-Instrument Score (MusicXML 3.1)",
                "filename": f"{job_id}.musicxml",
                "type": "musicxml",
                "url": f"/audio/{job_id}.musicxml"
            })

        sheets_dir = Path(f"generated_audio/sheets/{job_id}")
        if sheets_dir.exists() and sheets_dir.is_dir():
            for p in sorted(sheets_dir.glob("*.pdf")):
                sheets.append({
                    "name": p.stem.replace("_", " ").title(),
                    "filename": p.name,
                    "type": "pdf",
                    "url": f"/audio/sheets/{job_id}/{p.name}"
                })

        return sheets

    async def _fallback_transcription(
        self,
        audio_path: str,
        job_id: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        reason: str = "MuScriptor transcription failed",
    ) -> TranscriptionResult:
        """Report that transcription is unavailable.

        This method used to synthesise a hardcoded I–vi–IV–V progression at 124 BPM,
        write it to MIDI + MusicXML, and return it as if it were transcribed audio —
        so the piano roll and notation view rendered 8 seconds of C-major triads for
        any track whose transcription failed. Nothing is invented now: the caller gets
        an explicitly unavailable result and the job records the reason.
        """
        logger.warning(
            "MuScriptor transcription unavailable for job %s: %s (no MIDI/MusicXML written)",
            job_id, reason,
        )
        if progress_callback:
            progress_callback(4, 4, f"MuScriptor: transcription unavailable ({reason})")

        return TranscriptionResult(
            midi_path="",
            musicxml_path="",
            notes=[],
            beat_grid={},
            notes_json="[]",
            bpm=0.0,
            key="",
            source="unavailable",
            degraded=True,
            reason=reason,
        )

    def _generate_musicxml(self, notes: list[dict], bpm: float = 120.0, title: str = "Milimo Score") -> str:
        """Generates valid multi-part W3C MusicXML 3.1 score partitioning notes by instrument."""
        # Partition notes by instrument
        instruments_found = list(dict.fromkeys([n.get("instrument", "Piano") for n in notes])) or ["Piano"]

        part_list_xml = ["  <part-list>\n"]
        for idx, inst in enumerate(instruments_found):
            pid = f"P{idx + 1}"
            prog = INSTRUMENT_PROGRAM_MAP.get(inst, 0)
            part_list_xml.append(f"""    <score-part id="{pid}">
      <part-name>{inst}</part-name>
      <score-instrument id="{pid}-I1">
        <instrument-name>{inst}</instrument-name>
      </score-instrument>
      <midi-device id="{pid}-I1" port="1"></midi-device>
      <midi-instrument id="{pid}-I1">
        <midi-program>{min(128, prog + 1)}</midi-program>
      </midi-instrument>
    </score-part>\n""")
        part_list_xml.append("  </part-list>\n")

        header = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <work>
    <work-title>{title}</work-title>
  </work>
  <identification>
    <creator type="composer">Mainza AI (Milimo Music Studio)</creator>
    <encoding>
      <software>MuScriptor Neural Transcription Engine</software>
      <encoding-date>2026-08-21</encoding-date>
    </encoding>
  </identification>
{"".join(part_list_xml)}"""

        footer = "</score-partwise>\n"
        step_names = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
        alter_vals = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]

        measure_duration = (60.0 / bpm) * 4.0  # 4-beat bar duration in seconds
        max_time = max([n.get("end_time", 4.0) for n in notes]) if notes else 8.0
        num_measures = max(1, int(np.ceil(max_time / measure_duration)))

        parts_content_xml = []

        for p_idx, inst in enumerate(instruments_found):
            pid = f"P{p_idx + 1}"
            inst_notes = [n for n in notes if n.get("instrument", "Piano") == inst]
            is_piano = "Piano" in inst
            is_bass = "Bass" in inst
            clef_sign = "F" if is_bass else "G"
            clef_line = 4 if is_bass else 2

            part_xml = [f'  <part id="{pid}">\n']

            for m in range(num_measures):
                m_num = m + 1
                m_start = m * measure_duration
                m_end = (m + 1) * measure_duration
                m_notes = [n for n in inst_notes if m_start <= n.get("start_time", 0) < m_end]

                part_xml.append(f'    <measure number="{m_num}">\n')
                if m_num == 1:
                    staves_count = 2 if is_piano else 1
                    part_xml.append(f"""      <attributes>
        <divisions>4</divisions>
        <key>
          <fifths>0</fifths>
          <mode>major</mode>
        </key>
        <time>
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
        <staves>{staves_count}</staves>
        <clef number="1">
          <sign>{clef_sign}</sign>
          <line>{clef_line}</line>
        </clef>
""")
                    if is_piano:
                        part_xml.append("""        <clef number="2">
          <sign>F</sign>
          <line>4</line>
        </clef>
""")
                    part_xml.append(f"""      </attributes>
      <direction placement="above">
        <direction-type>
          <metronome>
            <beat-unit>quarter</beat-unit>
            <per-minute>{int(bpm)}</per-minute>
          </metronome>
        </direction-type>
        <sound tempo="{int(bpm)}"/>
      </direction>
""")

                if not m_notes:
                    part_xml.append("""      <note>
        <rest/>
        <duration>16</duration>
        <voice>1</voice>
        <type>whole</type>
        <staff>1</staff>
      </note>
""")
                else:
                    for note in m_notes[:8]:
                        p = note.get("pitch", 60)
                        step = step_names[p % 12]
                        alter = alter_vals[p % 12]
                        octave = (p // 12) - 1
                        staff = 2 if (is_piano and p < 60) else 1
                        dur = max(2, min(16, int(note.get("duration", 0.5) * 4)))

                        part_xml.append(f"""      <note>
        <pitch>
          <step>{step}</step>
          {"<alter>" + str(alter) + "</alter>" if alter != 0 else ""}
          <octave>{octave}</octave>
        </pitch>
        <duration>{dur}</duration>
        <voice>1</voice>
        <type>quarter</type>
        <staff>{staff}</staff>
      </note>
""")
                part_xml.append("    </measure>\n")

            part_xml.append("  </part>\n")
            parts_content_xml.append("".join(part_xml))

        return header + "".join(parts_content_xml) + footer


muscriptor_provider = MuScriptorProvider()

