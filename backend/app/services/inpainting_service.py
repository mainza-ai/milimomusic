import asyncio
import json
import logging
import math
import os
import shutil
import threading
import time
import uuid
from typing import Optional, Dict, Any
import numpy as np
import soundfile as sf
from sqlmodel import Session, select

from app.models import Job, JobStatus, GenerationRequest, TrackInpaintRequest
from app.services.music_service import music_service, event_manager
from app.providers.registry import provider_registry
from app.transcription.muscriptor_provider import muscriptor_provider
from app.transcription.real_separator import separate_sources, unload_model
from app.transcription.instrument_stems import render_instrument_parts
from app.transcription.karaoke import lyric_sync_engine, _resolve_audio_file
from app.core.paths import get_generated_audio_dir, get_repo_root
from app.core.hardware_lock import GlobalHardwareCoordinator
from app.providers.minimax_provider import (
    extract_audio_musical_attributes,
    build_locked_continuation_caption,
)

logger = logging.getLogger(__name__)


def _load_audio_np(file_path: str):
    """Load an audio file into a 2D numpy float32 array [channels, samples] and sample rate."""
    data, sr = sf.read(file_path, dtype="float32")
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)  # [1, samples]
    else:
        data = data.T  # [samples, channels] -> [channels, samples]
    return data, sr


def _save_audio_np(file_path: str, data: np.ndarray, sr: int):
    """Save [channels, samples] numpy float32 array as 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if data.ndim == 1:
        out = data
    else:
        out = data.T  # [samples, channels]
    sf.write(file_path, out, sr, subtype="PCM_16")


def _equal_power_crossfade(chunk_a: np.ndarray, chunk_b: np.ndarray) -> np.ndarray:
    """Equal-power sine/cosine crossfade between chunk_a (fading out) and chunk_b (fading in)."""
    n = min(chunk_a.shape[-1], chunk_b.shape[-1])
    if n == 0:
        return chunk_a
    ca = chunk_a[..., :n]
    cb = chunk_b[..., :n]
    t = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    if ca.ndim == 2:
        fade_out = fade_out[np.newaxis, :]
        fade_in = fade_in[np.newaxis, :]
    return ca * fade_out + cb * fade_in


class InpaintingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InpaintingService, cls).__new__(cls)
        return cls._instance

    def create_repair_job(self, parent_job_id: str, req: TrackInpaintRequest, db_engine) -> str:
        """Create and persist a child Job record for the segment repair, returning its new job ID."""
        with Session(db_engine) as session:
            parent_uuid = uuid.UUID(parent_job_id) if isinstance(parent_job_id, str) and "-" in parent_job_id else parent_job_id
            parent_job = session.get(Job, parent_uuid)
            if not parent_job:
                parent_job = session.exec(select(Job).where(Job.id == parent_uuid)).one_or_none()
            if not parent_job:
                raise ValueError(f"Parent track {parent_job_id} not found")

            new_job_uuid = uuid.uuid4()
            new_job_id = str(new_job_uuid)
            parent_title = parent_job.title or parent_job.prompt or "Track"
            repair_title = f"{parent_title} (Repaired)"

            new_job = Job(
                id=new_job_uuid,
                title=repair_title,
                prompt=req.prompt or parent_job.prompt,
                status=JobStatus.PROCESSING,
                parent_job_id=str(parent_job.id),
                duration_ms=parent_job.duration_ms,
                lyrics=parent_job.lyrics,
                tags=parent_job.tags,
                bpm=parent_job.bpm,
                beat_grid_json=parent_job.beat_grid_json,
                structured_caption_json=parent_job.structured_caption_json,
                project_id=parent_job.project_id,
                session_id=parent_job.session_id,
                cover_image_path=parent_job.cover_image_path,
                model_provider=parent_job.model_provider or "minimax_music3",
                is_repair=True,
                seed=((parent_job.seed or 42) + 7919) % 2147483647,
            )
            session.add(new_job)
            session.commit()

            # Publish SSE initial state
            event_manager.publish("job_update", {
                "job_id": new_job_id,
                "status": "processing",
                "title": repair_title,
                "parent_job_id": str(parent_job.id),
                "is_repair": True,
            })
            return new_job_id

    async def regenerate_segment(
        self,
        parent_job_id: str,
        repair_job_id: str,
        start_sec: float,
        end_sec: float,
        crossfade_sec: float = 1.0,
        prompt: Optional[str] = None,
        db_engine: Any = None,
    ):
        """Regenerate an audio time segment while preserving surrounding audio and musical continuity."""
        abort_event = threading.Event()
        music_service.active_jobs[repair_job_id] = abort_event
        music_service.job_started_monotonic[repair_job_id] = time.monotonic()

        gen_dir = get_generated_audio_dir()
        gen_dir.mkdir(parents=True, exist_ok=True)
        final_wav_path = str(gen_dir / f"{repair_job_id}.wav")

        try:
            # 1. Fetch parent job metadata
            with Session(db_engine) as session:
                p_uuid = uuid.UUID(parent_job_id) if isinstance(parent_job_id, str) and "-" in parent_job_id else parent_job_id
                pj = session.get(Job, p_uuid)
                if not pj:
                    pj = session.exec(select(Job).where(Job.id == p_uuid)).one_or_none()
                if not pj:
                    raise FileNotFoundError(f"Parent job {parent_job_id} not found in database")

                parent_audio_path = pj.audio_path
                parent_prompt = pj.prompt or ""
                parent_lyrics = pj.lyrics or ""
                parent_tags = pj.tags or ""
                parent_provider = pj.model_provider or "minimax_music3"
                parent_seed = pj.seed
                parent_bpm = pj.bpm
                parent_beat_grid_json = pj.beat_grid_json
                parent_sc_json = pj.structured_caption_json

            event_manager.publish("job_progress", {
                "job_id": repair_job_id,
                "step": 1,
                "total_steps": 4,
                "phase": "analysis",
                "progress": 15,
                "message": f"Analyzing parent audio and beat grid (repairing {start_sec:.1f}s - {end_sec:.1f}s)...",
            })

            # 2. Resolve parent audio on disk
            resolved_parent = _resolve_audio_file(parent_audio_path)
            if not resolved_parent or not os.path.exists(resolved_parent):
                raise FileNotFoundError(f"Parent audio file not found on disk: {parent_audio_path}")

            loop = asyncio.get_running_loop()
            parent_audio, parent_sr = await loop.run_in_executor(None, _load_audio_np, resolved_parent)
            num_channels, total_samples = parent_audio.shape
            total_duration_sec = total_samples / float(parent_sr)

            # 3. Beat-grid alignment & downbeat snapping
            musical_profile = extract_audio_musical_attributes(
                audio_path=resolved_parent,
                notes_json=None,
                beat_grid_json=parent_beat_grid_json,
                stored_bpm=parent_bpm,
            )
            bpm = musical_profile.get("bpm") or parent_bpm or 120.0

            parent_bg = None
            if parent_beat_grid_json:
                try:
                    parent_bg = json.loads(parent_beat_grid_json)
                except Exception:
                    pass
            if not parent_bg:
                parent_bg = {"bpm": bpm, "beats_per_bar": 4, "first_downbeat": 0.0}

            # Snap repair boundaries to musical downbeats/beats if plausible
            snapped_start = start_sec
            snapped_end = end_sec
            if parent_bg and parent_bg.get("bpm", 0.0) > 40.0:
                bg_bpm = float(parent_bg["bpm"])
                beat_dur = 60.0 / bg_bpm
                bpb = int(parent_bg.get("beats_per_bar", 4))
                f_down = float(parent_bg.get("first_downbeat", 0.0))

                # Snap start
                k_start = round((start_sec - f_down) / beat_dur)
                cand_start = f_down + k_start * beat_dur
                if 0.0 <= cand_start < total_duration_sec and abs(cand_start - start_sec) <= 0.6:
                    snapped_start = max(0.0, cand_start)

                # Snap end
                k_end = round((end_sec - f_down) / beat_dur)
                cand_end = f_down + k_end * beat_dur
                if snapped_start < cand_end <= total_duration_sec and abs(cand_end - end_sec) <= 0.6:
                    snapped_end = cand_end

            # Ensure valid bounds
            snapped_start = max(0.0, min(snapped_start, total_duration_sec - 0.2))
            snapped_end = max(snapped_start + 0.2, min(snapped_end, total_duration_sec))
            xfade_sec = max(0.05, min(crossfade_sec, 2.5, (snapped_end - snapped_start) / 2.0))

            logger.info(
                f"[InpaintingService] Repair region: {snapped_start:.2f}s - {snapped_end:.2f}s "
                f"(duration: {snapped_end - snapped_start:.2f}s, xfade: {xfade_sec:.2f}s)"
            )

            # 4. Generate Infill Segment
            event_manager.publish("job_progress", {
                "job_id": repair_job_id,
                "step": 1,
                "total_steps": 4,
                "phase": "generation",
                "progress": 30,
                "message": "Synthesizing infill audio segment with acoustic continuity...",
            })

            gap_sec = (snapped_end - snapped_start) + (2.0 * xfade_sec)
            infill_job_id = f"{repair_job_id}_infill"

            # Parse structured caption
            structured_meta = None
            if parent_sc_json:
                try:
                    structured_meta = json.loads(parent_sc_json)
                except Exception:
                    pass

            locked_caption = build_locked_continuation_caption(
                parent_prompt=parent_prompt,
                parent_tags=parent_tags,
                parent_structured_caption=structured_meta,
                musical_profile=musical_profile,
            )

            provider = provider_registry.get(parent_provider)
            infill_req_prompt = prompt.strip() if (prompt and prompt.strip()) else parent_prompt

            infill_result = await provider.generate(
                job_id=infill_job_id,
                prompt=infill_req_prompt,
                lyrics="",  # Instrumental infill to seamlessly fit accompaniment
                duration_ms=int(max(5.0, gap_sec) * 1000),
                tags=parent_tags,
                seed=((parent_seed or 42) + 7919) % 2147483647,
                structured_caption=locked_caption,
                cancel_event=abort_event,
            )

            if abort_event.is_set():
                raise asyncio.CancelledError("Repair cancelled by user")

            infill_audio_path = _resolve_audio_file(infill_result.audio_path)
            if not infill_audio_path or not os.path.exists(infill_audio_path):
                raise FileNotFoundError(f"Generated infill audio not found at {infill_result.audio_path}")

            infill_audio, infill_sr = await loop.run_in_executor(None, _load_audio_np, infill_audio_path)

            # Resample infill if sample rates differ
            if infill_sr != parent_sr:
                import soxr
                infill_resampled = []
                for ch in range(infill_audio.shape[0]):
                    res = soxr.resample(infill_audio[ch], infill_sr, parent_sr)
                    infill_resampled.append(res)
                infill_audio = np.stack(infill_resampled, axis=0)

            # Match channel count
            if infill_audio.shape[0] < num_channels:
                infill_audio = np.repeat(infill_audio, num_channels, axis=0)
            elif infill_audio.shape[0] > num_channels:
                infill_audio = infill_audio[:num_channels]

            # 5. Equal-Power Splicing into Parent Master Audio
            event_manager.publish("job_progress", {
                "job_id": repair_job_id,
                "step": 1,
                "total_steps": 4,
                "phase": "splicing",
                "progress": 45,
                "message": "Performing equal-power crossfade and splicing into master track...",
            })

            def _splice_audio():
                start_samp = int(snapped_start * parent_sr)
                end_samp = int(snapped_end * parent_sr)
                xfade_samp = int(xfade_sec * parent_sr)

                # Part 1: Head from 0 up to start_samp
                head = parent_audio[:, :start_samp]

                # Crossfade 1: transition from parent to infill
                p_leadout = parent_audio[:, start_samp : start_samp + xfade_samp]
                inf_leadin = infill_audio[:, :xfade_samp]
                xfade_1 = _equal_power_crossfade(p_leadout, inf_leadin)

                # Core infill
                inf_core_len = max(0, (end_samp - start_samp) - xfade_samp)
                inf_core = infill_audio[:, xfade_samp : xfade_samp + inf_core_len]

                # Crossfade 2: transition from infill back to parent
                inf_leadout = infill_audio[:, xfade_samp + inf_core_len : xfade_samp + inf_core_len + xfade_samp]
                p_leadin = parent_audio[:, end_samp - xfade_samp : end_samp]
                xfade_2 = _equal_power_crossfade(inf_leadout, p_leadin)

                # Part 3: Tail from end_samp to end
                tail = parent_audio[:, end_samp:]

                assembled = np.concatenate([head, xfade_1, inf_core, xfade_2, tail], axis=-1)

                # Ensure length matches total_samples exactly
                if assembled.shape[-1] > total_samples:
                    assembled = assembled[:, :total_samples]
                elif assembled.shape[-1] < total_samples:
                    pad = np.zeros((num_channels, total_samples - assembled.shape[-1]), dtype=np.float32)
                    assembled = np.concatenate([assembled, pad], axis=-1)

                _save_audio_np(final_wav_path, assembled, parent_sr)

                # Also mirror aliases
                try:
                    alt_wav = str(gen_dir / f"song_{repair_job_id}.wav")
                    if os.path.abspath(final_wav_path) != os.path.abspath(alt_wav):
                        shutil.copy2(final_wav_path, alt_wav)
                    backend_dir = get_repo_root() / "backend" / "generated_audio"
                    backend_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(final_wav_path, str(backend_dir / f"{repair_job_id}.wav"))
                    shutil.copy2(final_wav_path, str(backend_dir / f"song_{repair_job_id}.wav"))
                except Exception as _e:
                    logger.debug("Failed mirroring audio aliases: %s", _e)

            await loop.run_in_executor(None, _splice_audio)

            if abort_event.is_set():
                raise asyncio.CancelledError("Repair cancelled by user")

            # 6. Step 2: Neural stem separation (BS-Roformer 6-stem)
            event_manager.publish("job_progress", {
                "job_id": repair_job_id,
                "step": 2,
                "total_steps": 4,
                "phase": "stems",
                "progress": 60,
                "message": "Separating repaired master stems (BS-Roformer 6-stem)...",
            })

            real_stems = {}
            stems_dir = str(gen_dir / "stems")
            os.makedirs(stems_dir, exist_ok=True)
            try:
                separation_res = await loop.run_in_executor(
                    None, separate_sources, final_wav_path, stems_dir, repair_job_id, 1
                )
                if hasattr(separation_res, "stems"):
                    real_stems = dict(separation_res.stems)
                elif isinstance(separation_res, dict):
                    real_stems = dict(separation_res)
            except Exception as e:
                logger.warning(f"Stem separation failed for repair job {repair_job_id}: {e}")
            finally:
                try:
                    unload_model()
                    GlobalHardwareCoordinator.flush_memory()
                except Exception:
                    pass

            if abort_event.is_set():
                raise asyncio.CancelledError("Repair cancelled by user")

            # 7. Step 3: MuScriptor Neural Transcription
            event_manager.publish("job_progress", {
                "job_id": repair_job_id,
                "step": 3,
                "total_steps": 4,
                "phase": "transcription",
                "progress": 80,
                "message": "MuScriptor neural transcribing note events, chords & score...",
            })

            transcription_result = None
            try:
                transcription_result = await muscriptor_provider.transcribe(
                    audio_file_path=f"/audio/{repair_job_id}.wav",
                    job_id=repair_job_id,
                    progress_callback=lambda s, t, m: event_manager.publish("job_progress", {
                        "job_id": repair_job_id, "step": 3, "total_steps": 4, "phase": "transcription", "progress": 85, "message": m
                    })
                )
            except Exception as e:
                logger.warning(f"MuScriptor transcription failed for repair job {repair_job_id}: {e}")

            # 8. Step 4: Lyric synchronization & final DB commit
            event_manager.publish("job_progress", {
                "job_id": repair_job_id,
                "step": 4,
                "total_steps": 4,
                "phase": "lyrics",
                "progress": 95,
                "message": "Aligning timed lyrics and finalizing database records...",
            })

            vocal_stem_candidate = real_stems.get("vocals", "") or final_wav_path
            timed_lyrics = None
            try:
                timed_lyrics = lyric_sync_engine.align_lyrics(
                    lyrics=parent_lyrics or "",
                    duration_sec=total_duration_sec,
                    vocal_stem_path=vocal_stem_candidate,
                )
            except Exception as e:
                logger.warning(f"Lyric sync failed for repair job {repair_job_id}: {e}")

            repair_uuid = uuid.UUID(repair_job_id) if isinstance(repair_job_id, str) and "-" in repair_job_id else repair_job_id
            with Session(db_engine) as session:
                job = session.get(Job, repair_uuid)
                if not job:
                    job = session.exec(select(Job).where(Job.id == repair_uuid)).one_or_none()
                if job:
                    job.status = JobStatus.COMPLETED
                    job.audio_path = f"/audio/{repair_job_id}.wav"
                    job.duration_ms = int(total_duration_sec * 1000)
                    if real_stems:
                        job.stems_json = json.dumps(real_stems)
                    if timed_lyrics:
                        job.timed_lyrics_json = json.dumps(timed_lyrics)
                    if transcription_result:
                        job.midi_path = transcription_result.midi_path
                        job.musicxml_path = transcription_result.musicxml_path
                        job.notes_json = json.dumps(transcription_result.notes)
                    session.add(job)
                    session.commit()

            event_manager.publish("job_update", {
                "job_id": repair_job_id,
                "status": "completed",
                "audio_path": f"/audio/{repair_job_id}.wav",
                "duration_ms": int(total_duration_sec * 1000),
            })
            logger.info(f"Segment repair job {repair_job_id} successfully completed.")

        except asyncio.CancelledError:
            logger.info(f"Segment repair job {repair_job_id} cancelled.")
            repair_uuid = uuid.UUID(repair_job_id) if isinstance(repair_job_id, str) and "-" in repair_job_id else repair_job_id
            with Session(db_engine) as session:
                job = session.get(Job, repair_uuid)
                if not job:
                    job = session.exec(select(Job).where(Job.id == repair_uuid)).one_or_none()
                if job:
                    job.status = JobStatus.FAILED
                    job.error_msg = "Cancelled by user"
                    session.add(job)
                    session.commit()
            event_manager.publish("job_update", {"job_id": repair_job_id, "status": "failed", "error": "Cancelled"})

        except Exception as e:
            logger.error(f"Segment repair job {repair_job_id} failed: {e}", exc_info=True)
            repair_uuid = uuid.UUID(repair_job_id) if isinstance(repair_job_id, str) and "-" in repair_job_id else repair_job_id
            with Session(db_engine) as session:
                job = session.get(Job, repair_uuid)
                if not job:
                    job = session.exec(select(Job).where(Job.id == repair_uuid)).one_or_none()
                if job:
                    job.status = JobStatus.FAILED
                    job.error_msg = str(e)
                    session.add(job)
                    session.commit()
            event_manager.publish("job_update", {"job_id": repair_job_id, "status": "failed", "error": str(e)})

        finally:
            music_service.job_started_monotonic.pop(repair_job_id, None)
            music_service.active_jobs.pop(repair_job_id, None)


inpainting_service = InpaintingService()
