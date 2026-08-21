"""
End-to-End Orchestration Pipeline.
Coordinates: Model Generation -> Stem Separation -> MuScriptor Transcription -> Voice Conversion -> Smart History.
"""

import os
import json
import asyncio
import logging
from dataclasses import asdict
from typing import Optional, Dict, Any, Callable
from uuid import UUID
from sqlmodel import Session, select

from app.models import Job, JobStatus, GenerationRequest
from app.providers.registry import provider_registry
from app.transcription.muscriptor_provider import muscriptor_provider
from app.transcription.real_separator import separate_sources, unload_model
from app.transcription.instrument_stems import render_instrument_parts
from app.transcription.karaoke import lyric_sync_engine
from app.services.producer_service import extract_final_lyrics
from app.services.voice_service import voice_service

logger = logging.getLogger(__name__)


class GenerateAndTranscribePipeline:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GenerateAndTranscribePipeline, cls).__new__(cls)
        return cls._instance

    async def run(
        self,
        job_id: UUID,
        req: GenerationRequest,
        engine: Any,
        event_manager: Any,
        cancel_event: Optional[Any] = None
    ):
        """
        Execute the full production pipeline for a generation request.
        """
        job_id_str = str(job_id)
        logger.info(f"Starting orchestration pipeline for job {job_id_str}")

        # Update DB status: PROCESSING
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if not job:
                return
            job.status = JobStatus.PROCESSING
            session.add(job)
            session.commit()

        # Step 1: Model Generation
        try:
            event_manager.publish("job_update", {
                "job_id": job_id_str,
                "status": "processing"
            })
            event_manager.publish("job_progress", {
                "job_id": job_id_str,
                "step": 1,
                "total_steps": 4,
                "phase": "generation",
                "progress": 15,
                "message": f"Synthesizing track with {req.model_provider or 'MiniMax Music 3'}..."
            })

            provider = provider_registry.get_provider(req.model_provider)

            def _gen_progress(step: int, total: int, msg: str):
                prog_pct = min(40, int(15 + (step / max(1, total)) * 25))
                event_manager.publish("job_progress", {
                    "job_id": job_id_str,
                    "step": 1,
                    "total_steps": 4,
                    "phase": "generation",
                    "progress": prog_pct,
                    "message": msg
                })

            gen_result = await provider.generate(
                job_id=job_id_str,
                prompt=req.prompt,
                lyrics=req.lyrics,
                duration_ms=req.duration_ms,
                tags=req.tags,
                seed=req.seed,
                temperature=req.temperature,
                cfg_scale=req.cfg_scale,
                topk=req.topk,
                llm_model=req.llm_model,
                progress_callback=_gen_progress,
                cancel_event=cancel_event
            )

            # Update DB with audio path (and producer-enhanced inputs).
            with Session(engine) as session:
                job = session.get(Job, job_id)
                if job:
                    job.audio_path = gen_result.audio_path
                    if gen_result.structured_caption:
                        job.structured_caption_json = json.dumps(gen_result.structured_caption)
                    meta = gen_result.metadata or {}
                    # The producer may have enhanced a weak prompt / written real
                    # lyrics; surface those on the Job so the UI shows what was
                    # actually generated (the user's own inputs are preserved).
                    eff_lyrics = (meta.get("effective_lyrics") or "").strip()
                    eff_prompt = (meta.get("effective_prompt") or "").strip()
                    eff_tags = (meta.get("effective_tags") or "").strip()
                    if eff_lyrics:
                        # Persist the final, clean song (strip the Co-Writer's
                        # internal reasoning/thinking that leaks into raw output).
                        job.lyrics = extract_final_lyrics(eff_lyrics) or eff_lyrics
                    if eff_prompt:
                        job.prompt = eff_prompt
                    if eff_tags:
                        job.tags = eff_tags
                    session.add(job)
                    session.commit()

            # Step 2: Real Neural Source Separation (Demucs / HTDemucs).
            # Separates the ACTUAL generated master into genuine audio stems
            # (vocals, drums, bass, other) — real separated audio, not DSP
            # filter banks and never synthesized oscillators. Runs in a worker
            # thread because HTDemucs inference is CPU/GPU-heavy.
            event_manager.publish("job_progress", {
                "job_id": job_id_str,
                "step": 2,
                "total_steps": 4,
                "phase": "stems",
                "progress": 50,
                "message": "Real source separation (HTDemucs): vocals, drums, bass, other..."
            })

            local_master = gen_result.audio_path.replace("/audio/", "generated_audio/")

            # Run real neural separation off the event loop; if it ever fails
            # (heavy model load, missing weights, resource pressure) the job must
            # NOT die — we degrade to the per-instrument (MuScriptor) stems which
            # are rendered later, and the DAW falls back to whichever stem set is
            # available. Production-robust: separation is an enhancement, never a
            # single point of failure for the whole pipeline.
            real_stems: dict[str, str] = {}
            try:
                loop = asyncio.get_running_loop()
                real_stems = await loop.run_in_executor(
                    None, separate_sources, local_master,
                    "generated_audio/stems", job_id_str, 1,
                )
            except Exception as e:
                logger.warning(
                    f"HTDemucs separation failed for {job_id_str} ({e}); "
                    "continuing with per-instrument (MuScriptor) stems."
                )
                event_manager.publish("job_progress", {
                    "job_id": job_id_str,
                    "step": 2,
                    "total_steps": 4,
                    "phase": "stems",
                    "progress": 50,
                    "message": "Separation unavailable; using MuScriptor per-instrument parts.",
                })

            # Release the HTDemucs (torch) model so it isn't resident alongside the
            # MiniMax MLX model between generations — keeps memory footprint low.
            try:
                unload_model()
            except Exception as _u:
                logger.debug(f"HTDemucs unload skipped: {_u}")

            # Step 3: Optional Voice Identity Conversion (on the REAL vocal stem)
            final_vocal_path = real_stems.get("vocals", "")
            if req.voice_profile_id:
                event_manager.publish("job_progress", {
                    "job_id": job_id_str,
                    "step": 3,
                    "total_steps": 4,
                    "phase": "voice_conversion",
                    "progress": 70,
                    "message": f"Applying Voice Identity '{req.voice_profile_id}' to vocal stem..."
                })
                final_vocal_path = await voice_service.convert_vocals(
                    vocal_stem_path=real_stems.get("vocals", ""),
                    profile_id=req.voice_profile_id,
                    job_id=job_id_str
                )

            # Step 4: MuScriptor Automatic Music Transcription
            event_manager.publish("job_progress", {
                "job_id": job_id_str,
                "step": 4,
                "total_steps": 4,
                "phase": "transcription",
                "progress": 80,
                "message": "MuScriptor neural transcribing to MIDI, MusicXML & note events..."
            })

            transcription_result = await muscriptor_provider.transcribe(
                audio_file_path=gen_result.audio_path,
                job_id=job_id_str,
                progress_callback=lambda s, t, m: event_manager.publish("job_progress", {
                    "job_id": job_id_str, "step": 4, "total_steps": 4, "phase": "transcription", "progress": 90, "message": m
                })
            )

            # Generate Acoustic Lyric Sync / Karaoke
            effective_lyrics = ""
            with Session(engine) as session:
                j = session.get(Job, job_id)
                if j and j.lyrics:
                    effective_lyrics = j.lyrics
            if not effective_lyrics:
                effective_lyrics = req.lyrics or ""

            vocal_stem_candidate = final_vocal_path or real_stems.get("vocals", "") or local_master
            timed_lyrics = lyric_sync_engine.align_lyrics(
                lyrics=effective_lyrics,
                duration_sec=gen_result.duration_sec,
                vocal_stem_path=vocal_stem_candidate
            )

            # Dual-engine stems: real Demucs 4-master stems (neural separation
            # of the actual audio) AND optional MuScriptor-derived per-instrument
            # parts (one stem per distinct instrument in the transcription, with
            # its General MIDI program). The DAW lets the user choose which
            # source to view/hear; both are stored so nothing is lost.
            instrument_parts: dict[str, str] = {}
            instrument_programs: dict[str, int] = {}
            try:
                instrument_parts, instrument_programs = render_instrument_parts(
                    transcription_result.notes,
                    job_id_str,
                    duration_sec=gen_result.duration_sec,
                )
            except Exception as e:  # never let per-instrument rendering sink the job
                logger.warning(f"Per-instrument stem rendering skipped for {job_id_str}: {e}")

            # Finalize DB Record
            with Session(engine) as session:
                job = session.get(Job, job_id)
                if job:
                    job.status = JobStatus.COMPLETED
                    job.audio_path = gen_result.audio_path
                    job.midi_path = transcription_result.midi_path
                    job.musicxml_path = transcription_result.musicxml_path
                    job.notes_json = json.dumps(transcription_result.notes)
                    job.beat_grid_json = json.dumps(transcription_result.beat_grid)
                    # REAL separated stems (HTDemucs) + the transcription-driven
                    # DAW sources. MuScriptor remains the MIDI/notation engine;
                    # audio stems are the genuine separated audio.
                    job.stems_json = json.dumps({
                        "vocals": final_vocal_path or real_stems.get("vocals", ""),
                        "drums": real_stems.get("drums", ""),
                        "bass": real_stems.get("bass", ""),
                        "other": real_stems.get("other", ""),
                        "stems_source": "htdemucs",
                        # Per-instrument (MuScriptor-derived) stem set. The DAW
                        # can switch between "4 master stems" and these parts.
                        "instrumental_parts": instrument_parts,
                        "instrument_programs": instrument_programs,
                        "sources_available": ["muscriptor", "htdemucs"],
                        "default_source": "muscriptor",
                    })
                    job.timed_lyrics_json = json.dumps(timed_lyrics)
                    session.add(job)
                    session.commit()

            # Emit final completion event
            event_manager.publish("job_update", {
                "job_id": job_id_str,
                "status": "completed",
                "audio_path": gen_result.audio_path,
                "title": getattr(gen_result, 'title', None) or req.prompt
            })
            event_manager.publish("job_progress", {
                "job_id": job_id_str,
                "step": 4,
                "total_steps": 4,
                "phase": "done",
                "progress": 100,
                "message": "Track generation, stems, and MuScriptor transcription complete!"
            })
            logger.info(f"Pipeline completed successfully for job {job_id_str}")

        except asyncio.CancelledError:
            logger.info(f"Pipeline cancelled for job {job_id_str}")
            with Session(engine) as session:
                job = session.get(Job, job_id)
                if job:
                    job.status = JobStatus.FAILED
                    job.error_msg = "Cancelled by user"
                    session.add(job)
                    session.commit()
            event_manager.publish("job_update", {"job_id": job_id_str, "status": "failed", "error": "Cancelled by user"})

        except Exception as e:
            logger.error(f"Pipeline failed for job {job_id_str}: {e}", exc_info=True)
            with Session(engine) as session:
                job = session.get(Job, job_id)
                if job:
                    job.status = JobStatus.FAILED
                    job.error_msg = str(e)
                    session.add(job)
                    session.commit()
            event_manager.publish("job_update", {"job_id": job_id_str, "status": "failed", "error": str(e)})


pipeline_orchestrator = GenerateAndTranscribePipeline()
