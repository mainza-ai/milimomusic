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
from app.services.music_service import music_service
from app.transcription.muscriptor_provider import muscriptor_provider
from app.transcription.real_separator import separate_sources, unload_model
from app.transcription.instrument_stems import render_instrument_parts
from app.transcription.karaoke import lyric_sync_engine
from app.services.producer_service import extract_final_lyrics
from app.services.voice_service import voice_service
from app.core.hardware_lock import GlobalHardwareCoordinator
from app.core import provenance as provenance_util
from app.core.recovery import touch_heartbeat as provenance_util_touch

logger = logging.getLogger(__name__)


def _job_status(engine, job_id) -> Optional[str]:
    """Fresh DB read — the single source of truth for lifecycle decisions."""
    with Session(engine) as session:
        job = session.get(Job, job_id)
        return job.status.value if job else None


def _abort_if_terminal(engine, job_id, cancel_event=None, stage: str = "") -> None:
    """Raise CancelledError when the job is cancelled/failed/interrupted or the
    cancel event fired. Called BETWEEN heavy steps so orphaned work can never
    resurrect a dead job or burn GPU after external termination.

    A missing row is terminal too: `DELETE /jobs/{id}` removes the row while the
    pipeline is still running, and the old code treated `None` as "keep going"
    and wrote stems/MIDI for a job that no longer existed.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise asyncio.CancelledError("Cancelled by request")
    status = _job_status(engine, job_id)
    terminal = {
        JobStatus.FAILED.value,
        JobStatus.CANCELLED.value,
        JobStatus.INTERRUPTED.value,
        None,
    }
    if status in terminal:
        raise asyncio.CancelledError(
            f"Job already terminal ({status}); aborting {stage} stage")


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
        The GPU lock is acquired for the WHOLE run — generation, separation and
        transcription all contend for the same device (previously only
        inpainting took this lock, so concurrent runs raced unsynchronized).
        """
        job_id_str = str(job_id)
        logger.info(f"Starting orchestration pipeline for job {job_id_str}")
        await GlobalHardwareCoordinator.acquire_device("Audio Pipeline", modality="audio_gen")
        try:
            async with music_service.gpu_lock:
                return await self._run_locked(job_id, req, engine, event_manager, cancel_event)
        finally:
            GlobalHardwareCoordinator.release_device("Audio Pipeline", modality="audio_gen")

    async def _run_locked(
        self,
        job_id: UUID,
        req: GenerationRequest,
        engine: Any,
        event_manager: Any,
        cancel_event: Optional[Any] = None
    ):
        job_id_str = str(job_id)

        # Atomic claim: only a job that is still QUEUED may start. The old code
        # wrote PROCESSING unconditionally, so cancelling a queued job marked it
        # FAILED and the background task then flipped it back to PROCESSING and
        # ran to completion — the user's cancel visibly "un-cancelled".
        from sqlalchemy import update as _sql_update
        with Session(engine) as session:
            claim = session.execute(
                _sql_update(Job)
                .where(Job.id == job_id, Job.status == JobStatus.QUEUED)
                .values(status=JobStatus.PROCESSING)
            )
            session.commit()
            if claim.rowcount == 0:
                logger.info(f"Job {job_id_str} is no longer QUEUED; skipping pipeline start.")
                return

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
            _abort_if_terminal(engine, job_id, cancel_event, stage="pre-generation")

            _last_beat = [0.0]

            def _beat():
                """Liveness for the stall sweeper (throttled to 10s)."""
                import time as _t
                now = _t.monotonic()
                if now - _last_beat[0] >= 10.0:
                    _last_beat[0] = now
                    provenance_util_touch(engine, job_id)

            def _gen_progress(step: int, total: int, msg: str):
                _beat()
                prog_pct = min(40, int(15 + (step / max(1, total)) * 25))
                event_manager.publish("job_progress", {
                    "job_id": job_id_str,
                    "step": 1,
                    "total_steps": 4,
                    "phase": "generation",
                    "progress": prog_pct,
                    "message": msg
                })

            extra_gen_kwargs = {}
            if getattr(req, "is_cover", False):
                extra_gen_kwargs.update({
                    "ref_audio_path": req.ref_audio_path,
                    "melody_midi_path": req.melody_midi_path,
                    "chord_midi_path": req.chord_midi_path,
                    "drum_midi_path": req.drum_midi_path,
                    "bpm": req.bpm,
                    "transcription_engine": getattr(req, "transcription_engine", "milimo_neural"),
                })

            if getattr(req, "is_extension", False) and getattr(req, "parent_audio_path", None):
                # Only call extend() on a provider that can genuinely continue
                # audio. Every provider inherits the base class, so `hasattr`
                # was always true and four of them generated a fresh song.
                _caps = provider.get_capabilities()
                if not _caps.supports_track_extension:
                    raise RuntimeError(
                        f"{_caps.display_name} cannot extend existing audio. "
                        "Choose a provider with track-extension support (MiniMax Music 3)."
                    )
                gen_result = await provider.extend(
                    job_id=job_id_str,
                    parent_audio_path=req.parent_audio_path,
                    extend_ms=req.duration_ms,
                    lyrics=req.lyrics,
                    prompt=req.prompt,
                    extend_from_sec=getattr(req, "extend_from_sec", None),
                    crossfade_sec=getattr(req, "crossfade_sec", 1.5),
                    seed=req.seed,
                    tags=req.tags,
                    temperature=getattr(req, "temperature", 1.0),
                    cfg_scale=getattr(req, "cfg_scale", 1.5),
                    topk=getattr(req, "topk", 50),
                    llm_model=getattr(req, "llm_model", None),
                    structured_caption=req.structured_caption or None,
                    progress_callback=_gen_progress,
                    cancel_event=cancel_event,
                    beat_grid=getattr(req, "beat_grid", None),
                    target_duration_sec=getattr(req, "target_duration_sec", None),
                    parent_prompt=getattr(req, "parent_prompt", None),
                    parent_lyrics=getattr(req, "parent_lyrics", None),
                    parent_structured_caption=getattr(req, "parent_structured_caption", None),
                    parent_job_id=getattr(req, "parent_job_id", None),
                    tail_lyrics=getattr(req, "tail_lyrics", None),
                    **extra_gen_kwargs
                )
            else:
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
                    cancel_event=cancel_event,
                    structured_caption=req.structured_caption or None,
                    **extra_gen_kwargs
                )

            # Orphan-work guard: the blocking inference thread cannot be
            # interrupted mid-call — discard its output if we woke up dead.
            _abort_if_terminal(engine, job_id, cancel_event, stage="post-generation")

            # Update DB with audio path (and producer-enhanced inputs).
            with Session(engine) as session:
                job = session.get(Job, job_id)
                if job:
                    job.audio_path = gen_result.audio_path
                    if gen_result.duration_sec:
                        job.duration_ms = int(gen_result.duration_sec * 1000)
                    if gen_result.structured_caption:
                        job.structured_caption_json = json.dumps(gen_result.structured_caption)
                    # Generation provenance: persist whether real MiniMax inference
                    # produced the audio or the procedural fallback synth did, plus
                    # the reason — surfaced to the UI so fallbacks are never silent.
                    job.used_fallback_synth = gen_result.used_fallback_synth
                    job.fallback_reason = gen_result.fallback_reason
                    provenance_util.merge_stage(job, gen_result.provenance())
                    meta = gen_result.metadata or {}
                    if getattr(req, "is_extension", False) and meta.get("extension_mode"):
                        job.extension_mode = meta["extension_mode"]
                    if meta.get("symbolic_midi"):
                        sym = meta["symbolic_midi"]
                        if isinstance(sym, dict):
                            job.melody_midi_path = sym.get("melody")
                            job.chord_midi_path = sym.get("chord")
                            job.drum_midi_path = sym.get("drums")
                    if meta.get("bpm"):
                        job.bpm = meta["bpm"]
                    # EXACT model conditioning, stored verbatim and NEVER
                    # lossy-transformed: a later extension replays from these
                    # strings byte-identically instead of reconstructing (and
                    # silently diverging from) the parent run's inputs.
                    # (job.lyrics/prompt below stay human-readable for UI/sync.)
                    if meta.get("model_caption"):
                        job.model_prompt = meta["model_caption"]
                    if meta.get("model_lyrics") is not None:
                        job.model_lyrics = meta["model_lyrics"]
                    if getattr(req, "is_cover", False):
                        job.is_cover = True
                        job.cover_mode = "audio_reference" if req.ref_audio_path else "symbolic_midi"
                    # The producer may have enhanced a weak prompt / written real
                    # lyrics; surface those on the Job so the UI shows what was
                    # actually generated (the user's own inputs are preserved).
                    eff_lyrics = (meta.get("effective_lyrics") or "").strip()
                    eff_prompt = (meta.get("effective_prompt") or "").strip()
                    eff_tags = (meta.get("effective_tags") or "").strip()
                    if eff_lyrics and not getattr(req, "is_extension", False):
                        # Persist the final, clean song (strip the Co-Writer's
                        # internal reasoning/thinking that leaks into raw output).
                        job.lyrics = extract_final_lyrics(eff_lyrics) or eff_lyrics
                    if eff_prompt:
                        job.prompt = eff_prompt
                    if eff_tags:
                        job.tags = eff_tags
                    session.add(job)
                    session.commit()

            # Immediate Eviction of Generation Model weights to maximize VRAM for Separation
            try:
                if hasattr(provider, "unload"):
                    provider.unload()
                GlobalHardwareCoordinator.flush_memory()
            except Exception as _u:
                logger.debug(f"Generation provider unload skipped: {_u}")

            # Step 2: SOTA Neural Source Separation (BS-Roformer / MelBand-Roformer).

            # Separates the ACTUAL generated master into genuine audio stems
            # (vocals, drums, bass, guitar, piano, other) — real separated audio,
            # never synthesized oscillators. Runs in a worker thread.
            event_manager.publish("job_progress", {
                "job_id": job_id_str,
                "step": 2,
                "total_steps": 4,
                "phase": "stems",
                "progress": 50,
                "message": "Separating the master into neural stems (drums, bass, other, vocals)..."
            })

            from app.transcription.karaoke import _resolve_audio_file
            from app.core.paths import get_generated_audio_dir

            local_master = _resolve_audio_file(gen_result.audio_path)
            if not local_master or not os.path.exists(local_master):
                local_master = str(get_generated_audio_dir() / os.path.basename(gen_result.audio_path))

            # Run real neural separation off the event loop; if it ever fails
            # (heavy model load, missing weights, resource pressure) the job must
            # NOT die — we degrade to the per-instrument (MuScriptor) stems which
            # are rendered later, and the DAW falls back to whichever stem set is
            # available. Production-robust: separation is an enhancement, never a
            # single point of failure for the whole pipeline.
            real_stems: dict[str, str] = {}
            stems_source_id = "bs_roformer_6stem"
            stems_dir = str(get_generated_audio_dir() / "stems")
            os.makedirs(stems_dir, exist_ok=True)
            try:
                loop = asyncio.get_running_loop()
                separation_res = await loop.run_in_executor(
                    None, separate_sources, local_master,
                    stems_dir, job_id_str, 1,
                )
                if hasattr(separation_res, "stems"):
                    real_stems = dict(separation_res.stems)
                    stems_source_id = getattr(separation_res, "source_id", "unknown")
                elif isinstance(separation_res, dict):
                    real_stems = dict(separation_res)
                    stems_source_id = "unknown"
                with Session(engine) as _s:
                    _j = _s.get(Job, job_id)
                    if _j:
                        provenance_util.merge_stage(_j, {
                            "stage": "separation",
                            "source": "model" if real_stems else "placeholder",
                            "degraded": not bool(real_stems),
                            "reason": None if real_stems else "separator returned no stems",
                            "engine": stems_source_id,
                            "stems": sorted(real_stems.keys()),
                        })
                        _s.add(_j)
                        _s.commit()
            except Exception as e:
                logger.warning(
                    f"Neural separation failed for {job_id_str} ({e}); "
                    "continuing with per-instrument (MuScriptor) stems."
                )
                event_manager.publish("job_progress", {
                    "job_id": job_id_str,
                    "step": 2,
                    "total_steps": 4,
                    "phase": "stems",
                    "progress": 50,
                    "message": "Neural separation unavailable; using MuScriptor per-instrument parts.",
                })
                with Session(engine) as _s:
                    _j = _s.get(Job, job_id)
                    if _j:
                        provenance_util.merge_stage(_j, {
                            "stage": "separation", "source": "placeholder", "degraded": True,
                            "reason": f"separator failed: {str(e)[:160]}", "stems": [],
                        })
                        _s.add(_j)
                        _s.commit()

            # Release the separation model from memory
            try:
                unload_model()
                GlobalHardwareCoordinator.flush_memory()
            except Exception as _u:
                logger.debug(f"Separation unload skipped: {_u}")

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
                # Voice conversion is an enhancement, never a single point of
                # failure: separation is already optional and may leave no vocal
                # stem, which used to raise FileNotFoundError and fail a job whose
                # generation had succeeded.
                try:
                    final_vocal_path = await voice_service.convert_vocals(
                        vocal_stem_path=real_stems.get("vocals", ""),
                        profile_id=req.voice_profile_id,
                        job_id=job_id_str
                    )
                except Exception as _vc_err:
                    final_vocal_path = ""
                    logger.warning(
                        f"Voice conversion skipped for {job_id_str}: {_vc_err}. "
                        "Continuing with the original vocal."
                    )
                    with Session(engine) as _s:
                        _j = _s.get(Job, job_id)
                        if _j:
                            provenance_util.merge_stage(_j, {
                                "stage": "voice",
                                "source": "placeholder",
                                "degraded": True,
                                "reason": str(_vc_err)[:200],
                                "method": "failed",
                                "profile_id": req.voice_profile_id,
                            })
                            _s.add(_j)
                            _s.commit()

            # If voice conversion occurred, remix final master track audio with converted vocals
            final_master_path = gen_result.audio_path
            if req.voice_profile_id and final_vocal_path:
                with Session(engine) as _s:
                    _j = _s.get(Job, job_id)
                    if _j:
                        provenance_util.merge_stage(_j, getattr(voice_service, "last_conversion", None))
                        _s.add(_j)
                        _s.commit()
                try:
                    final_master_path = voice_service.remix_master_with_vocal(
                        original_audio_path=gen_result.audio_path,
                        converted_vocal_path=final_vocal_path,
                        stems_dict=real_stems,
                        output_filename=f"{job_id_str}_remixed_master.wav"
                    )
                except Exception as e:
                    logger.warning(f"Master track remix with converted vocal failed: {e}. Keeping gen_result.audio_path.")

            # Step 4: MuScriptor Automatic Music Transcription
            event_manager.publish("job_progress", {
                "job_id": job_id_str,
                "step": 4,
                "total_steps": 4,
                "phase": "transcription",
                "progress": 80,
                "message": "MuScriptor neural transcribing to MIDI, MusicXML & note events..."
            })

            # Transcribe what will actually be delivered: step 3 may have replaced
            # the vocal stem and the finalize step swaps the master to the remix, so
            # transcribing `gen_result.audio_path` produced MIDI for audio the user
            # never receives.
            _transcription_target = final_master_path or gen_result.audio_path
            transcription_result = await muscriptor_provider.transcribe(
                audio_file_path=_transcription_target,
                job_id=job_id_str,
                progress_callback=lambda s, t, m: event_manager.publish("job_progress", {
                    "job_id": job_id_str, "step": 4, "total_steps": 4, "phase": "transcription", "progress": 90, "message": m
                }),
                cancel_event=cancel_event,
            )
            GlobalHardwareCoordinator.flush_memory()

            # Generate Acoustic Lyric Sync / Karaoke
            effective_lyrics = ""
            with Session(engine) as session:
                j = session.get(Job, job_id)
                if j and j.lyrics:
                    effective_lyrics = j.lyrics
            if not effective_lyrics:
                effective_lyrics = req.lyrics or ""

            vocal_stem_candidate = final_vocal_path or real_stems.get("vocals", "") or local_master
            # MMS_FA model load + full-file forward pass: blocking, and it used to
            # stall the event loop (SSE heartbeats, polling, cancel requests).
            timed_lyrics = await asyncio.to_thread(
                lyric_sync_engine.align_lyrics,
                lyrics=effective_lyrics,
                duration_sec=gen_result.duration_sec,
                vocal_stem_path=vocal_stem_candidate,
            )

            # Dual-engine stems: real Demucs 4-master stems (neural separation
            # of the actual audio) AND optional MuScriptor-derived per-instrument
            # parts (one stem per distinct instrument in the transcription, with
            # its General MIDI program). The DAW lets the user choose which
            # source to view/hear; both are stored so nothing is lost.
            instrument_parts: dict[str, str] = {}
            instrument_programs: dict[str, int] = {}
            try:
                instrument_parts, instrument_programs = await asyncio.to_thread(
                    render_instrument_parts,
                    transcription_result.notes,
                    job_id_str,
                    duration_sec=gen_result.duration_sec,
                )
            except Exception as e:  # never let per-instrument rendering sink the job
                logger.warning(f"Per-instrument stem rendering skipped for {job_id_str}: {e}")

            # Release transcription model from memory
            try:
                muscriptor_provider.unload()
                GlobalHardwareCoordinator.flush_memory()
            except Exception as _u:
                logger.debug(f"Transcription unload skipped: {_u}")

            _abort_if_terminal(engine, job_id, cancel_event, stage="finalize")

            # Auto-generate album artwork if enabled and no manual cover is set
            should_auto_cover = getattr(req, "auto_generate_cover", True)
            final_cover_path = getattr(req, "cover_image_path", None)
            if should_auto_cover and not final_cover_path:
                try:
                    from app.services.image_service import image_service
                    from app.services.llm_service import LLMService
                    # generate_cover_prompt never raises: worst case is a flagged
                    # fallback, and the cover must never fail the music job.
                    cover_prompt_res = LLMService.generate_cover_prompt(
                        title=getattr(req, "title", None),
                        tags=getattr(req, "tags", None),
                        lyrics=getattr(req, "lyrics", None),
                    )
                    cover_prompt = cover_prompt_res["prompt"]
                    if not cover_prompt_res["llm_used"]:
                        logger.warning(f"Job {job_id_str} auto-cover uses generic prompt (LLM unavailable).")
                    cover_style = req.tags or "cinematic album cover"
                    cover_model_id = getattr(req, "cover_image_model_id", None) or image_service.get_default_image_model()
                    cover_res = image_service.generate_cover(
                        prompt=cover_prompt,
                        style=cover_style,
                        aspect_ratio="1:1",
                        model_id=cover_model_id,
                        title=(getattr(req, "title", None) or "").strip() or None,
                    )
                    final_cover_path = cover_res.get("url")
                    logger.info(f"Auto-generated album cover for job {job_id_str}: {final_cover_path}")
                    # Immediately unload image diffusion models
                    try:
                        image_service.unload_models()
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"Auto-cover generation skipped for {job_id_str}: {e}")


            # Finalize DB Record
            _tail_ok_out = None
            _rv_out = None
            with Session(engine) as session:
                job = session.get(Job, job_id)
                if job:
                    job.status = JobStatus.COMPLETED
                    job.audio_path = final_master_path
                    if final_cover_path and not job.cover_image_path:
                        job.cover_image_path = final_cover_path
                    # Never store fabricated notation: an unavailable transcription
                    # leaves these empty and records why (plan 1.3).
                    job.midi_path = transcription_result.midi_path or None
                    job.musicxml_path = transcription_result.musicxml_path or None
                    job.notes_json = json.dumps(transcription_result.notes) if transcription_result.available else None
                    if transcription_result.beat_grid:
                        job.beat_grid_json = json.dumps(transcription_result.beat_grid)
                    provenance_util.merge_stage(job, transcription_result.provenance())
                    # REAL separated stems (BS-Roformer) + the transcription-driven
                    # DAW sources. MuScriptor remains the MIDI/notation engine;
                    # audio stems are the genuine separated audio.
                    dynamic_stems_payload = {
                        "stems_source": stems_source_id,
                        "instrumental_parts": instrument_parts,
                        "instrument_programs": instrument_programs,
                        "sources_available": [stems_source_id, "muscriptor"],
                        "default_source": stems_source_id if real_stems else "muscriptor",
                    }
                    # Merge all real neural stems dynamically (vocals, drums, bass, guitar, piano, other, etc.)
                    for stem_k, stem_v in real_stems.items():
                        dynamic_stems_payload[stem_k] = stem_v
                    if final_vocal_path:
                        dynamic_stems_payload["vocals"] = final_vocal_path

                    job.stems_json = json.dumps(dynamic_stems_payload)
                    job.timed_lyrics_json = json.dumps(timed_lyrics)
                    # Extensions: verify the tail actually sings the unsung
                    # remainder (post-hoc, never blocks completion).
                    try:
                        if getattr(req, "is_extension", False) and getattr(req, "parent_job_id", None):
                            from app.services.lyrics_utils import verify_extension_tail
                            _parent = session.get(Job, UUID(str(req.parent_job_id)))
                            _parent_timed = _parent.timed_lyrics_json if _parent else None
                            _ok, _detail = verify_extension_tail(
                                _parent_timed,
                                job.timed_lyrics_json,
                                float(getattr(req, "extend_from_sec", 0.0) or 0.0),
                            )
                            job.tail_lyrics_ok = _ok
                            if _ok is False:
                                job.fallback_reason = (
                                    (job.fallback_reason + " | " if job.fallback_reason else "")
                                    + f"Tail lyric mismatch: {_detail}"
                                )
                                logger.warning(f"Extension {job_id_str} tail check FAILED: {_detail}")
                            elif _ok is True:
                                logger.info(f"Extension {job_id_str} tail check passed: {_detail}")
                    except Exception as e:
                        logger.warning(f"Extension tail verification skipped for {job_id_str}: {e}")
                    # Persist replay fidelity (full-timeline renders only): a
                    # diverged replay used to sail through silently because the
                    # splice masks the head. None = not a full render / unchecked.
                    try:
                        _rv = (gen_result.metadata or {}).get("replay_verified")
                        if getattr(req, "is_extension", False) and _rv is not None:
                            job.replay_verified = bool(_rv)
                            if _rv is False:
                                job.fallback_reason = (
                                    (job.fallback_reason + " | " if job.fallback_reason else "")
                                    + f"Replay fidelity check failed: {(gen_result.metadata or {}).get('replay_check_detail', 'groove mismatch')}"
                                )
                                logger.warning(f"Extension {job_id_str} replay check FAILED")
                    except Exception as e:
                        logger.warning(f"Extension replay flag persist skipped for {job_id_str}: {e}")
                    session.add(job)
                    session.commit()
                    # Capture for the SSE payload below (object detaches after commit).
                    _tail_ok_out = job.tail_lyrics_ok if getattr(req, "is_extension", False) else None
                    _rv_out = job.replay_verified if getattr(req, "is_extension", False) else None
                    _quality_out = provenance_util.git_quality(job)
                    _provenance_out = provenance_util.load(job)

            # Emit final completion event (DB-persisted master path, not the
            # pre-voice-remix generator path, so consumers never get a stale URL)
            event_manager.publish("job_update", {
                "job_id": job_id_str,
                "status": "completed",
                "audio_path": final_master_path,
                "title": getattr(gen_result, 'title', None) or req.prompt,
                "cover_image_path": final_cover_path or getattr(req, "cover_image_path", None),
                "extension_mode": (gen_result.metadata or {}).get("extension_mode") if getattr(req, "is_extension", False) else None,
                "tail_lyrics_ok": _tail_ok_out,
                "replay_verified": _rv_out,
                # Provenance verdict: the UI badges anything that is not "neural".
                "quality": _quality_out,
                "provenance": _provenance_out,
            })
            event_manager.publish("job_progress", {
                "job_id": job_id_str,
                "step": 4,
                "total_steps": 4,
                "phase": "done",
                "progress": 100,
                "message": (
                    "Track ready (degraded: "
                    + ", ".join(sorted(k for k, v in (_provenance_out or {}).items() if v.get("degraded")))
                    + ")"
                    if _quality_out and _quality_out != "neural"
                    else "Track generation, stems, and MuScriptor transcription complete!"
                )
            })
            logger.info(f"Pipeline completed successfully for job {job_id_str}")

        except asyncio.CancelledError:
            logger.info(f"Pipeline cancelled for job {job_id_str}")
            with Session(engine) as session:
                job = session.get(Job, job_id)
                if job and job.status not in (JobStatus.FAILED, JobStatus.COMPLETED):
                    job.status = JobStatus.CANCELLED
                    job.error_msg = "Cancelled by user"
                    session.add(job)
                    session.commit()
            event_manager.publish("job_update", {"job_id": job_id_str, "status": JobStatus.CANCELLED.value, "error": "Cancelled by user"})

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
