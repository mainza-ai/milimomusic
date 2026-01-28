
import asyncio
import os
import torch
import logging
from app.models import Job, JobStatus
from sqlmodel import Session, select
from heartlib import HeartMuLaGenPipeline
from app.services.music_service import music_service, event_manager # Share loaded pipeline

logger = logging.getLogger(__name__)

class InpaintingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InpaintingService, cls).__new__(cls)
        return cls._instance

    async def regenerate_segment(self, job_id: str, start_sec: float, end_sec: float, db_engine):
        """Regenerate a specific time range of an existing job."""
        
        # Borrow pipeline from MusicService
        pipeline = music_service.pipeline
        if not pipeline:
            logger.error("Pipeline not loaded. Cannot inpaint.")
            return
            
        async with music_service.gpu_lock:
            try:
                # 1. Fetch Job
                with Session(db_engine) as session:
                    parent_job = session.exec(select(Job).where(Job.id == job_id)).one_or_none()
                    if not parent_job:
                        logger.error(f"Job {job_id} not found.")
                        return
                    
                    # Create NEW Child Job for the result
                    import uuid
                    new_job_id = str(uuid.uuid4())
                    new_job = Job(
                        id=new_job_id,
                        prompt=f"Repair of {parent_job.title}",
                        status=JobStatus.PROCESSING,
                        parent_job_id=job_id,
                        duration_ms=parent_job.duration_ms, # Same duration
                        title=f"{parent_job.title} (Repaired)",
                        lyrics=parent_job.lyrics, # Copy lyrics
                        tags=parent_job.tags,     # Copy tags
                        seed=int(torch.seed() % 2147483647) # Safe 32-bit int for SQLite
                    )
                    session.add(new_job)
                    session.commit()
                    

                    # Capture safely inside session
                    parent_duration_secs = parent_job.duration_ms / 1000.0
                    parent_lyrics = parent_job.lyrics
                    parent_tags = parent_job.tags
                
                # Notify Start
                event_manager.publish("job_update", {"job_id": new_job_id, "status": "processing"})
                event_manager.publish("job_progress", {"job_id": new_job_id, "progress": 0, "msg": "Starting repair..."})

                # 2. Load Tokens
                tokens_path = os.path.join(os.getcwd(), "generated_tokens", f"{job_id}.pt")
                if not os.path.exists(tokens_path):
                    raise FileNotFoundError(f"Tokens not found at {tokens_path}")
                
                # Explicitly set weights_only=False because we are loading tensor data that might rely on pickle for some internals,
                # though usually safe with True for pure tensors. We suppress the warning.
                codes = torch.load(tokens_path, map_location=music_service.device, weights_only=False)
                
                # Fix dimensions if needed (should be [1, 8, T])
                # If loaded as [8, T], unsqueeze
                if codes.dim() == 2:
                    codes = codes.unsqueeze(0)
                
                # 3. Calculate Frames
                # HeartCodec is 12.5 Hz approx? Need to verify rate.
                # In detokenize: min_samples = duration * 12.5. So 12.5 Hz.
                fps = 12.5
                start_frame = int(start_sec * fps)
                end_frame = int(end_sec * fps)
                
                # 4. Generate Auto-Title
                import threading
                abort_event = threading.Event()
                # Register with MusicService so it can be tracked/cancelled
                music_service.active_jobs[new_job_id] = abort_event
                
                try:
                    # 4b. Load Parent Audio
                    # We need the original wav to mix the seamless segment into.
                    import torchaudio
                    audio_path = os.path.join(os.getcwd(), "generated_audio", f"song_{job_id}.mp3")
                    if not os.path.exists(audio_path):
                         raise FileNotFoundError(f"Original audio not found at {audio_path}")
                    
                    # Check device for crossfade operations
                    mix_device = "cpu" # Perform mixing on CPU to avoid VRAM issues
                    
                    parent_wav_tensor, parent_sr = torchaudio.load(audio_path)
                    # Ensure 48k or pipeline SR
                    if parent_sr != pipeline.audio_codec.config.sample_rate:
                        # Resample if needed
                        resampler = torchaudio.transforms.Resample(parent_sr, pipeline.audio_codec.config.sample_rate)
                        parent_wav_tensor = resampler(parent_wav_tensor)
                    
                    parent_wav_tensor = parent_wav_tensor.to(mix_device)
                    
                    # 5. Run In-Painting
                    loop = asyncio.get_running_loop()
                    
                    def _progress_callback(progress):
                        if abort_event.is_set():
                            raise InterruptedError("Job Cancelled")
                        pct = int(progress * 100)
                        msg = f"Repairing... {pct}%"
                        loop.call_soon_threadsafe(
                            event_manager.publish,
                            "job_progress",
                            {"job_id": new_job_id, "progress": pct, "msg": msg}
                        )

                    # Strategy: Windowed In-Painting (Context + Generation)
                    # We define a "Window" of context around the repair area.
                    # We run 'inpaint' on this window, asking it to KEEP the context and GENERATE the gap.
                    
                    # Parameters
                    CONTEXT_SEC = 8.0 
                    CROSSFADE_SEC = 0.1
                    
                    total_frames = codes.shape[-1]
                    context_frames = int(CONTEXT_SEC * 12.5)
                    
                    # Window Definition
                    win_start = max(0, start_frame - context_frames)
                    win_end = min(total_frames, end_frame + context_frames)
                    
                    # Extract Context Codes
                    window_codes = codes[:, :, win_start:win_end].clone()
                    
                    # Calculate Relative Repair Region within the Window
                    rel_start = start_frame - win_start
                    rel_end = end_frame - win_start
                    
                    print(f"[InpaintingService] Windowed In-Painting: {win_start}-{win_end} (Repairing {rel_start}-{rel_end})")


                    def _run_generate_and_mix():
                        if abort_event.is_set(): return None
                        
                        # Strategy: LM-Guided Repair (Generate New Tokens -> Splice -> Decode)
                        logger.info(f"Generating new token content for gap: {rel_start} to {rel_end}")
                        
                        # 1. Prepare History (Context up to repair start)
                        history_codes = codes[..., :start_frame].clone() 
                        
                        # Calculate duration in MS
                        gap_frames = rel_end - rel_start
                        gap_ms = int((gap_frames / 12.5) * 1000)
                        
                        history_ms = int((history_codes.shape[-1] / 12.5) * 1000)
                        total_ms = history_ms + gap_ms
                        
                        # Fix History Shape for Pipeline: [1, 8, T] -> [1, T, 8]
                        hist_in = history_codes.permute(0, 2, 1)
                        if music_service.pipeline and hasattr(music_service.pipeline, "_parallel_number"):
                            expected = music_service.pipeline._parallel_number
                            if hist_in.shape[-1] == expected - 1:
                                padding = torch.zeros((hist_in.shape[0], hist_in.shape[1], 1), device=hist_in.device, dtype=hist_in.dtype)
                                hist_in = torch.cat([hist_in, padding], dim=-1)
                        
                        # 2. Call Pipeline (Sync, Blocking)
                        # We use the ORIGINAL lyrics/tags to maintain style and voice.
                        output = music_service.pipeline(
                            {
                                "lyrics": parent_lyrics or "...", 
                                "tags": parent_tags or "pop, continuation", 
                            },
                            max_audio_length_ms=total_ms,
                            history_tokens=hist_in, 
                            temperature=0.2,   # Near-deterministic (Vocal Stability)
                            topk=30,           # Even narrower for precise matching
                            cfg_scale=1.0,     # Audio Dominance (Trust History)
                            save_path=None, 
                        )
                        
                        if output and "tokens" in output:
                            # 3. Extract New Tokens
                            generated_tokens = output["tokens"]
                            
                            # Normalize [8, T] -> [1, 8, T]
                            if generated_tokens.dim() == 2:
                                generated_tokens = generated_tokens.unsqueeze(0)
                            # Normalize [B, T, 8] -> [B, 8, T]
                            elif generated_tokens.dim() == 3 and generated_tokens.shape[1] != 8 and generated_tokens.shape[2] == 8:
                                generated_tokens = generated_tokens.permute(0, 2, 1)

                            # Remove 9th channel if present [B, 9, T]
                            if generated_tokens.shape[1] > 8:
                                generated_tokens = generated_tokens[:, :8, :]
                                
                            hist_len = history_codes.shape[-1]
                            new_content = generated_tokens[..., hist_len:]
                            
                            if new_content.shape[-1] >= gap_frames:
                                new_content = new_content[..., :gap_frames]
                            else:
                                diff = gap_frames - new_content.shape[-1]
                                if diff > 0 and new_content.shape[-1] > 0:
                                     new_content = torch.cat([new_content, new_content[..., -1:]], dim=-1) # primitive padding if really short
                            
                            # 4. Splice into Window Codes
                            new_content = new_content.to(window_codes.device)
                            
                            # Size check
                            write_len = min(new_content.shape[-1], rel_end - rel_start)
                            window_codes[:, :, rel_start:rel_start+write_len] = new_content[..., :write_len]
                            logger.info("Splice successful. Decoding...")
                        
                        if abort_event.is_set(): return None

                        win_duration = (win_end - win_start) / 12.5
                        
                        # 5. Decode (Mask 2 = Keep Tokens)
                        new_wav = pipeline.audio_codec.inpaint(
                            window_codes[0],
                            start_frame=0, 
                            end_frame=0,
                            duration=win_duration,
                            device=music_service.device.type,
                            mask_mode=2, 
                        )
                        
                        if abort_event.is_set(): return None

                        # FIX: Clone to detach from InferenceMode
                        new_wav = new_wav.clone()
                        
                        # Normalize to [C, T] or [T]
                        # We used to squeeze blindly, which failed for Stereo [2, T].
                        # Let's trust proper robust slicing instead.

                        # Extract the Gap + small sync buffer for crossfade
                        sr = pipeline.audio_codec.config.sample_rate
                        ratio = sr / 12.5
                        
                        # Calculate Sample Indices relative to the Window Start
                        samp_rel_start = int(rel_start * ratio)
                        samp_rel_end = int(rel_end * ratio)
                        
                        # Extract THE GENERATED SEGMENT
                        # USE ELLIPSIS slicing to ensure we slice the LAST dimension (Time)
                        generated_segment = new_wav[..., samp_rel_start : samp_rel_end]
                        print(f"[Debug] Extracted Segment: {generated_segment.shape} (from {new_wav.shape})")
                        
                        # Now Crossfade into Parent
                        # Parent global indices
                        samp_global_start = int(start_frame * ratio)
                        
                        final_wav = parent_wav_tensor.clone()
                        
                        channels = 1
                        if final_wav.dim() == 2:
                            channels = final_wav.shape[0]
                        
                        # Match Channels
                        # If generated is [T], unsqueeze to [1, T] then match
                        if generated_segment.dim() == 1:
                             generated_segment = generated_segment.unsqueeze(0)
                        
                        # If generated is [1, T] and final is [2, T], repeat
                        if channels > 1 and generated_segment.shape[0] == 1:
                            generated_segment = generated_segment.repeat(channels, 1)
                        # If generated is [C, T] but channels=1, mean?
                        elif channels == 1 and generated_segment.shape[0] > 1:
                            generated_segment = generated_segment.mean(dim=0, keepdim=True)
                            
                        # Insert Logic
                        insert_len = generated_segment.shape[-1]
                        
                        # Bounds check
                        if samp_global_start + insert_len > final_wav.shape[-1]:
                             valid_len = final_wav.shape[-1] - samp_global_start
                             if generated_segment.dim() == 2:
                                 generated_segment = generated_segment[:, :valid_len]
                             else:
                                 generated_segment = generated_segment[:valid_len]
                             insert_len = valid_len
                             
                        xfade_samps = int(CROSSFADE_SEC * sr)
                        
                        generated_segment = generated_segment.to(mix_device)
                        final_wav = final_wav.to(mix_device)
                        
                        if xfade_samps > 0 and insert_len > xfade_samps:
                            fade_in = torch.linspace(0, 1, xfade_samps, device=mix_device)
                            fade_out = torch.linspace(1, 0, xfade_samps, device=mix_device)
                            
                            
                            if channels > 1:
                                # Ensure fade vectors have channel dim [1, N]
                                if fade_in.dim() == 1:
                                    fade_in = fade_in.unsqueeze(0)
                                    fade_out = fade_out.unsqueeze(0)
                            
                            old_chunk = final_wav[..., samp_global_start : samp_global_start+xfade_samps]
                            new_chunk = generated_segment[..., :xfade_samps]
                            
                            # Safety check for empty chunks
                            if old_chunk.shape[-1] == 0 or new_chunk.shape[-1] == 0:
                                print("[Debug] Skipping start crossfade due to empty chunk")
                            else:
                                # Ensure lengths match exactly (trim to min)
                                min_len = min(old_chunk.shape[-1], new_chunk.shape[-1])
                                
                                # Resize/Trim
                                old_chunk = old_chunk[..., :min_len]
                                new_chunk = new_chunk[..., :min_len]
                                fade_in = fade_in[..., :min_len] # This might retain shape [1, N]
                                fade_out = fade_out[..., :min_len]

                                print(f"[Debug-Crossfade] Old: {old_chunk.shape} | New: {new_chunk.shape} | FadeIn: {fade_in.shape} | FadeOut: {fade_out.shape}")

                                mixed_start = old_chunk * fade_out + new_chunk * fade_in
                                generated_segment[..., :min_len] = mixed_start
                            
                            # End Boundary
                            end_pos = samp_global_start + insert_len
                            old_end_chunk = final_wav[..., end_pos-xfade_samps : end_pos]
                            new_end_chunk = generated_segment[..., -xfade_samps:]
                            
                            mixed_end = old_end_chunk * fade_in + new_end_chunk * fade_out
                            generated_segment[..., -xfade_samps:] = mixed_end

                        final_wav[..., samp_global_start : samp_global_start+insert_len] = generated_segment
                        
                        return final_wav
                    
                    wav_tensor = await loop.run_in_executor(None, _run_generate_and_mix)
                    
                    if wav_tensor is None or abort_event.is_set():
                        raise InterruptedError("Job Cancelled")

                    # 5. Save Result
                    output_filename = f"song_{new_job_id}.mp3"
                    save_path = os.path.abspath(f"generated_audio/{output_filename}")
                    
                    import torchaudio
                    torchaudio.save(save_path, wav_tensor, 48000)
                    
                    # 6. Complete
                    with Session(db_engine) as session:
                        job = session.exec(select(Job).where(Job.id == new_job_id)).one_or_none()
                        if job:
                            job.status = JobStatus.COMPLETED
                            job.audio_path = f"/audio/{output_filename}"
                            session.add(job)
                            session.commit()
                            
                            final_job_path = job.audio_path
                            final_job_title = job.title
                        
                    event_manager.publish("job_update", {"job_id": new_job_id, "status": "completed", "audio_path": final_job_path, "title": final_job_title})
                    
                except InterruptedError:
                    logger.info(f"Repair Job {new_job_id} cancelled.")
                    # Mark as failed/cancelled
                    with Session(db_engine) as session:
                        job = session.exec(select(Job).where(Job.id == new_job_id)).one_or_none()
                        if job:
                            job.status = JobStatus.FAILED
                            job.error_msg = "Cancelled"
                            session.add(job)
                            session.commit()
                    event_manager.publish("job_update", {"job_id": new_job_id, "status": "failed", "error": "Cancelled"})

                except Exception as e:
                    logger.error(f"In-painting failed: {e}", exc_info=True)
                    
                    with Session(db_engine) as session:
                        job = session.exec(select(Job).where(Job.id == new_job_id)).one_or_none()
                        if job:
                            job.status = JobStatus.FAILED
                            job.error_msg = str(e)
                            session.add(job)
                            session.commit()
                            
                            event_manager.publish("job_update", {
                                "job_id": new_job_id, 
                                "status": "failed", 
                                "error": str(e)
                            })
                finally:
                    # Cleanup
                    if 'new_job_id' in locals() and new_job_id in music_service.active_jobs:
                        del music_service.active_jobs[new_job_id]

            except Exception as e:
                logger.error(f"Inpainting setup/lifecycle failed: {e}", exc_info=True)
                # Ensure we don't leave zombie jobs if possible, but hard to recover without ID context


inpainting_service = InpaintingService()
