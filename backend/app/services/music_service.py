import asyncio
import os
import torch
import logging
from typing import Optional
from app.models import GenerationRequest, Job, JobStatus
from sqlmodel import Session, select
from heartlib import HeartMuLaGenPipeline

logger = logging.getLogger(__name__)

class MusicService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MusicService, cls).__new__(cls)
            cls._instance.pipeline = None
            cls._instance.gpu_lock = asyncio.Lock()
            cls._instance.is_loading = False
            cls._instance.active_jobs = {} # Map job_id -> threading.Event
        return cls._instance
        return cls._instance

    async def initialize(self, model_path: str = "../heartlib/ckpt", version: str = "3B"):
        if self.pipeline is not None or self.is_loading:
            return

        self.is_loading = True
        logger.info(f"Loading Heartlib model from {model_path}...")
        try:
            # Run blocking load in executor to avoid freezing async loop
            loop = asyncio.get_running_loop()
            self.pipeline = await loop.run_in_executor(
                None,
                lambda: HeartMuLaGenPipeline.from_pretrained(
                    model_path,
                    device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
                    dtype=torch.bfloat16,
                    version=version
                )
            )
            logger.info("Heartlib model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Heartlib model: {e}")
            raise e
        finally:
            self.is_loading = False

    async def generate_task(self, job_id: str, request: GenerationRequest, db_engine):
        """Background task to generate music."""
        job_id = str(job_id) # Ensure string for dictionary keys
        
        # 1. Acquire GPU Lock
        async with self.gpu_lock:
            logger.info(f"Starting generation for job {job_id}")
            
            # 2. Update status to PROCESSING
            try:
                with Session(db_engine) as session:
                    # check if job still exists
                    job = session.exec(select(Job).where(Job.id == job_id)).one_or_none()
                    if not job:
                        logger.warning(f"Job {job_id} was deleted before processing started. Aborting.")
                        return
                    
                    job.status = JobStatus.PROCESSING
                    session.add(job)
                    session.commit()
            except Exception as e:
                logger.error(f"Failed to update job status to PROCESSING: {e}")
                return

            try:
                # 3. Create unique filename
                output_filename = f"song_{job_id}.mp3"
                save_path = os.path.abspath(f"generated_audio/{output_filename}")
                
                # Create Cancellation Event
                import threading
                abort_event = threading.Event()
                self.active_jobs[job_id] = abort_event
                
                # 4. Generate Auto-Title (Robust)
                from app.services.llm_service import LLMService
                
                # Use lyrics for context if available, otherwise prompt
                context_source = request.lyrics if request.lyrics and len(request.lyrics) > 10 else request.prompt
                # Truncate to first 1000 chars to avoid token limits, but enough for context
                context_source = context_source[:1000]
                
                auto_title = "Untitled Track"
                try:
                    # Logic: If no specific model requested, pass None.
                    # LLMService.generate_title will resolve it to the active configured model.
                    auto_title = LLMService.generate_title(context_source, model=request.llm_model)
                except Exception as e:
                    logger.warning(f"Auto-title generation failed: {e}. Using default.")
                
                # 5. Run Generation (Blocking, run in executor)
                
                # Phase 10: Set Seed (Moved to outer scope)
                seed_to_use = request.seed
                if seed_to_use is None:
                    # Fallback if not passed (though we should have it)
                    import random
                    seed_to_use = random.randint(0, 2**32 - 1)
                
                # Note: heartlib's pipeline is not async, so we wrap it
                loop = asyncio.get_running_loop()
                
                # Progress Callback for Pipeline
                def _pipeline_callback(progress, msg):
                    # Suppress MPS autocast warning spam if mostly benign (it just disables autocast for unsupported ops)
                    import warnings
                    warnings.filterwarnings("ignore", message="In MPS autocast, but the target dtype is not supported")

                    loop.call_soon_threadsafe(
                        event_manager.publish, 
                        "job_progress", 
                        {"job_id": str(job_id), "progress": progress, "msg": msg}
                    )

                def _run_pipeline():
                    # Set fallback for MPS conv1d limit just in case
                    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                    
                    # Logic: 
                    # request.tags -> Sound Description (e.g. "Afrobeat") -> Heartlib 'tags'
                    # request.prompt -> User's Concept (e.g. "Song about rain") -> Not used by Heartlib generation, just for history/title
                    
                    # If user didn't provide sound tags, use the prompt as a fallback tag
                    sound_tags = request.tags if request.tags and request.tags.strip() else "pop music"

                    # Phase 9: Load History Tokens if extending
                    history_tokens = None
                    if request.parent_job_id:
                        try:
                            parent_token_path = os.path.join(os.getcwd(), "generated_tokens", f"{request.parent_job_id}.pt")
                            if os.path.exists(parent_token_path):
                                logger.info(f"Loading history tokens from {parent_token_path}")
                                history_tokens = torch.load(parent_token_path, map_location=self.device)
                                # Ensure correct shape/device if needed
                                if history_tokens.device != self.device:
                                     history_tokens = history_tokens.to(self.device)
                            else:
                                logger.warning(f"Parent token file not found: {parent_token_path}")
                        except Exception as e:
                            logger.error(f"Failed to load history tokens: {e}")

                    logger.info(f"Setting random seed to {seed_to_use}")
                    torch.manual_seed(seed_to_use)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed_to_use)
                    import random
                    random.seed(seed_to_use)
                    import numpy as np
                    np.random.seed(seed_to_use)



                    with torch.no_grad():
                        output = self.pipeline(
                            {
                                "lyrics": request.lyrics,
                                "tags": sound_tags,
                            },
                            max_audio_length_ms=request.duration_ms,
                            save_path=save_path,
                            topk=request.topk,
                            temperature=request.temperature, 
                            cfg_scale=request.cfg_scale,
                            callback=_pipeline_callback,  # Pass our new callback
                            abort_event=abort_event,      # Pass cancellation signal
                            history_tokens=history_tokens, # Phase 9
                        )
                        
                        # Save tokens if returned (Phase 9)
                        if "tokens" in output and output["tokens"] is not None:
                            try:
                                tokens_dir = os.path.join(os.getcwd(), "generated_tokens")
                                os.makedirs(tokens_dir, exist_ok=True)
                                token_path = os.path.join(tokens_dir, f"{job_id}.pt")
                                torch.save(output["tokens"], token_path)
                                logger.info(f"Saved tokens to {token_path}")
                                
                                # Update Job with token path (Requires DB schema update or just implicit knowledge)
                                # For now, we assume implicit path based on ID, but ideally we add to DB.
                                # Let's update the job object later in the session block.
                            except Exception as e:
                                logger.error(f"Failed to save tokens: {e}")
                    
                    return output
                
                # Notify Start
                event_manager.publish("job_update", {"job_id": str(job_id), "status": "processing"})
                event_manager.publish("job_progress", {"job_id": str(job_id), "progress": 0, "msg": "Starting generation pipeline..."})

                # output variable capture
                output = await loop.run_in_executor(None, _run_pipeline)

                # 6. Update status to COMPLETED
                with Session(db_engine) as session:
                    # Re-fetch to avoid stale object
                    job = session.exec(select(Job).where(Job.id == job_id)).one_or_none()
                    if not job:
                         logger.warning(f"Job {job_id} was deleted during generation. Discarding result.")
                         return

                    job.status = JobStatus.COMPLETED
                    job.audio_path = f"/audio/{output_filename}"
                    job.title = auto_title
                    job.seed = seed_to_use # Ensure saved
                    session.add(job)
                    session.commit()
                    # Extract values while attached to session
                    final_audio_path = job.audio_path
                    final_title = job.title
                
                logger.info(f"Job {job_id} completed. Saved to {save_path}")
                event_manager.publish("job_update", {"job_id": str(job_id), "status": "completed", "audio_path": final_audio_path, "title": final_title})
                event_manager.publish("job_progress", {"job_id": str(job_id), "progress": 100, "msg": "Done!"})

            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}")
                with Session(db_engine) as session:
                    job = session.exec(select(Job).where(Job.id == job_id)).one()
                    job.status = JobStatus.FAILED
                    job.error_msg = str(e)
                    session.add(job)
                    session.commit()
                event_manager.publish("job_update", {"job_id": str(job_id), "status": "failed", "error": str(e)})

            finally:
                # Cleanup cancellation event
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]

    def cancel_job(self, job_id: str):
        if job_id in self.active_jobs:
            logger.info(f"Cancelling job {job_id}")
            self.active_jobs[job_id].set()
            return True
        return False

    def shutdown_all(self):
        """Cancel all active jobs."""
        logger.info(f"Shutting down MusicService. Cancelling {len(self.active_jobs)} active jobs.")
        for job_id, event in list(self.active_jobs.items()):
            event.set()


class EventManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventManager, cls).__new__(cls)
            cls._instance.subscribers = []
        return cls._instance

    def subscribe(self):
        q = asyncio.Queue()
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        if q in self.subscribers:
            self.subscribers.remove(q)

    def publish(self, event_type: str, data: dict):
        import json
        msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        for q in self.subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    def shutdown(self):
        """Broadcast shutdown signal to all subscribers to release connections."""
        msg = "event: shutdown\ndata: {}\n\n"
        for q in self.subscribers:
             try:
                q.put_nowait(msg)
             except asyncio.QueueFull:
                pass


music_service = MusicService()
event_manager = EventManager()
