import asyncio
import os
import time
import torch
import logging
from typing import Optional, Any
from uuid import UUID
from app.models import GenerationRequest, Job, JobStatus
from sqlmodel import Session, select

logger = logging.getLogger(__name__)


class EventManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventManager, cls).__new__(cls)
            cls._instance.subscribers = []
        return cls._instance

    def subscribe(self):
        # Bounded: a stalled SSE client must not accumulate every frame during
        # hours-long runs. Drops stale progress frames on overflow; terminal
        # states are recovered by polling.
        q = asyncio.Queue(maxsize=512)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        if q in self.subscribers:
            self.subscribers.remove(q)

    def publish(self, event_type: Any, data: Optional[dict] = None):
        import json
        if isinstance(event_type, dict) and data is None:
            data = event_type
            evt_name = "job_progress"
        else:
            evt_name = str(event_type)
            data = data or {}

        msg = f"event: {evt_name}\ndata: {json.dumps(data)}\n\n"
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


event_manager = EventManager()


class MusicService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MusicService, cls).__new__(cls)
            cls._instance.pipeline = None
            cls._instance.gpu_lock = asyncio.Lock()
            cls._instance.is_loading = False
            cls._instance.active_jobs = {} # Map job_id -> threading.Event
            cls._instance.job_started_monotonic = {}
        return cls._instance

    async def initialize(self, model_path: str = None, version: str = "3B"):
        """Initialize provider registry with MiniMax Music 3 and HeartMuLa."""
        from app.providers.registry import provider_registry
        # Initialize default provider (MiniMax Music 3)
        minimax = provider_registry.get_provider("minimax_music3")
        await minimax.initialize()

        # Also initialize HeartMuLa if weights exist
        try:
            heartmula = provider_registry.get_provider("heartmula")
            await heartmula.initialize(model_path)
            self.pipeline = getattr(heartmula, "pipeline", None)
        except Exception as e:
            logger.warning(f"HeartMuLa optional init deferred: {e}")

    def load_lora_checkpoint(self, checkpoint_path: str) -> bool:
        """Forward LoRA loading to active provider if supported."""
        from app.providers.registry import provider_registry
        provider = provider_registry.get_provider("heartmula")
        if hasattr(provider, "load_lora_checkpoint"):
            return provider.load_lora_checkpoint(checkpoint_path)
        return False

    def unload_lora(self) -> bool:
        from app.providers.registry import provider_registry
        provider = provider_registry.get_provider("heartmula")
        if hasattr(provider, "unload_lora"):
            return provider.unload_lora()
        return True

    async def generate_task(self, job_id: UUID, req: GenerationRequest, db_engine: Any):
        """Run complete generation & transcription pipeline."""
        import threading
        job_id_str = str(job_id)
        cancel_event = threading.Event()
        self.active_jobs[job_id_str] = cancel_event
        self.job_started_monotonic[job_id_str] = time.monotonic()

        from app.orchestration.pipeline import pipeline_orchestrator
        try:
            await pipeline_orchestrator.run(
                job_id=job_id,
                req=req,
                engine=db_engine,
                event_manager=event_manager,
                cancel_event=cancel_event
            )
        finally:
            self.job_started_monotonic.pop(job_id_str, None)
            if job_id_str in self.active_jobs:
                del self.active_jobs[job_id_str]

    def cancel_job(self, job_id: str):
        if job_id in self.active_jobs:
            logger.info(f"Cancelling job {job_id}")
            self.active_jobs[job_id].set()
            return True
        return False

    def active_status(self) -> dict:
        """Elapsed per active job. ETA derives from measured RTF (~130x M3 Max)."""
        out = {}
        for jid, started in self.job_started_monotonic.items():
            out[jid] = {"elapsed_s": int(time.monotonic() - started)}
        return {"active": len(self.active_jobs), "jobs": out,
                "rtf_estimate": float(os.environ.get("MILIMO_RTF_ESTIMATE", "3"))}

    def shutdown_all(self):
        """Signal every active job to cancel. NOTE: threads already inside a
        blocking MLX call cannot be preempted — the pipeline's terminal-state
        guards discard their outputs when they surface. Returns job count."""
        n = len(self.active_jobs)
        if n:
            logger.warning(f"MusicService shutdown: signalling {n} active job(s): "
                           f"{list(self.active_jobs.keys())}")
        for job_id, event in list(self.active_jobs.items()):
            event.set()
        return n


music_service = MusicService()
