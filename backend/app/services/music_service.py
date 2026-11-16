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


import threading


class EventManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventManager, cls).__new__(cls)
            cls._instance.subscribers = []
            cls._instance._lock = threading.Lock()
            # Replay ring: bounded, shared by every subscriber.
            cls._instance._seq = 0
            cls._instance._ring = []
            cls._instance._ring_lock = threading.Lock()
        return cls._instance

    def subscribe(self):
        # Bounded: a stalled SSE client must not accumulate every frame during
        # hours-long runs. Drops stale progress frames on overflow; terminal
        # states are recovered by polling.
        q = asyncio.Queue(maxsize=512)
        with self._lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    # ── Replay buffer ───────────────────────────────────────────────────────
    # An EventSource reconnect (laptop sleep, network blip, proxy timeout) used to
    # lose every event in the gap, so clients had to poll /jobs/{id} to stay
    # correct — defeating the point of SSE. Events now carry a monotonic id and a
    # bounded ring replays whatever the client missed via Last-Event-ID.
    RING_SIZE = 500

    def _ring_append(self, event_type: str, payload_json: str) -> int:
        with self._ring_lock:
            self._seq += 1
            seq = self._seq
            self._ring.append((seq, event_type, payload_json))
            if len(self._ring) > self.RING_SIZE:
                del self._ring[: len(self._ring) - self.RING_SIZE]
            return seq

    def replay_since(self, last_event_id: int) -> list:
        """Events with id > last_event_id, oldest first (bounded by the ring)."""
        with self._ring_lock:
            return [item for item in self._ring if item[0] > last_event_id]

    @staticmethod
    def format_event(seq: int, event_type: str, payload_json: str) -> str:
        return f"id: {seq}\nevent: {event_type}\ndata: {payload_json}\n\n"

    @staticmethod
    def _safe_put(q: asyncio.Queue, msg: str):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    def publish(self, event_type: Any, data: Optional[dict] = None):
        import json
        if isinstance(event_type, dict) and data is None:
            data = event_type
            evt_name = "job_progress"
        else:
            evt_name = str(event_type)
            data = data or {}

        payload_json = json.dumps(data)
        seq = self._ring_append(evt_name, payload_json)
        msg = self.format_event(seq, evt_name, payload_json)
        with self._lock:
            subscribers_snapshot = list(self.subscribers)

        running_loop = None
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        for q in subscribers_snapshot:
            # If current thread is the loop thread, put_nowait directly
            if running_loop and running_loop.is_running():
                self._safe_put(q, msg)
            else:
                # Dispatched from background worker thread (e.g. run_in_executor)
                target_loop = getattr(q, "_loop", None)
                if target_loop and target_loop.is_running():
                    target_loop.call_soon_threadsafe(self._safe_put, q, msg)
                else:
                    self._safe_put(q, msg)

    def shutdown(self):
        """Broadcast shutdown signal to all subscribers to release connections."""
        seq = self._ring_append("shutdown", "{}")
        msg = self.format_event(seq, "shutdown", "{}")
        with self._lock:
            subscribers_snapshot = list(self.subscribers)
        for q in subscribers_snapshot:
            self._safe_put(q, msg)


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
        """Initialize provider registry with MiniMax Music 3."""
        from app.providers.registry import provider_registry
        # Initialize default provider (MiniMax Music 3)
        minimax = provider_registry.get_provider("minimax_music3")
        await minimax.initialize()

    def load_lora_checkpoint(self, checkpoint_path: str) -> bool:
        """Forward LoRA loading to active provider if supported."""
        from app.providers.registry import provider_registry
        provider = provider_registry.get_provider()
        if hasattr(provider, "load_lora_checkpoint"):
            return provider.load_lora_checkpoint(checkpoint_path)
        return False

    def unload_lora(self) -> bool:
        from app.providers.registry import provider_registry
        provider = provider_registry.get_provider()
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
