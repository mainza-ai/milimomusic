"""Album/agent run orchestration — lifecycle engine for multi-step runs.

Reuses every proven primitive:
  * instant-return routes (the /generate/music pattern)
  * threading.Event cancellation registry (cloned from MusicService.active_jobs)
  * EventManager progress events using the EXACT job_progress payload vocabulary
    the frontend already renders
  * transactional state cursor per step → resumable after restart

The GPU lock serializes per-track generation children automatically.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class RunRegistry:
    """threading.Event cancel registry for active runs (per-run lifecycle)."""

    def __init__(self) -> None:
        self._events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def register(self, run_id: str) -> threading.Event:
        ev = threading.Event()
        with self._lock:
            self._events[run_id] = ev
        return ev

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            ev = self._events.get(run_id)
        if ev is not None:
            ev.set()
            return True
        return False

    def unregister(self, run_id: str) -> None:
        with self._lock:
            self._events.pop(run_id, None)

    def shutdown_all(self) -> None:
        with self._lock:
            for ev in self._events.values():
                ev.set()
            self._events.clear()


@dataclass
class BudgetState:
    max_tokens_in: Optional[int] = None
    max_tokens_out: Optional[int] = None
    deadline_s: Optional[float] = None
    tokens_in: int = 0
    tokens_out: int = 0
    elapsed_s: float = 0.0

    def consume(self, tokens_in: int, tokens_out: int, elapsed_s: float) -> Optional[str]:
        """Accumulate child usage; return breach code or None."""
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.elapsed_s = elapsed_s
        if self.max_tokens_in is not None and self.tokens_in > self.max_tokens_in:
            return "budget_tokens_in_exceeded"
        if self.max_tokens_out is not None and self.tokens_out > self.max_tokens_out:
            return "budget_tokens_out_exceeded"
        if self.deadline_s is not None and self.elapsed_s > self.deadline_s:
            return "budget_deadline_exceeded"
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "elapsed_s": round(self.elapsed_s, 1),
            "caps": {"max_tokens_in": self.max_tokens_in,
                     "max_tokens_out": self.max_tokens_out,
                     "deadline_s": self.deadline_s},
        }


@dataclass
class OrchestratorStep:
    """One unit of an album plan. `kind` selects the executor:
      'experiencer' | 'songwriter' (future) | 'generation' (future)."""
    kind: str
    label: str
    seed_index: Optional[int] = None       # for per-track steps
    payload: Dict[str, Any] = field(default_factory=dict)


class AlbumRunHandle:
    """Live handle the background coroutine carries; also the cancel surface."""

    def __init__(self, run_id: str, total_steps: int):
        self.run_id = run_id
        self.total_steps = total_steps
        self.cancel_event = threading.Event()
        self.started_monotonic = time.monotonic()
        self.budget = BudgetState()

    def check_cancel(self) -> bool:
        return self.cancel_event.is_set()

    def elapsed(self) -> float:
        return time.monotonic() - self.started_monotonic


def make_step_runner(
    publish: Callable[[str, Dict[str, Any]], None],
) -> Callable[..., Any]:
    """Returns a step-execution decorator bound to an event publisher.

    Kept minimal for this phase: the album orchestrator (R4) composes
    experiencer/songwriter/generation executors on top of it.
    """
    def run_step(step_label: str, fn: Callable[[], Any], *, step: int, total: int, run_id: str) -> Any:
        publish("run_progress", {
            "run_id": run_id, "step": step, "total_steps": total,
            "phase": step_label, "progress": int(100 * (step - 1) / max(1, total)),
            "message": f"{step_label}…",
        })
        value = fn()
        publish("run_progress", {
            "run_id": run_id, "step": step, "total_steps": total,
            "phase": step_label, "progress": int(100 * step / max(1, total)),
            "message": f"{step_label} done",
        })
        return value
    return run_step
