"""Usage accounting for agent runs (audit gap G5).

Every provider attempt is recorded — success or failure — with latency and
token counts, so multi-agent runs can attribute spend per agent and per run
instead of burning quota invisibly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AttemptRecord:
    provider: str
    model: str
    ok: bool
    latency_ms: int = 0
    error_type: str = ""          # typed class name when ok=False
    error_message: str = ""
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> Dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "error_type": self.error_type,
            "error_message": self.error_message[:500],
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }


@dataclass
class RunUsage:
    """Aggregated accounting across every attempt in one run."""

    attempts: List[AttemptRecord] = field(default_factory=list)

    def add(self, attempt: AttemptRecord) -> None:
        self.attempts.append(attempt)

    @property
    def total_tokens_in(self) -> int:
        return sum(a.tokens_in for a in self.attempts)

    @property
    def total_tokens_out(self) -> int:
        return sum(a.tokens_out for a in self.attempts)

    @property
    def total_latency_ms(self) -> int:
        return sum(a.latency_ms for a in self.attempts)

    @property
    def provider_calls(self) -> int:
        return len(self.attempts)

    def to_dict(self) -> Dict:
        return {
            "provider_calls": self.provider_calls,
            "tokens_in": self.total_tokens_in,
            "tokens_out": self.total_tokens_out,
            "total_latency_ms": self.total_latency_ms,
            "attempts": [a.to_dict() for a in self.attempts],
        }
