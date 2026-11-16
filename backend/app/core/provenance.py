"""
Per-stage provenance for every artifact a job produces.

One rule, enforced mechanically: **no artifact may reach COMPLETED without saying
what produced it and how degraded it is.**

Each pipeline stage records a small dict:

    {"stage": "transcription", "source": "fallback_placeholder",
     "degraded": True, "reason": "MuScriptor weights unavailable", ...}

`merge_stage()` folds it into `job.provenance_json` and recomputes `job.quality`,
which is the single field the UI filters and badges on.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("milimo.provenance")

# Ordered worst-first so the most severe stage wins the job verdict.
_SOURCE_RANK = {"placeholder": 3, "dsp": 2, "model": 1}

QUALITY_NEURAL = "neural"
QUALITY_DEGRADED = "degraded"
QUALITY_PLACEHOLDER = "placeholder"


def load(job) -> Dict[str, Any]:
    raw = getattr(job, "provenance_json", None)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        logger.warning("Discarding unparseable provenance_json for job %s", getattr(job, "id", "?"))
        return {}


def compute_quality(stages: Dict[str, Dict[str, Any]]) -> str:
    worst = 0
    for record in stages.values():
        if not isinstance(record, dict):
            continue
        if not record.get("degraded"):
            worst = max(worst, _SOURCE_RANK["model"])
            continue
        worst = max(worst, _SOURCE_RANK.get(str(record.get("source", "dsp")), 2))
    if worst >= _SOURCE_RANK["placeholder"]:
        return QUALITY_PLACEHOLDER
    if worst >= _SOURCE_RANK["dsp"]:
        return QUALITY_DEGRADED
    return QUALITY_NEURAL


def merge_stage(job, record: Optional[Dict[str, Any]]) -> Optional[str]:
    """Fold one stage record into the job and return the new quality verdict."""
    if not record or not record.get("stage"):
        return git_quality(job)

    stages = load(job)
    stages[str(record["stage"])] = record
    job.provenance_json = json.dumps(stages, ensure_ascii=False, default=str)
    job.quality = compute_quality(stages)
    return job.quality


def git_quality(job) -> str:
    """Current quality without mutating (falls back to the legacy synced flags)."""
    stages = load(job)
    if stages:
        return compute_quality(stages)
    return QUALITY_PLACEHOLDER if getattr(job, "used_fallback_synth", False) else QUALITY_NEURAL


def summarise(job) -> Dict[str, Any]:
    """Compact payload for SSE / API responses."""
    return {
        "quality": git_quality(job),
        "provenance": load(job),
        "fallback_reason": getattr(job, "fallback_reason", None),
    }
