"""Release lifecycle: status state machine + album slot-cursor resolution.

Single source of truth so the orchestrator, API endpoints, and the UI agree on:
- which release status transitions are legal, and
- which Job row "won" each album seed slot (retry deduplication).

The orchestrator's ``state_json`` is the authoritative cursor: ``slot_jobs``
maps seed index → winning Job id, ``failed_jobs`` maps seed index → attempt
Job ids that did not complete. A failed attempt is only visible in the
tracklist while its slot has no winner; once a retry succeeds the failed row
is superseded and hidden.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("milimo.release")

RELEASE_STATUSES = ("planned", "in_progress", "completed")

# planned → in_progress → completed; completed may reopen for re-production.
# Reverting in_progress → planned is allowed at the state-machine level, but
# callers must independently guarantee no active run holds the release.
VALID_RELEASE_TRANSITIONS: Dict[str, set] = {
    "planned": {"in_progress", "completed"},
    "in_progress": {"completed", "planned"},
    "completed": {"in_progress"},
}

# AgentRun.status values that hold a release / profile hostage.
ACTIVE_RUN_STATUSES = ("queued", "running", "awaiting_approval")


def can_transition(current: str, nxt: str) -> bool:
    return nxt in VALID_RELEASE_TRANSITIONS.get(current, set())


def transition_release(release, new_status: str) -> None:
    """Validate + apply a status transition in place. Raises ValueError."""
    current = str(getattr(release, "status", "planned"))
    if current == new_status:
        return
    if not can_transition(current, new_status):
        raise ValueError(f"Invalid release status transition: {current} → {new_status}")
    release.status = new_status


def album_run_release_id(run: Any) -> Optional[str]:
    """Album runs carry their release id in input_json (tolerate legacy shapes)."""
    for blob in (getattr(run, "input_json", None), getattr(run, "state_json", None)):
        try:
            rid = (json.loads(blob or "{}") or {}).get("release_id")
            if rid:
                return str(rid)
        except Exception:
            continue
    return None


def resolve_track_rows(
    rows: List[Any], album_runs: List[Any], release_id: str
) -> List[Tuple[Any, Optional[int]]]:
    """Deduplicate a release's Job rows into (job, seed_slot) pairs.

    rows: Job objects for this release, chronological (created_at asc).
    album_runs: AgentRun rows with agent_name == 'album_orchestrator',
    newest first. Only runs whose input/state point at ``release_id`` count.

    - Non-orchestrated release: every row passes through, slot=None.
    - Orchestrated: the slot cursor is the truth — one row per slot (the
      winner, in slot order) plus failed attempts whose slot has no winner
      yet. Superseded retries are hidden.

    Legacy cursors (``job_ids`` array, pre-slot) map array position → slot.
    """
    slot_winner: Dict[int, str] = {}
    slot_failed: Dict[int, List[str]] = {}
    for run in album_runs:
        if album_run_release_id(run) != str(release_id):
            continue
        try:
            state = json.loads(getattr(run, "state_json", None) or "{}") or {}
        except Exception:
            state = {}
        # Legacy positional mapping (job_ids appended in seed order).
        for pos, jid in enumerate(state.get("job_ids") or []):
            slot_winner.setdefault(int(pos), str(jid))
        for slot, jid in (state.get("slot_jobs") or {}).items():
            slot_winner.setdefault(int(slot), str(jid))
        for slot, jids in (state.get("failed_jobs") or {}).items():
            bucket = slot_failed.setdefault(int(slot), [])
            for jid in jids if isinstance(jids, list) else [jids]:
                if str(jid) not in bucket:
                    bucket.append(str(jid))
    if not slot_winner and not slot_failed:
        return [(job, None) for job in rows]

    by_id = {str(getattr(job, "id")): job for job in rows}
    out: List[Tuple[Any, Optional[int]]] = []
    emitted: set = set()
    for slot in sorted(slot_winner):
        job = by_id.get(slot_winner[slot])
        if job is not None:
            out.append((job, slot))
            emitted.add(slot_winner[slot])
    for slot in sorted(slot_failed):
        if slot in slot_winner:
            continue  # superseded — the slot's winner is the truth
        for jid in slot_failed[slot]:
            if jid in by_id and jid not in emitted:
                out.append((by_id[jid], slot))
                emitted.add(jid)
    return out
