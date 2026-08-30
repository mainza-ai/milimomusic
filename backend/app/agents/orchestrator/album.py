"""Album Orchestrator: vision → per-seed songwriter+generation children.

Design (locked): sequential children (GPU lock serializes anyway) · gated-by-default
with autopilot toggle · transactional state_json cursor after every step · budget
fed from child attempts · cancel checked between steps · resumable via cursor.
"""
from __future__ import annotations

import json
import logging
import time
import uuid as uuidlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlmodel import Session

from app.agents.orchestrator import AlbumRunHandle, BudgetState, RunRegistry
from app.agents.orchestrator.bridge import create_track_from_seed
from app.agents.runtime.context import RunContext
from app.agents.runtime.overrides import resolve_chain_head
from app.agents.runtime.policy import ResiliencePolicy
from app.core.llm_contracts import AllProvidersFailedError
from app.experiencer_bridge import run_experiencer_for_release
from app.models import AgentRun, Release
from app.release_state import transition_release

logger = logging.getLogger("milimo.agents.album")


class BudgetExceeded(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AlbumOrchestrator:
    def __init__(self, registry: RunRegistry, publish):
        self.registry = registry
        self.publish = publish  # (event_type, data_dict)

    # ------------------------------------------------------------------ events
    def _emit(self, run_id: str, payload: Dict[str, Any], event: str = "run_progress"):
        self.publish(event, {"run_id": run_id, **payload})

    # ------------------------------------------------------------------ cursor
    @staticmethod
    def _load_state(run: AgentRun) -> Dict[str, Any]:
        try:
            return json.loads(run.state_json or "{}")
        except Exception:
            return {}

    @staticmethod
    def _save_state(session: Session, run: AgentRun, state: Dict[str, Any]):
        run.state_json = json.dumps(state)
        session.add(run)
        session.commit()

    # ------------------------------------------------------------------ main
    async def execute(
        self,
        parent_run_id,
        release_id,
        autopilot: bool,
        engine,
        brief_payload: Optional[Dict[str, Any]] = None,
        budget: Optional[BudgetState] = None,
    ) -> None:
        """Background coroutine. Owns the parent AgentRun row lifecycle."""
        handle = None
        from app.main import engine as _eng  # late bind avoids circulars at import
        eng = engine or _eng

        with Session(eng) as session:
            run = session.get(AgentRun, parent_run_id)
            if run is None:
                logger.error(f"[album] parent run {parent_run_id} missing")
                return
            release = session.get(Release, release_id)
            if release is None:
                run.status = "failed"
                run.error_type = "not_found"
                run.error_message = "Release not found."
                run.finished_at = _now()
                session.add(run); session.commit()
                return

            run.status = "running"
            session.add(run); session.commit()

            # Release lifecycle: producing flips planned/completed → in_progress.
            if getattr(release, "status", "planned") in ("planned", "completed"):
                release.status = "in_progress"
                session.add(release); session.commit()

            handle = AlbumRunHandle(str(parent_run_id), total_steps=0)
            handle.budget = budget or BudgetState(**json.loads(run.budget_json or "{}").get("caps", {}))
            self.registry.register(handle.run_id)

            state = self._load_state(run)
            completed_seeds: List[int] = state.get("completed_seeds", [])
            # Cursor → caller brief → previously persisted release vision (free reuse).
            persisted_vision = None
            if getattr(release, "vision_json", None):
                try:
                    persisted_vision = json.loads(release.vision_json)
                except Exception:
                    persisted_vision = None
            # NOTE: `a or b or c` returns the LAST FALSY operand ({}) when all
            # are empty — which silently skipped the vision step. Pick first
            # NON-EMPTY explicitly.
            vision_payload: Optional[Dict[str, Any]] = next(
                (v for v in (state.get("vision"), brief_payload, persisted_vision) if v),
                None,
            )
            awaiting = state.get("awaiting_approval_at")  # gated resume point

            logger.info(f"[album] cursor={state} persisted_type={type(persisted_vision).__name__} "
                        f"persisted_truthy={bool(persisted_vision)} release_vj={str(getattr(release, 'vision_json', None))[:60]}")
            try:
                # ---- Step 0: vision (skipped if cursor has it) -------------
                if vision_payload is None:
                    self._emit(handle.run_id, {"step": 0, "total_steps": "?",
                                               "phase": "experiencer",
                                               "progress": 2, "message": "Imagining the journey…"})
                    vision_payload = await run_experiencer_for_release(
                        release=release, session=session, engine=eng)
                    state["vision"] = vision_payload
                    self._save_state(session, run, state)

                seeds: List[Dict[str, Any]] = vision_payload.get("song_seeds", [])
                if not seeds:
                    raise ValueError("Vision contains no song seeds; nothing to produce.")
                album_context = {
                    "album_title": vision_payload.get("journey_title") or release.title,
                    "album_concept": vision_payload.get("concept_statement", ""),
                    "artist_name": getattr(release, "artist_name", "") or "",
                }
                total = len(seeds)
                handle.total_steps = total
                run.progress = int(100 * len(completed_seeds) / max(1, total))
                session.add(run); session.commit()

                policy = ResiliencePolicy(
                    chain_head=resolve_chain_head(session, str(getattr(release, "profile_id", "") or "") or None, "songwriter"))
                current_slot: Optional[int] = None  # slot being attempted (failure capture)

                for idx, seed in enumerate(seeds):
                    if idx in completed_seeds:
                        continue
                    if handle.check_cancel():
                        raise KeyboardInterrupt()

                    # Gated mode: pause here until approval (unless mid-resume).
                    if not autopilot and awaiting != idx:
                        state["awaiting_approval_at"] = idx
                        self._save_state(session, run, state)
                        run.status = "awaiting_approval"
                        session.add(run); session.commit()
                        self._emit(handle.run_id, {
                            "step": idx, "total_steps": total,
                            "phase": "awaiting_approval", "progress": run.progress,
                            "message": f"Track {idx + 1}/{total} ready to produce — approve to continue.",
                        })
                        return  # resume endpoint continues from cursor

                    state.pop("awaiting_approval_at", None)
                    current_slot = idx
                    track_start = time.monotonic()
                    self._emit(handle.run_id, {
                        "step": idx + 1, "total_steps": total,
                        "phase": "track", "progress": int(100 * idx / total),
                        "message": f"Producing '{seed.get('working_title', f'track {idx + 1}')}…",
                    })

                    ctx = RunContext(
                        agent_name="songwriter", run_id=str(parent_run_id),
                        project_id=str(getattr(release, "project_id", None) or ""),
                        artist_profile_id=getattr(release, "profile_id", None),
                    )
                    job = await create_track_from_seed(
                        seed={**seed, "target_duration_s": 180},
                        album_context=album_context,
                        artist_profile_id=getattr(release, "profile_id", None),
                        release_id=str(release.id),
                        project_id=str(getattr(release, "project_id", None) or "") or None,
                        provider_name="minimax_music3",
                        ctx=ctx, policy=policy,
                        engine=eng,
                        progress_cb=lambda phase, msg, i=idx, t=total: self._emit(
                            handle.run_id,
                            {"step": i + 1, "total_steps": t, "phase": phase,
                             "progress": int(100 * (i + 0.5) / t), "message": msg}),
                    )

                    completed_seeds.append(idx)
                    state["completed_seeds"] = completed_seeds
                    state.setdefault("job_ids", []).append(str(job.id))
                    # Slot cursor: seed index → winning Job. The tracklist
                    # endpoint resolves superseded retries against this.
                    state.setdefault("slot_jobs", {})[str(idx)] = str(job.id)
                    # Budget: deadline + wall-clock enforced per track; token
                    # totals arrive from child ledger rows at resume points.
                    breach = handle.budget.consume(
                        tokens_in=0, tokens_out=0,
                        elapsed_s=handle.elapsed())
                    self._save_state(session, run, state)
                    run.progress = int(100 * len(completed_seeds) / total)
                    session.add(run); session.commit()

                    self._emit(handle.run_id, {
                        "step": idx + 1, "total_steps": total, "phase": "track_done",
                        "progress": run.progress,
                        "message": f"'{job.title}' complete ({round(time.monotonic() - track_start)}s).",
                    })
                    if breach:
                        raise BudgetExceeded(breach)

                # ---- Done --------------------------------------------------
                run.status = "succeeded"
                run.progress = 100
                run.finished_at = _now()
                run.budget_json = json.dumps(handle.budget.to_dict())
                transition_release(release, "completed")
                session.add(run); session.add(release); session.commit()
                self._emit(handle.run_id, {"step": total, "total_steps": total,
                                           "phase": "done", "progress": 100,
                                           "message": "Album produced."}, event="run_update")

            except KeyboardInterrupt:
                run.status = "cancelled"
                run.error_type = "cancelled"
                run.finished_at = _now()
                session.add(run); session.commit()
                self._emit(handle.run_id, {"phase": "cancelled", "message": "Run cancelled."},
                           event="run_update")
            except BudgetExceeded as exc:
                run.status = "budget_exceeded"
                run.error_type = str(exc)
                run.error_message = f"Budget cap hit: {exc}"
                run.finished_at = _now()
                run.budget_json = json.dumps(handle.budget.to_dict())
                session.add(run); session.commit()
                self._emit(handle.run_id, {"phase": "budget_exceeded",
                                           "message": run.error_message}, event="run_update")
            except Exception as exc:
                # Honest ledger: which seed died, so resume/UI can show gaps.
                try:
                    cur = self._load_state(run)
                    cur.setdefault("failed_seeds", []).append(
                        {"error": str(exc)[:300], "at": run.progress})
                    # A created-but-failed Job is pinned to its slot so the
                    # tracklist can show the failure until a retry wins.
                    failed_job_id = getattr(exc, "job_id", None)
                    if current_slot is not None and failed_job_id:
                        bucket = cur.setdefault("failed_jobs", {}).setdefault(str(current_slot), [])
                        if str(failed_job_id) not in bucket:
                            bucket.append(str(failed_job_id))
                    self._save_state(session, run, cur)
                except Exception:
                    pass
                # Error handling must NEVER crash the task (that would strand
                # the row at 'running'). Attempts summary best-effort.
                try:
                    attempts_summary = ""
                    if isinstance(exc, AllProvidersFailedError):
                        attempts_summary = " | attempts: " + "; ".join(
                            f"{a.get('provider')}/{a.get('model')}:{a.get('error_type')}"
                            f":{str(a.get('error_message'))[:100]}"
                            for a in (exc.attempts or [])
                        )
                    logger.exception(f"[album] run failed{attempts_summary}")
                except Exception:
                    logger.exception("[album] run failed (summary unavailable)")
                run.status = "failed"
                run.error_type = type(exc).__name__
                run.error_message = str(exc)[:2000]
                run.finished_at = _now()
                session.add(run); session.commit()
                self._emit(handle.run_id, {"phase": "failed", "message": run.error_message},
                           event="run_update")
            finally:
                if handle:
                    self.registry.unregister(handle.run_id)


# -------------------------------------------------------------------------- single-seed retry
async def retry_single_seed(
    *,
    parent_run_id,
    release_id,
    engine,
    orchestrator: "AlbumOrchestrator",
) -> None:
    """Re-produce ONE failed album seed. Background coroutine; owns its
    parent AgentRun row. On success the album run's slot cursor is updated so
    the new job becomes the slot winner (the retried failure is un-pinned);
    on failure the new attempt joins the slot's failed set."""
    from app.main import engine as _eng  # late bind avoids circulars at import
    eng = engine or _eng
    emit = orchestrator._emit

    with Session(eng) as session:
        run = session.get(AgentRun, parent_run_id)
        if run is None:
            logger.error(f"[album-retry] parent run {parent_run_id} missing")
            return
        release = session.get(Release, release_id)
        if release is None:
            run.status = "failed"
            run.error_type = "not_found"
            run.error_message = "Release not found."
            run.finished_at = _now()
            session.add(run); session.commit()
            return
        try:
            cfg = json.loads(run.input_json or "{}")
        except Exception:
            cfg = {}
        try:
            seed_slot = int(cfg.get("seed_slot", -1))
        except (TypeError, ValueError):
            seed_slot = -1
        album_run_id = cfg.get("album_run_id")

        try:
            vision = json.loads(release.vision_json or "{}")
        except Exception:
            vision = {}
        seeds: List[Dict[str, Any]] = vision.get("song_seeds", [])

        if seed_slot < 0 or seed_slot >= len(seeds):
            run.status = "failed"
            run.error_type = "invalid_state"
            run.error_message = "The seed for this track is no longer present in the release vision."
            run.finished_at = _now()
            session.add(run); session.commit()
            emit(str(parent_run_id), {"phase": "failed", "message": run.error_message}, event="run_update")
            return

        run.status = "running"
        session.add(run); session.commit()
        if getattr(release, "status", "planned") in ("planned", "completed"):
            release.status = "in_progress"
            session.add(release); session.commit()

        handle = AlbumRunHandle(str(parent_run_id), total_steps=1)
        orchestrator.registry.register(handle.run_id)
        policy = ResiliencePolicy(
            chain_head=resolve_chain_head(session, str(getattr(release, "profile_id", "") or "") or None, "songwriter"))
        try:
            ctx = RunContext(
                agent_name="songwriter", run_id=str(parent_run_id),
                project_id=str(getattr(release, "project_id", None) or ""),
                artist_profile_id=getattr(release, "profile_id", None),
            )
            album_context = {
                "album_title": vision.get("journey_title") or release.title,
                "album_concept": vision.get("concept_statement", ""),
                "artist_name": getattr(release, "artist_name", "") or "",
            }
            emit(handle.run_id, {"step": 1, "total_steps": 1, "phase": "track", "progress": 5,
                                  "message": f"Reproducing '{seeds[seed_slot].get('working_title', 'track')}…"})
            job = await create_track_from_seed(
                seed={**seeds[seed_slot], "target_duration_s": 180},
                album_context=album_context,
                artist_profile_id=getattr(release, "profile_id", None),
                release_id=str(release.id),
                project_id=str(getattr(release, "project_id", None) or "") or None,
                provider_name="minimax_music3",
                ctx=ctx, policy=policy, engine=eng,
                progress_cb=lambda phase, msg: emit(
                    handle.run_id,
                    {"step": 1, "total_steps": 1, "phase": phase, "progress": 50, "message": msg}),
            )

            run.status = "succeeded"
            run.progress = 100
            run.finished_at = _now()
            session.add(run)
            if album_run_id:
                try:
                    album_run = session.get(AgentRun, uuidlib.UUID(str(album_run_id)))
                    if album_run is not None:
                        st = AlbumOrchestrator._load_state(album_run)
                        st.setdefault("slot_jobs", {})[str(seed_slot)] = str(job.id)
                        completed = st.get("completed_seeds", [])
                        if seed_slot not in completed:
                            completed.append(seed_slot)
                        st["completed_seeds"] = completed
                        retried_failure = str(cfg.get("job_id") or "")
                        bucket = st.setdefault("failed_jobs", {}).setdefault(str(seed_slot), [])
                        st["failed_jobs"][str(seed_slot)] = [j for j in bucket if j != retried_failure]
                        AlbumOrchestrator._save_state(session, album_run, st)
                except Exception:
                    logger.exception("[album-retry] cursor update failed (job still recorded)")
            session.commit()
            emit(handle.run_id, {"step": 1, "total_steps": 1, "phase": "done", "progress": 100,
                                  "message": f"'{job.title}' complete."}, event="run_update")

        except Exception as exc:  # noqa: BLE001 — ledger must never strand 'running'
            run.status = "failed"
            run.error_type = type(exc).__name__
            run.error_message = str(exc)[:2000]
            run.finished_at = _now()
            session.add(run); session.commit()
            failed_job_id = getattr(exc, "job_id", None)
            if failed_job_id and album_run_id:
                try:
                    album_run = session.get(AgentRun, uuidlib.UUID(str(album_run_id)))
                    if album_run is not None:
                        st = AlbumOrchestrator._load_state(album_run)
                        bucket = st.setdefault("failed_jobs", {}).setdefault(str(seed_slot), [])
                        if str(failed_job_id) not in bucket:
                            bucket.append(str(failed_job_id))
                        AlbumOrchestrator._save_state(session, album_run, st)
                except Exception:
                    logger.exception("[album-retry] failure recording failed")
            emit(handle.run_id, {"phase": "failed", "message": str(exc)[:300]}, event="run_update")
        finally:
            orchestrator.registry.unregister(handle.run_id)
