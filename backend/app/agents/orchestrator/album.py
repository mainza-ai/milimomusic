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

from sqlmodel import Session, select

from app.agents.orchestrator import AlbumRunHandle, BudgetState, RunRegistry
from app.agents.orchestrator.bridge import create_track_from_seed
from app.agents.runtime.context import RunContext
from app.agents.runtime.overrides import load_artist_lore, resolve_chain_head
from app.agents.runtime.policy import ResiliencePolicy
from app.core.llm_contracts import AllProvidersFailedError
from app.experiencer_bridge import run_experiencer_for_release
from app.models import AgentRun, ArtistProfile, Release
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
            session.add(run)

            if getattr(release, "status", "planned") in ("planned", "completed"):
                release.status = "in_progress"
                session.add(release)
            session.commit()

            state = self._load_state(run)
            completed_seeds: List[int] = list(state.get("completed_seeds", []))
            if not completed_seeds:
                # Inherit previously completed slots from prior runs on this release
                prev_runs = session.exec(
                    select(AgentRun).where(
                        AgentRun.agent_name == "album_orchestrator",
                        AgentRun.release_id == str(release_id),
                        AgentRun.id != run.id,
                    ).order_by(AgentRun.created_at.desc())
                ).all()
                for pr in prev_runs:
                    pst = self._load_state(pr)
                    if pst.get("completed_seeds"):
                        completed_seeds = list(pst["completed_seeds"])
                        state["completed_seeds"] = completed_seeds
                        for k, v in (pst.get("slot_jobs") or {}).items():
                            state.setdefault("slot_jobs", {})[k] = v
                        for k, v in (pst.get("reviews") or {}).items():
                            state.setdefault("reviews", {})[k] = v
                        self._save_state(session, run, state)
                        session.commit()
                        break

            try:
                _cfg = json.loads(run.input_json or "{}")
            except Exception:
                _cfg = {}
            _crew = _cfg.get("crew") if isinstance(_cfg.get("crew"), dict) else {}
            crew_flags = {"stylist": bool(_crew.get("stylist")), "critic": bool(_crew.get("critic"))}

            persisted_vision = None
            if getattr(release, "vision_json", None):
                try:
                    persisted_vision = json.loads(release.vision_json)
                except Exception:
                    persisted_vision = None
            vision_payload: Optional[Dict[str, Any]] = next(
                (v for v in (state.get("vision"), brief_payload, persisted_vision) if v),
                None,
            )
            awaiting = state.get("awaiting_approval_at")

            artist_profile = session.get(ArtistProfile, uuidlib.UUID(str(release.profile_id))) if getattr(release, "profile_id", None) else None
            artist_name = (getattr(artist_profile, "name", "") or "").strip()
            artist_voice = getattr(artist_profile, "voice_profile_id", None) or None
            artist_lore = load_artist_lore(session, str(getattr(release, "profile_id", "") or "") or None)
            chain_head = resolve_chain_head(session, str(getattr(release, "profile_id", "") or "") or None, "songwriter")
            budget_caps = json.loads(run.budget_json or "{}").get("caps", {})
            release_profile_id_val = getattr(release, "profile_id", None)
            project_id_val = getattr(release, "project_id", None)

        handle = AlbumRunHandle(str(parent_run_id), total_steps=0)
        handle.budget = budget or BudgetState(**budget_caps)
        self.registry.register(handle.run_id)

        try:
            # ---- Step 0: vision (skipped if cursor has it) -------------
            if vision_payload is None:
                self._emit(handle.run_id, {"step": 0, "total_steps": "?",
                                           "phase": "experiencer",
                                           "progress": 2, "message": "Imagining the journey…"})
                with Session(eng) as session:
                    rel = session.get(Release, release_id)
                    vision_payload = await run_experiencer_for_release(
                        release=rel, session=session, engine=eng)
                    r = session.get(AgentRun, parent_run_id)
                    if r:
                        state["vision"] = vision_payload
                        self._save_state(session, r, state)
                        session.commit()

            seeds: List[Dict[str, Any]] = vision_payload.get("song_seeds", [])
            if not seeds:
                raise ValueError("Vision contains no song seeds; nothing to produce.")

            album_context = {
                "album_title": vision_payload.get("journey_title") or release.title,
                "album_concept": vision_payload.get("concept_statement", ""),
                "artist_name": artist_name,
                "artist_lore": artist_lore,
            }
            total = len(seeds)
            handle.total_steps = total

            with Session(eng) as session:
                r = session.get(AgentRun, parent_run_id)
                if r:
                    r.progress = int(100 * len(completed_seeds) / max(1, total))
                    session.add(r); session.commit()

            policy = ResiliencePolicy(chain_head=chain_head)
            current_slot: Optional[int] = None

            for idx, seed in enumerate(seeds):
                if idx in completed_seeds:
                    continue
                if handle.check_cancel():
                    raise KeyboardInterrupt()

                # Gated mode: pause here until approval (unless mid-resume).
                if not autopilot and awaiting != idx:
                    with Session(eng) as session:
                        r = session.get(AgentRun, parent_run_id)
                        if r:
                            state["awaiting_approval_at"] = idx
                            self._save_state(session, r, state)
                            r.status = "awaiting_approval"
                            session.add(r); session.commit()
                    self._emit(handle.run_id, {
                        "step": idx, "total_steps": total,
                        "phase": "awaiting_approval", "progress": int(100 * len(completed_seeds) / total),
                        "message": f"Track {idx + 1}/{total} ready to produce — approve to continue.",
                    })
                    return

                state.pop("awaiting_approval_at", None)
                current_slot = idx
                review_sink: Dict[str, Any] = {}
                track_start = time.monotonic()
                self._emit(handle.run_id, {
                    "step": idx + 1, "total_steps": total,
                    "phase": "track", "progress": int(100 * idx / total),
                    "message": f"Producing '{seed.get('working_title', f'track {idx + 1}')}…",
                })

                ctx = RunContext(
                    agent_name="songwriter", run_id=str(parent_run_id),
                    project_id=str(project_id_val or ""),
                    artist_profile_id=release_profile_id_val,
                )
                job = await create_track_from_seed(
                    seed={**seed, "target_duration_s": seed.get("target_duration_s", 180)},
                    album_context=album_context,
                    artist_profile_id=release_profile_id_val,
                    release_id=str(release_id),
                    project_id=str(project_id_val or "") or None,
                    provider_name="minimax_music3",
                    voice_profile_id=artist_voice,
                    crew_flags=crew_flags,
                    review_sink=review_sink,
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
                state.setdefault("slot_jobs", {})[str(idx)] = str(job.id)
                if review_sink.get("review"):
                    state.setdefault("reviews", {})[str(idx)] = review_sink["review"]

                breach = handle.budget.consume(
                    tokens_in=0, tokens_out=0,
                    elapsed_s=handle.elapsed())

                with Session(eng) as session:
                    r = session.get(AgentRun, parent_run_id)
                    if r:
                        self._save_state(session, r, state)
                        r.progress = int(100 * len(completed_seeds) / total)
                        session.add(r); session.commit()

                self._emit(handle.run_id, {
                    "step": idx + 1, "total_steps": total, "phase": "track_done",
                    "progress": int(100 * len(completed_seeds) / total),
                    "message": f"'{job.title}' complete ({round(time.monotonic() - track_start)}s).",
                })
                if breach:
                    raise BudgetExceeded(breach)

            # ---- Done --------------------------------------------------
            with Session(eng) as session:
                r = session.get(AgentRun, parent_run_id)
                rel = session.get(Release, release_id)
                if r:
                    r.status = "succeeded"
                    r.progress = 100
                    r.finished_at = _now()
                    r.budget_json = json.dumps(handle.budget.to_dict())
                    session.add(r)
                if rel:
                    transition_release(rel, "completed")
                    session.add(rel)
                session.commit()

            self._emit(handle.run_id, {"step": total, "total_steps": total,
                                       "phase": "done", "progress": 100,
                                       "message": "Album produced."}, event="run_update")

        except KeyboardInterrupt:
            with Session(eng) as session:
                r = session.get(AgentRun, parent_run_id)
                if r:
                    r.status = "cancelled"
                    r.error_type = "cancelled"
                    r.finished_at = _now()
                    session.add(r); session.commit()
            self._emit(handle.run_id, {"phase": "cancelled", "message": "Run cancelled."},
                       event="run_update")
        except BudgetExceeded as exc:
            with Session(eng) as session:
                r = session.get(AgentRun, parent_run_id)
                if r:
                    r.status = "budget_exceeded"
                    r.error_type = str(exc)
                    r.error_message = f"Budget cap hit: {exc}"
                    r.finished_at = _now()
                    r.budget_json = json.dumps(handle.budget.to_dict())
                    session.add(r); session.commit()
            self._emit(handle.run_id, {"phase": "budget_exceeded",
                                       "message": f"Budget cap hit: {exc}"}, event="run_update")
        except Exception as exc:
            with Session(eng) as session:
                r = session.get(AgentRun, parent_run_id)
                if r:
                    try:
                        cur = self._load_state(r)
                        cur.setdefault("failed_seeds", []).append(
                            {"error": str(exc)[:300], "at": r.progress})
                        failed_job_id = getattr(exc, "job_id", None)
                        if current_slot is not None and failed_job_id:
                            bucket = cur.setdefault("failed_jobs", {}).setdefault(str(current_slot), [])
                            if str(failed_job_id) not in bucket:
                                bucket.append(str(failed_job_id))
                        self._save_state(session, r, cur)
                    except Exception:
                        pass
                    r.status = "failed"
                    r.error_type = type(exc).__name__
                    r.error_message = str(exc)[:2000]
                    r.finished_at = _now()
                    session.add(r); session.commit()
            logger.exception(f"[album] run failed: {exc}")
            self._emit(handle.run_id, {"phase": "failed", "message": str(exc)[:2000]},
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
        session.add(run)
        if getattr(release, "status", "planned") in ("planned", "completed"):
            release.status = "in_progress"
            session.add(release)
        session.commit()

        retry_profile = session.get(ArtistProfile, uuidlib.UUID(str(release.profile_id))) if getattr(release, "profile_id", None) else None
        retry_voice = getattr(retry_profile, "voice_profile_id", None) or None
        artist_name = (getattr(retry_profile, "name", "") or "").strip()
        artist_lore = load_artist_lore(session, str(getattr(release, "profile_id", "") or "") or None)
        chain_head = resolve_chain_head(session, str(getattr(release, "profile_id", "") or "") or None, "songwriter")
        release_title = release.title
        release_id_str = str(release.id)
        release_profile_id_val = getattr(release, "profile_id", None)
        project_id_val = getattr(release, "project_id", None)

    handle = AlbumRunHandle(str(parent_run_id), total_steps=1)
    orchestrator.registry.register(handle.run_id)
    policy = ResiliencePolicy(chain_head=chain_head)
    _crew = cfg.get("crew") if isinstance(cfg.get("crew"), dict) else {}
    crew_flags = {"stylist": bool(_crew.get("stylist")), "critic": bool(_crew.get("critic"))}
    review_sink: Dict[str, Any] = {}
    try:
        ctx = RunContext(
            agent_name="songwriter", run_id=str(parent_run_id),
            project_id=str(project_id_val or ""),
            artist_profile_id=release_profile_id_val,
        )
        album_context = {
            "album_title": vision.get("journey_title") or release_title,
            "album_concept": vision.get("concept_statement", ""),
            "artist_name": artist_name,
            "artist_lore": artist_lore,
        }
        emit(handle.run_id, {"step": 1, "total_steps": 1, "phase": "track", "progress": 5,
                              "message": f"Reproducing '{seeds[seed_slot].get('working_title', 'track')}…"})
        job = await create_track_from_seed(
            seed={**seeds[seed_slot], "target_duration_s": seeds[seed_slot].get("target_duration_s", 180)},
            album_context=album_context,
            artist_profile_id=release_profile_id_val,
            release_id=release_id_str,
            project_id=str(project_id_val or "") or None,
            provider_name="minimax_music3",
            voice_profile_id=retry_voice,
            crew_flags=crew_flags,
            review_sink=review_sink,
            ctx=ctx, policy=policy, engine=eng,
            progress_cb=lambda phase, msg: emit(
                handle.run_id,
                {"step": 1, "total_steps": 1, "phase": phase, "progress": 50, "message": msg}),
        )

        with Session(eng) as session:
            r = session.get(AgentRun, parent_run_id)
            if r:
                r.status = "succeeded"
                r.progress = 100
                r.finished_at = _now()
                session.add(r)
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
                        if review_sink.get("review"):
                            st.setdefault("reviews", {})[str(seed_slot)] = review_sink["review"]
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
        with Session(eng) as session:
            r = session.get(AgentRun, parent_run_id)
            if r:
                r.status = "failed"
                r.error_type = type(exc).__name__
                r.error_message = str(exc)[:2000]
                r.finished_at = _now()
                session.add(r)
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
            session.commit()
        emit(handle.run_id, {"phase": "failed", "message": str(exc)[:300]}, event="run_update")
    finally:
        orchestrator.registry.unregister(handle.run_id)
