"""
Video Orchestrator — End-to-End Production Music Video Assembly Pipeline.
Integrates VideoDirector, LivePortrait singing avatar, Wan 2.1 diffusion,
per-scene regeneration, beat transitions, ASS karaoke burning, and master remuxing.
"""

import os
import time
import re
import math
import json
import uuid
import shutil
import asyncio
import logging
import threading
from typing import List, Dict, Optional, Any, Tuple

from app.models import Job
from app.core.paths import (
    get_generated_audio_dir, get_data_dir,
    resolve_audio_file, resolve_stem_file, resolve_image_file
)
from app.transcription.karaoke import lyric_sync_engine
from app.services.video.types import (
    SceneClip, SceneType, VideoPlan, VideoTaskStatusInfo,
    VideoProviderInfo, VideoProviderType, LipSyncEngineType, VideoModelType
)
from app.services.video.video_director import video_director, STYLE_PALETTES
from app.services.video.subtitle_styles import (
    generate_karaoke_ass_script, get_fonts_dir, resolve_preset, SUBTITLE_PRESETS,
    detect_hardware_encoder, find_ffmpeg_executable, has_subtitles_filter
)
from app.services.video.lip_sync.base import BaseLipSyncProvider
from app.services.video.lip_sync.live_portrait import LivePortraitProvider
from app.services.video.lip_sync.fallback import SmoothVisemeFallbackProvider
from app.services.video.generators.base import BaseVideoGenerator
from app.services.video.generators.diffusers_wan import DiffusersWanGenerator
from app.services.video.generators.diffusers_ltx import DiffusersLTXGenerator
from app.services.video.generators.cloud_video import CloudVideoGenerator
from app.services.video.generators.procedural import ProceduralVideoGenerator
from app.core.hardware_lock import GlobalHardwareCoordinator
from app.core.task_queue import task_queue
from app.services.video.stem_audio_reactive import (
    build_stem_reactive_filter,
    extract_stem_reactive_modulation,
)

logger = logging.getLogger(__name__)

VIDEO_DIR = str(get_generated_audio_dir() / "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)
TEMP_DIR = str(get_data_dir() / "video_cache")
os.makedirs(TEMP_DIR, exist_ok=True)
KEYFRAMES_DIR = str(get_generated_audio_dir() / "videos" / "keyframes")
os.makedirs(KEYFRAMES_DIR, exist_ok=True)


class VideoOrchestrator:
    _instance = None
    _tasks: Dict[str, VideoTaskStatusInfo] = {}
    _video_cancels: Dict[str, asyncio.Event] = {}
    _keyframe_cancels: Dict[str, asyncio.Event] = {}
    _plan_cancels: Dict[str, asyncio.Event] = {}
    _active_render_tasks: Dict[str, asyncio.Task] = {}
    _active_procs: Dict[str, Any] = {}
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VideoOrchestrator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._fallback_lipsync = SmoothVisemeFallbackProvider()
        self._local_wan_14b = DiffusersWanGenerator(model_size="14b")
        self._local_wan_1_3b = DiffusersWanGenerator(model_size="1.3b")
        self._local_ltx = DiffusersLTXGenerator()
        self._procedural = ProceduralVideoGenerator()

    def begin_task(
        self,
        task_id: str,
        job_id: str,
        *,
        step: str,
        total_clips: int = 1,
    ) -> None:
        """Register a background task so `GET /videos/tasks/{id}` can report it.

        Export and variant jobs returned a task id that no endpoint could resolve,
        so the UI could not show progress or completion for them.
        """
        with self._lock:
            self._tasks[task_id] = VideoTaskStatusInfo(
                id=task_id,
                job_id=str(job_id),
                status="processing",
                step=step,
                progress=5,
                total_clips=int(total_clips),
                current_clip=0,
            )

    def end_task(
        self,
        task_id: str,
        *,
        status: str,
        step: Optional[str] = None,
        video_url: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark a `begin_task` task terminal (kept for the poller, not deleted)."""
        payload: Dict[str, Any] = {
            "status": status,
            "step": step or ("Done" if status == "completed" else (error or status)),
            "video_url": video_url,
            "error": error,
        }
        if status == "completed":
            payload["progress"] = 100
        self.update_task(task_id, **payload)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            t = self._tasks.get(task_id)
            if t:
                return t.to_dict()
        # Query durable SQLite task queue if not in memory
        pt = task_queue.get_task(task_id)
        if pt:
            d = pt.to_dict()
            if "id" not in d:
                d["id"] = d.get("task_id", task_id)
            if "step" not in d:
                d["step"] = d.get("message", "processing")
            if "total_clips" not in d:
                d["total_clips"] = 0
            if "current_clip" not in d:
                d["current_clip"] = 0
            if pt.result and isinstance(pt.result, dict):
                if "clips" in pt.result and not d.get("clips"):
                    d["clips"] = pt.result["clips"]
                if "treatment" in pt.result and not d.get("treatment"):
                    d["treatment"] = pt.result["treatment"]
            return d
        return None

    def update_task(self, task_id: str, **kwargs):
        with self._lock:
            if task_id in self._tasks:
                t = self._tasks[task_id]
                for k, v in kwargs.items():
                    if hasattr(t, k):
                        setattr(t, k, v)
        # Update persistent SQLite queue
        try:
            status = kwargs.get("status")
            progress = kwargs.get("progress")
            step = kwargs.get("step") or kwargs.get("error")
            result = kwargs.get("result")
            task_queue.update_task(
                task_id=task_id,
                status=status,
                progress=progress,
                message=step,
                result=result,
                error=kwargs.get("error"),
            )
        except Exception as e:
            logger.debug(f"Failed to sync task {task_id} to durable queue: {e}")

    def register_render_task(self, task_id: str, task: asyncio.Task) -> None:
        with self._lock:
            self._active_render_tasks[task_id] = task

    def unregister_render_task(self, task_id: str) -> None:
        with self._lock:
            self._active_render_tasks.pop(task_id, None)

    def cancel_video_task(self, task_id: str) -> bool:
        with self._lock:
            ev = self._video_cancels.get(task_id)
            if ev:
                ev.set()
            ev_plan = self._plan_cancels.get(task_id)
            if ev_plan:
                ev_plan.set()

            # Immediately terminate running FFmpeg / subprocess if any
            proc = self._active_procs.get(task_id)
            if proc and proc.returncode is None:
                try:
                    proc.kill()
                    logger.info(f"VideoOrchestrator: Killed active subprocess for task {task_id}.")
                except Exception as e:
                    logger.debug(f"Error killing subprocess for {task_id}: {e}")

            bg_task = self._active_render_tasks.get(task_id)
            if bg_task and not bg_task.done():
                try:
                    bg_task.cancel()
                    logger.info(f"VideoOrchestrator: Cancelled background asyncio task for {task_id}.")
                except Exception as e:
                    logger.debug(f"Error cancelling asyncio task {task_id}: {e}")
        self.update_task(task_id, status="cancelled", step="Video rendering cancelled by user.", progress=0)
        try:
            from app.core.hardware_lock import GlobalHardwareCoordinator
            GlobalHardwareCoordinator.flush_memory()
        except Exception:
            pass
        try:
            from app.services.video.generator_registry import VideoGeneratorRegistry
            VideoGeneratorRegistry.unload_all()
        except Exception:
            pass
        try:
            from app.services.llm_service import LLMService
            LLMService.unload_local_model()
        except Exception:
            pass
        return True

    def cancel_keyframe_generation(self, job_id: str) -> bool:
        clean_target = str(job_id).replace("-", "").strip().lower()
        matching_tasks = []
        with self._lock:
            # 1. Trigger all matching cancel events (raw key or normalized UUID)
            for k, ev in list(self._keyframe_cancels.items()):
                if str(k).replace("-", "").strip().lower() == clean_target or str(k) == str(job_id):
                    ev.set()

            # 2. Cancel matching background asyncio tasks and collect task IDs
            for tid, tinfo in self._tasks.items():
                t_job = str(tinfo.job_id).replace("-", "").strip().lower()
                if (t_job == clean_target or tinfo.job_id == str(job_id)) and tid.startswith("kf_"):
                    matching_tasks.append(tid)
                    bg = self._active_render_tasks.get(tid)
                    if bg and not bg.done():
                        try:
                            bg.cancel()
                            logger.info(f"VideoOrchestrator: Cancelled background task for keyframes {tid}")
                        except Exception as e:
                            logger.debug(f"Error cancelling asyncio keyframe task {tid}: {e}")

        # 3. Update task status in memory and durable SQLite queue
        for tid in matching_tasks:
            self.update_task(tid, status="cancelled", step="Keyframe generation cancelled by user.", progress=0)
            try:
                task_queue.fail_task(tid, "Cancelled by user")
            except Exception:
                pass

        # 4. Abort in-flight MLX/diffusers diffusion immediately (< 500ms)
        try:
            from app.services.image_service import image_service
            # Scope the cancel to this job's keyframes: a global cancel aborted
            # another track's cover generation mid-flight.
            # The cancel is already scoped by this method's `job_id` argument.
            image_service.cancel_active_diffusion(job_id=str(job_id))
            from app.core.hardware_lock import GlobalHardwareCoordinator
            GlobalHardwareCoordinator.flush_memory()
        except Exception as e:
            logger.debug(f"Error aborting active diffusion: {e}")

        return True

    def resolve_audio_path(self, path: Optional[str]) -> Optional[str]:
        return resolve_audio_file(path)

    def resolve_vocals_stem(self, job: Job) -> Optional[str]:
        stems_val = getattr(job, "stems_json", None) or getattr(job, "stem_paths", None)
        return resolve_stem_file(job.id, "vocals", stems_val)

    def resolve_stem(self, job: Job, stem_name: str) -> Optional[str]:
        stems_val = getattr(job, "stems_json", None) or getattr(job, "stem_paths", None)
        return resolve_stem_file(job.id, stem_name, stems_val)

    def resolve_face_image(self, job: Job, custom_image: Optional[str] = None) -> Optional[str]:
        """Resolve face/performer portrait image for avatar lip-sync.

        Only returns an image if an explicit character portrait is supplied.
        Album cover artwork is never treated as a face portrait.
        """
        if custom_image:
            img = resolve_image_file(custom_image) or resolve_audio_file(custom_image)
            if img:
                return img
        char_img = getattr(job, "character_image_path", None)
        if char_img:
            img = resolve_image_file(char_img) or resolve_audio_file(char_img)
            if img:
                return img
        return None

    @staticmethod
    def _has_video_weights(token: str) -> bool:
        """True when an installed video model matching `token` exists on disk.

        A provider chip must mean "this can run now". `DiffusersWanGenerator
        .is_available` only checks that `diffusers` exposes a pipeline *class*,
        which is true on any machine with the package installed.
        """
        try:
            from app.services.model_manager import model_manager

            for model in model_manager.get_model_tree():
                if model.get("category") != "video" or not model.get("is_installed"):
                    continue
                if token in str(model.get("id", "")):
                    return True
        except Exception:
            return False
        return False

    def get_video_providers(self) -> List[Dict[str, Any]]:
        """Return catalog of local and cloud providers with availability status."""
        return [
            VideoProviderInfo(
                id="local_wan_14b",
                name="Local M3 Max Flagship (Wan 2.1 14B + LivePortrait)",
                provider_type=VideoProviderType.LOCAL,
                description="High-fidelity 14B text-to-video / image-to-video with LivePortrait singing avatar on Apple Silicon.",
                # A "flagship" chip must reflect downloaded weights, not just
                # whether `diffusers` exposes the pipeline class.
                is_available=self._has_video_weights("14"),
                supported_models=["wan_14b", "live_portrait"],
                default_for_tier=True
            ).to_dict(),
            VideoProviderInfo(
                id="local_fast",
                name="Local Fast Studio (Wan 1.3B / LTX-Video + LivePortrait)",
                provider_type=VideoProviderType.LOCAL,
                description="Ultra-fast local preview generation with rapid clip turnaround.",
                # Only claim availability when the engine can actually run: the
                # procedural generator always can, but a diffusion model needs
                # weights. `is_available` on the generator is the honest answer.
                is_available=self._has_video_weights("ltx") or self._has_video_weights("1_3"),
                supported_models=["wan_1.3b", "ltx_video", "live_portrait"]
            ).to_dict(),
            VideoProviderInfo(
                id="cloud_fal",
                name="Cloud GPU Accelerated Studio (Fal.ai)",
                provider_type=VideoProviderType.CLOUD_FAL,
                description="Broadcast studio 1080p generation in parallel on Cloud H100 GPUs.",
                is_available=bool(os.environ.get("FAL_KEY")),
                supported_models=["wan_14b", "live_portrait", "kling"],
                has_api_key=bool(os.environ.get("FAL_KEY")),
                requires_api_key=True
            ).to_dict(),
        ]

    def generate_karaoke_ass(
        self,
        timed_lines: List[Dict[str, Any]],
        width: int = 1280,
        height: int = 720,
        style: str = "neon-cyberpunk",
        subtitle_style: str = "neon",
        aspect_ratio: str = "16:9",
        font_family: Optional[str] = None,
        font_size_override: Optional[int] = None,
    ) -> str:
        """
        Generate Advanced SubStation Alpha (.ass) subtitle content with word-level karaoke highlights.
        """
        palette = STYLE_PALETTES.get(style, STYLE_PALETTES.get("neon-cyberpunk", {}))
        colors = None
        if palette and "primary_color" in palette and "accent_color" in palette:
            colors = (palette["primary_color"], palette["accent_color"])

        return generate_karaoke_ass_script(
            timed_lines=timed_lines,
            width=width,
            height=height,
            visual_style=style,
            subtitle_style=subtitle_style,
            aspect_ratio=aspect_ratio,
            font_family=font_family,
            font_size_override=font_size_override,
            palette_colors=colors,
        )

    def get_job_keyframes(self, job_id: str) -> Dict[int, str]:
        """Scan KEYFRAMES_DIR and return existing keyframes map {clip_index: url} for job."""
        kf_map: Dict[int, str] = {}
        prefix = f"keyframe_{job_id}_"
        if not os.path.isdir(KEYFRAMES_DIR):
            return kf_map

        for fname in os.listdir(KEYFRAMES_DIR):
            if fname.startswith(prefix) and fname.endswith(".png"):
                suffix = fname[len(prefix):-4]
                try:
                    clip_idx = int(suffix)
                    full_p = os.path.join(KEYFRAMES_DIR, fname)
                    mtime = int(os.path.getmtime(full_p))
                    kf_map[clip_idx] = f"/audio/videos/keyframes/{fname}?v={mtime}"
                except (ValueError, OSError):
                    continue
        return kf_map

    async def generate_scene_keyframes(
        self,
        job: Job,
        visual_style: str = "neon-cyberpunk",
        width: int = 1280,
        height: int = 720,
        custom_style_prompt: Optional[str] = None,
        force_regenerate: bool = False,
        user_scenes: Optional[List[Dict[str, Any]]] = None,
        task_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate visual keyframe stills for each scene in the storyboard breakdown.
        Enables user to inspect and approve director concepts before full video diffusion.
        Every scene (vocal performance, narrative, B-roll, instrumental solo) receives
        a dedicated scene still rendered from its director prompt.
        """
        cancel_event = asyncio.Event()
        with self._lock:
            self._keyframe_cancels[str(job.id)] = cancel_event
            if task_id:
                tinfo = VideoTaskStatusInfo(
                    id=task_id,
                    job_id=str(job.id),
                    status="processing",
                    step="Conceiving Scene Breakdown & Director Prompts",
                    progress=5,
                    total_clips=len(user_scenes) if user_scenes else 8,
                    current_clip=0
                )
                self._tasks[task_id] = tinfo
                try:
                    task_queue.enqueue_task(
                        task_id=task_id,
                        task_type="keyframe_generation",
                        payload={"job_id": str(job.id)},
                        initial_status="processing",
                        initial_message="Conceiving Scene Breakdown & Director Prompts"
                    )
                except Exception as _e:
                    logger.debug(f"Could not enqueue keyframe task to durable queue: {_e}")

        # If force-regenerating, purge any existing keyframe files for this job to prevent stale caches
        if force_regenerate and os.path.isdir(KEYFRAMES_DIR):
            prefix = f"keyframe_{job.id}_"
            for fname in os.listdir(KEYFRAMES_DIR):
                if fname.startswith(prefix) and fname.endswith(".png"):
                    try:
                        os.remove(os.path.join(KEYFRAMES_DIR, fname))
                    except OSError:
                        pass

        # Ensure LLM is unloaded before starting segmentation or diffusion
        try:
            from app.services.llm_service import llm_service
            llm_service.unload_local_model()
        except Exception:
            pass

        plan = await asyncio.to_thread(
            video_director.segment_song,
            job=job,
            max_clip_duration=15.0,
            visual_style=visual_style,
            custom_style_prompt=custom_style_prompt,
            user_scenes=user_scenes
        )

        # Unload LLM immediately after director treatment generation to yield unified memory to diffusion
        try:
            from app.services.llm_service import llm_service
            llm_service.unload_local_model()
        except Exception:
            pass

        results = []
        try:
            total_scenes = len(plan.clips)
            for idx, clip in enumerate(plan.clips):
                if cancel_event.is_set():
                    logger.info(f"generate_scene_keyframes: Cancelled by user for job {job.id}")
                    if task_id:
                        self.update_task(task_id, status="cancelled", step="Keyframe generation cancelled by user.", progress=0)
                    break

                clip_idx = int(clip.clip_index or (idx + 1))
                if task_id:
                    self.update_task(
                        task_id,
                        current_clip=idx + 1,
                        total_clips=total_scenes,
                        step=f"Pre-building scene backgrounds... {idx + 1}/{total_scenes} ({'🎤 Vocal' if clip.is_vocal else '🎥 Cinematic'})",
                        progress=10 + int(85 * (idx / max(1, total_scenes)))
                    )

                kf_filename = f"keyframe_{job.id}_{clip_idx:03d}.png"
                kf_path = os.path.join(KEYFRAMES_DIR, kf_filename)

                def _clip_dict(c, c_idx: int) -> Dict[str, Any]:
                    has_file = os.path.isfile(kf_path)
                    mtime = int(os.path.getmtime(kf_path)) if has_file else int(time.time())
                    return {
                        "clip_index": c_idx,
                        "start_time": c.start_time,
                        "end_time": c.end_time,
                        "duration": c.duration,
                        "time_str": c.time_str,
                        "is_vocal": c.is_vocal,
                        "scene_type": c.scene_type,
                        "prompt": c.prompt,
                        "camera": c.camera,
                        "lighting": c.lighting,
                        "directors_note": getattr(c, "directors_note", None),
                        "visual_action": getattr(c, "visual_action", None),
                        "musical_energy": getattr(c, "musical_energy", 3),
                        "section_label": getattr(c, "section_label", None),
                        "lyrics": getattr(c, "lyrics", ""),
                        "keyframe_path": kf_path if has_file else None,
                        "keyframe_url": f"/audio/videos/keyframes/{kf_filename}?v={mtime}" if has_file else None
                    }

                # Preserve existing valid still unless forced or if it is a stale cover copy
                if not force_regenerate and os.path.isfile(kf_path) and os.path.getsize(kf_path) > 0:
                    is_stale_cover = False
                    if job.cover_image_path:
                        cover_full = resolve_image_file(job.cover_image_path) or resolve_audio_file(job.cover_image_path)
                        if cover_full and os.path.isfile(cover_full):
                            try:
                                if os.path.getsize(kf_path) == os.path.getsize(cover_full):
                                    import hashlib
                                    with open(kf_path, "rb") as f1, open(cover_full, "rb") as f2:
                                        if hashlib.md5(f1.read()).digest() == hashlib.md5(f2.read()).digest():
                                            is_stale_cover = True
                            except Exception:
                                pass

                    # If existing still is square legacy cover but widescreen/vertical was requested
                    if not is_stale_cover and width != height:
                        try:
                            from PIL import Image
                            with Image.open(kf_path) as im:
                                w_kf, h_kf = im.size
                                if abs(w_kf - h_kf) <= 2:
                                    is_stale_cover = True
                        except Exception:
                            pass

                    # Cinematic plates must carry the people-free pre-build sidecar: a still
                    # without a matching fingerprint was diffused from a different (or
                    # unconstrained) prompt and must not seed Wan i2v.
                    if not is_stale_cover and not clip.is_vocal:
                        if not self._scene_plate_is_current(kf_path, width, height, str(clip.prompt or ""), visual_style):
                            is_stale_cover = True

                    if not is_stale_cover:
                        results.append(_clip_dict(clip, clip_idx))
                        continue

                # Ensure clean slate for this keyframe file
                if os.path.isfile(kf_path):
                    try:
                        os.remove(kf_path)
                    except OSError:
                        pass

                # Render the scene still from the clip's director prompt. Cinematic scenes
                # go through the people-free scene pre-build so the frame the user approves
                # here is exactly the plate Wan later seeds from; vocal scenes keep the
                # performer-visible still, since lip-sync (not Wan) drives those shots.
                try:
                    from app.services.image_service import image_service
                    if clip.is_vocal:
                        res = image_service.generate_scene_background(
                            prompt=clip.prompt,
                            style=visual_style,
                            width=width,
                            height=height,
                            auto_unload=False,
                            cancel_check=lambda: cancel_event.is_set(),
                        )
                        staged_src = res.get("dest_path") if (res.get("ok") and res.get("dest_path")) else None
                    else:
                        plate_batch = image_service.pregenerate_scene_backgrounds(
                            job_id=job.id,
                            prompts=[str(clip.prompt or "")],
                            style=visual_style,
                            width=width,
                            height=height,
                            index_start=clip_idx,
                            auto_unload=False,
                            cancel_check=lambda: cancel_event.is_set(),
                        )
                        staged_src = (plate_batch.get("stills") or {}).get(0)

                    if cancel_event.is_set():
                        logger.info(f"generate_scene_keyframes: Cancelled during clip {clip_idx}")
                        break

                    if staged_src and os.path.isfile(staged_src):
                        shutil.copy(staged_src, kf_path)
                        if not clip.is_vocal:
                            image_service.stamp_scene_plate(kf_path, str(clip.prompt or ""), visual_style)
                    else:
                        # Procedural fallback still if diffusion is unavailable
                        image_service._generate_raster_cover(
                            prompt=f"{clip.prompt} (Scene {clip_idx})",
                            style=visual_style,
                            width=width,
                            height=height,
                            dest_path=kf_path
                        )
                except (asyncio.CancelledError, KeyboardInterrupt):
                    logger.info(f"generate_scene_keyframes: Cancelled by user at clip {clip_idx}")
                    if task_id:
                        self.update_task(task_id, status="cancelled", step="Keyframe generation cancelled by user.", progress=0)
                    break
                except Exception as e:
                    logger.warning(f"Failed to generate keyframe image for clip {clip_idx} ({e})")
                    try:
                        from app.services.image_service import image_service
                        image_service._generate_raster_cover(
                            prompt=f"{clip.prompt} (Scene {clip_idx})",
                            style=visual_style,
                            width=width,
                            height=height,
                            dest_path=kf_path
                        )
                    except Exception as ex:
                        logger.error(f"Fallback keyframe still generation also failed: {ex}")

                results.append(_clip_dict(clip, clip_idx))

            if task_id and not cancel_event.is_set():
                self.update_task(
                    task_id,
                    status="completed",
                    step="Scene keyframes generated successfully.",
                    progress=100
                )
        finally:
            with self._lock:
                self._keyframe_cancels.pop(str(job.id), None)
            # Batch keyframe rendering complete: release image generation weights from unified memory
            try:
                from app.services.image_service import image_service
                image_service.unload_models()
                from app.core.hardware_lock import GlobalHardwareCoordinator
                GlobalHardwareCoordinator.flush_memory()
            except Exception as e:
                logger.warning(f"Error during post-keyframe cleanup: {e}")

        return results

    @staticmethod
    def _scene_plate_is_current(kf_path: str, width: int, height: int, prompt: str = "", style: str = "") -> bool:
        """True when a staged keyframe can be reused as this scene's Wan plate.

        Reuse needs the pixels to actually fit the requested frame: a square
        leftover (album-cover copy) is never a widescreen plate, and a plate
        rendered at another resolution/aspect gets re-diffused. It also needs the
        `.prompt` sidecar the people-free pre-build writes — a plate without a
        matching sidecar was diffused from a different (or unconstrained) prompt,
        so it is not allowed to seed Wan.
        """
        try:
            if not os.path.isfile(kf_path) or os.path.getsize(kf_path) <= 0:
                return False
            if prompt:
                sidecar = kf_path + ".prompt"
                if not os.path.isfile(sidecar):
                    return False
                from app.services.image_service import image_service
                with open(sidecar, "r", encoding="utf-8") as fh:
                    cached_fp = fh.read().strip()
                # `approved:` = a frame the user hand-picked via a scene retake. It is
                # final by definition, so a differing director prompt never invalidates
                # it — only the pixel checks below can send it back to diffusion.
                if not cached_fp.startswith("approved:"):
                    if cached_fp != image_service.prompt_fingerprint(prompt, style):
                        return False
            from PIL import Image
            with Image.open(kf_path) as im:
                w, h = im.size
        except Exception:
            return False

        req_w, req_h = max(1, int(width)), max(1, int(height))
        if abs(w - h) <= 2 and abs(req_w - req_h) > 2:
            return False
        req_ar, cur_ar = req_w / req_h, w / h
        if abs(req_ar - cur_ar) > 0.08 * max(req_ar, cur_ar):
            return False
        # Anything much smaller upscales into mush once Wan seeds from it.
        return min(w, h) >= int(0.5 * min(req_w, req_h))

    async def _prebuild_scene_plates(
        self,
        job: Job,
        clips: List[Any],
        task_id: str,
        style: str,
        width: int,
        height: int,
        cancel_event: asyncio.Event,
    ) -> Dict[int, str]:
        """Diffuse every missing scene plate in ONE image-model load.

        Runs ahead of the clip loop so the image pipeline loads once, renders all
        N plates and unloads once — replacing the old per-clip load then one still
        then unload cycle that dominated wall time — and emits real
        "Pre-building scene backgrounds... i/N" progress while it works.
        Returns {clip_index: staged_keyframe_path}.
        """
        from app.services.image_service import image_service

        pending: List[Tuple[int, Any]] = []
        for idx, clip in enumerate(clips):
            if getattr(clip, "is_vocal", False):
                continue
            clip_idx = int(clip.clip_index or (idx + 1))
            kf_path = os.path.join(KEYFRAMES_DIR, f"keyframe_{job.id}_{clip_idx:03d}.png")
            if not self._scene_plate_is_current(kf_path, width, height, str(getattr(clip, "prompt", "") or ""), style):
                pending.append((clip_idx, clip))

        if not pending:
            logger.info(f"Scene pre-build skipped for job {job.id}: every scene plate already current")
            return {}

        total = len(pending)
        logger.info(f"Pre-building {total} scene background plates for job {job.id} in a single image load")

        def _on_prebuild_progress(evt: Dict[str, Any]) -> None:
            done = int(evt.get("scene_index", 0) or 0) + 1
            status = str(evt.get("status") or "rendering")
            label = {
                "reused": "reusing cached plate",
                "rendering": "diffusing",
                "rendered": "plate ready",
                "failed": "diffusion unavailable",
            }.get(status, status)
            try:
                self.update_task(
                    task_id,
                    step=f"Pre-building scene backgrounds... {done}/{total} ({label})",
                    progress=12 + int(8 * done / max(1, total)),
                )
            except Exception as _e:
                logger.debug(f"Scene pre-build progress update skipped: {_e}")

        async with GlobalHardwareCoordinator.scoped_device(
            f"Scene Background Pre-Build ({total} plates)", modality="image_gen"
        ):
            loop = asyncio.get_running_loop()
            batch = await loop.run_in_executor(
                None,
                lambda: image_service.pregenerate_scene_backgrounds(
                    job_id=job.id,
                    prompts=[str(getattr(clip, "prompt", "") or "") for _cidx, clip in pending],
                    style=style,
                    width=width,
                    height=height,
                    progress_cb=_on_prebuild_progress,
                    cancel_check=lambda: cancel_event.is_set(),
                )
            )

        raw_stills = batch.get("stills") or {}
        staged: Dict[int, str] = {}
        for order, (clip_idx, clip) in enumerate(pending):
            src = raw_stills.get(order, raw_stills.get(str(order)))
            kf_path = os.path.join(KEYFRAMES_DIR, f"keyframe_{job.id}_{clip_idx:03d}.png")
            if not src or not os.path.isfile(src):
                logger.warning(f"Scene {clip_idx}: no background plate produced; Wan falls back per-scene")
                continue
            try:
                shutil.copy(src, kf_path)
                image_service.stamp_scene_plate(kf_path, str(getattr(clip, "prompt", "") or ""), style)
                staged[clip_idx] = kf_path
            except OSError as exc:
                logger.warning(f"Failed staging scene plate for scene {clip_idx}: {exc}")

        if batch.get("cancelled"):
            logger.info(f"Scene pre-build cancelled for job {job.id} after {len(staged)} plates")
        logger.info(
            f"Scene pre-build job {job.id}: {batch.get('generated', 0)} rendered, "
            f"{batch.get('reused', 0)} reused, {batch.get('failed', 0)} failed"
        )
        return staged

    async def render_advanced_music_video(
        self,
        job: Job,
        task_id: str,
        config: Dict[str, Any]
    ) -> str:
        """
        Execute full production music video orchestration.
        """
        task_info = VideoTaskStatusInfo(
            id=task_id,
            job_id=str(job.id),
            status="processing",
            step="Analyzing Track & Ingesting Stems",
            progress=5,
            total_clips=0,
            current_clip=0
        )
        cancel_event = asyncio.Event()
        with self._lock:
            self._video_cancels[task_id] = cancel_event
            self._video_cancels[str(job.id)] = cancel_event
            self._tasks[task_id] = task_info

        try:
            task_queue.enqueue_task(
                task_id=task_id,
                task_type="video_generation",
                payload={"job_id": str(job.id), "config": config},
                initial_status="processing",
                initial_message="Analyzing Track & Ingesting Stems",
            )
            if job.audio_path:
                task_queue.create_task_workspace(task_id, [job.audio_path])
        except Exception as e:
            logger.warning(f"Failed persisting task {task_id} to SQLite queue: {e}")

        rendered_clips: List[str] = []
        try:
            resolved_master = self.resolve_audio_path(job.audio_path)
            if not resolved_master:
                raise FileNotFoundError(f"Master audio not found for track: {job.audio_path}")

            style = config.get("visual_style", "neon-cyberpunk")
            resolution = config.get("resolution", "720p")
            aspect_ratio = config.get("aspect_ratio", "16:9")
            model_name = config.get("model_name", "wan_14b")
            provider_type = config.get("provider", "local")
            enable_lip_sync = config.get("enable_lip_sync", True)
            burn_lyrics = config.get("burn_lyrics", True)
            subtitle_style = config.get("subtitle_style", "neon")
            transition_style = config.get("transition_style", "beat_cut")
            from app.services.video.consistency import (
                apply_character_bible,
                character_bible,
                chain_allowed,
                extract_last_frame,
                normalise_mode,
                scene_seed,
            )

            consistency_mode = normalise_mode(config.get("consistency_mode"))
            chain_scope = config.get("chain_scope")

            if aspect_ratio == "9:16":
                w, h = (1080, 1920) if resolution == "1080p" else (720, 1280)
            elif aspect_ratio == "1:1":
                w, h = (1080, 1080) if resolution == "1080p" else (720, 720)
            elif aspect_ratio == "21:9":
                w, h = (2560, 1080) if resolution == "1080p" else (1680, 720)
            else: # 16:9 widescreen default
                w, h = (1920, 1080) if resolution == "1080p" else (1280, 720)

            # One frame rate for the whole render (plan 4.8). Extraction used a
            # hardcoded 24 while slicing used the model's fps (16 for Wan), so
            # beat pulses landed off-frame.
            from app.services.video.model_specs import get_model_spec

            model_spec = get_model_spec(model_name)
            project_fps = int(getattr(model_spec, "fps", 24) or 24)

            vocal_stem = self.resolve_vocals_stem(job)
            face_image = self.resolve_face_image(job, config.get("character_image_path") or config.get("face_image_path"))

            # Step 1: Song & Scene Planning
            self.update_task(task_id, step="Segmenting Song into Musical Scenes", progress=12)

            # Extract stem audio reactive curves for lip-sync and dynamic camera pulse
            stem_reactivity = None
            try:
                drums_stem = self.resolve_stem(job, "drums")
                bass_stem = self.resolve_stem(job, "bass")
                stem_reactivity = extract_stem_reactive_modulation(
                    drums_stem_path=drums_stem,
                    bass_stem_path=bass_stem,
                    fps=project_fps,
                )
                logger.info(f"Stem audio-reactivity analysis completed ({stem_reactivity.get('total_frames')} frames)")
            except Exception as e:
                logger.warning(f"Stem audio-reactivity analysis skipped: {e}")
            plan = await asyncio.to_thread(
                video_director.segment_song,
                job=job,
                max_clip_duration=config.get("max_clip_duration"),
                model_name=model_name,
                bpm=config.get("bpm"),
                visual_style=style,
                vocal_stem_path=vocal_stem or resolved_master,
                character_desc=config.get("character_desc") or config.get("characterPromptNote"),
                custom_style_prompt=config.get("custom_style_prompt"),
                pacing_bias=int(config.get("pacing_bias", 0)),
                visible_cast=config.get("visible_cast"),
                user_scenes=config.get("scenes") or config.get("clips")
            )
            total_clips = plan.total_clips
            # Audit E6: one character description, injected into every scene, so the
            # text conditioning does not describe eight different people.
            bible = character_bible(config, plan)
            if bible and normalise_mode(config.get("consistency_mode")) != "off":
                for clip in plan.clips:
                    clip.prompt = apply_character_bible(clip.prompt, bible)
                logger.info("Character bible applied to %d scenes (mode=%s)", len(plan.clips), consistency_mode)
            self.update_task(task_id, total_clips=total_clips, clips=[c.to_dict() for c in plan.clips])

            from app.services.video.generator_registry import VideoGeneratorRegistry

            # model_spec was resolved before reactive analysis so one fps flows
            # through planning, extraction and rendering (plan 4.8).
            video_generator = VideoGeneratorRegistry.resolve(model_name, provider_type)

            # Plan 7.27: honour the engine the request asked for. It used to be
            # ignored, so the UI offered a choice that changed nothing.
            from app.services.video.lip_sync.registry import (
                resolve_cloud_provider,
                resolve_local_provider,
            )

            requested_engine = config.get("lip_sync_engine") or "live_portrait"
            lipsync_fallback_reason: Optional[str] = None
            cloud_provider = resolve_cloud_provider(provider_type)
            if cloud_provider is not None:
                lipsync_provider = cloud_provider
            else:
                lipsync_provider, lipsync_fallback_reason = resolve_local_provider(requested_engine)

            if lipsync_fallback_reason:
                task_info.fallback_used = True
                task_info.fallback_reason = lipsync_fallback_reason
                logger.info("Lip sync: %s", lipsync_fallback_reason)
            task_info.lip_sync_engine = getattr(lipsync_provider, "name", requested_engine)

            is_local = (provider_type not in ("cloud_fal", "cloud_replicate", "cloud_minimax"))

            # Phase A: build every missing scene plate in ONE image-model load, while
            # the image model is the only thing on the accelerator (progress 12→20%).
            # The clip loop below never loads the image pipeline again.
            if is_local:
                scene_plates = await self._prebuild_scene_plates(
                    job=job,
                    clips=plan.clips,
                    task_id=task_id,
                    style=style,
                    width=w,
                    height=h,
                    cancel_event=cancel_event,
                )
                if scene_plates:
                    self.update_task(task_id, step="Scene background plates ready", progress=20)

            # Step 2: Render individual scene clips
            self.update_task(task_id, step="Rendering Video Scenes & Lip-Sync Performance", progress=20)
            if cancel_event.is_set():
                logger.info("render_advanced_music_video: Cancelled before scene rendering.")
                self.update_task(task_id, status="cancelled", step="Video rendering cancelled by user.", progress=0)
                return ""

            try:
                _is_cancelled = lambda: cancel_event.is_set()

                previous_clip_file: Optional[str] = None
                previous_section: Optional[str] = None

                async def _render_all_scenes():
                    nonlocal previous_clip_file, previous_section
                    for idx, clip in enumerate(plan.clips):
                        if cancel_event.is_set():
                            logger.info(f"render_advanced_music_video: Cancelled before rendering scene {idx + 1}.")
                            self.update_task(task_id, status="cancelled", step="Video rendering cancelled by user.", progress=0)
                            raise asyncio.CancelledError("Video rendering cancelled by user.")

                        clip_file = os.path.join(TEMP_DIR, f"clip_{task_id}_{idx:03d}.mp4")
                        self.update_task(
                            task_id,
                            current_clip=idx + 1,
                            current_clip_type=clip.scene_type,
                            step=f"Rendering Scene {idx + 1}/{total_clips} ({'🎤 Vocal Performance' if clip.is_vocal else '🎥 Cinematic B-Roll'})",
                            progress=20 + int(60 * (idx / total_clips))
                        )

                        # Vocal Singing Scene
                        vocal_audio_source = vocal_stem or resolved_master
                        if clip.is_vocal and enable_lip_sync and face_image and vocal_audio_source:
                            logger.info(f"Rendering singing performance for Scene {idx + 1} with {lipsync_provider.name} (audio: {os.path.basename(vocal_audio_source)})...")
                            success = await lipsync_provider.render_lip_sync(
                                face_image_path=face_image,
                                vocal_audio_path=vocal_audio_source,
                                start_time=clip.start_time,
                                duration=clip.duration,
                                out_path=clip_file,
                                width=w, height=h,
                                cancel_event=cancel_event,
                                cancel_check=_is_cancelled
                            )
                            clip_outcome = dict(getattr(lipsync_provider, "last_outcome", {}) or {})
                            if not success or not os.path.isfile(clip_file) or os.path.getsize(clip_file) == 0:
                                if cancel_event.is_set():
                                    raise asyncio.CancelledError("Video rendering cancelled by user.")
                                # Fallback to the smooth (jaw-warp) provider. This is
                                # NOT a neural avatar: say so, the same way the video
                                # diffusion path records its fallback.
                                task_info.fallback_used = True
                                task_info.fallback_reason = (
                                    clip_outcome.get("reason")
                                    or f"Neural lip sync unavailable for clip {idx + 1}; "
                                       "used the procedural jaw-warp animator."
                                )
                                await self._fallback_lipsync.render_lip_sync(
                                    face_image_path=face_image,
                                    vocal_audio_path=vocal_audio_source,
                                    start_time=clip.start_time,
                                    duration=clip.duration,
                                    out_path=clip_file,
                                    width=w, height=h,
                                    cancel_event=cancel_event,
                                    cancel_check=_is_cancelled
                                )
                                task_info.lip_sync_engine = "jaw_warp"
                            elif clip_outcome.get("used_fallback"):
                                # The provider succeeded, but by delegating to a
                                # different engine. That is still a fallback and the
                                # user must be told (audit E4).
                                task_info.fallback_used = True
                                task_info.fallback_reason = (
                                    clip_outcome.get("reason")
                                    or "Neural lip sync unavailable; used the procedural animatic."
                                )
                                task_info.lip_sync_engine = clip_outcome.get("engine")
                            else:
                                task_info.lip_sync_engine = clip_outcome.get("engine") or lipsync_provider.name

                        # Cinematic Scene
                        else:
                            clip_idx = int(clip.clip_index or (idx + 1))
                            logger.info(f"Rendering visual scene {clip_idx} with {video_generator.name}...")
                            # Check if an approved pre-rendered keyframe exists on disk
                            pre_rendered_kf = os.path.join(KEYFRAMES_DIR, f"keyframe_{job.id}_{clip_idx:03d}.png")
                            # Continuity: the previous clip's final frame carries the
                            # grade and framing into this scene (mode="chain").
                            chained_frame = None
                            if (
                                consistency_mode == "chain"
                                and previous_clip_file
                                and chain_allowed(previous_section, clip.section_label, chain_scope)
                            ):
                                chained_frame = extract_last_frame(
                                    previous_clip_file,
                                    os.path.join(TEMP_DIR, f"chain_{task_id}_{idx:03d}.png"),
                                )
                            scene_bg = scene_seed(
                                approved_plate=pre_rendered_kf,
                                chained_frame=chained_frame,
                                face_image=face_image,
                                mode=consistency_mode,
                            )

                            fb_meta: Dict[str, Any] = {}
                            success = await video_generator.generate_clip(
                                prompt=clip.prompt,
                                duration=clip.duration,
                                out_path=clip_file,
                                width=w, height=h,
                                image_path=scene_bg,
                                negative_prompt=clip.negative_prompt,
                                visual_style=style,
                                provider=provider_type,
                                fallback_metadata=fb_meta,
                                cancel_event=cancel_event,
                                cancel_check=_is_cancelled
                            )
                            if cancel_event.is_set():
                                raise asyncio.CancelledError("Video rendering cancelled by user.")
                            if fb_meta.get("fallback_used"):
                                task_info.fallback_used = True
                                task_info.fallback_reason = fb_meta.get("error", "Procedural fallback used")
                            if not success or not os.path.isfile(clip_file) or os.path.getsize(clip_file) == 0:
                                task_info.fallback_used = True
                                task_info.fallback_reason = fb_meta.get("error") or f"{video_generator.name} output empty; falling back to procedural animatic."
                                # Fallback to procedural generator
                                await self._procedural.generate_clip(
                                    prompt=clip.prompt,
                                    duration=clip.duration,
                                    out_path=clip_file,
                                    width=w, height=h,
                                    image_path=scene_bg,
                                    visual_style=style,
                                    fps=model_spec.fps,
                                    cancel_event=cancel_event,
                                    cancel_check=_is_cancelled
                                )

                        if os.path.isfile(clip_file) and os.path.getsize(clip_file) > 0:
                            # The beat pulse is applied inside the normalise pass
                            # below, not as a separate `ultrafast` re-encode (audit M5).
                            rendered_clips.append(clip_file)
                            clip.rendered_clip_path = clip_file
                            clip.status = "completed"
                            previous_clip_file = clip_file
                            previous_section = clip.section_label

                if is_local:
                    async with GlobalHardwareCoordinator.scoped_device(
                        f"Video Generation: {video_generator.name} ({total_clips} scenes)",
                        modality="video_gen"
                    ):
                        await _render_all_scenes()
                else:
                    await _render_all_scenes()

            finally:
                with self._lock:
                    self._video_cancels.pop(task_id, None)
                    self._video_cancels.pop(str(job.id), None)
                    self._active_render_tasks.pop(task_id, None)
                if is_local and hasattr(video_generator, "unload"):
                    try:
                        video_generator.unload()
                        GlobalHardwareCoordinator.flush_memory()
                    except Exception as _e:
                        logger.debug(f"Video generator unload skipped: {_e}")
                # Intermediates are not needed once the master is remuxed.
                self._cleanup_task_temp(task_id)

            if cancel_event.is_set():
                logger.info(f"render_advanced_music_video: Cancelled after scenes render.")
                self.update_task(task_id, status="cancelled", step="Video rendering cancelled by user.", progress=0)
                return ""

            if not rendered_clips:
                raise RuntimeError("No video scenes were successfully rendered.")

            # Step 2.5: Normalise every clip to its planned duration and the project
            # geometry. Diffusion only emits frame counts on a lattice and rounds
            # down (a 3.7 s plan renders 3.56 s); nothing consumed the trim the
            # planner computed, so a 40-clip video finished seconds before the song
            # and every later cut drifted off its beat.
            # Plan 7.3: report the resolution the generator really produced. On MPS
            # Wan clamps to 832x480, and the "1080p" master is an upscale — the user
            # must be able to see that.
            try:
                from app.services.video.export_presets import probe_streams as _probe_streams

                first = _probe_streams(rendered_clips[0]) if rendered_clips else {}
                if first.get("width") and first.get("height"):
                    self.update_task(
                        task_id,
                        source_resolution=f"{first['width']}x{first['height']}",
                        delivered_resolution=f"{w}x{h}",
                    )
                    if (first["width"], first["height"]) != (w, h):
                        logger.info(
                            "Delivered %sx%s from %sx%s source (upscale %.2fx)",
                            w, h, first["width"], first["height"],
                            w / max(1, first["width"]),
                        )
            except Exception as _res_err:
                logger.debug("Could not probe source resolution: %s", _res_err)

            self.update_task(task_id, step="Normalising scene durations & geometry", progress=78)
            # A crossfade overlaps neighbours, so the stitched video would be
            # (n-1)*trans_dur shorter than the sum of its clips even with perfect
            # per-clip durations. Give every non-final clip that much extra so the
            # assembled picture lands on the song length.
            rendered_clips = await self._normalise_rendered_clips(
                rendered_clips=rendered_clips,
                plan_clips=plan.clips,
                width=w, height=h, fps=model_spec.fps,
                task_id=task_id,
                cancel_event=cancel_event,
                transition_overlap=self._transition_overlap(transition_style, plan.clips),
                stem_reactivity=stem_reactivity if config.get("enable_audio_reactive", True) else None,
            )

            # Step 3: Scene Assembly & Concat
            self.update_task(task_id, step=f"Assembling & Stitching Video Scenes ({transition_style})", progress=82)
            stitched_video = os.path.join(TEMP_DIR, f"stitched_{task_id}.mp4")
            stitch_ok = await self._stitch_video_segments_with_transitions(
                rendered_clips=rendered_clips,
                transition_style=transition_style,
                stitched_video=stitched_video,
                task_id=task_id
            )
            if not stitch_ok or not os.path.isfile(stitched_video) or os.path.getsize(stitched_video) == 0:
                raise RuntimeError("Video scene stitching failed.")

            # Step 4: Subtitle Burning & Master Audio Remuxing
            self.update_task(task_id, step="Burning Synchronized Karaoke Subtitles & Master Remux", progress=92)
            out_filename = f"{job.id}_master_mv.mp4"
            out_path = os.path.join(VIDEO_DIR, out_filename)

            # Check if subtitle burning requested
            ass_path = None
            if burn_lyrics and job.lyrics:
                try:
                    timed_lines = lyric_sync_engine.align_lyrics(
                        lyrics=job.lyrics,
                        duration_sec=float((job.duration_ms or 180000) / 1000.0),
                        vocal_stem_path=vocal_stem or resolved_master
                    )
                    ass_content = self.generate_karaoke_ass(
                        timed_lines=timed_lines,
                        width=w, height=h,
                        style=style,
                        subtitle_style=subtitle_style,
                        aspect_ratio=aspect_ratio,
                    )
                    ass_path = os.path.join(TEMP_DIR, f"subtitles_{task_id}.ass")
                    with open(ass_path, "w", encoding="utf-8") as f_sub:
                        f_sub.write(ass_content)
                except Exception as ex:
                    logger.warning(f"Could not generate ASS subtitles: {ex}")
                    ass_path = None

            # The song is the master clock. `-shortest` truncated the song to the
            # (slightly short) picture; instead fix the output length to the audio
            # and pad the video with its last frame when assembly came up short.
            from app.services.video.clip_normalise import probe_duration as _probe
            from app.services.video.clip_normalise import verify_assembly as _verify

            song_duration = _probe(resolved_master)
            stitched_duration = _probe(stitched_video)
            assembly_report = _verify(stitched_duration, song_duration)
            if assembly_report and not assembly_report["ok"]:
                logger.warning(
                    "Assembly drift: video %.3fs vs song %.3fs (%.3fs). Padding to the song.",
                    assembly_report["video_duration_s"], assembly_report["song_duration_s"],
                    assembly_report["drift_s"],
                )

            delivery = await self._finalise_master(
                stitched_video=stitched_video,
                audio_path=resolved_master,
                out_path=out_path,
                width=w,
                height=h,
                song_duration=song_duration,
                ass_path=ass_path,
                timed_lines=timed_lines if burn_lyrics and job.lyrics else [],
                preset_id=config.get("export_preset") or "default",
                delivery_fps=float(config["delivery_fps"]) if config.get("delivery_fps") else None,
            )
            qc_report = delivery["qc"]
            self.update_task(task_id, qc=qc_report)

            # Per-scene clips and the plan stay on disk for incremental retakes
            # (plan 7.11) instead of being deleted after assembly.
            render_id = config.get("render_id")
            if render_id:
                try:
                    from app.services.video import takes as takes_service

                    takes_service.save_clip_plan(
                        render_id,
                        {
                            "task_id": task_id,
                            "job_id": str(job.id),
                            "model_name": model_name,
                            "bpm": getattr(plan, "bpm", None),
                            "meter": getattr(plan, "meter", 4),
                            "structure_source": getattr(plan, "structure_source", "energy_windows"),
                            "scenes": [c.to_dict() for c in plan.clips],
                            "config": config,
                        },
                        rendered_clips,
                    )
                except Exception as _persist_err:
                    logger.debug("Could not persist clips for take %s: %s", render_id, _persist_err)

            video_url = f"/audio/videos/{out_filename}"
            if render_id:
                self._record_take(
                    render_id=render_id,
                    kind="advanced",
                    video_url=video_url,
                    file_path=out_path,
                    poster_path=os.path.join(VIDEO_DIR, "previews", out_filename.replace(".mp4", "_poster.jpg")),
                    proxy_path=os.path.join(VIDEO_DIR, "previews", out_filename.replace(".mp4", "_480p.mp4")),
                    qc=locals().get("qc_report"),
                )
                self.update_task(task_id, render_id=render_id)

            self.update_task(
                task_id,
                status="completed",
                progress=100,
                step=(
                    "Production Video Complete"
                    if not assembly_report or assembly_report["ok"]
                    else f"Video complete (drift {assembly_report['drift_s']:+.2f}s vs song)"
                ),
                video_url=video_url
            )
            return video_url

        except asyncio.CancelledError:
            logger.info(f"render_advanced_music_video: Task {task_id} successfully cancelled.")
            self.update_task(task_id, status="cancelled", step="Video rendering cancelled by user.", progress=0)
            return ""
        except Exception as e:
            if cancel_event.is_set():
                logger.info(f"render_advanced_music_video: Task {task_id} aborted cleanly due to cancellation.")
                self.update_task(task_id, status="cancelled", step="Video rendering cancelled by user.", progress=0)
                return ""
            logger.error(f"Advanced video generation failed: {e}", exc_info=True)
            self.update_task(task_id, status="error", error=str(e), step=f"Error: {str(e)[:120]}")
            raise e
        finally:
            # Clean up temp files
            for cf in rendered_clips:
                if os.path.isfile(cf):
                    try: os.remove(cf)
                    except: pass

    async def render_scene_retake(
        self,
        *,
        job: Job,
        task_id: str,
        render_id: str,
        clip_index: int,
        config: Dict[str, Any],
    ) -> str:
        """Re-render ONE scene from a completed take and re-assemble (plan 7.11).

        Every other scene is reused from the take's persisted clips, so a retake
        costs one diffusion (or lip-sync) pass instead of the whole film. The result
        is a *new* take: the previous master is never overwritten (plan 7.4).
        """
        from app.services.video import takes as takes_service
        from app.services.video.clip_normalise import probe_duration, verify_assembly

        plan = await asyncio.to_thread(takes_service.load_clip_plan, render_id)
        if not plan or not plan.get("clips"):
            raise ValueError(
                "This take has no per-scene clips on disk; render the full video once "
                "before retaking a scene."
            )

        scenes: List[Dict[str, Any]] = list(plan.get("scenes") or [])
        target_idx = None
        for idx, scene in enumerate(scenes):
            if int(scene.get("clip_index") or idx + 1) == int(clip_index):
                target_idx = idx
                break
        if target_idx is None:
            raise ValueError(f"Scene {clip_index} is not in this take's plan.")

        scene = dict(scenes[target_idx])
        if config.get("prompt"):
            scene["prompt"] = str(config["prompt"])
        style = config.get("visual_style") or plan.get("config", {}).get("visual_style") or "neon-cyberpunk"
        model_name = plan.get("config", {}).get("model_name") or plan.get("model_name") or "wan_14b"
        provider_type = plan.get("config", {}).get("provider") or "local"
        transition_style = config.get("transition_style") or plan.get("config", {}).get("transition_style") or "beat_cut"
        burn_lyrics = bool(config.get("burn_lyrics", plan.get("config", {}).get("burn_lyrics", True)))

        width = int(config.get("width") or plan.get("config", {}).get("width") or 1280)
        height = int(config.get("height") or plan.get("config", {}).get("height") or 720)
        from app.services.video.model_specs import get_model_spec

        spec = get_model_spec(model_name)
        fps = int(getattr(spec, "fps", 24) or 24)

        self.update_task(task_id, step=f"Re-rendering scene {clip_index}", progress=10)
        cancel_event = asyncio.Event()
        with self._lock:
            self._video_cancels[task_id] = cancel_event

        clip_file = os.path.join(TEMP_DIR, f"retake_{task_id}_{clip_index:03d}.mp4")
        rendered = False
        try:
            if scene.get("is_vocal") and config.get("enable_lip_sync", True):
                face = self.resolve_face_image(job)
                vocal = self.resolve_vocals_stem(job) or self.resolve_audio_path(job.audio_path)
                if face and vocal:
                    from app.services.video.lip_sync.live_portrait import LivePortraitProvider

                    provider = LivePortraitProvider()
                    rendered = await provider.render_lip_sync(
                        face_image_path=face,
                        vocal_audio_path=vocal,
                        start_time=float(scene.get("start_time") or 0.0),
                        duration=float(scene.get("duration") or 0.0),
                        out_path=clip_file,
                        width=width, height=height,
                        cancel_event=cancel_event,
                        cancel_check=lambda: cancel_event.is_set(),
                    )
                    outcome = dict(getattr(provider, "last_outcome", {}) or {})
                    if outcome.get("used_fallback"):
                        self.update_task(
                            task_id,
                            fallback_used=True,
                            fallback_reason=outcome.get("reason"),
                            lip_sync_engine=outcome.get("engine"),
                        )

            if not rendered:
                from app.services.video.generator_registry import VideoGeneratorRegistry

                generator = VideoGeneratorRegistry.resolve(model_name, provider_type)
                kf_path = os.path.join(KEYFRAMES_DIR, f"keyframe_{job.id}_{clip_index:03d}.png")
                rendered = await generator.generate_clip(
                    prompt=str(scene.get("prompt") or ""),
                    duration=float(scene.get("duration") or spec.default_duration),
                    out_path=clip_file,
                    width=width, height=height,
                    image_path=kf_path if os.path.isfile(kf_path) else self.resolve_face_image(job),
                    negative_prompt=str(scene.get("negative_prompt") or ""),
                    visual_style=style,
                    provider=provider_type,
                    cancel_event=cancel_event,
                    cancel_check=lambda: cancel_event.is_set(),
                )

            if not rendered or not os.path.isfile(clip_file) or os.path.getsize(clip_file) == 0:
                raise RuntimeError(f"Scene {clip_index} produced no output.")

            # Normalise the new clip to the plan's duration and the take's geometry.
            from app.services.video.clip_normalise import normalise_clip

            normalised = os.path.join(TEMP_DIR, f"retake_norm_{task_id}_{clip_index:03d}.mp4")
            target_duration = float(scene.get("duration") or 0.0)
            self.update_task(task_id, step=f"Normalising scene {clip_index}", progress=45)
            ok = await asyncio.to_thread(
                normalise_clip, clip_file, normalised,
                duration_s=target_duration, width=width, height=height, fps=fps,
            )
            if not ok:
                raise RuntimeError(f"Scene {clip_index} could not be normalised.")

            clip_paths = list(plan["clips"])
            clip_paths[target_idx] = normalised
            scenes[target_idx] = scene

            # Re-assemble from the persisted clips (no re-diffusion of other scenes).
            self.update_task(task_id, step="Re-assembling timeline", progress=60)
            stitched = os.path.join(TEMP_DIR, f"retake_stitch_{task_id}.mp4")
            stitched_ok = await self._stitch_video_segments_with_transitions(
                rendered_clips=clip_paths,
                transition_style=transition_style,
                stitched_video=stitched,
                task_id=task_id,
            )
            if not stitched_ok:
                raise RuntimeError("Re-assembly failed.")

            audio_path = self.resolve_audio_path(job.audio_path)
            song_duration = probe_duration(audio_path) if audio_path else None
            verify_assembly(probe_duration(stitched), song_duration)

            # Subtitles for the re-assembled master.
            ass_path = None
            timed_lines: List[Dict[str, Any]] = []
            if burn_lyrics and job.lyrics:
                try:
                    timed_lines = lyric_sync_engine.align_lyrics(
                        lyrics=job.lyrics,
                        duration_sec=float(song_duration or (job.duration_ms or 180000) / 1000.0),
                        vocal_stem_path=self.resolve_vocals_stem(job) or audio_path,
                    )
                    ass_content = self.generate_karaoke_ass(
                        timed_lines=timed_lines, width=width, height=height,
                        style=style,
                        subtitle_style=config.get("subtitle_style") or "neon",
                        aspect_ratio=config.get("aspect_ratio") or "16:9",
                    )
                    ass_path = os.path.join(TEMP_DIR, f"retake_{task_id}.ass")
                    with open(ass_path, "w", encoding="utf-8") as fh:
                        fh.write(ass_content)
                except Exception as exc:
                    logger.warning("Retake subtitle generation skipped: %s", exc)

            tag = uuid.uuid4().hex[:8]
            out_filename = f"{job.id}_master_mv_{tag}.mp4"
            out_path = os.path.join(VIDEO_DIR, out_filename)
            self.update_task(task_id, step="Encoding retake master", progress=80)
            delivery = await self._finalise_master(
                stitched_video=stitched,
                audio_path=audio_path,
                out_path=out_path,
                width=width, height=height,
                song_duration=song_duration,
                ass_path=ass_path,
                timed_lines=timed_lines,
                preset_id=config.get("export_preset") or plan.get("config", {}).get("export_preset") or "default",
            )

            # New take row, selected; the previous master stays on disk.
            new_render_id = None
            try:
                from sqlmodel import Session

                from app.main import engine

                with Session(engine) as session:
                    take = takes_service.create_take(
                        session,
                        job_id=str(job.id),
                        task_id=task_id,
                        kind="retake",
                        config={**plan.get("config", {}), "retake_clip": clip_index,
                                "source_render": render_id},
                    )
                    new_render_id = str(take.id)
                    takes_service.finish_take(
                        session, take, status="completed",
                        video_path=f"/audio/videos/{out_filename}",
                        file_path=out_path,
                        qc=delivery["qc"],
                    )
                    takes_service.select_take(session, str(job.id), new_render_id)
                    takes_service.save_clip_plan(
                        new_render_id,
                        {**plan, "scenes": scenes, "retake_clip": clip_index},
                        clip_paths,
                    )
            except Exception as exc:
                logger.warning("Could not record the retake as a take: %s", exc)

            video_url = f"/audio/videos/{out_filename}"
            self.update_task(
                task_id, status="completed", progress=100,
                step=f"Scene {clip_index} re-rendered", video_url=video_url,
                qc=delivery["qc"], render_id=new_render_id,
            )
            return video_url
        except asyncio.CancelledError:
            self.update_task(task_id, status="cancelled", step="Retake cancelled.", progress=0)
            return ""
        except Exception as exc:
            logger.error("Scene retake failed: %s", exc, exc_info=True)
            self.update_task(task_id, status="error", error=str(exc)[:300])
            raise
        finally:
            with self._lock:
                self._video_cancels.pop(task_id, None)
            for path in (clip_file,):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except OSError:
                    pass

    async def plan_music_video_async(
        self,
        job_id: str,
        task_id: str,
        max_clip_duration: Optional[float] = None,
        model_name: Optional[str] = "wan_14b",
        bpm: Optional[float] = None,
        visual_style: str = "neon-cyberpunk",
        custom_style_prompt: Optional[str] = None,
        pacing_bias: int = 0,
        character_desc: Optional[str] = None,
        visible_cast: Optional[List[str]] = None,
        user_scenes: Optional[List[Dict[str, Any]]] = None,
        use_llm: bool = True,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Asynchronously plan a multi-scene music video with real-time HUD progress updates.
        Updates task status across:
          1. 10%: Audio analysis & BPM detection
          2. 25%: Lyric alignment & vocal cadence
          3. 50%: AI Visual Director LLM conceptualization
          4. 85%: Storyboard lattice snapping & prompt formatting
          5. 100%: Persistence to SQLite DB & completion with clips and treatment
        """
        from app.main import engine, get_job_by_id
        from app.models import Job
        from sqlmodel import Session
        from app.services.video.video_director import video_director

        cancel_event = asyncio.Event()
        with self._lock:
            self._plan_cancels[task_id] = cancel_event
            tinfo = VideoTaskStatusInfo(
                id=task_id,
                job_id=job_id,
                status="processing",
                step="Analyzing Audio Signal & Musical Downbeats",
                progress=10,
                total_clips=0,
                current_clip=0
            )
            self._tasks[task_id] = tinfo
            try:
                task_queue.enqueue_task(
                    task_id=task_id,
                    task_type="video_planning",
                    payload={"job_id": job_id},
                    initial_status="processing",
                    initial_message="Analyzing Audio Signal & Musical Downbeats"
                )
            except Exception as _e:
                logger.debug(f"Could not enqueue planning task to durable queue: {_e}")

        def _progress_cb(step_desc: str, pct: int):
            if cancel_event.is_set():
                raise asyncio.CancelledError("Planning cancelled by user.")
            self.update_task(task_id, step=step_desc, progress=pct)

        try:
            with Session(engine) as session:
                job = get_job_by_id(session, job_id)
                if not job:
                    raise ValueError(f"Job {job_id} not found")

            if cancel_event.is_set():
                raise asyncio.CancelledError("Planning cancelled by user.")

            model_name = model_name or "wan_14b"
            model_max = video_director.get_model_max_duration(model_name)
            if max_clip_duration is not None and float(max_clip_duration) > 0:
                clip_dur = max(1.0, min(float(max_clip_duration), model_max))
            else:
                clip_dur = model_max

            # Execute segmentation in worker thread so event loop remains responsive
            plan = await asyncio.to_thread(
                video_director.segment_song,
                job=job,
                max_clip_duration=clip_dur,
                model_name=model_name,
                bpm=bpm,
                visual_style=visual_style or "neon-cyberpunk",
                vocal_stem_path=self.resolve_vocals_stem(job),
                character_desc=character_desc,
                custom_style_prompt=custom_style_prompt,
                pacing_bias=pacing_bias or 0,
                visible_cast=visible_cast,
                user_scenes=user_scenes,
                use_llm=use_llm,
                force_refresh=force_refresh,
                progress_callback=_progress_cb,
                cancel_check=lambda: cancel_event.is_set()
            )

            if cancel_event.is_set():
                raise asyncio.CancelledError("Planning cancelled by user.")

            plan_dict = plan.to_dict()
            clips = plan_dict.get("clips", [])
            treatment = plan_dict.get("treatment")
            fallback_used = getattr(plan, "fallback_used", False)
            fallback_reason = getattr(plan, "fallback_reason", None)

            # Persist plan to SQLite Job database
            _progress_cb("Saving Scene Storyboard to Project", 95)
            with Session(engine) as session:
                j = session.get(Job, job.id)
                if j:
                    existing_cfg = {}
                    if j.video_config_json:
                        try:
                            existing_cfg = json.loads(j.video_config_json)
                        except Exception:
                            pass
                    if treatment:
                        existing_cfg["director_treatment"] = treatment
                    existing_cfg["scenes"] = clips
                    existing_cfg["visual_style"] = visual_style or "neon-cyberpunk"
                    existing_cfg["model_name"] = model_name
                    j.video_config_json = json.dumps(existing_cfg, default=str)
                    session.add(j)
                    session.commit()

            self.update_task(
                task_id,
                status="completed",
                progress=100,
                step="Scene Planning Complete",
                total_clips=len(clips),
                clips=clips,
                treatment=treatment,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                result={"clips": clips, "treatment": treatment}
            )
            return {
                "status": "completed",
                "task_id": task_id,
                "job_id": job_id,
                "clips": clips,
                "treatment": treatment
            }

        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info(f"Scene planning task {task_id} cancelled.")
            self.update_task(task_id, status="cancelled", step="Planning cancelled by user.")
            try:
                from app.services.llm_service import LLMService
                LLMService.unload_local_model()
            except Exception:
                pass
            return {"status": "cancelled", "task_id": task_id}

        except Exception as e:
            logger.error(f"Scene planning task {task_id} failed: {e}", exc_info=True)
            self.update_task(task_id, status="failed", error=str(e), step=f"Planning Failed: {str(e)[:100]}")
            return {"status": "failed", "task_id": task_id, "error": str(e)}

        finally:
            with self._lock:
                self._plan_cancels.pop(task_id, None)

    async def _probe_clip_duration(self, file_path: str) -> float:
        """Probe video file duration in seconds using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, _ = await proc.communicate()
            val = float(out.decode().strip())
            return max(0.5, val)
        except Exception:
            return 3.0

    async def _finalise_master(
        self,
        *,
        stitched_video: str,
        audio_path: str,
        out_path: str,
        width: int,
        height: int,
        song_duration: Optional[float],
        ass_path: Optional[str],
        timed_lines: List[Dict[str, Any]],
        preset_id: str = "default",
        delivery_fps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Encode the delivered master: one preset-targeted pass, QC, sidecars.

        Shared by the full render and the single-scene retake (plan 7.11) so both
        produce identical delivery characteristics (plans 7.12-7.16).
        """
        from app.services.video.export_presets import (
            audio_args,
            container_args,
            get_preset,
            loudnorm_filter,
            probe_streams,
            video_args,
        )
        from app.services.video.clip_normalise import interpolation_filter
        from app.services.video.qc import run_qc

        preset = get_preset(preset_id)
        master_streams = probe_streams(audio_path) or {}
        picture_streams = probe_streams(stitched_video) or {}
        source_is_aac = (master_streams.get("audio_codec") or "").lower() == "aac"
        has_audio = bool(master_streams.get("audio_codec"))

        video_chain: List[str] = []
        if ass_path and os.path.isfile(ass_path):
            escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
            fonts_dir = get_fonts_dir()
            fonts_opt = ""
            if fonts_dir:
                fonts_safe = fonts_dir.replace("\\", "/").replace(":", "\\:")
                fonts_opt = f":fontsdir='{fonts_safe}'"
            video_chain.append(f"subtitles='{escaped_ass}'{fonts_opt}")
        # Audit G5: 16 fps model output must reach a delivery frame rate with
        # synthesised frames, not duplicates. Explicit request wins; otherwise only
        # low-rate sources are lifted to 24.
        source_fps = picture_streams.get("fps")
        target_fps = delivery_fps
        if target_fps is None and source_fps and source_fps <= 18:
            target_fps = 24.0
        interp = interpolation_filter(source_fps, target_fps)
        if interp:
            video_chain.append(interp)
        if song_duration:
            # Clone the final frame so the picture never ends before the song.
            video_chain.append(f"tpad=stop_mode=clone:stop_duration={song_duration:.3f}")

        filters: List[str] = []
        audio_map = "1:a:0"
        if video_chain:
            filters.append("[0:v]" + ",".join(video_chain) + "[vout]")
        loudness = loudnorm_filter(preset)
        if loudness and has_audio:
            filters.append(f"[1:a]{loudness}[aout]")
            audio_map = "[aout]"

        cmd = ["ffmpeg", "-y", "-i", stitched_video, "-i", audio_path]
        if filters:
            cmd += ["-filter_complex", ";".join(filters)]
        cmd += ["-map", "[vout]" if video_chain else "0:v:0"]
        if has_audio:
            cmd += ["-map", audio_map]
        cmd += video_args(preset)
        if has_audio:
            # `-filter_complex ... [aout]` re-encodes audio, so passthrough is off
            # whenever loudness normalisation is in the graph.
            cmd += audio_args(preset, source_is_aac=source_is_aac and not (loudness and has_audio))
        cmd += container_args(preset)
        if song_duration:
            cmd += ["-t", f"{song_duration:.3f}"]
        cmd += [out_path]

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, err = await proc.communicate()

        if proc.returncode != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            logger.warning(
                "Delivery encode failed (%s): %s",
                proc.returncode, (err or b"").decode("utf-8", errors="ignore")[-300:],
            )
            fallback = ["ffmpeg", "-y", "-i", stitched_video, "-i", audio_path, "-map", "0:v:0"]
            if has_audio:
                fallback += ["-map", "1:a:0"]
            fallback += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                         "-pix_fmt", "yuv420p"]
            if has_audio:
                fallback += ["-c:a", "aac", "-b:a", "320k"]
            fallback += ["-movflags", "+faststart"]
            if song_duration:
                fallback += ["-t", f"{song_duration:.3f}"]
            fallback += [out_path]
            proc_fb = await asyncio.create_subprocess_exec(
                *fallback, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc_fb.communicate()

        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("Failed to generate master music video output file.")

        captions = self._write_caption_sidecars(
            out_path, timed_lines,
            ass_path=ass_path if ass_path and os.path.isfile(ass_path) else None,
        )
        qc_report = await asyncio.to_thread(
            run_qc,
            out_path,
            expected_duration=song_duration,
            expected_width=width,
            expected_height=height,
            target_lufs=preset.loudness_lufs,
        )
        if not qc_report.get("ok"):
            raise RuntimeError("QC failed: " + ", ".join(qc_report.get("fatal") or ["unknown"]))

        return {"qc": qc_report, "captions": captions, "preset": preset.id}

    @staticmethod
    def _write_caption_sidecars(
        master_path: str,
        timed_lines: List[Dict[str, Any]],
        ass_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """Write SRT/VTT (and the ASS when available) beside the master (plan 7.16).

        Burned pixels cannot be reused: a caption file is a deliverable in its own
        right (YouTube sidecar, TTML conversion, translation).
        """
        written: Dict[str, str] = {}
        if not master_path or not timed_lines:
            return written
        stem = os.path.splitext(master_path)[0]
        try:
            from app.transcription.karaoke import lyric_sync_engine

            srt = lyric_sync_engine.generate_srt(timed_lines)
            vtt = lyric_sync_engine.generate_vtt(timed_lines)
            for suffix, content in ((".srt", srt), (".vtt", vtt)):
                if not content:
                    continue
                path = stem + suffix
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(content)
                written[suffix.lstrip(".")] = path
            if ass_path and os.path.isfile(ass_path):
                dest = stem + ".ass"
                if os.path.abspath(dest) != os.path.abspath(ass_path):
                    import shutil

                    shutil.copyfile(ass_path, dest)
                written["ass"] = dest
        except Exception as exc:
            logger.warning("Could not write caption sidecars for %s: %s", master_path, exc)
        return written

    @staticmethod
    def _record_take(
        *,
        render_id: str,
        kind: str,
        video_url: str,
        file_path: str,
        poster_path: Optional[str] = None,
        proxy_path: Optional[str] = None,
        qc: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a render take complete and select it (plan 7.4)."""
        try:
            from sqlmodel import Session

            from app.main import engine
            from app.services.video import takes as takes_service

            with Session(engine) as session:
                take = takes_service.get_take(session, render_id)
                if not take:
                    logger.warning("Render take %s vanished before completion", render_id)
                    return
                takes_service.finish_take(
                    session, take,
                    status="completed",
                    video_path=video_url,
                    file_path=file_path,
                    qc=qc,
                    poster_path=poster_path if poster_path and os.path.isfile(poster_path) else None,
                    proxy_path=proxy_path if proxy_path and os.path.isfile(proxy_path) else None,
                )
                takes_service.select_take(session, str(take.job_id), render_id)
        except Exception as exc:  # a take is bookkeeping; never fail a finished render
            logger.warning("Could not record render take %s: %s", render_id, exc)

    def _cleanup_task_temp(task_id: str) -> int:
        """Delete every intermediate file this render created.

        Only `rendered_clips` used to be removed — and after the reactive pass that
        pointed at `reactive_*.mp4`, so the raw `clip_*`, `stitched_*`, `norm_*`,
        `concat_*.txt` and `subtitles_*.ass` files stayed behind. A 3-minute render
        left hundreds of MB per run.
        """
        removed = 0
        try:
            entries = list(os.scandir(TEMP_DIR))
        except OSError:
            return 0
        for entry in entries:
            try:
                if not entry.is_file():
                    continue
                name = entry.name
                # Match the task id, and the scene-plate/subtitle side files.
                if f"_{task_id}" in name or name.startswith(f"subtitles_{task_id}"):
                    os.remove(entry.path)
                    removed += 1
            except OSError:
                continue
        if removed:
            logger.info("Cleaned up %d intermediate video file(s) for task %s", removed, task_id)
        return removed

    @staticmethod
    def _transition_overlap(transition_style: str, plan_clips: List[Any]) -> float:
        """The crossfade duration the stitcher will use, or 0 for a hard cut.

        Mirrors `_stitch_video_segments_with_transitions` (0.5 s cap, 25% of the
        shortest clip, floored at 0.2 s) so normalisation and assembly agree.
        """
        style = (transition_style or "").lower()
        if style not in {"crossfade", "dissolve", "whip_pan", "flash", "glitch"}:
            return 0.0
        durations = [float(getattr(c, "duration", 0.0) or 0.0) for c in plan_clips]
        durations = [d for d in durations if d > 0]
        if len(durations) < 2:
            return 0.0
        return min(0.5, max(0.2, min(durations) * 0.25))

    async def _normalise_rendered_clips(
        self,
        rendered_clips: List[str],
        plan_clips: List[Any],
        width: int,
        height: int,
        fps: int,
        task_id: str,
        cancel_event: Optional[asyncio.Event] = None,
        transition_overlap: float = 0.0,
        stem_reactivity: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Rewrite each rendered clip to its planned duration and the project geometry.

        See `services/video/clip_normalise.py`: lattice rounding loses ~0.1-0.25 s
        per clip, and mixed model frame rates/resolutions cannot be concatenated.
        A clip that cannot be normalised is kept as-is (better a slightly short
        video than no video) and the drift is reported on the task.
        """
        from app.services.video.clip_normalise import normalise_clip, probe_duration

        normalised: List[str] = []
        for idx, clip_path in enumerate(rendered_clips):
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError("Video rendering cancelled during normalisation")

            target = plan_clips[idx] if idx < len(plan_clips) else None
            target_duration = float(getattr(target, "duration", 0.0) or 0.0) if target else 0.0
            if target_duration <= 0:
                normalised.append(clip_path)
                continue
            # xfade eats `transition_overlap` from every non-final clip.
            if transition_overlap and idx < len(rendered_clips) - 1:
                target_duration += transition_overlap

            dest = os.path.join(TEMP_DIR, f"norm_{task_id}_{idx:03d}.mp4")
            before = probe_duration(clip_path)
            extra_vf = None
            if stem_reactivity is not None and target is not None:
                extra_vf = build_stem_reactive_filter(
                    start_time=float(getattr(target, "start_time", 0.0) or 0.0),
                    duration=target_duration,
                    stem_reactivity=stem_reactivity,
                    fps=fps,
                )
            ok = await asyncio.to_thread(
                normalise_clip,
                clip_path, dest,
                duration_s=target_duration,
                width=width, height=height, fps=fps,
                extra_vf=extra_vf,
            )
            if ok:
                normalised.append(dest)
                if before is not None and abs(before - target_duration) > 0.05:
                    logger.info(
                        "Clip %d: %.3fs rendered, trimmed/padded to %.3fs",
                        idx + 1, before, target_duration,
                    )
            else:
                logger.warning(
                    "Clip %d could not be normalised; keeping the raw render "
                    "(assembly may drift).", idx + 1,
                )
                normalised.append(clip_path)

        return normalised

    async def _stitch_video_segments_with_transitions(
        self,
        rendered_clips: List[str],
        transition_style: str,
        stitched_video: str,
        task_id: str,
    ) -> bool:
        """Stitch video clips using either FFmpeg xfade (dissolve, wipe, flash, glitch)
        or sample-accurate concat demuxer (beat_cut)."""
        if not rendered_clips:
            return False

        if len(rendered_clips) == 1:
            cmd = ["ffmpeg", "-y", "-i", rendered_clips[0], "-c", "copy", stitched_video]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            return os.path.isfile(stitched_video) and os.path.getsize(stitched_video) > 0

        # Map transition names to FFmpeg xfade transition types
        xfade_map = {
            "crossfade": "fade",
            "dissolve": "dissolve",
            "whip_pan": "smoothleft",
            "flash": "fadewhite",
            "glitch": "pixelize",
        }
        xfade_type = xfade_map.get(transition_style.lower())

        if xfade_type:
            try:
                durations: List[float] = []
                for cf in rendered_clips:
                    dur = await self._probe_clip_duration(cf)
                    durations.append(dur)

                min_dur = min(durations) if durations else 2.0
                trans_dur = min(0.5, max(0.2, min_dur * 0.25))

                inputs: List[str] = []
                for cf in rendered_clips:
                    inputs.extend(["-i", os.path.abspath(cf)])

                filter_parts: List[str] = []
                last_v = "0:v"
                current_offset = durations[0] - trans_dur

                for idx in range(1, len(rendered_clips)):
                    out_v = f"v{idx}" if idx < len(rendered_clips) - 1 else "v_out"
                    filter_parts.append(
                        f"[{last_v}][{idx}:v]xfade=transition={xfade_type}:duration={trans_dur:.2f}:offset={max(0.1, current_offset):.2f}[{out_v}]"
                    )
                    last_v = out_v
                    if idx < len(rendered_clips) - 1:
                        current_offset += durations[idx] - trans_dur

                filter_complex = ";".join(filter_parts)
                cmd_xfade = [
                    "ffmpeg", "-y",
                    *inputs,
                    "-filter_complex", filter_complex,
                    "-map", "[v_out]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                    "-c:a", "aac", "-b:a", "192k",
                    stitched_video
                ]
                logger.info(f"Stitching {len(rendered_clips)} scenes with FFmpeg xfade ({transition_style} -> {xfade_type})...")
                proc = await asyncio.create_subprocess_exec(
                    *cmd_xfade, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                _, err = await proc.communicate()
                if os.path.isfile(stitched_video) and os.path.getsize(stitched_video) > 0:
                    logger.info(f"Successfully stitched scenes with {transition_style} xfade.")
                    return True
                logger.warning(f"xfade transition failed ({err.decode('utf-8', errors='ignore')[:150]}); falling back to concat demuxer.")
            except Exception as e:
                logger.warning(f"Error executing xfade stitching: {e}; falling back to concat demuxer.")

        # Fallback / beat_cut: Fast concat demuxer
        concat_list_path = os.path.join(TEMP_DIR, f"concat_{task_id}.txt")
        with open(concat_list_path, "w") as f:
            for cf in rendered_clips:
                f.write(f"file '{os.path.abspath(cf)}'\n")

        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "192k",
            stitched_video
        ]
        proc = await asyncio.create_subprocess_exec(*cmd_concat, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, err_concat = await proc.communicate()
        return os.path.isfile(stitched_video) and os.path.getsize(stitched_video) > 0

    async def render_lyric_music_video(
        self,
        job: Job,
        task_id: str,
        config: Dict[str, Any]
    ) -> str:
        """
        Fast-path local lyric music video rendering pipeline (< 45s).
        Combines hardware-accelerated video encoding (VideoToolbox / NVENC / CPU libx264),
        smooth Ken Burns camera drift on artwork, audio-reactive frequency spectrum visualizer,
        and pixel-perfect word-level karaoke ASS subtitles.
        """
        cancel_event = asyncio.Event()
        with self._lock:
            self._video_cancels[task_id] = cancel_event
            tinfo = VideoTaskStatusInfo(
                id=task_id,
                job_id=str(job.id),
                status="processing",
                step="Initializing Fast-Path Lyric Video Engine",
                progress=5,
                total_clips=1,
                current_clip=1
            )
            self._tasks[task_id] = tinfo
            try:
                task_queue.enqueue_task(
                    task_id=task_id,
                    task_type="lyric_video_rendering",
                    payload={"job_id": str(job.id), "config": config},
                    initial_status="processing",
                    initial_message="Initializing Fast-Path Lyric Video Engine"
                )
            except Exception as _e:
                logger.debug(f"Could not enqueue lyric video task: {_e}")

        try:
            if cancel_event.is_set():
                raise asyncio.CancelledError("Lyric video rendering cancelled by user.")

            resolved_master = self.resolve_audio_path(job.audio_path)
            if not resolved_master or not os.path.isfile(resolved_master):
                raise FileNotFoundError(f"Master audio file not found for job: {job.audio_path}")

            total_duration = 180.0
            try:
                import soundfile as sf
                total_duration = float(sf.info(resolved_master).duration)
            except Exception:
                if getattr(job, "duration_ms", None):
                    total_duration = float(job.duration_ms) / 1000.0

            aspect_ratio = config.get("aspect_ratio", "16:9")
            resolution = config.get("resolution", "720p")
            if aspect_ratio == "9:16":
                w, h = (1080, 1920) if resolution == "1080p" else (720, 1280)
            elif aspect_ratio == "1:1":
                w, h = (1080, 1080) if resolution == "1080p" else (720, 720)
            elif aspect_ratio == "21:9":
                w, h = (2560, 1080) if resolution == "1080p" else (1680, 720)
            else:
                w, h = (1920, 1080) if resolution == "1080p" else (1280, 720)

            # Generate Subtitles
            self.update_task(task_id, step="Aligning Word Timestamps & Formatting Typography", progress=20)
            timed_lines = []
            if job.timed_lyrics_json:
                try:
                    timed_lines = json.loads(job.timed_lyrics_json)
                except Exception:
                    timed_lines = []

            if not timed_lines and job.lyrics:
                stems_dict = json.loads(job.stems_json) if job.stems_json else {}
                vocal_stem = stems_dict.get("vocals") or stems_dict.get("part_vocals") or resolved_master
                timed_lines = lyric_sync_engine.align_lyrics(
                    job.lyrics, duration_sec=total_duration, vocal_stem_path=vocal_stem
                )

            style_preset = config.get("style_preset", "neon")
            burn_lyrics = config.get("burn_lyrics", True)
            ass_path = None

            if burn_lyrics and timed_lines:
                try:
                    ass_content = self.generate_karaoke_ass(
                        timed_lines=timed_lines,
                        width=w,
                        height=h,
                        style=config.get("visual_style", "neon-cyberpunk"),
                        subtitle_style=style_preset,
                        aspect_ratio=aspect_ratio,
                        font_family=config.get("font_family"),
                        font_size_override=config.get("font_size_override"),
                    )
                    ass_path = os.path.join(TEMP_DIR, f"lyric_{task_id}.ass")
                    with open(ass_path, "w", encoding="utf-8") as f_ass:
                        f_ass.write(ass_content)
                except Exception as ex:
                    logger.warning(f"Could not generate ASS subtitles for lyric video: {ex}")
                    ass_path = None

            if cancel_event.is_set():
                raise asyncio.CancelledError("Lyric video rendering cancelled by user.")

            self.update_task(task_id, step="Building Motion Background & Audio Reactive Visualizer", progress=40)

            # Detect optimal ffmpeg executable and hardware acceleration encoder
            ffmpeg_bin = find_ffmpeg_executable()
            encoder, enc_flags = detect_hardware_encoder(ffmpeg_bin)
            logger.info(f"Lyric Video: Using ffmpeg '{ffmpeg_bin}', video encoder '{encoder}' with flags {enc_flags}")

            bg_mode = config.get("background_mode", "cover_art")
            include_spectrum = config.get("include_spectrum", False)

            # Resolve cover image reliably from config, job, or filesystem
            cover_raw = config.get("cover_image_path") or getattr(job, "cover_image_path", None)
            cover_path = resolve_image_file(cover_raw) if cover_raw else None
            if not cover_path or not os.path.isfile(cover_path):
                if cover_raw and os.path.isfile(cover_raw):
                    cover_path = cover_raw
                else:
                    cover_path = self.resolve_face_image(job)
            if not cover_path or not os.path.isfile(cover_path):
                if getattr(job, "cover_image_path", None):
                    cover_cand = resolve_image_file(job.cover_image_path)
                    if cover_cand and os.path.isfile(cover_cand):
                        cover_path = cover_cand

            out_filename = f"{job.id}_lyric.mp4"
            out_path = os.path.join(VIDEO_DIR, out_filename)
            tmp_out_path = os.path.join(TEMP_DIR, f"{job.id}_lyric_{task_id}.mp4")

            # Always save a permanent sidecar ASS subtitle file alongside output video
            if ass_path and os.path.isfile(ass_path):
                try:
                    sidecar_ass_path = os.path.join(VIDEO_DIR, f"{job.id}_lyric.ass")
                    import shutil
                    shutil.copyfile(ass_path, sidecar_ass_path)
                except Exception as _e:
                    logger.debug(f"Could not copy ASS sidecar: {_e}")

            # Check if active ffmpeg binary supports burning in subtitles via libass
            can_hardsub = has_subtitles_filter(ffmpeg_bin) and bool(ass_path and os.path.isfile(ass_path))
            sub_filter = ""
            if can_hardsub and ass_path:
                escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
                fonts_dir = get_fonts_dir()
                fonts_opt = f":fontsdir='{fonts_dir.replace('\\', '/').replace(':', '\\:')}'" if fonts_dir else ""
                sub_filter = f",subtitles='{escaped_ass}'{fonts_opt}"

            def _build_ffmpeg_cmd(use_hardsub: bool, enc: str, e_flags: List[str]) -> List[str]:
                current_sub_filter = sub_filter if (use_hardsub and can_hardsub) else ""
                has_soft_sub = (not current_sub_filter) and bool(ass_path and os.path.isfile(ass_path))

                if cover_path and os.path.isfile(cover_path) and bg_mode != "spectrum":
                    target_w = int(w * 1.25)
                    target_h = int(h * 1.25)
                    total_frames = max(1, int(math.ceil(total_duration * 30)))
                    filter_complex = (
                        f"[0:v]scale=w={target_w}:h={target_h}:force_original_aspect_ratio=increase,"
                        f"crop={target_w}:{target_h},setsar=1,"
                        f"zoompan=z='min(zoom+0.0006,1.20)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={total_frames}:s={w}x{h}:fps=30[bg]"
                    )
                    if include_spectrum:
                        filter_complex += (
                            f";[1:a]showwaves=s={w}x{int(h * 0.22)}:mode=line:colors=0x00f0ff|0x7000ff:scale=sqrt[waves]"
                            f";[bg][waves]overlay=(W-w)/2:H-h-60[comp]{current_sub_filter},setsar=1,format=yuv420p[v_out]"
                        )
                    else:
                        filter_complex += f";[bg]null{current_sub_filter},setsar=1,format=yuv420p[v_out]"
                    # The song is the master: normalise once, here (plan 7.13).
                    filter_complex += ";[1:a]loudnorm=I=-14:TP=-1:LRA=11[a_out]"

                    c = [
                        ffmpeg_bin, "-y",
                        "-i", cover_path,
                        "-i", resolved_master,
                    ]
                    if has_soft_sub:
                        c.extend(["-i", ass_path])
                    c.extend([
                        "-filter_complex", filter_complex,
                        "-map", "[v_out]", "-map", "[a_out]"
                    ])
                    if has_soft_sub:
                        c.extend(["-map", "2:s:0", "-c:s", "mov_text"])
                    c.extend([
                        "-c:v", enc, *e_flags,
                        "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "256k",
                        "-movflags", "+faststart",
                        "-t", f"{total_duration:.3f}",
                        "-shortest",
                        tmp_out_path
                    ])
                    return c
                else:
                    palette = STYLE_PALETTES.get(config.get("visual_style", "neon-cyberpunk"), STYLE_PALETTES["neon-cyberpunk"])
                    colors = palette.get("colors", "0x00f0ff|0x7000ff")
                    filter_complex = (
                        f"color=c=0x0b0f19:s={w}x{h}:d={total_duration:.3f}:r=30[bg];"
                        f"[0:a]showwaves=s={w}x{int(h * 0.35)}:mode=line:colors={colors}:scale=sqrt[waves];"
                        f"[bg][waves]overlay=(W-w)/2:(H-h)/2[comp]{current_sub_filter},setsar=1,format=yuv420p[v_out]"
                        ";[0:a]loudnorm=I=-14:TP=-1:LRA=11[a_out]"
                    )
                    c = [
                        ffmpeg_bin, "-y",
                        "-i", resolved_master,
                    ]
                    if has_soft_sub:
                        c.extend(["-i", ass_path])
                    c.extend([
                        "-filter_complex", filter_complex,
                        "-map", "[v_out]", "-map", "[a_out]"
                    ])
                    if has_soft_sub:
                        c.extend(["-map", "1:s:0", "-c:s", "mov_text"])
                    c.extend([
                        "-c:v", enc, *e_flags,
                        "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "256k",
                        "-movflags", "+faststart",
                        "-t", f"{total_duration:.3f}",
                        "-shortest",
                        tmp_out_path
                    ])
                    return c

            cmd = _build_ffmpeg_cmd(use_hardsub=can_hardsub, enc=encoder, e_flags=enc_flags)

            self.update_task(task_id, step="Encoding Lyric Video with Hardware Acceleration", progress=60)
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            with self._lock:
                self._active_procs[task_id] = proc

            # Polling loop for cancellation check
            while proc.returncode is None:
                if cancel_event.is_set():
                    try:
                        proc.kill()
                        await proc.wait()
                    except Exception:
                        pass
                    raise asyncio.CancelledError("Lyric video rendering cancelled by user.")
                try:
                    await asyncio.wait_for(proc.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass

            stdout_b, stderr_b = await proc.communicate()

            # Fallback re-encode with libx264 (retaining hardsub subtitles first) if initial encode failed
            if proc.returncode != 0 or not os.path.isfile(tmp_out_path) or os.path.getsize(tmp_out_path) == 0:
                err_msg = stderr_b.decode("utf-8", errors="ignore")[:300]
                logger.warning(f"Initial lyric video encode failed ({proc.returncode}): {err_msg}; retrying with libx264 fallback encoder...")
                cmd_fb = _build_ffmpeg_cmd(use_hardsub=can_hardsub, enc="libx264", e_flags=["-preset", "veryfast", "-crf", "20"])
                proc_fb = await asyncio.create_subprocess_exec(
                    *cmd_fb, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                with self._lock:
                    self._active_procs[task_id] = proc_fb
                while proc_fb.returncode is None:
                    if cancel_event.is_set():
                        try:
                            proc_fb.kill()
                            await proc_fb.wait()
                        except Exception:
                            pass
                        raise asyncio.CancelledError("Lyric video rendering cancelled by user.")
                    try:
                        await asyncio.wait_for(proc_fb.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        pass
                _, err_fb = await proc_fb.communicate()

                # Last-resort fallback with soft subs if hardsub filter itself failed
                if (proc_fb.returncode != 0 or not os.path.isfile(tmp_out_path) or os.path.getsize(tmp_out_path) == 0) and can_hardsub:
                    logger.warning(f"Hardsub fallback also failed ({proc_fb.returncode}); falling back to soft subtitles...")
                    cmd_soft = _build_ffmpeg_cmd(use_hardsub=False, enc="libx264", e_flags=["-preset", "veryfast", "-crf", "20"])
                    proc_soft = await asyncio.create_subprocess_exec(
                        *cmd_soft, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                    )
                    with self._lock:
                        self._active_procs[task_id] = proc_soft
                    while proc_soft.returncode is None:
                        if cancel_event.is_set():
                            try:
                                proc_soft.kill()
                                await proc_soft.wait()
                            except Exception:
                                pass
                            raise asyncio.CancelledError("Lyric video rendering cancelled by user.")
                        try:
                            await asyncio.wait_for(proc_soft.wait(), timeout=0.5)
                        except asyncio.TimeoutError:
                            pass
                    await proc_soft.communicate()

            if not os.path.isfile(tmp_out_path) or os.path.getsize(tmp_out_path) == 0:
                raise RuntimeError(f"Failed to generate lyric music video output file: {tmp_out_path}")

            # Atomic rename into final output path to avoid race conditions with browser stream
            import shutil
            shutil.move(tmp_out_path, out_path)

            render_id = config.get("render_id")
            if render_id:
                self._record_take(
                    render_id=render_id,
                    kind="lyric",
                    video_url=f"/audio/videos/{out_filename}",
                    file_path=out_path,
                    qc=locals().get("qc_report"),
                )
                self.update_task(task_id, render_id=render_id)

            # Delivery gate for the lyric master too (plans 7.14/7.16).
            from app.services.video.qc import run_qc

            qc_report = await asyncio.to_thread(
                run_qc, out_path, expected_duration=total_duration,
                expected_width=w, expected_height=h, target_lufs=-14.0,
            )
            self.update_task(task_id, qc=qc_report)
            if not qc_report.get("ok"):
                raise RuntimeError("QC failed: " + ", ".join(qc_report.get("fatal") or ["unknown"]))
            self._write_caption_sidecars(
                out_path, timed_lines,
                ass_path=os.path.join(VIDEO_DIR, f"{job.id}_lyric.ass")
                if os.path.isfile(os.path.join(VIDEO_DIR, f"{job.id}_lyric.ass")) else None,
            )

            import time
            timestamp = int(time.time())
            video_url = f"/audio/videos/{out_filename}?t={timestamp}"
            self.update_task(
                task_id,
                status="completed",
                progress=100,
                step="Lyric Video Render Complete",
                video_url=video_url,
                result={"video_url": video_url}
            )
            return video_url

        except asyncio.CancelledError:
            logger.info(f"render_lyric_music_video: Task {task_id} successfully cancelled.")
            if 'proc' in locals() and proc and proc.returncode is None:
                try:
                    proc.kill()
                except Exception:
                    pass
            if 'proc_fb' in locals() and proc_fb and proc_fb.returncode is None:
                try:
                    proc_fb.kill()
                except Exception:
                    pass
            self.update_task(task_id, status="cancelled", step="Lyric video rendering cancelled by user.", progress=0)
            return ""
        except Exception as e:
            if cancel_event.is_set():
                logger.info(f"render_lyric_music_video: Task {task_id} aborted cleanly due to cancellation.")
                self.update_task(task_id, status="cancelled", step="Lyric video rendering cancelled by user.", progress=0)
                return ""
            logger.error(f"render_lyric_music_video: Failed for task {task_id}: {e}", exc_info=True)
            self.update_task(task_id, status="failed", error=str(e), progress=0)
            raise e
        finally:
            with self._lock:
                self._active_procs.pop(task_id, None)
                self._video_cancels.pop(task_id, None)
            if 'proc' in locals() and proc and proc.returncode is None:
                try:
                    proc.kill()
                except Exception:
                    pass
            if 'proc_fb' in locals() and proc_fb and proc_fb.returncode is None:
                try:
                    proc_fb.kill()
                except Exception:
                    pass
            if 'tmp_out_path' in locals() and os.path.isfile(tmp_out_path):
                try:
                    os.remove(tmp_out_path)
                except OSError:
                    pass


video_orchestrator = VideoOrchestrator()

