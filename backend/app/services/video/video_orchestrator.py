"""
Video Orchestrator — End-to-End Production Music Video Assembly Pipeline.
Integrates VideoDirector, LivePortrait singing avatar, Wan 2.1 diffusion,
per-scene regeneration, beat transitions, ASS karaoke burning, and master remuxing.
"""

import os
import re
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
from app.services.video.lip_sync.base import BaseLipSyncProvider
from app.services.video.lip_sync.live_portrait import LivePortraitProvider
from app.services.video.lip_sync.cloud_lipsync import CloudLipSyncProvider
from app.services.video.lip_sync.fallback import SmoothVisemeFallbackProvider
from app.services.video.generators.base import BaseVideoGenerator
from app.services.video.generators.diffusers_wan import DiffusersWanGenerator
from app.services.video.generators.diffusers_ltx import DiffusersLTXGenerator
from app.services.video.generators.cloud_video import CloudVideoGenerator
from app.services.video.generators.procedural import ProceduralVideoGenerator
from app.core.hardware_lock import GlobalHardwareCoordinator
from app.core.task_queue import task_queue
from app.services.video.stem_audio_reactive import extract_stem_reactive_modulation

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
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VideoOrchestrator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._local_lipsync = LivePortraitProvider()
        self._fallback_lipsync = SmoothVisemeFallbackProvider()
        self._local_wan_14b = DiffusersWanGenerator(model_size="14b")
        self._local_wan_1_3b = DiffusersWanGenerator(model_size="1.3b")
        self._local_ltx = DiffusersLTXGenerator()
        self._procedural = ProceduralVideoGenerator()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            t = self._tasks.get(task_id)
            if t:
                return t.to_dict()
        # Query durable SQLite task queue if not in memory
        pt = task_queue.get_task(task_id)
        if pt:
            return pt.to_dict()
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
            task_queue.update_task(
                task_id=task_id,
                status=status,
                progress=progress,
                message=step,
                error=kwargs.get("error"),
            )
        except Exception as e:
            logger.debug(f"Failed to sync task {task_id} to durable queue: {e}")

    def cancel_video_task(self, task_id: str) -> bool:
        with self._lock:
            ev = self._video_cancels.get(task_id)
            if ev:
                ev.set()
        self.update_task(task_id, status="cancelled", step="Video rendering cancelled by user.")
        try:
            from app.core.hardware_lock import GlobalHardwareCoordinator
            GlobalHardwareCoordinator.flush_memory()
        except Exception:
            pass
        return True

    def cancel_keyframe_generation(self, job_id: str) -> bool:
        matching_tasks = []
        with self._lock:
            ev = self._keyframe_cancels.get(str(job_id))
            if ev:
                ev.set()
            for tid, tinfo in self._tasks.items():
                if tinfo.job_id == str(job_id) and tid.startswith("kf_"):
                    matching_tasks.append(tid)

        for tid in matching_tasks:
            self.update_task(tid, status="cancelled", step="Keyframe generation cancelled by user.")

        try:
            from app.services.image_service import image_service
            image_service.unload_models()
            from app.core.hardware_lock import GlobalHardwareCoordinator
            GlobalHardwareCoordinator.flush_memory()
        except Exception:
            pass
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

    def get_video_providers(self) -> List[Dict[str, Any]]:
        """Return catalog of local and cloud providers with availability status."""
        return [
            VideoProviderInfo(
                id="local_wan_14b",
                name="Local M3 Max Flagship (Wan 2.1 14B + LivePortrait)",
                provider_type=VideoProviderType.LOCAL,
                description="High-fidelity 14B text-to-video / image-to-video with LivePortrait singing avatar on Apple Silicon.",
                is_available=self._local_wan_14b.is_available,
                supported_models=["wan_14b", "live_portrait"],
                default_for_tier=True
            ).to_dict(),
            VideoProviderInfo(
                id="local_fast",
                name="Local Fast Studio (Wan 1.3B / LTX-Video + LivePortrait)",
                provider_type=VideoProviderType.LOCAL,
                description="Ultra-fast local preview generation with rapid clip turnaround.",
                is_available=True,
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
            VideoProviderInfo(
                id="cloud_replicate",
                name="Cloud GPU Replicate Studio",
                provider_type=VideoProviderType.CLOUD_REPLICATE,
                description="Scalable cloud inference across Wan 2.1 and LivePortrait endpoints.",
                is_available=bool(os.environ.get("REPLICATE_API_TOKEN")),
                supported_models=["wan_14b", "live_portrait"],
                has_api_key=bool(os.environ.get("REPLICATE_API_TOKEN")),
                requires_api_key=True
            ).to_dict(),
        ]

    def generate_karaoke_ass(
        self,
        timed_lines: List[Dict[str, Any]],
        width: int = 1280,
        height: int = 720,
        style: str = "neon-cyberpunk",
        subtitle_style: str = "neon"
    ) -> str:
        """
        Generate Advanced SubStation Alpha (.ass) subtitle content with karaoke highlights.
        """
        palette = STYLE_PALETTES.get(style, STYLE_PALETTES["neon-cyberpunk"])
        r, g, b = palette["primary_color"]
        primary_ass = f"&H00{b:02X}{g:02X}{r:02X}"
        ar, ag, ab = palette["accent_color"]
        accent_ass = f"&H00{ab:02X}{ag:02X}{ar:02X}"

        font_size = int(height * 0.048)
        margin_v = int(height * 0.08)

        ass_lines = [
            "[Script Info]",
            "Title: Milimo Music Synchronized Video",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,Arial,{font_size},{primary_ass},{accent_ass},&H00090A10,&H80000000,1,0,0,0,100,100,0,0,1,2.5,1.5,2,40,40,{margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ]

        def fmt_ass_time(sec: float) -> str:
            m = int(sec // 60)
            s = int(sec % 60)
            cs = int(round((sec - int(sec)) * 100))
            h = m // 60
            m = m % 60
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        for line in timed_lines:
            text = line.get("text", "").strip()
            if not text:
                continue
            start_t = fmt_ass_time(line.get("start", 0.0))
            end_t = fmt_ass_time(line.get("end", 0.0))
            # If karaoke style is enabled, wrap text with karaoke duration tag
            if subtitle_style == "karaoke":
                dur_cs = int(round((line.get("end", 0.0) - line.get("start", 0.0)) * 100))
                ass_lines.append(f"Dialogue: 0,{start_t},{end_t},Default,,0,0,0,,{{\\k{dur_cs}}}{text}")
            else:
                ass_lines.append(f"Dialogue: 0,{start_t},{end_t},Default,,0,0,0,,{text}")

        return "\n".join(ass_lines)

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

        plan = video_director.segment_song(
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
                        step=f"Diffusing Scene Still {idx + 1}/{total_scenes} ({'🎤 Vocal' if clip.is_vocal else '🎥 Cinematic'})",
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

                    if not is_stale_cover:
                        results.append(_clip_dict(clip, clip_idx))
                        continue

                # Ensure clean slate for this keyframe file
                if os.path.isfile(kf_path):
                    try:
                        os.remove(kf_path)
                    except OSError:
                        pass

                # Render dedicated scene still from the clip's director prompt
                try:
                    from app.services.image_service import image_service
                    res = image_service.generate_scene_background(
                        prompt=clip.prompt,
                        style=visual_style,
                        width=width,
                        height=height,
                        auto_unload=False
                    )
                    if res.get("ok") and res.get("dest_path") and os.path.isfile(res["dest_path"]):
                        shutil.copy(res["dest_path"], kf_path)
                    else:
                        # Procedural fallback still if diffusion is unavailable
                        image_service._generate_raster_cover(
                            prompt=f"{clip.prompt} (Scene {clip_idx})",
                            style=visual_style,
                            width=width,
                            height=height,
                            dest_path=kf_path
                        )
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

            if aspect_ratio == "9:16":
                w, h = (1080, 1920) if resolution == "1080p" else (720, 1280)
            elif aspect_ratio == "1:1":
                w, h = (1080, 1080) if resolution == "1080p" else (720, 720)
            elif aspect_ratio == "21:9":
                w, h = (2560, 1080) if resolution == "1080p" else (1680, 720)
            else: # 16:9 widescreen default
                w, h = (1920, 1080) if resolution == "1080p" else (1280, 720)

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
                    vocal_stem_path=vocal_stem or resolved_master,
                    drums_stem_path=drums_stem,
                    bass_stem_path=bass_stem,
                    fps=24
                )
                logger.info(f"Stem audio-reactivity analysis completed ({stem_reactivity.get('total_frames')} frames)")
            except Exception as e:
                logger.warning(f"Stem audio-reactivity analysis skipped: {e}")
            plan = video_director.segment_song(
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
            self.update_task(task_id, total_clips=total_clips, clips=[c.to_dict() for c in plan.clips])

            # Select generator and lip-sync providers
            if provider_type == "cloud_fal":
                lipsync_provider = CloudLipSyncProvider("fal")
                video_generator = CloudVideoGenerator("fal", model="wan-t2v")
            elif provider_type == "cloud_replicate":
                lipsync_provider = CloudLipSyncProvider("replicate")
                video_generator = CloudVideoGenerator("replicate", model="wan-14b")
            else:
                lipsync_provider = self._local_lipsync
                if "1.3" in model_name:
                    video_generator = self._local_wan_1_3b
                elif "ltx" in model_name:
                    video_generator = self._local_ltx
                else:
                    video_generator = self._local_wan_14b

            # Step 2: Render individual scene clips
            self.update_task(task_id, step="Rendering Video Scenes & Lip-Sync Performance", progress=20)
            is_local = (provider_type not in ("cloud_fal", "cloud_replicate"))

            # Phase A: Batch Pre-Render Missing Keyframe Stills (Image Modality)
            if is_local:
                async with GlobalHardwareCoordinator.scoped_device("Keyframe Stills Generation", modality="image_gen"):
                    for idx, clip in enumerate(plan.clips):
                        if cancel_event.is_set():
                            logger.info(f"render_advanced_music_video: Cancelled during Phase A stills generation.")
                            self.update_task(task_id, status="cancelled", step="Video rendering cancelled by user.", progress=0)
                            return ""

                        if not clip.is_vocal:
                            clip_idx = int(clip.clip_index or (idx + 1))
                            pre_rendered_kf = os.path.join(KEYFRAMES_DIR, f"keyframe_{job.id}_{clip_idx:03d}.png")
                            if not (os.path.isfile(pre_rendered_kf) and os.path.getsize(pre_rendered_kf) > 0):
                                try:
                                    from app.services.image_service import image_service
                                    bg = image_service.generate_scene_background(
                                        prompt=clip.prompt,
                                        style=style,
                                        width=w, height=h,
                                        auto_unload=False
                                    )
                                    if bg.get("ok") and bg.get("dest_path") and os.path.isfile(bg["dest_path"]):
                                        shutil.copy(bg["dest_path"], pre_rendered_kf)
                                except Exception as e:
                                    logger.warning(f"Keyframe batch generation skipped for scene {clip_idx}: {e}")
                    # Immediately unload image diffusion models and clear MLX/CUDA memory
                    try:
                        from app.services.image_service import image_service
                        image_service.unload_models()
                    except Exception:
                        pass

            try:
                for idx, clip in enumerate(plan.clips):
                    if cancel_event.is_set():
                        logger.info(f"render_advanced_music_video: Cancelled before rendering scene {idx + 1}.")
                        self.update_task(task_id, status="cancelled", step="Video rendering cancelled by user.", progress=0)
                        return ""

                    clip_file = os.path.join(TEMP_DIR, f"clip_{task_id}_{idx:03d}.mp4")
                    self.update_task(
                        task_id,
                        current_clip=idx + 1,
                        current_clip_type=clip.scene_type,
                        step=f"Rendering Scene {idx + 1}/{total_clips} ({'🎤 Vocal Performance' if clip.is_vocal else '🎥 Cinematic B-Roll'})",
                        progress=20 + int(60 * (idx / total_clips))
                    )

                    async def _render_clip():
                        # Vocal Singing Scene
                        # Prefer isolated vocal stem; fall back to resolved master audio so lip-sync succeeds even without stem separation
                        vocal_audio_source = vocal_stem or resolved_master
                        if clip.is_vocal and enable_lip_sync and face_image and vocal_audio_source:
                            logger.info(f"Rendering singing performance for Scene {idx + 1} with {lipsync_provider.name} (audio: {os.path.basename(vocal_audio_source)})...")
                            success = await lipsync_provider.render_lip_sync(
                                face_image_path=face_image,
                                vocal_audio_path=vocal_audio_source,
                                start_time=clip.start_time,
                                duration=clip.duration,
                                out_path=clip_file,
                                width=w, height=h
                            )
                            if not success or not os.path.isfile(clip_file) or os.path.getsize(clip_file) == 0:
                                # Fallback to smooth provider
                                await self._fallback_lipsync.render_lip_sync(
                                    face_image_path=face_image,
                                    vocal_audio_path=vocal_audio_source,
                                    start_time=clip.start_time,
                                    duration=clip.duration,
                                    out_path=clip_file,
                                    width=w, height=h
                                )

                        # Cinematic Scene
                        else:
                            clip_idx = int(clip.clip_index or (idx + 1))
                            logger.info(f"Rendering visual scene {clip_idx} with {video_generator.name}...")
                            # Check if an approved pre-rendered keyframe exists on disk
                            pre_rendered_kf = os.path.join(KEYFRAMES_DIR, f"keyframe_{job.id}_{clip_idx:03d}.png")
                            scene_bg = pre_rendered_kf if (os.path.isfile(pre_rendered_kf) and os.path.getsize(pre_rendered_kf) > 0) else face_image

                            fb_meta: Dict[str, Any] = {}
                            success = await video_generator.generate_clip(
                                prompt=clip.prompt,
                                duration=clip.duration,
                                out_path=clip_file,
                                width=w, height=h,
                                image_path=scene_bg,
                                negative_prompt=clip.negative_prompt,
                                visual_style=style,
                                fallback_metadata=fb_meta
                            )
                            if fb_meta.get("fallback_used"):
                                task_info.fallback_used = True
                                task_info.fallback_reason = fb_meta.get("error", "Procedural fallback used")
                            if not success or not os.path.isfile(clip_file) or os.path.getsize(clip_file) == 0:
                                task_info.fallback_used = True
                                task_info.fallback_reason = "Video generator output empty; falling back to procedural animatic."
                                # Fallback to procedural generator
                                await self._procedural.generate_clip(
                                    prompt=clip.prompt,
                                    duration=clip.duration,
                                    out_path=clip_file,
                                    width=w, height=h,
                                    image_path=scene_bg,
                                    visual_style=style
                                )

                    if is_local:
                        async with GlobalHardwareCoordinator.scoped_device(
                            f"Wan 2.1 Video Clip {idx + 1}/{total_clips}",
                            modality="video_gen"
                        ):
                            await _render_clip()
                    else:
                        await _render_clip()

                    if os.path.isfile(clip_file) and os.path.getsize(clip_file) > 0:
                        rendered_clips.append(clip_file)
                        clip.rendered_clip_path = clip_file
                        clip.status = "completed"

            finally:
                with self._lock:
                    self._video_cancels.pop(task_id, None)
                    self._video_cancels.pop(str(job.id), None)
                if is_local and hasattr(video_generator, "unload"):
                    try:
                        video_generator.unload()
                        GlobalHardwareCoordinator.flush_memory()
                    except Exception as _e:
                        logger.debug(f"Video generator unload skipped: {_e}")

            if not rendered_clips:
                raise RuntimeError("No video scenes were successfully rendered.")


            # Step 3: Scene Assembly & Concat
            self.update_task(task_id, step="Assembling & Stitching Video Scenes", progress=82)
            concat_list_path = os.path.join(TEMP_DIR, f"concat_{task_id}.txt")
            with open(concat_list_path, "w") as f:
                for cf in rendered_clips:
                    f.write(f"file '{os.path.abspath(cf)}'\n")

            stitched_video = os.path.join(TEMP_DIR, f"stitched_{task_id}.mp4")
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

            if not os.path.isfile(stitched_video) or os.path.getsize(stitched_video) == 0:
                raise RuntimeError(f"Video scene stitching failed: {err_concat.decode('utf-8', errors='ignore')[:200]}")

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
                        subtitle_style=subtitle_style
                    )
                    ass_path = os.path.join(TEMP_DIR, f"subtitles_{task_id}.ass")
                    with open(ass_path, "w", encoding="utf-8") as f_sub:
                        f_sub.write(ass_content)
                except Exception as ex:
                    logger.warning(f"Could not generate ASS subtitles: {ex}")
                    ass_path = None

            # Execute master remux with burned subtitles if available
            if ass_path and os.path.isfile(ass_path):
                # Escape path for FFmpeg filter
                escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
                cmd_final = [
                    "ffmpeg", "-y",
                    "-i", stitched_video,
                    "-i", resolved_master,
                    "-filter_complex", f"[0:v]subtitles='{escaped_ass}'[v]",
                    "-map", "[v]", "-map", "1:a:0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                    "-c:a", "aac", "-b:a", "256k",
                    "-shortest",
                    out_path
                ]
            else:
                cmd_final = [
                    "ffmpeg", "-y",
                    "-i", stitched_video,
                    "-i", resolved_master,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "256k",
                    "-shortest",
                    out_path
                ]

            proc_final = await asyncio.create_subprocess_exec(*cmd_final, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _, err_final = await proc_final.communicate()

            # If stream copy or filter failed, try safe re-encode fallback
            if proc_final.returncode != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
                logger.warning("Stream mux failed, trying safe re-encode fallback...")
                cmd_fb = [
                    "ffmpeg", "-y",
                    "-i", stitched_video,
                    "-i", resolved_master,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                    "-c:a", "aac", "-b:a", "256k",
                    "-shortest",
                    out_path
                ]
                proc_fb = await asyncio.create_subprocess_exec(*cmd_fb, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await proc_fb.communicate()

            if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
                raise RuntimeError("Failed to generate master music video output file.")

            video_url = f"/audio/videos/{out_filename}"
            self.update_task(
                task_id,
                status="completed",
                progress=100,
                step="Production Video Complete",
                video_url=video_url
            )
            return video_url

        except Exception as e:
            logger.error(f"Advanced video generation failed: {e}", exc_info=True)
            self.update_task(task_id, status="error", error=str(e), step=f"Error: {str(e)[:120]}")
            raise e
        finally:
            # Clean up temp files
            for cf in rendered_clips:
                if os.path.isfile(cf):
                    try: os.remove(cf)
                    except: pass


video_orchestrator = VideoOrchestrator()
