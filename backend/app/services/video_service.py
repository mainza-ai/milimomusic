"""
AI Music Video Studio Service — Production Pipeline.
Provides:
1. Song Segmentation & Duration Constraint Management (Wan2.1 5s, CogVideoX 6s, Hailuo H3 8s).
2. Vocal Stem Extraction & Character Lip-Sync Generation.
3. Beat-Matched Scene Storyboard & Cinematic B-Roll Planning.
4. Synchronized Lyric & Karaoke Subtitle Burning.
5. Multi-Clip Video Assembly & Master Audio-Video Remuxing.
"""

import os
import re
import json
import math
import uuid
import shutil
import asyncio
import logging
import threading
import subprocess
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import scipy.io.wavfile as wavfile
except ImportError:
    wavfile = None

from app.models import Job
from app.services.llm_service import LLMService
from app.transcription.karaoke import lyric_sync_engine

from app.core.paths import get_generated_audio_dir, get_data_dir

logger = logging.getLogger(__name__)

VIDEO_DIR = str(get_generated_audio_dir() / "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)
TEMP_DIR = str(get_data_dir() / "video_cache")
os.makedirs(TEMP_DIR, exist_ok=True)

from app.services.video.video_director import STYLE_PALETTES

# Video palette key -> cinematic image descriptor for scene-background stills.
SCENE_STYLE_DESCRIPTORS: Dict[str, str] = {
    k: v.get("atmosphere", "cinematic atmosphere")
    for k, v in STYLE_PALETTES.items()
}

MODEL_MAX_DURATIONS: Dict[str, float] = {
    "wan_14b": 5.0,
    "wan_1.3b": 5.0,
    "wan2.1": 5.0,  # backwards compatibility alias
    "ltx_video": 10.0,
    "cogvideox": 10.0,
    "hailuo_h3": 15.0,
    "hunyuan": 15.0,
    "audioreactive": 120.0,
}

# Model-manager video model -> Music Videos page engine key.
# Specific variants (1.3b, 14b, ltx) are matched first to ensure precision.
VIDEO_ENGINE_HINTS = (
    ("wan_1.3b", ("1.3b", "1_3b", "t2v-1.3b", "t2v_1_3b", "1.3")),
    ("wan_14b", ("14b", "14-b", "t2v-14b", "t2v_14b", "wan2", "wan-2", "wanvideo", "wan_2", "wan")),
    ("ltx_video", ("ltx", "lightricks")),
    ("cogvideox", ("cogvideo", "cogvideox")),
    ("hailuo_h3", ("hailuo", "h3", "minimax")),
    ("hunyuan", ("hunyuan",)),
    ("audioreactive", ("audioreactive", "reactive")),
)

DEFAULT_VIDEO_ENGINE = "wan_14b"


def resolve_engine_for_video_model(model_info: Optional[Dict[str, Any]]) -> str:
    """Map a model-manager video model entry to a Music Videos engine key.

    Blob = id + repo_id + name lowercased; first keyword-family hit wins.
    Returns DEFAULT_VIDEO_ENGINE when nothing maps (backwards compatible).
    """
    if not model_info:
        return DEFAULT_VIDEO_ENGINE

    model_id = str(model_info.get("id") or "").lower()
    if model_id in MODEL_MAX_DURATIONS:
        return "wan_14b" if model_id == "wan2.1" else model_id

    blob = " ".join(
        str(model_info.get(k) or "")
        for k in ("id", "repo_id", "name", "local_path")
    ).lower()
    if not blob.strip():
        return DEFAULT_VIDEO_ENGINE

    # Explicit Wan discrimination
    if "wan" in blob:
        if "1.3" in blob or "1_3" in blob:
            return "wan_1.3b"
        return "wan_14b"

    for engine, keywords in VIDEO_ENGINE_HINTS:
        if any(k in blob for k in keywords):
            return engine
    return DEFAULT_VIDEO_ENGINE


class VideoService:
    _instance = None
    _tasks: Dict[str, Dict[str, Any]] = {}
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VideoService, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_model_max_duration(cls, model_name: Optional[str] = None) -> float:
        """Resolve maximum architectural clip duration for a video model (e.g. H3 15s, Hunyuan 15s, CogVideoX 10s, Wan 5s)."""
        if not model_name:
            return 5.0
        m = model_name.lower().strip()
        if "ltx" in m:
            return 10.0
        if "cog" in m:
            return 10.0
        if "hailuo" in m or "h3" in m or "minimax" in m:
            return 15.0
        if "hunyuan" in m:
            return 15.0
        if "audioreactive" in m or "reactive" in m:
            return 120.0
        if "wan" in m:
            return 5.0
        return MODEL_MAX_DURATIONS.get(m, 5.0)

    @classmethod
    def get_available_video_models(cls) -> Dict[str, Any]:
        """Query model_manager for installed weights and return dynamic model registry."""
        from app.services.model_manager import model_manager

        installed_models: Dict[str, Dict[str, Any]] = {}
        try:
            tree = model_manager.get_model_tree()
            for m in tree:
                if m.get("category") == "video":
                    installed_models[m["id"]] = m
        except Exception as e:
            logger.warning(f"Could not query model tree for video models: {e}")

        def is_installed(*model_ids: str) -> bool:
            for mid in model_ids:
                item = installed_models.get(mid)
                if item and item.get("is_installed"):
                    return True
            return False

        has_wan_14b = is_installed("wan2_1_t2v_14b")
        has_wan_1_3b = is_installed("wan2_1_t2v_1_3b")
        has_h3 = is_installed("minimax_h3", "minimax_h3_gguf", "minimax_h3_mlx_8bit")
        has_cog = is_installed("cogvideox_5b")
        has_hunyuan = is_installed("hunyuan_video")
        has_ltx = any("ltx" in mid.lower() and m.get("is_installed") for mid, m in installed_models.items())

        catalog = {
            "wan_14b": {
                "id": "wan_14b",
                "name": "Wan 2.1 14B Flagship",
                "max_duration": 5.0,
                "local_weights_present": has_wan_14b,
                "weights_path": (installed_models.get("wan2_1_t2v_14b") or {}).get("local_path"),
                "family": "dit",
                "description": "Alibaba Wan 2.1 14B DiT with 3D temporal diffusion & keyframe I2V"
            },
            "wan_1.3b": {
                "id": "wan_1.3b",
                "name": "Wan 2.1 1.3B Fast",
                "max_duration": 5.0,
                "local_weights_present": has_wan_1_3b,
                "weights_path": (installed_models.get("wan2_1_t2v_1_3b") or {}).get("local_path"),
                "family": "t2v",
                "description": "Lightweight text-to-video diffusion for rapid local preview"
            },
            "ltx_video": {
                "id": "ltx_video",
                "name": "LTX-Video 0.9B Realtime",
                "max_duration": 10.0,
                "local_weights_present": has_ltx,
                "family": "dit",
                "description": "Lightricks 0.9B real-time DiT (24 fps) for quick scene rendering"
            },
            "cogvideox": {
                "id": "cogvideox",
                "name": "THUDM CogVideoX 1.5 (5B)",
                "max_duration": 10.0,
                "local_weights_present": has_cog,
                "family": "3d-vae",
                "description": "5B 3D causal VAE model with emotive cinematic depth zooms"
            },
            "hailuo_h3": {
                "id": "hailuo_h3",
                "name": "MiniMax Hailuo H3 (33B DiT)",
                "max_duration": 15.0,
                "local_weights_present": has_h3,
                "weights_path": (installed_models.get("minimax_h3_mlx_8bit") or installed_models.get("minimax_h3") or {}).get("local_path"),
                "family": "dit",
                "description": "33B Omni-Modal DiT flagship with high visual fidelity and beat-matched rhythm"
            },
            "hunyuan": {
                "id": "hunyuan",
                "name": "Tencent HunyuanVideo (13B DiT)",
                "max_duration": 15.0,
                "local_weights_present": has_hunyuan,
                "family": "dit",
                "description": "Open-source 13B visual DiT sequence renderer with wide panoramic sweeps"
            },
            "audioreactive": {
                "id": "audioreactive",
                "name": "Audio-Reactive Full Visualizer",
                "max_duration": 120.0,
                "local_weights_present": True,
                "family": "procedural",
                "description": "Continuous full-timeline audio reactive spectrum & waveform visualizer"
            },
            # Backwards compatibility alias
            "wan2.1": {
                "id": "wan_14b",
                "name": "Wan-AI Wan 2.1 (1.3B/14B)",
                "max_duration": 5.0,
                "local_weights_present": has_wan_14b or has_wan_1_3b,
                "family": "t2v",
                "description": "Wan 2.1 text-to-video diffusion"
            }
        }
        return catalog

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        from app.services.video.video_orchestrator import video_orchestrator
        t = video_orchestrator.get_task(task_id)
        if t:
            return t
        with self._lock:
            return self._tasks.get(task_id)

    def get_video_providers(self) -> List[Dict[str, Any]]:
        from app.services.video.video_orchestrator import video_orchestrator
        return video_orchestrator.get_video_providers()

    async def generate_scene_keyframes(self, job: Job, visual_style: str = "neon-cyberpunk", width: int = 1280, height: int = 720, custom_style_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        from app.services.video.video_orchestrator import video_orchestrator
        return await video_orchestrator.generate_scene_keyframes(job=job, visual_style=visual_style, width=width, height=height, custom_style_prompt=custom_style_prompt)

    @classmethod
    def get_active_video_engine(cls) -> Dict[str, Any]:
        """Resolve the engine the Music Videos page should default to.

        Reads the model the user marked active in Models & HW
        (model_manager, persisted in active_models.json under 'video') and maps
        it to a page engine key. Falls back to the hardcoded default engine
        when none is active or the model doesn't map.
        """
        from app.services.model_manager import model_manager  # lazy: avoids import cycle

        try:
            active = model_manager.get_active_model("video")
        except Exception as e:
            logger.warning(f"Could not resolve active video model: {e}")
            active = None

        engine = resolve_engine_for_video_model(active) if active else DEFAULT_VIDEO_ENGINE
        return {
            "engine": engine,
            "model_id": (active or {}).get("id"),
            "name": (active or {}).get("name"),
            "weights_present": bool(active and active.get("is_installed")),
        }

    @classmethod
    def set_active_video_engine(cls, engine_or_model_id: str) -> Dict[str, Any]:
        """Activate a video engine and persist selection to active_models.json."""
        from app.services.model_manager import model_manager

        target = (engine_or_model_id or "").strip().lower()

        # Engine to catalog ID mapping
        engine_to_model_map = {
            "wan_14b": "wan2_1_t2v_14b",
            "wan_1.3b": "wan2_1_t2v_1_3b",
            "wan2.1": "wan2_1_t2v_14b",
            "cogvideox": "cogvideox_5b",
            "hailuo_h3": "minimax_h3",
            "hunyuan": "hunyuan_video",
        }

        model_id = engine_to_model_map.get(target, target)

        tree = model_manager.get_model_tree()
        match = next((m for m in tree if m["id"] == model_id or m.get("repo_id") == model_id), None)

        if not match:
            video_models = [m for m in tree if m.get("category") == "video"]
            if "1.3" in target:
                match = next((m for m in video_models if "1_3" in m["id"] or "1.3" in m["id"]), None)
            elif "wan" in target:
                match = next((m for m in video_models if "14" in m["id"] or "wan" in m["id"]), None)
            elif "cog" in target:
                match = next((m for m in video_models if "cog" in m["id"]), None)
            elif "h3" in target or "hailuo" in target or "minimax" in target:
                match = next((m for m in video_models if "h3" in m["id"] or "minimax" in m["id"]), None)
            elif "hunyuan" in target:
                match = next((m for m in video_models if "hunyuan" in m["id"]), None)
            elif "ltx" in target:
                match = next((m for m in video_models if "ltx" in m["id"]), None)

        if match:
            model_manager.set_active_model(match["id"])
            resolved_engine = resolve_engine_for_video_model(match)
            return {
                "status": "ok",
                "engine": resolved_engine,
                "model_id": match["id"],
                "name": match["name"],
                "weights_present": bool(match.get("is_installed")),
            }
        else:
            if "reactive" in target:
                active_dict = model_manager._load_active_models()
                active_dict["video"] = "audioreactive"
                model_manager._save_active_models(active_dict)
                return {
                    "status": "ok",
                    "engine": "audioreactive",
                    "model_id": "audioreactive",
                    "name": "Audio-Reactive Full Visualizer",
                    "weights_present": True,
                }
            raise ValueError(f"Unknown video model or engine: '{engine_or_model_id}'")

    def _update_task(self, task_id: str, **kwargs):
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(kwargs)

    def resolve_audio_path(self, path: Optional[str]) -> Optional[str]:
        """Find readable audio file on disk."""
        if not path:
            return None
        candidates = [
            path,
            path.lstrip("/"),
            path.replace("/audio/", "generated_audio/"),
            os.path.join("generated_audio", os.path.basename(path))
        ]
        for c in candidates:
            if os.path.isfile(c) and os.path.getsize(c) > 0:
                return os.path.abspath(c)
        return None

    def resolve_vocals_stem(self, job: Job) -> Optional[str]:
        """Locate isolated vocals stem or return None."""
        stems_json = getattr(job, "stems_json", None)
        if stems_json:
            try:
                data = json.loads(stems_json) if isinstance(stems_json, str) else stems_json
                if isinstance(data, dict) and data.get("vocals"):
                    p = self.resolve_audio_path(data["vocals"])
                    if p:
                        return p
            except Exception:
                pass

        # Look in generated_audio/stems/{job.id}/vocals.mp3 or .wav
        for ext in [".mp3", ".wav"]:
            cand = os.path.join("generated_audio", "stems", str(job.id), f"vocals{ext}")
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return os.path.abspath(cand)

        return None

    def resolve_face_image(self, job: Job, custom_image: Optional[str] = None) -> Optional[str]:
        """Locate character or artist face image."""
        if custom_image:
            cand = self.resolve_audio_path(custom_image) or os.path.join("data", "covers", os.path.basename(custom_image))
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return os.path.abspath(cand)

        if job.cover_image_path:
            candidates = [
                job.cover_image_path,
                job.cover_image_path.lstrip("/"),
                os.path.join("data", "covers", os.path.basename(job.cover_image_path)),
                os.path.join("generated_audio", os.path.basename(job.cover_image_path))
            ]
            for c in candidates:
                if os.path.isfile(c) and os.path.getsize(c) > 0:
                    return os.path.abspath(c)
        return None

    def segment_song_for_video(
        self,
        job: Job,
        max_clip_duration: Optional[float] = None,
        model_name: Optional[str] = "wan_14b",
        bpm: Optional[float] = None,
        visual_style: str = "neon-cyberpunk",
        custom_style_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Segment the entire song into consecutive clips respecting model duration constraints.
        Snaps cuts to musical bars and lyric pauses, and tags scenes as VOCAL vs CINEMATIC B-ROLL.
        """
        total_duration = float((job.duration_ms or 180000) / 1000.0)
        track_bpm = float(bpm or 120.0)
        seconds_per_bar = (60.0 / track_bpm) * 4.0

        model_max = self.get_model_max_duration(model_name)
        if max_clip_duration is None or max_clip_duration <= 0:
            effective_max = model_max
        else:
            effective_max = max(1.0, min(float(max_clip_duration), model_max))

        # Calculate target clip duration clamped to musical bars
        bars_per_clip = max(1, int(round(effective_max / seconds_per_bar)))
        target_clip_len = bars_per_clip * seconds_per_bar
        if target_clip_len > effective_max and bars_per_clip > 1:
            target_clip_len = (bars_per_clip - 1) * seconds_per_bar
        target_clip_len = max(1.5, min(target_clip_len, effective_max))

        # Retrieve timed lyrics via lyric_sync_engine
        vocals_path = self.resolve_vocals_stem(job)
        timed_lines = lyric_sync_engine.align_lyrics(
            lyrics=job.lyrics or "",
            duration_sec=total_duration,
            vocal_stem_path=vocals_path
        )

        clips: List[Dict[str, Any]] = []
        cur_time = 0.0
        clip_idx = 1
        if custom_style_prompt and custom_style_prompt.strip():
            palette = {
                "colors": "0x14b8a6|0x06b6d4",
                "bg": "0x0a0f1d",
                "primary_color": (20, 184, 166),
                "accent_color": (6, 182, 212),
                "desc": "Custom Directing",
                "atmosphere": custom_style_prompt.strip(),
                "negative": "blurry, low resolution, watermark, bad hands, distorted anatomy"
            }
        else:
            palette = STYLE_PALETTES.get(visual_style, STYLE_PALETTES["neon-cyberpunk"])

        cameras = [
            "Medium orbital shot focusing on performer",
            "Slow cinematic tracking crane down",
            "Wide atmospheric environmental sweep",
            "Dutch angle low push-in with rim light flare",
            "Tight emotive close-up with soft depth of field"
        ]

        while cur_time < total_duration:
            clip_end = min(cur_time + target_clip_len, total_duration)
            if (total_duration - clip_end) < 2.0:
                clip_end = total_duration

            # Check for active lyrics in this window
            overlapping_lyrics = []
            for line in timed_lines:
                l_start = line.get("start", 0.0)
                l_end = line.get("end", 0.0)
                if max(cur_time, l_start) < min(clip_end, l_end):
                    overlapping_lyrics.append(line.get("text", "").strip())

            has_vocals = len(overlapping_lyrics) > 0
            scene_type = "VOCAL_PERFORMANCE" if has_vocals else "CINEMATIC_BROLL"
            lyric_snippet = " / ".join(overlapping_lyrics) if overlapping_lyrics else ""

            s_m, s_s = int(cur_time // 60), int(cur_time % 60)
            e_m, e_s = int(clip_end // 60), int(clip_end % 60)
            time_label = f"{s_m}:{s_s:02d} - {e_m}:{e_s:02d}"

            if has_vocals:
                prompt = f"{palette['desc']}: Singer performing passionately in {job.tags or 'urban studio'}. Lyrics: \"{lyric_snippet[:60]}\". Volumetric lighting and particle atmosphere."
            else:
                prompt = f"{palette['desc']}: Cinematic B-roll scenery, sonic wave pulse through cityscape, rhythmic strobe reflections and atmospheric haze."

            clips.append({
                "clip_index": clip_idx,
                "start_time": round(cur_time, 2),
                "end_time": round(clip_end, 2),
                "duration": round(clip_end - cur_time, 2),
                "time_str": time_label,
                "is_vocal": has_vocals,
                "scene_type": scene_type,
                "lyrics": lyric_snippet,
                "prompt": prompt,
                "camera": cameras[(clip_idx - 1) % len(cameras)],
                "lighting": "Cyan and magenta anamorphic rim lighting"
            })

            clip_idx += 1
            cur_time = clip_end

        return clips

    def generate_karaoke_ass(
        self,
        timed_lines: List[Dict[str, Any]],
        width: int = 1280,
        height: int = 720,
        style: str = "neon-cyberpunk"
    ) -> str:
        """
        Generate an Advanced SubStation Alpha (.ass) subtitle file
        with karaoke highlight tags and studio typography.
        """
        palette = STYLE_PALETTES.get(style, STYLE_PALETTES["neon-cyberpunk"])
        r, g, b = palette["primary_color"]
        # In ASS color format &HAABBGGRR
        primary_ass = f"&H00{b:02X}{g:02X}{r:02X}"
        ar, ag, ab = palette["accent_color"]
        accent_ass = f"&H00{ab:02X}{ag:02X}{ar:02X}"

        font_size = int(height * 0.045)
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
            ass_lines.append(f"Dialogue: 0,{start_t},{end_t},Default,,0,0,0,,{text}")

        return "\n".join(ass_lines)

    async def render_lip_sync_clip(
        self,
        face_image_path: str,
        vocal_audio_path: str,
        start_time: float,
        duration: float,
        out_path: str,
        width: int = 1280,
        height: int = 720
    ):
        """
        Renders an audio-driven singing vocal performance clip from an artist portrait
        and the isolated vocal audio slice using vocal-energy viseme deformation.
        """
        # 1. Extract audio slice for this clip (16-bit mono 44.1kHz for analysis & muxing)
        slice_audio = os.path.join(TEMP_DIR, f"vocal_slice_{uuid.uuid4().hex[:8]}.wav")
        cmd_cut = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", vocal_audio_path,
            "-ar", "44100", "-ac", "1",
            slice_audio
        ]
        proc = await asyncio.create_subprocess_exec(*cmd_cut, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()

        fps = 25
        total_frames = max(1, int(round(duration * fps)))
        frames_dir = os.path.join(TEMP_DIR, f"frames_{uuid.uuid4().hex[:8]}")
        os.makedirs(frames_dir, exist_ok=True)

        try:
            # 2. Extract vocal energy envelope
            envelopes: List[float] = []
            if wavfile is not None and os.path.isfile(slice_audio):
                try:
                    sr, audio_data = wavfile.read(slice_audio)
                    if audio_data.ndim > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    elif audio_data.dtype == np.int32:
                        audio_data = audio_data.astype(np.float32) / 2147483648.0
                    else:
                        audio_data = audio_data.astype(np.float32)

                    frame_len = max(1, int(sr / fps))
                    for f_i in range(total_frames):
                        s_idx = f_i * frame_len
                        e_idx = min(len(audio_data), (f_i + 1) * frame_len)
                        if e_idx > s_idx:
                            rms = float(np.sqrt(np.mean(audio_data[s_idx:e_idx] ** 2)))
                        else:
                            rms = 0.0
                        envelopes.append(rms)
                except Exception as ex:
                    logger.warning(f"Error computing vocal envelope: {ex}")

            if not envelopes:
                envelopes = [0.0] * total_frames

            # Dynamic range normalization & ballistic smoothing
            max_e = max(envelopes) if envelopes else 0.0
            if max_e > 1e-4:
                p95 = float(np.percentile(envelopes, 95))
                scale = max(p95, 1e-3)
                norm_env = [min(1.0, e / scale) for e in envelopes]
            else:
                norm_env = [0.0] * total_frames

            smoothed_env = []
            c_val = 0.0
            for val in norm_env:
                if val > c_val:
                    c_val = c_val * 0.35 + val * 0.65  # Fast vocal attack
                else:
                    c_val = c_val * 0.70 + val * 0.30  # Smooth acoustic release
                smoothed_env.append(c_val)

            # 3. Detect facial structure & mouth coordinates
            if cv2 is not None:
                base_bgr = cv2.imread(face_image_path)
                if base_bgr is None:
                    pil_img = Image.open(face_image_path).convert("RGB")
                    base_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                base_bgr = cv2.resize(base_bgr, (width, height), interpolation=cv2.INTER_LANCZOS4)
                gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)

                cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                faces = []
                if os.path.isfile(cascade_path):
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=4, minSize=(int(height * 0.12), int(height * 0.12))
                    )

                if len(faces) > 0:
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    fx, fy, fw, fh = faces[0]
                    cx = fx + fw // 2
                    mx = cx
                    my = fy + int(0.74 * fh)
                    mw = int(0.42 * fw)
                    mh = int(0.20 * fh)
                else:
                    mx = width // 2
                    my = int(height * 0.64)
                    mw = int(width * 0.18)
                    mh = int(height * 0.08)

                # 4. Generate viseme mouth-deformed video frames
                for i in range(total_frames):
                    viseme = smoothed_env[i]
                    frame = base_bgr.copy()

                    if viseme > 0.04:
                        # Opening displacement scaled by vocal energy
                        open_h = int(mh * 0.65 * viseme)
                        open_w = int(mw * 0.40 * (1.0 + 0.18 * viseme))

                        if open_h >= 2 and open_w >= 3:
                            lip_y1 = max(0, my)
                            lip_y2 = min(height, my + mh + open_h)
                            lip_x1 = max(0, mx - open_w)
                            lip_x2 = min(width, mx + open_w)

                            if (lip_y2 - lip_y1) > open_h and (lip_x2 - lip_x1) > 0:
                                # Extract original lower lip ROI
                                lower_lip_slice = base_bgr[lip_y1 : lip_y2 - open_h, lip_x1 : lip_x2].copy()

                                # Render deep oral cavity
                                cv2.ellipse(
                                    frame,
                                    (mx, my + open_h // 2),
                                    (open_w, open_h),
                                    0, 0, 360,
                                    (22, 14, 42), -1, cv2.LINE_AA
                                )
                                # Subtle upper teeth curve
                                teeth_y = my + max(1, open_h // 4)
                                cv2.ellipse(
                                    frame,
                                    (mx, teeth_y),
                                    (int(open_w * 0.58), max(1, open_h // 5)),
                                    0, 0, 180,
                                    (210, 218, 222), -1, cv2.LINE_AA
                                )

                                # Blend translated lower lip over aperture with smooth vertical alpha
                                target_slice = frame[lip_y1 + open_h : lip_y2, lip_x1 : lip_x2]
                                if target_slice.shape == lower_lip_slice.shape:
                                    alpha_mask = np.linspace(0.35, 1.0, target_slice.shape[0], dtype=np.float32)[:, None, None]
                                    blended = (lower_lip_slice * alpha_mask + target_slice * (1.0 - alpha_mask)).astype(np.uint8)
                                    frame[lip_y1 + open_h : lip_y2, lip_x1 : lip_x2] = blended

                    # Studio rim flare & subtle breathing motion
                    osc = math.sin(i * 0.35)
                    rim_intensity = int(12 + 8 * osc + 15 * viseme)
                    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (166, 184, 20), max(1, rim_intensity // 4))

                    frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                    cv2.imwrite(frame_path, frame)

            else:
                # PIL Fallback
                base_img = Image.open(face_image_path).convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
                for i in range(total_frames):
                    frame_img = base_img.copy()
                    draw = ImageDraw.Draw(frame_img)
                    osc = math.sin(i * 0.4)
                    draw.rectangle([(0, 0), (width, height)], outline=(20, 184, 166, int(25 + 15 * osc)), width=6)
                    frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                    frame_img.save(frame_path)

            # 5. Assemble frames + vocal slice into clip MP4
            cmd_clip = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(frames_dir, "frame_%04d.png"),
                "-i", slice_audio,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                "-c:a", "aac", "-b:a", "192k",
                "-t", str(duration),
                out_path
            ]
            proc_clip = await asyncio.create_subprocess_exec(*cmd_clip, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc_clip.communicate()

        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)
            if os.path.isfile(slice_audio):
                os.remove(slice_audio)

    async def render_broll_clip(
        self,
        style: str,
        duration: float,
        out_path: str,
        width: int = 1280,
        height: int = 720,
        bg_image: Optional[str] = None,
        prompt: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Renders a dynamic cinematic B-roll scene clip with camera motion and lighting.
        Supports diffusion models if weights are present, with high-end procedural
        Ken Burns and atmospheric particle field generation.
        """
        palette = STYLE_PALETTES.get(style, STYLE_PALETTES["neon-cyberpunk"])
        fps = 25
        total_d = max(1, int(round(duration * fps)))

        prompt_str = (prompt or "").lower()
        if "close-up" in prompt_str or "tight" in prompt_str:
            zoom_expr = "min(zoom+0.0018,1.28)"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif "pan" in prompt_str or "sweep" in prompt_str or "environmental" in prompt_str:
            zoom_expr = "1.15"
            x_expr = "if(lte(on,1),(iw-iw/zoom)/2,x+0.8)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif "crane" in prompt_str or "low" in prompt_str or "dutch" in prompt_str:
            zoom_expr = "min(zoom+0.0012,1.20)"
            x_expr = "iw/2-(iw/zoom/2)+sin(in/20)*25"
            y_expr = "ih/2-(ih/zoom/2)+cos(in/25)*30"
        else:
            zoom_expr = "min(zoom+0.0014,1.22)"
            x_expr = "iw/2-(iw/zoom/2)+sin(in/25)*30"
            y_expr = "ih/2-(ih/zoom/2)+cos(in/30)*20"

        if bg_image and os.path.isfile(bg_image):
            filter_str = (
                f"scale={int(width * 1.25)}:{int(height * 1.25)},"
                f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={total_d}:s={width}x{height},"
                f"eq=contrast=1.10:saturation=1.20:brightness=0.01"
            )
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", bg_image,
                "-f", "lavfi", "-t", str(duration), "-i", "anullsrc=r=44100:cl=stereo",
                "-vf", filter_str,
                "-t", str(duration),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path
            ]
        else:
            # Procedural dynamic visual synthesizer with flowing chromatic plasma & atmospheric haze
            pr, pg, pb = palette["primary_color"]
            ar, ag, ab = palette["accent_color"]
            filter_str = (
                f"nullsrc=s={width}x{height}:d={duration},"
                f"geq=r='{pr // 6}+{pr // 4}*sin(X/120+T*2.2)+{ar // 4}*cos(Y/140+T*1.8)':"
                f"g='{pg // 6}+{pg // 4}*cos(Y/130+T*2.0)+{ag // 4}*sin(X/150+T*2.4)':"
                f"b='{pb // 6}+{pb // 4}*sin((X+Y)/160+T*2.6)+{ab // 4}*cos(X/110+T*1.9)',"
                f"boxblur=luma_radius=12:luma_power=2,"
                f"eq=contrast=1.12:saturation=1.25"
            )
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", filter_str,
                "-f", "lavfi", "-t", str(duration), "-i", "anullsrc=r=44100:cl=stereo",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                "-t", str(duration),
                out_path
            ]

        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, err = await proc.communicate()
        if proc.returncode != 0 or not os.path.isfile(out_path):
            logger.error(f"render_broll_clip failed ({proc.returncode}): {err.decode('utf-8', errors='ignore')}")

    async def generate_storyboard(
        self,
        job: Job,
        visual_style: str = "neon-cyberpunk",
        custom_style_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a beat-matched, musical scene storyboard sequence for a job.
        Used by the AI Music Video Studio for directing notes and clip planning.
        """
        clips = self.segment_song_for_video(
            job=job,
            max_clip_duration=15.0,
            bpm=120.0,
            visual_style=visual_style,
            custom_style_prompt=custom_style_prompt
        )
        scenes = []
        for c in clips:
            scenes.append({
                "time": c["time_str"],
                "prompt": c["prompt"],
                "camera": c["camera"],
                "lighting": c.get("lighting", "Cyan and magenta anamorphic rim lighting")
            })
        return scenes

    async def render_audio_reactive_video(
        self,
        job: Job,
        visual_style: str = "neon-cyberpunk",
        resolution: str = "720p"
    ) -> str:
        """
        Render an audio-reactive visualizer music video using ffmpeg.
        Outputs an MP4 with stereo audio and returns the static URL path.
        """
        resolved_master = self.resolve_audio_path(job.audio_path)
        if not resolved_master or not os.path.isfile(resolved_master):
            raise FileNotFoundError(f"Master audio file not found for job: {job.audio_path}")

        palette = STYLE_PALETTES.get(visual_style, STYLE_PALETTES["neon-cyberpunk"])
        colors = palette.get("colors", "0x14b8a6|0x06b6d4")
        width, height = (1920, 1080) if resolution == "1080p" else (1280, 720)

        out_filename = f"{job.id}_reactive.mp4"
        out_path = os.path.join(VIDEO_DIR, out_filename)

        face_image = self.resolve_face_image(job)
        if face_image and os.path.isfile(face_image):
            filter_complex = (
                f"[1:a]showwaves=s={width}x{int(height * 0.35)}:mode=line:colors={colors}:scale=sqrt[wv];"
                f"[0:v]scale={width}:{height},boxblur=4:1[bg];"
                f"[bg][wv]overlay=(W-w)/2:H-h-40[v]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", face_image,
                "-i", resolved_master,
                "-filter_complex", filter_complex,
                "-map", "[v]", "-map", "1:a",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path
            ]
        else:
            filter_complex = (
                f"[0:a]showcqt=s={width}x{height}:csp=bt709:bar_g=2:basefreq=40:endfreq=12000[v]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", resolved_master,
                "-filter_complex", filter_complex,
                "-map", "[v]", "-map", "0:a",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path
            ]

        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()

        video_url = f"/audio/videos/{out_filename}"
        return video_url

    async def render_advanced_music_video(
        self,
        job: Job,
        task_id: str,
        config: Dict[str, Any]
    ) -> str:
        """
        Orchestrate multi-scene music video production via VideoOrchestrator:
        1. Segment track to match duration constraints & musical beats
        2. Render lip-synced singing avatar clips (LivePortrait / Cloud / Smooth Viseme)
        3. Diffuse cinematic B-roll scenes (Wan 2.1 14B / 1.3B / LTX-Video / Cloud)
        4. Assemble clips with beat-synchronized transitions
        5. Burn synchronized lyric / karaoke subtitles
        6. Remux master stereo audio with sample-accurate sync
        """
        from app.services.video.video_orchestrator import video_orchestrator
        return await video_orchestrator.render_advanced_music_video(
            job=job,
            task_id=task_id,
            config=config
        )


video_service = VideoService()
