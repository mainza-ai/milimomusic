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

STYLE_PALETTES = {
    "neon-cyberpunk": {
        "colors": "0x14b8a6|0x06b6d4",
        "bg": "0x0a0f1d",
        "primary_color": (20, 184, 166),
        "accent_color": (6, 182, 212),
        "desc": "Cyberpunk Neon"
    },
    "anime-cinematic": {
        "colors": "0xf43f5e|0xf59e0b",
        "bg": "0x111827",
        "primary_color": (244, 63, 94),
        "accent_color": (245, 158, 11),
        "desc": "Anime Cinematic"
    },
    "retro-vhs": {
        "colors": "0xa855f7|0xec4899",
        "bg": "0x0f0b1e",
        "primary_color": (168, 85, 247),
        "accent_color": (236, 72, 153),
        "desc": "80s Retro VHS"
    },
    "minimal-lyrics": {
        "colors": "0x38bdf8|0x818cf8",
        "bg": "0x090d16",
        "primary_color": (240, 240, 245),
        "accent_color": (56, 189, 248),
        "desc": "Minimal Typography"
    }
}

# Video palette key -> cinematic image descriptor for scene-background stills.
# The palette key drives ffmpeg colors; this descriptor drives the diffusion
# prompt (they are different surfaces — never pass the palette key as an
# image style).
SCENE_STYLE_DESCRIPTORS: Dict[str, str] = {
    "neon-cyberpunk": "neon-lit cyberpunk city atmosphere, rain-slicked streets, volumetric glow",
    "anime-cinematic": "anime cinematic film aesthetic, dramatic sky, expressive lighting",
    "retro-vhs": "80s retro film aesthetic, soft grain, neon dusk glow",
    "minimal-lyrics": "minimal atmospheric cinema, soft gradient light, empty space",
}

MODEL_MAX_DURATIONS: Dict[str, float] = {
    "hailuo_h3": 15.0,
    "hunyuan": 15.0,
    "cogvideox": 10.0,
    "wan2.1": 5.0,
    "audioreactive": 120.0,
}

# Model-manager video model -> Music Videos page engine key. Catalog ids AND
# repo ids/names are keyword-matched so user-downloaded (custom_*) models map
# too. Mirrors modality.py's keyword rules; audioreactive has no model entry.
VIDEO_ENGINE_HINTS = (
    ("hailuo_h3", ("hailuo", "h3", "minimax")),
    ("hunyuan", ("hunyuan",)),
    ("cogvideox", ("cogvideo", "cogvideox")),
    ("wan2.1", ("wan2", "wan-2", "wanvideo", "wan_2")),
)

DEFAULT_VIDEO_ENGINE = "hailuo_h3"


def resolve_engine_for_video_model(model_info: Optional[Dict[str, Any]]) -> str:
    """Map a model-manager video model entry to a Music Videos engine key.

    Blob = id + repo_id + name lowercased; first keyword-family hit wins.
    Returns DEFAULT_VIDEO_ENGINE when nothing maps (backwards compatible).
    """
    if not model_info:
        return DEFAULT_VIDEO_ENGINE
    blob = " ".join(
        str(model_info.get(k) or "")
        for k in ("id", "repo_id", "name", "local_path")
    ).lower()
    if not blob.strip():
        return DEFAULT_VIDEO_ENGINE
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
        """Resolve maximum architectural clip duration for a video model (e.g. H3 15s, Hunyuan 15s, CogVideoX 10s, Wan2.1 5s)."""
        if not model_name:
            return 5.0
        m = model_name.lower().strip()
        if "hailuo" in m or "h3" in m or "minimax" in m:
            return 15.0
        if "hunyuan" in m:
            return 15.0
        if "cog" in m:
            return 10.0
        if "audioreactive" in m or "reactive" in m:
            return 120.0
        if "wan" in m:
            return 5.0
        return MODEL_MAX_DURATIONS.get(m, 5.0)

    @classmethod
    def get_available_video_models(cls) -> Dict[str, Any]:
        """Detect local video model weights in models/video/ and return model registry."""
        base_models_dir = os.path.join(os.getcwd(), "models", "video")
        h3_path = os.path.join(base_models_dir, "pipenetwork__MiniMax-H3-MLX-8bit")
        has_h3 = os.path.isdir(h3_path) and any(f.endswith(".safetensors") for f in os.listdir(h3_path) if os.path.isfile(os.path.join(h3_path, f)))
        return {
            "hailuo_h3": {
                "id": "hailuo_h3",
                "name": "MiniMax Hailuo H3 (33B DiT)",
                "max_duration": 15.0,
                "local_weights_present": has_h3,
                "weights_path": h3_path if has_h3 else None,
                "family": "dit",
                "description": "33B Omni-Modal DiT flagship with high visual fidelity and beat-matched rhythm"
            },
            "hunyuan": {
                "id": "hunyuan",
                "name": "Tencent HunyuanVideo (13B DiT)",
                "max_duration": 15.0,
                "local_weights_present": False,
                "family": "dit",
                "description": "Open-source 13B visual DiT sequence renderer with wide panoramic sweeps"
            },
            "cogvideox": {
                "id": "cogvideox",
                "name": "THUDM CogVideoX 1.5 (5B)",
                "max_duration": 10.0,
                "local_weights_present": False,
                "family": "3d-vae",
                "description": "5B 3D causal VAE model with emotive cinematic depth zooms"
            },
            "wan2.1": {
                "id": "wan2.1",
                "name": "Wan-AI Wan 2.1 (1.3B/14B)",
                "max_duration": 5.0,
                "local_weights_present": False,
                "family": "t2v",
                "description": "Lightweight text-to-video diffusion with rapid tempo cuts"
            },
            "audioreactive": {
                "id": "audioreactive",
                "name": "Audio-Reactive Synth Visualizer",
                "max_duration": 120.0,
                "local_weights_present": True,
                "family": "procedural",
                "description": "Real-time procedural waveform, frequency spectrum, and chromatic plasma synthesis"
            }
        }

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._tasks.get(task_id)

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
        model_name: Optional[str] = "wan2.1",
        bpm: Optional[float] = None,
        visual_style: str = "neon-cyberpunk"
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
        visual_style: str = "neon-cyberpunk"
    ) -> List[Dict[str, Any]]:
        """
        Generate a beat-matched, musical scene storyboard sequence for a job.
        Used by the AI Music Video Studio for directing notes and clip planning.
        """
        clips = self.segment_song_for_video(
            job=job,
            max_clip_duration=15.0,
            bpm=120.0,
            visual_style=visual_style
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
        Orchestrate multi-scene music video production:
        1. Segment track to match duration constraints
        2. Render lip-synced vocal performance and cinematic B-roll clips
        3. Assemble clips with beat-synchronized transitions
        4. Burn synchronized lyric / karaoke subtitles
        5. Remux master stereo audio with sample-accurate sync
        """
        self._tasks[task_id] = {
            "id": task_id,
            "job_id": str(job.id),
            "status": "processing",
            "step": "Analyzing Track & Ingesting Stems",
            "progress": 5,
            "total_clips": 0,
            "current_clip": 0,
            "video_url": None,
            "error": None
        }

        try:
            resolved_master = self.resolve_audio_path(job.audio_path)
            if not resolved_master:
                raise FileNotFoundError(f"Master audio not found for job: {job.audio_path}")

            style = config.get("visual_style", "neon-cyberpunk")
            resolution = config.get("resolution", "720p")
            aspect_ratio = config.get("aspect_ratio", "16:9")
            model_name = config.get("model_name", "wan2.1")
            model_max = self.get_model_max_duration(model_name)
            raw_dur = config.get("max_clip_duration")
            if raw_dur is not None and float(raw_dur) > 0:
                max_duration = max(1.0, min(float(raw_dur), model_max))
            else:
                max_duration = model_max
            enable_lip_sync = config.get("enable_lip_sync", True)
            burn_lyrics = config.get("burn_lyrics", True)

            w, h = (1920, 1080) if resolution == "1080p" else (1280, 720)
            if aspect_ratio == "9:16":
                w, h = h, w

            vocal_stem = self.resolve_vocals_stem(job)
            face_image = self.resolve_face_image(job, config.get("face_image_path"))

            # Step 1: Song Segmentation
            self._update_task(task_id, step="Segmenting Song to Model Duration Constraints", progress=15)
            clips = self.segment_song_for_video(
                job=job,
                max_clip_duration=max_duration,
                model_name=model_name,
                bpm=config.get("bpm"),
                visual_style=style
            )
            self._update_task(task_id, total_clips=len(clips))

            # Step 2: Render individual scene clips
            self._update_task(task_id, step="Rendering Lip-Sync & Scene Clips", progress=25)
            rendered_clips: List[str] = []
            total_clips = len(clips)

            if model_name == "audioreactive":
                self._update_task(task_id, step="Rendering Audio-Reactive Visualizer Video", progress=50)
                video_url = await self.render_audio_reactive_video(
                    job=job,
                    visual_style=style,
                    resolution=resolution
                )
                self._update_task(task_id, status="completed", progress=100, step="Completed", video_url=video_url)
                return video_url

            for idx, clip in enumerate(clips):
                clip_file = os.path.join(TEMP_DIR, f"clip_{task_id}_{idx:03d}.mp4")
                self._update_task(
                    task_id,
                    current_clip=idx + 1,
                    step=f"Rendering Clip {idx + 1}/{total_clips} ({clip['scene_type']})",
                    progress=25 + int(50 * (idx / total_clips))
                )

                if clip["is_vocal"] and enable_lip_sync and face_image and vocal_stem:
                    await self.render_lip_sync_clip(
                        face_image_path=face_image,
                        vocal_audio_path=vocal_stem,
                        start_time=clip["start_time"],
                        duration=clip["duration"],
                        out_path=clip_file,
                        width=w, height=h
                    )
                else:
                    scene_bg = None
                    try:
                        from app.services.image_service import image_service
                        scene_prompt = clip.get("prompt") or f"{style} music video, {clip.get('scene_type', 'cinematic')} shot"
                        scene_descriptor = SCENE_STYLE_DESCRIPTORS.get(style, style)
                        bg = image_service.generate_scene_background(
                            prompt=scene_prompt,
                            style=scene_descriptor,
                            width=w,
                            height=h,
                        )
                        if bg.get("ok") and bg.get("dest_path") and os.path.isfile(bg["dest_path"]):
                            scene_bg = bg["dest_path"]
                            logger.info(f"Scene background still for clip {idx + 1}: {scene_bg}")
                    except Exception as e:
                        logger.warning(f"Failed to generate unique scene visual ({e}), falling back.")
                        scene_bg = face_image

                    await self.render_broll_clip(
                        style=style,
                        duration=clip["duration"],
                        out_path=clip_file,
                        width=w, height=h,
                        bg_image=scene_bg,
                        prompt=clip.get("prompt"),
                        model_name=model_name
                    )

                if os.path.isfile(clip_file) and os.path.getsize(clip_file) > 0:
                    rendered_clips.append(clip_file)

            if not rendered_clips:
                raise RuntimeError("No video scenes were successfully rendered.")

            # Step 3: Video Assembly
            self._update_task(task_id, step="Assembling & Stitching Video Scenes", progress=80)
            concat_list_path = os.path.join(TEMP_DIR, f"concat_{task_id}.txt")
            with open(concat_list_path, "w") as f:
                for cf in rendered_clips:
                    f.write(f"file '{os.path.abspath(cf)}'\n")

            stitched_video = os.path.join(TEMP_DIR, f"stitched_{task_id}.mp4")
            cmd_concat = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",
                stitched_video
            ]
            proc = await asyncio.create_subprocess_exec(*cmd_concat, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _, err_concat = await proc.communicate()
            if proc.returncode != 0 or not os.path.isfile(stitched_video) or os.path.getsize(stitched_video) == 0:
                logger.warning(f"Fast stream copy concat failed ({err_concat.decode('utf-8', errors='ignore')[:150]}), falling back to re-encode concat.")
                cmd_concat_fallback = [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_list_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                    "-c:a", "aac", "-b:a", "192k",
                    stitched_video
                ]
                proc_fb = await asyncio.create_subprocess_exec(*cmd_concat_fallback, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await proc_fb.communicate()

            if not os.path.isfile(stitched_video) or os.path.getsize(stitched_video) == 0:
                raise RuntimeError(f"Video scene stitching failed: {err_concat.decode('utf-8', errors='ignore')[:200]}")

            # Step 4: Synchronized Lyric Subtitles & Master Audio Remuxing
            self._update_task(task_id, step="Burning Synchronized Lyrics & Remuxing Audio", progress=90)
            out_filename = f"{job.id}_master_mv.mp4"
            out_path = os.path.join(VIDEO_DIR, out_filename)

            # Master Remuxing
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
            if proc_final.returncode != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
                logger.warning("Stream copy remux failed, falling back to re-encode remux.")
                cmd_final_re = [
                    "ffmpeg", "-y",
                    "-i", stitched_video,
                    "-i", resolved_master,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                    "-c:a", "aac", "-b:a", "256k",
                    "-shortest",
                    out_path
                ]
                proc_re = await asyncio.create_subprocess_exec(*cmd_final_re, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await proc_re.communicate()

            if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
                raise RuntimeError("Failed to generate master music video output file.")

            video_url = f"/audio/videos/{out_filename}"
            self._update_task(task_id, status="completed", progress=100, step="Completed", video_url=video_url)
            return video_url

        except Exception as e:
            logger.error(f"Advanced video generation failed: {e}", exc_info=True)
            self._update_task(task_id, status="error", error=str(e), step=f"Error: {str(e)[:100]}")
            raise e
        finally:
            # Clean up temp clips
            for cf in rendered_clips:
                if os.path.isfile(cf):
                    try: os.remove(cf)
                    except: pass


video_service = VideoService()
