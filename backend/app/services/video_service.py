"""
AI Music Video Studio Service.
Provides LLM-driven structured storyboard synthesis and real FFmpeg audio-reactive
music video rendering with waveform visualizers and cover art overlays.
"""

import os
import json
import asyncio
import logging
import subprocess
from typing import List, Dict, Optional, Any
from app.models import Job
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

VIDEO_DIR = "generated_audio/videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

STYLE_PALETTES = {
    "neon-cyberpunk": {"colors": "0x14b8a6|0x06b6d4", "bg": "0x0a0f1d", "desc": "Cyberpunk Neon"},
    "anime-cinematic": {"colors": "0xf43f5e|0xf59e0b", "bg": "0x111827", "desc": "Anime Cinematic"},
    "retro-vhs": {"colors": "0xa855f7|0xec4899", "bg": "0x0f0b1e", "desc": "80s Retro VHS"},
    "minimal-lyrics": {"colors": "0x38bdf8|0x818cf8", "bg": "0x090d16", "desc": "Minimal Typography"}
}


class VideoService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VideoService, cls).__new__(cls)
        return cls._instance

    async def generate_storyboard(self, job: Job, visual_style: str = "neon-cyberpunk") -> List[Dict[str, str]]:
        """
        Synthesize multi-scene storyboard synchronized to song structure and lyrics.
        """
        style_meta = STYLE_PALETTES.get(visual_style, STYLE_PALETTES["neon-cyberpunk"])
        duration_sec = int((job.duration_ms or 240000) / 1000)
        minutes = duration_sec // 60
        seconds = duration_sec % 60
        duration_str = f"{minutes}:{seconds:02d}"

        # Segment song into 3-5 cinematic timeline windows
        seg_duration = max(15, duration_sec // 4)
        time_windows = []
        for start_s in range(0, duration_sec, seg_duration):
            end_s = min(start_s + seg_duration, duration_sec)
            s_m, s_s = start_s // 60, start_s % 60
            e_m, e_s = end_s // 60, end_s % 60
            time_windows.append(f"{s_m}:{s_s:02d} - {e_m}:{e_s:02d}")
            if len(time_windows) >= 4:
                if end_s < duration_sec:
                    time_windows[-1] = f"{s_m}:{s_s:02d} - {minutes}:{seconds:02d}"
                break

        system_prompt = (
            "You are a visionary music video director and visual prompt engineer. "
            "Generate a sequential, cinematic visual storyboard matching the song's energy and narrative. "
            "Output MUST be valid JSON only: an array of scene objects with keys 'time', 'prompt', 'camera', 'lighting'."
        )
        user_prompt = (
            f"Song Title: {job.title or 'Studio Production'}\n"
            f"Musical Style: {job.tags or 'Electronic / Ambient'}\n"
            f"Concept / Prompt: {job.prompt}\n"
            f"Visual Aesthetic: {style_meta['desc']}\n"
            f"Duration: {duration_str}\n"
            f"Timeline Windows: {', '.join(time_windows)}\n"
            f"Lyrics:\n{job.lyrics or 'Instrumental track with evocative sonic texture.'}\n\n"
            "Return exactly one scene per timeline window in JSON format."
        )

        try:
            raw_res = await llm_service.generate_text(prompt=f"{system_prompt}\n\n{user_prompt}")
            if raw_res:
                text = raw_res.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                start_idx = text.find("[")
                end_idx = text.rfind("]")
                if start_idx != -1 and end_idx != -1:
                    scenes = json.loads(text[start_idx:end_idx + 1])
                    if isinstance(scenes, list) and len(scenes) > 0:
                        return scenes
        except Exception as e:
            logger.warning(f"LLM storyboard generation fell back to programmatic director: {e}")

        cameras = [
            "Slow cinematic drone tracking forward",
            "360-degree orbital medium close-up",
            "Dynamic Dutch angle low-altitude push",
            "Smooth crane descent with shallow depth of field"
        ]
        scenes = []
        tags_snippet = (job.tags or "atmospheric soundscape").split(",")[0].strip()
        for idx, tw in enumerate(time_windows):
            scenes.append({
                "time": tw,
                "prompt": f"{style_meta['desc']}: Scene {idx + 1} embodying {tags_snippet}. {job.prompt[:80]} with volumetric atmospheric haze and ray-traced reflections.",
                "camera": cameras[idx % len(cameras)],
                "lighting": "Anamorphic cyan and magenta chromatic rim lighting"
            })
        return scenes

    async def render_audio_reactive_video(
        self,
        job: Job,
        visual_style: str = "neon-cyberpunk",
        resolution: str = "720p"
    ) -> str:
        """
        Renders broadcast-quality audio-reactive MP4 video using FFmpeg.
        Combines background cover art, dynamic reactive waveform visualizer, and stereo AAC.
        """
        if not job.audio_path:
            raise ValueError("Job has no audio_path to render.")

        palette = STYLE_PALETTES.get(visual_style, STYLE_PALETTES["neon-cyberpunk"])
        out_filename = f"{job.id}_music_video.mp4"
        out_path = os.path.join(VIDEO_DIR, out_filename)

        # Resolve local audio path
        possible_audio = [
            job.audio_path,
            job.audio_path.lstrip("/"),
            job.audio_path.replace("/audio/", "generated_audio/"),
            os.path.join("generated_audio", os.path.basename(job.audio_path))
        ]
        resolved_audio = None
        for p in possible_audio:
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                resolved_audio = p
                break

        if not resolved_audio:
            raise FileNotFoundError(f"Source audio not found for job: {job.audio_path}")

        # Check for cover art
        resolved_cover = None
        if job.cover_image_path:
            possible_cover = [
                job.cover_image_path,
                job.cover_image_path.lstrip("/"),
                os.path.join("data/covers", os.path.basename(job.cover_image_path)),
                os.path.join("generated_audio", os.path.basename(job.cover_image_path))
            ]
            for c in possible_cover:
                if os.path.isfile(c) and os.path.getsize(c) > 0:
                    resolved_cover = c
                    break

        w, h = (1920, 1080) if resolution == "1080p" else (1280, 720)
        wave_h = int(h * 0.38)

        if resolved_cover:
            filter_complex = (
                f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=8:4[bg];"
                f"[1:a]showwaves=s={w}x{wave_h}:mode=line:colors={palette['colors']}:scale=cbrt[wave];"
                f"[bg][wave]overlay=0:H-h-40[v]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", resolved_cover,
                "-i", resolved_audio,
                "-filter_complex", filter_complex,
                "-map", "[v]", "-map", "1:a",
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path
            ]
        else:
            filter_complex = (
                f"[1:a]showwaves=s={w}x{wave_h}:mode=line:colors={palette['colors']}:scale=cbrt[wave];"
                f"[0:v][wave]overlay=0:(H-h)/2[v]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c={palette['bg']}:s={w}x{h}:r=25",
                "-i", resolved_audio,
                "-filter_complex", filter_complex,
                "-map", "[v]", "-map", "1:a",
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path
            ]

        logger.info(f"Rendering audio-reactive video for {job.id} -> {out_path}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            err_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
            logger.error(f"FFmpeg video render failed: {err_msg}")
            raise RuntimeError(f"FFmpeg rendering error: {err_msg[:200]}")

        return f"/audio/videos/{out_filename}"


video_service = VideoService()
