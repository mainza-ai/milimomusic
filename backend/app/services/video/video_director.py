"""
Video Director Engine — Song Segmentation, Beat Snapping & Directing Prompt Generation.
"""

import os
import re
import math
import logging
from typing import List, Dict, Optional, Any, Tuple

from app.models import Job
from app.transcription.karaoke import lyric_sync_engine
from app.services.video.types import SceneClip, SceneType, VideoPlan

logger = logging.getLogger(__name__)

STYLE_PALETTES = {
    "neon-cyberpunk": {
        "colors": "0x14b8a6|0x06b6d4",
        "bg": "0x0a0f1d",
        "primary_color": (20, 184, 166),
        "accent_color": (6, 182, 212),
        "desc": "Cyberpunk Neon",
        "atmosphere": "neon-lit cyberpunk city, rain-slicked pavement, atmospheric volumetric fog, high-contrast cyan and magenta lighting",
        "negative": "washed out, daylight, cartoon, blurry, low resolution, watermark"
    },
    "anime-cinematic": {
        "colors": "0xf43f5e|0xf59e0b",
        "bg": "0x111827",
        "primary_color": (244, 63, 94),
        "accent_color": (245, 158, 11),
        "desc": "Anime Cinematic",
        "atmosphere": "high-end anime cinematic film still, Makoto Shinkai aesthetic, golden hour twilight sky, lens flare, expressive emotional atmosphere",
        "negative": "realistic 3d render, live action, flat colors, blurry, artifacting"
    },
    "retro-vhs": {
        "colors": "0xa855f7|0xec4899",
        "bg": "0x0f0b1e",
        "primary_color": (168, 85, 247),
        "accent_color": (236, 72, 153),
        "desc": "80s Retro VHS",
        "atmosphere": "1980s synthwave music video aesthetic, nostalgic film grain, CRT phosphor glow, dreamlike pastel dusk haze",
        "negative": "modern digital clean, high contrast harsh daylight, 4k digital sharpness"
    },
    "minimal-stage": {
        "colors": "0x38bdf8|0x818cf8",
        "bg": "0x090d16",
        "primary_color": (240, 240, 245),
        "accent_color": (56, 189, 248),
        "desc": "Minimalist Noir Stage",
        "atmosphere": "moody Scandinavian cinema, soft diffused daylight, vast negative space, elegant architectural silhouettes, understated melancholy",
        "negative": "cluttered, chaotic, neon, bright garish colors, overexposed"
    },
    "minimal-lyrics": {
        "colors": "0x38bdf8|0x818cf8",
        "bg": "0x090d16",
        "primary_color": (240, 240, 245),
        "accent_color": (56, 189, 248),
        "desc": "Minimal Typography Stage",
        "atmosphere": "moody Scandinavian cinema, soft diffused daylight, vast negative space, elegant architectural silhouettes, understated melancholy",
        "negative": "cluttered, chaotic, neon, bright garish colors, overexposed"
    },
    "film-noir-35mm": {
        "colors": "0xd1d5db|0x9ca3af",
        "bg": "0x030712",
        "primary_color": (220, 220, 230),
        "accent_color": (160, 165, 180),
        "desc": "Classic Film Noir (35mm)",
        "atmosphere": "classic 1940s film noir, black and white 35mm cinematography, dramatic chiaroscuro shadow patterns through Venetian blinds, drifting cigarette smoke, wet asphalt reflections",
        "negative": "color, modern digital clean, oversaturated, neon, flat lighting"
    },
    "golden-hour-folk": {
        "colors": "0xf59e0b|0xd97706",
        "bg": "0x1c1006",
        "primary_color": (245, 158, 11),
        "accent_color": (217, 119, 6),
        "desc": "Golden Hour Folk",
        "atmosphere": "sun-drenched golden hour acoustic cinema, warm floating dust particles, anamorphic amber rim flare, organic 70s film warmth, intimate natural landscape",
        "negative": "cold blue, harsh artificial lighting, neon, sterile studio, modern digital sharpness"
    },
    "hyper-scifi": {
        "colors": "0x06b6d4|0x3b82f6",
        "bg": "0x030712",
        "primary_color": (6, 182, 212),
        "accent_color": (59, 130, 246),
        "desc": "Interstellar Sci-Fi",
        "atmosphere": "monolithic hard sci-fi aesthetics, sterile architectural titanium interiors, deep cosmic nebula backdrop, volumetric anamorphic cobalt blue lighting, Stanley Kubrick precision",
        "negative": "medieval, rustic, fantasy, earthy, low-tech, grainy low-res"
    },
    "gothic-dark": {
        "colors": "0xef4444|0x7c3aed",
        "bg": "0x0f0514",
        "primary_color": (239, 68, 68),
        "accent_color": (124, 58, 237),
        "desc": "Dark Gothic Cathedral",
        "atmosphere": "dark romantic gothic fantasy, candlelit cathedral vaults, obsidian velvet shadows, misty moonlight through stained glass, deep crimson and violet undertones",
        "negative": "cheerful, bright sunny daylight, pastel, cartoon, modern technology"
    },
    "vintage-kodak": {
        "colors": "0xf97316|0xeab308",
        "bg": "0x1c1208",
        "primary_color": (249, 115, 22),
        "accent_color": (234, 179, 8),
        "desc": "70s Kodachrome 35mm",
        "atmosphere": "authentic 1970s Kodachrome color science, warm nostalgic amber and terracotta saturation, subtle analog gate weave, vintage Panavision lenses, sun-kissed Americana",
        "negative": "modern digital video, sterile, cold blue tint, 8k crisp digital sharpness"
    },
    "kpop-holographic": {
        "colors": "0xec4899|0x8b5cf6",
        "bg": "0x180b24",
        "primary_color": (236, 72, 153),
        "accent_color": (139, 92, 246),
        "desc": "K-Pop Prism Gloss",
        "atmosphere": "ultra-high-budget K-pop music video studio, holographic chromatic aberration, dynamic pastel iridescent LED tunnel, pristine commercial gloss, vibrant choreography lighting",
        "negative": "dark muddy shadows, rustic, vintage grain, muted desaturated colors"
    },
    "psychedelic-surreal": {
        "colors": "0x10b981|0xec4899",
        "bg": "0x091410",
        "primary_color": (16, 185, 129),
        "accent_color": (236, 72, 153),
        "desc": "Surrealist Psychedelic",
        "atmosphere": "surrealist 1960s liquid oil light show, morphing dreamscape physics, kaleidoscopic spectral color diffusion, shimmering velvet textures, otherworldly optical distortions",
        "negative": "mundane reality, realistic documentary, sterile corporate, flat lighting"
    },
    "urban-street-grime": {
        "colors": "0xeab308|0x71717a",
        "bg": "0x121008",
        "primary_color": (234, 179, 8),
        "accent_color": (113, 113, 122),
        "desc": "Urban Street Grime",
        "atmosphere": "90s East Coast hip-hop music video, raw brutalist concrete architecture, amber sodium vapor streetlights, low-angle fisheye lens perspective, atmospheric smoke and gritty textures",
        "negative": "clean corporate office, pastel, fantasy, fairy tale, oversaturated cartoon"
    },
    "claymation-stopmo": {
        "colors": "0xf43f5e|0x10b981",
        "bg": "0x1c0d0a",
        "primary_color": (244, 63, 94),
        "accent_color": (16, 185, 129),
        "desc": "Claymation Art",
        "atmosphere": "tactile stop-motion claymation animated aesthetic, visible handmade plasticine fingerprint textures, physical miniature studio lighting, charming stop-motion framerate, rich organic depth",
        "negative": "smooth 3d CGI, realistic live-action human video, glossy digital render"
    },
    "wes-anderson-pastel": {
        "colors": "0xfbbf24|0x34d399",
        "bg": "0x18180c",
        "primary_color": (251, 191, 36),
        "accent_color": (52, 211, 153),
        "desc": "Symmetrical Pastel",
        "atmosphere": "meticulous Wes Anderson storybook aesthetic, perfect central one-point perspective symmetry, whimsical muted pastel palette of mustard yellow and mint green, vintage retro props",
        "negative": "chaotic Dutch angles, shaky handheld cam, gritty dark shadows, high-contrast neon"
    }
}

MODEL_MAX_DURATIONS: Dict[str, float] = {
    "wan_14b": 5.0,
    "wan_1.3b": 5.0,
    "wan2.1": 5.0,
    "ltx_video": 10.0,
    "cogvideox": 10.0,
    "hailuo_h3": 15.0,
    "hunyuan": 15.0,
    "audioreactive": 120.0,
}

CAMERA_MOTIONS = [
    "Slow cinematic tracking crane down toward performer",
    "360-degree orbital medium shot emphasizing musical intensity",
    "Wide establishing atmospheric sweep panning across environment",
    "Dutch angle low push-in with anamorphic rim light flare",
    "Intimate emotive close-up with shallow depth of field and bokeh",
    "Dynamic forward dolly zoom following rhythmic cadence",
    "Floating handheld aesthetic with subtle organic camera sway"
]

LIGHTING_DESIGNS = [
    "Anamorphic cyan rim lighting with deep violet environmental shadows",
    "Volumetric warm amber stage spotlights piercing atmospheric haze",
    "Dynamic rhythmic stroboscopic backlight matching the percussion",
    "Golden hour sunset backlighting with soft lens flare and particle glow",
    "Soft diffused silhouette with high-contrast edge illumination"
]


class VideoDirector:
    """
    Directs music video scene breakdown, beat synchronization, and visual scene prompts.
    """

    @classmethod
    def get_model_max_duration(cls, model_name: Optional[str] = None) -> float:
        if not model_name:
            return 5.0
        m = model_name.lower().strip()
        if "wan" in m:
            return 5.0
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
        return MODEL_MAX_DURATIONS.get(m, 5.0)

    @classmethod
    def detect_bpm(cls, audio_path: Optional[str] = None, default_bpm: float = 120.0) -> float:
        """
        Detect the tempo (BPM) of the audio track using librosa or Muscriptor's beat-this grid.
        """
        if not audio_path or not os.path.isfile(audio_path):
            return default_bpm

        try:
            # Try Muscriptor BeatGrid first
            from muscriptor.muscriptor.utils.beats import detect_beat_grid_for
            grid = detect_beat_grid_for(audio_path, mode="best-effort")
            if grid and grid.bpm and 40.0 <= grid.bpm <= 240.0:
                logger.info(f"Muscriptor detected BPM: {grid.bpm:.1f}")
                return float(grid.bpm)
        except Exception:
            pass

        try:
            # Try librosa
            import librosa
            y, sr = librosa.load(audio_path, sr=22050, duration=45.0)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm_val = float(tempo[0] if hasattr(tempo, "__len__") else tempo)
            if 40.0 <= bpm_val <= 240.0:
                logger.info(f"Librosa detected BPM: {bpm_val:.1f}")
                return round(bpm_val, 1)
        except Exception as ex:
            logger.debug(f"Librosa tempo estimation skipped: {ex}")

        return default_bpm

    def segment_song(
        self,
        job: Job,
        max_clip_duration: Optional[float] = None,
        model_name: Optional[str] = "wan_14b",
        bpm: Optional[float] = None,
        visual_style: str = "neon-cyberpunk",
        vocal_stem_path: Optional[str] = None,
        character_desc: Optional[str] = None,
        custom_style_prompt: Optional[str] = None
    ) -> VideoPlan:
        """
        Segment song into bar-aligned musical scenes classified as Vocal Performance vs B-Roll.
        """
        total_duration = float((job.duration_ms or 180000) / 1000.0)
        resolved_bpm = float(bpm) if bpm and bpm > 0 else self.detect_bpm(job.audio_path, default_bpm=120.0)
        seconds_per_bar = (60.0 / resolved_bpm) * 4.0

        model_max = self.get_model_max_duration(model_name)
        if max_clip_duration is None or max_clip_duration <= 0:
            effective_max = model_max
        else:
            effective_max = max(1.0, min(float(max_clip_duration), model_max))

        # Align target clip duration to integer musical bars
        bars_per_clip = max(1, int(round(effective_max / seconds_per_bar)))
        target_clip_len = bars_per_clip * seconds_per_bar
        if target_clip_len > effective_max and bars_per_clip > 1:
            target_clip_len = (bars_per_clip - 1) * seconds_per_bar
        target_clip_len = max(1.5, min(target_clip_len, effective_max))

        # Retrieve timed lyrics via lyric_sync_engine
        timed_lines = lyric_sync_engine.align_lyrics(
            lyrics=job.lyrics or "",
            duration_sec=total_duration,
            vocal_stem_path=vocal_stem_path
        )

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

        clips: List[SceneClip] = []
        cur_time = 0.0
        clip_idx = 1

        genre_hint = (job.tags or job.prompt or "cinematic modern music").strip()
        char_hint = f"featuring {character_desc.strip()}, " if character_desc and character_desc.strip() else ""

        while cur_time < total_duration:
            clip_end = min(cur_time + target_clip_len, total_duration)
            if (total_duration - clip_end) < 2.0:
                clip_end = total_duration

            # Find active lyrics overlapping this clip window
            overlapping_lyrics = []
            for line in timed_lines:
                l_start = line.get("start", 0.0)
                l_end = line.get("end", 0.0)
                if max(cur_time, l_start) < min(clip_end, l_end):
                    overlapping_lyrics.append(line.get("text", "").strip())

            has_vocals = len(overlapping_lyrics) > 0
            scene_type = SceneType.VOCAL_PERFORMANCE.value if has_vocals else SceneType.CINEMATIC_BROLL.value
            lyric_snippet = " / ".join(overlapping_lyrics) if overlapping_lyrics else ""

            s_m, s_s = int(cur_time // 60), int(cur_time % 60)
            e_m, e_s = int(clip_end // 60), int(clip_end % 60)
            time_label = f"{s_m}:{s_s:02d} - {e_m}:{e_s:02d}"

            camera = CAMERA_MOTIONS[(clip_idx - 1) % len(CAMERA_MOTIONS)]
            lighting = LIGHTING_DESIGNS[(clip_idx - 1) % len(LIGHTING_DESIGNS)]

            if has_vocals:
                prompt = (
                    f"{palette['desc']} music video: Singer {char_hint}performing with passionate expression and singing mouth movement, "
                    f"style of {genre_hint}. Lyrics: \"{lyric_snippet[:70]}\". {camera}, {lighting}. {palette['atmosphere']}."
                )
            else:
                prompt = (
                    f"{palette['desc']} music video B-roll: Cinematic narrative visual scene capturing the musical mood of {genre_hint}. "
                    f"{camera}, {lighting}. {palette['atmosphere']}, dynamic organic movement."
                )

            clips.append(SceneClip(
                clip_index=clip_idx,
                start_time=round(cur_time, 2),
                end_time=round(clip_end, 2),
                duration=round(clip_end - cur_time, 2),
                time_str=time_label,
                is_vocal=has_vocals,
                scene_type=scene_type,
                lyrics=lyric_snippet,
                prompt=prompt,
                negative_prompt=palette["negative"],
                camera=camera,
                lighting=lighting,
            ))

            clip_idx += 1
            cur_time = clip_end

        vocal_count = sum(1 for c in clips if c.is_vocal)
        broll_count = len(clips) - vocal_count

        return VideoPlan(
            job_id=str(job.id),
            total_clips=len(clips),
            vocal_clips_count=vocal_count,
            broll_clips_count=broll_count,
            max_clip_duration=target_clip_len,
            model_max_duration=model_max,
            model_name=model_name or "wan_14b",
            bpm=round(resolved_bpm, 1),
            visual_style=visual_style,
            clips=clips
        )

    def generate_storyboard_scenes(
        self,
        job: Job,
        visual_style: str = "neon-cyberpunk",
        custom_style_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate human-readable scene storyboard cards for the frontend directing notes.
        """
        plan = self.segment_song(
            job=job,
            max_clip_duration=15.0,
            bpm=120.0,
            visual_style=visual_style,
            custom_style_prompt=custom_style_prompt
        )
        return [
            {
                "time": c.time_str,
                "prompt": c.prompt,
                "camera": c.camera,
                "lighting": c.lighting,
                "scene_type": c.scene_type,
                "is_vocal": c.is_vocal,
                "lyrics": c.lyrics
            }
            for c in plan.clips
        ]


video_director = VideoDirector()
