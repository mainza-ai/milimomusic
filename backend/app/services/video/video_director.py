"""
Video Director Engine — AI Cinematographer, Song Segmentation, Beat Snapping & Directing Prompt Generation.

Empowers an LLM-driven AI Visual Director to conceptualize narrative arcs, visual metaphors,
character continuity, and cinematography, fully integrated with musical downbeats and frame lattices.
"""

from __future__ import annotations

import os
import re
import math
import json
import logging
from typing import List, Dict, Optional, Any, Tuple

from app.models import Job
from app.transcription.karaoke import lyric_sync_engine
from app.services.video.types import SceneClip, SceneType, VideoPlan, VideoDirectorTreatment
from app.services.video.audio_analysis import AudioSignalAnalyzer, AudioAnalysisResult
from app.services.video.director_music_timing import DirectorMusicTiming, PlannedMusicClip
from app.services.video.music_performance import MusicPerformanceDirector
from app.services.video.prompt_enhancer import VideoPromptEnhancer

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
    Powered by the LLM Visual Director & Cinematographer with resilient local fallback.
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
        Detect the tempo (BPM) of the audio track using librosa or Muscriptor's beat grid.
        """
        if not audio_path or not os.path.isfile(audio_path):
            return default_bpm

        try:
            from muscriptor.muscriptor.utils.beats import detect_beat_grid_for
            grid = detect_beat_grid_for(audio_path, mode="best-effort")
            if grid and grid.bpm and 40.0 <= grid.bpm <= 240.0:
                logger.info(f"Muscriptor detected BPM: {grid.bpm:.1f}")
                return float(grid.bpm)
        except Exception:
            pass

        try:
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

    def generate_director_treatment(
        self,
        job: Job,
        model_name: Optional[str] = "wan_14b",
        max_clip_duration: Optional[float] = None,
        visual_style: str = "neon-cyberpunk",
        custom_style_prompt: Optional[str] = None,
        pacing_bias: int = 0,
        visible_cast: Optional[List[str]] = None,
        character_desc: Optional[str] = None,
        vocal_stem_path: Optional[str] = None,
        use_llm: bool = True
    ) -> VideoDirectorTreatment:
        """
        Conceptualize and script an entire music video using the AI Visual Director (LLMService).
        Translates lyrical themes, mood, and musical downbeats into a cohesive visual treatment
        with persistent character continuity, metaphorical scene planning, and camera choreography.
        """
        total_duration = float((job.duration_ms or 180000) / 1000.0)
        model_max = self.get_model_max_duration(model_name)
        effective_max = max(1.0, min(float(max_clip_duration or model_max), model_max))

        # 1. Audio Signal Analysis & Timed Lyrics Alignment
        timed_lines = lyric_sync_engine.align_lyrics(
            lyrics=job.lyrics or "",
            duration_sec=total_duration,
            vocal_stem_path=vocal_stem_path
        )

        audio_file = job.audio_path if job.audio_path and os.path.isfile(job.audio_path) else None
        if audio_file:
            analysis = AudioSignalAnalyzer.analyze_audio(
                audio_path=audio_file,
                vocal_path=vocal_stem_path,
                timed_lyrics=timed_lines
            )
        else:
            analysis = AudioSignalAnalyzer._fallback_analysis(audio_file or "unknown.mp3", duration_sec=total_duration)

        # 2. Lattice-Snapped Musical Clip Boundaries
        planned_clips: List[PlannedMusicClip] = DirectorMusicTiming.plan_capped_music_clips(
            analysis=analysis,
            model_name=model_name or "wan2.1",
            pacing_bias=pacing_bias,
            model_max_duration=effective_max
        )

        if custom_style_prompt and custom_style_prompt.strip():
            custom_text = custom_style_prompt.strip()
            style_desc = f"Custom Directed: {custom_text}"
            atmosphere_desc = custom_text
            palette = {
                "colors": "0x14b8a6|0x06b6d4",
                "bg": "0x0a0f1d",
                "primary_color": (20, 184, 166),
                "accent_color": (6, 182, 212),
                "desc": style_desc,
                "atmosphere": atmosphere_desc,
                "negative": "blurry, low resolution, watermark, bad hands, distorted anatomy"
            }
        else:
            palette = dict(STYLE_PALETTES.get(visual_style, STYLE_PALETTES["neon-cyberpunk"]))
            style_desc = palette["desc"]
            atmosphere_desc = palette["atmosphere"]

        # Map timed lyrics to each clip window
        clip_metadata_list: List[Dict[str, Any]] = []
        for c in planned_clips:
            overlapping = []
            for line in timed_lines:
                l_start = line.get("start", 0.0)
                l_end = line.get("end", 0.0)
                if max(c.start_time, l_start) < min(c.end_time, l_end):
                    overlapping.append(line.get("text", "").strip())
            lyric_snippet = " / ".join(overlapping) if overlapping else ""
            clip_metadata_list.append({
                "clip_index": c.clip_index,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "duration": c.target_duration,
                "section": c.section_label,
                "is_vocal": c.is_vocal_active,
                "lyrics": lyric_snippet
            })

        # 3. Call LLM for Visual Directing or Fast Fallback
        should_use_llm = use_llm and os.environ.get("MILIMO_FAST_TEST_MODE") != "1"
        if not should_use_llm:
            return self._generate_intelligent_fallback_treatment(
                job=job,
                analysis=analysis,
                clips_meta=clip_metadata_list,
                style_desc=style_desc,
                atmosphere_desc=atmosphere_desc,
                character_desc=character_desc,
                visible_cast=visible_cast,
                visual_style=visual_style,
                palette=palette
            )

        treatment = self._call_llm_visual_director(
            job=job,
            analysis=analysis,
            clips_meta=clip_metadata_list,
            style_desc=style_desc,
            atmosphere_desc=atmosphere_desc,
            character_desc=character_desc,
            visible_cast=visible_cast,
            visual_style=visual_style,
            palette=palette
        )

        return treatment

    def _call_llm_visual_director(
        self,
        job: Job,
        analysis: AudioAnalysisResult,
        clips_meta: List[Dict[str, Any]],
        style_desc: str,
        atmosphere_desc: str,
        character_desc: Optional[str] = None,
        visible_cast: Optional[List[str]] = None,
        visual_style: str = "neon-cyberpunk",
        palette: Optional[Dict[str, Any]] = None
    ) -> VideoDirectorTreatment:
        """Invoke LLMService with structured visual director prompt and fall back cleanly on error."""
        from app.services.llm_service import LLMService

        if palette is None:
            if visual_style in STYLE_PALETTES:
                palette = dict(STYLE_PALETTES[visual_style])
            else:
                palette = {
                    "colors": "0x14b8a6|0x06b6d4",
                    "bg": "0x0a0f1d",
                    "primary_color": (20, 184, 166),
                    "accent_color": (6, 182, 212),
                    "desc": style_desc,
                    "atmosphere": atmosphere_desc,
                    "negative": "blurry, low resolution, watermark, bad hands, distorted anatomy"
                }
        else:
            palette = dict(palette)
        if atmosphere_desc:
            palette["atmosphere"] = atmosphere_desc
        if style_desc:
            palette["desc"] = style_desc

        prompt_enhancer = VideoPromptEnhancer(max_fidelity_retries=1, auto_continue_on_fail=True)

        system_instruction = (
            "You are an award-winning cinematic Music Video Director and Director of Photography (DP).\n"
            "Your job is to conceptualize, plan, and direct a cohesive, visually stunning music video based on "
            "song metadata, musical structure, mood, and lyrics.\n\n"
            "CRITICAL DIRECTING RULES:\n"
            "1. NEVER literally quote or append lyrics into visual prompts! Diffusion models cannot render words; "
            "translate lyrical metaphors, themes, and emotional subtext into physical visual scenes, lighting, and character actions.\n"
            "2. CHARACTER CONTINUITY: Define a single, precise physical profile for the protagonist (wardrobe, hairstyle, "
            "facial characteristics) and weave these exact keywords into every shot featuring the character.\n"
            "3. RESPECT SONG DYNAMICS:\n"
            "   - Intro/Verses: Intimate, contemplative, slower camera movements (slow push-in, subtle pan).\n"
            "   - Chorus/Drops: Dynamic camera energy (sweeping crane, orbit, low-angle tracking), dramatic contrast lighting, musical energy 4-5.\n"
            "   - Instrumental Solos: Focus on instrument playing technique or explosive metaphorical visuals with mouth_movement: closed.\n"
            "4. TWO-TIER VOCAL BYPASS:\n"
            "   - VOCAL_PERFORMANCE shots: Character lip-syncs with emotive facial delivery.\n"
            "   - All other shots (NARRATIVE, METAPHOR, B-ROLL, SOLOS): Explicitly specify mouth_movement: closed.\n"
            "5. OUTPUT FORMAT: Respond ONLY with a valid JSON object matching the requested schema."
        )

        clips_overview = []
        for c in clips_meta:
            clips_overview.append(
                f"Scene #{c['clip_index']}: [{c['start_time']}s - {c['end_time']}s] "
                f"Section: {c['section']}, Vocal Active: {c['is_vocal']}, "
                f"Lyrics snippet: \"{c['lyrics'][:60]}\""
            )

        user_content = (
            f"DIRECT THIS MUSIC VIDEO:\n"
            f"Song Title: {job.title or 'Untitled Track'}\n"
            f"Genre/Style Tags: {job.tags or job.prompt or 'Modern Production'}\n"
            f"Tempo: {analysis.tempo_bpm} BPM | Duration: {analysis.duration_sec}s\n"
            f"Visual Aesthetic: {style_desc}\n"
            f"Atmosphere: {atmosphere_desc}\n"
            f"Character Seed Note: {character_desc or 'Default lead performer'}\n"
            f"Visible Cast: {', '.join(visible_cast) if visible_cast else 'Lead Performer'}\n\n"
            f"Full Lyrics:\n{job.lyrics or 'Instrumental track'}\n\n"
            f"SCENE BREAKDOWN WINDOWS ({len(clips_meta)} scenes):\n"
            + "\n".join(clips_overview) + "\n\n"
            "Provide the Director's Treatment in the following JSON schema:\n"
            "{\n"
            '  "concept_title": "Short creative title",\n'
            '  "logline": "1-2 sentence dramatic visual summary",\n'
            '  "visual_metaphor": "Central visual metaphor embodying the song",\n'
            '  "color_palette_arc": "Progression of color grade across song",\n'
            '  "character_profile": "Detailed immutable physical description of the lead protagonist",\n'
            '  "scenes": [\n'
            "    {\n"
            '      "clip_index": 1,\n'
            '      "scene_type": "VOCAL_PERFORMANCE" | "NARRATIVE_STORY" | "METAPHORICAL_VISUAL" | "INSTRUMENTAL_FOCUS" | "ENVIRONMENTAL_BROLL",\n'
            '      "musical_energy": 1-5,\n'
            '      "visual_action": "Concrete physical action in scene",\n'
            '      "diffusion_prompt": "Highly detailed video diffusion prompt without lyric words",\n'
            '      "camera_motion": "Cinematic camera movement and focal lens",\n'
            '      "lighting_and_atmosphere": "Lighting design and environmental mood",\n'
            '      "directors_note": "Creative rationale for this shot"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        full_prompt = f"{system_instruction}\n\n{user_content}"

        try:
            response_text, provider, model = LLMService.generate_text_via_active(
                full_prompt,
                options={"temperature": 0.7}
            )
            if response_text and response_text.strip():
                parsed = self._extract_json_treatment(response_text)
                if parsed and parsed.get("scenes") and len(parsed["scenes"]) > 0:
                    logger.info(f"AI Visual Director treatment successfully generated via {provider}/{model} ({len(parsed['scenes'])} scenes).")
                    scenes = self._normalize_director_scenes(
                        parsed_scenes=parsed["scenes"],
                        clips_meta=clips_meta,
                        character_profile=parsed.get("character_profile", character_desc or "Lead performer"),
                        visible_cast=visible_cast,
                        palette=palette,
                        prompt_enhancer=prompt_enhancer
                    )
                    return VideoDirectorTreatment(
                        job_id=str(job.id),
                        concept_title=parsed.get("concept_title", f"{job.title or 'Track'} Visuals"),
                        logline=parsed.get("logline", "Cinematic music video narrative"),
                        visual_metaphor=parsed.get("visual_metaphor", "Expressive visual journey"),
                        color_palette_arc=parsed.get("color_palette_arc", atmosphere_desc),
                        character_profile=parsed.get("character_profile", character_desc or "Lead performer"),
                        scenes=scenes,
                        llm_used=True,
                        provider=provider,
                        model=model
                    )
        except Exception as e:
            logger.warning(f"AI Visual Director LLM invocation failed ({e}), using intelligent fallback.")

        # Fallback to intelligent rule-based directing
        return self._generate_intelligent_fallback_treatment(
            job=job,
            analysis=analysis,
            clips_meta=clips_meta,
            style_desc=style_desc,
            atmosphere_desc=atmosphere_desc,
            character_desc=character_desc,
            visible_cast=visible_cast,
            visual_style=visual_style,
            palette=palette
        )

    def _extract_json_treatment(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON object from LLM response text, handling fences and formatting."""
        # 1. Try direct parse
        try:
            return json.loads(text.strip())
        except Exception:
            pass

        # 2. Look for ```json ... ``` code fence
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except Exception:
                pass

        # 3. Look for outermost { ... }
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0).strip())
            except Exception:
                pass

        return None

    def _normalize_director_scenes(
        self,
        parsed_scenes: List[Dict[str, Any]],
        clips_meta: List[Dict[str, Any]],
        character_profile: str,
        visible_cast: Optional[List[str]],
        palette: Dict[str, Any],
        prompt_enhancer: VideoPromptEnhancer
    ) -> List[Dict[str, Any]]:
        """Align parsed scenes with physical clip intervals and apply performance & bypass rules."""
        results: List[Dict[str, Any]] = []
        parsed_by_idx = {s.get("clip_index"): s for s in parsed_scenes if s.get("clip_index")}

        for meta in clips_meta:
            idx = meta["clip_index"]
            raw_s = parsed_by_idx.get(idx) or (parsed_scenes[idx - 1] if idx - 1 < len(parsed_scenes) else {})

            is_vocal = bool(meta["is_vocal"])
            default_type = SceneType.VOCAL_PERFORMANCE.value if is_vocal else SceneType.CINEMATIC_BROLL.value
            scene_type = raw_s.get("scene_type", default_type)

            visual_action = raw_s.get("visual_action") or (
                f"Performer singing with emotional resonance" if is_vocal else "Atmospheric cinematic scenery"
            )
            camera = raw_s.get("camera_motion") or CAMERA_MOTIONS[(idx - 1) % len(CAMERA_MOTIONS)]
            lighting = raw_s.get("lighting_and_atmosphere") or LIGHTING_DESIGNS[(idx - 1) % len(LIGHTING_DESIGNS)]
            directors_note = raw_s.get("directors_note") or f"Section: {meta['section']}"

            # Format performer instruction
            shot_intent = "performance" if scene_type == SceneType.VOCAL_PERFORMANCE.value else "narrative"
            performer_rule = MusicPerformanceDirector.format_performer_prompt_instructions(
                visible_cast=visible_cast,
                is_vocal_active=is_vocal,
                section_label=meta["section"],
                is_instrumental_solo=(scene_type == SceneType.INSTRUMENTAL_FOCUS.value),
                shot_intent=shot_intent
            )

            # Assemble clean diffusion prompt
            base_prompt = raw_s.get("diffusion_prompt", "")
            if not base_prompt or len(base_prompt.strip()) < 15:
                base_prompt = f"{visual_action}. {camera}. {lighting}. {palette['atmosphere']}."

            cleaned_prompt = prompt_enhancer.strip_accidental_dialogue(base_prompt)
            # Prepend character profile for shots involving performance or narrative
            if character_profile and scene_type in (SceneType.VOCAL_PERFORMANCE.value, SceneType.NARRATIVE_STORY.value):
                if character_profile[:30].lower() not in cleaned_prompt.lower():
                    cleaned_prompt = f"Featuring {character_profile}. {cleaned_prompt}"

            s_m, s_s = int(meta["start_time"] // 60), int(meta["start_time"] % 60)
            e_m, e_s = int(meta["end_time"] // 60), int(meta["end_time"] % 60)
            time_label = f"{s_m}:{s_s:02d} - {e_m}:{e_s:02d}"

            results.append({
                "clip_index": idx,
                "start_time": meta["start_time"],
                "end_time": meta["end_time"],
                "duration": meta["duration"],
                "time_str": time_label,
                "is_vocal": is_vocal,
                "scene_type": scene_type,
                "section_label": meta["section"],
                "musical_energy": int(raw_s.get("musical_energy", 3)),
                "visual_action": visual_action,
                "prompt": cleaned_prompt,
                "negative_prompt": palette.get("negative", ""),
                "camera": camera,
                "lighting": lighting,
                "performer_rule": performer_rule,
                "directors_note": directors_note,
                "lyrics": meta["lyrics"]
            })

        return results

    def _generate_intelligent_fallback_treatment(
        self,
        job: Job,
        analysis: AudioAnalysisResult,
        clips_meta: List[Dict[str, Any]],
        style_desc: str,
        atmosphere_desc: str,
        character_desc: Optional[str] = None,
        visible_cast: Optional[List[str]] = None,
        visual_style: str = "neon-cyberpunk",
        palette: Optional[Dict[str, Any]] = None
    ) -> VideoDirectorTreatment:
        """
        Intelligent deterministic fallback when LLM is offline or unreachable.
        Generates structured, section-aware scenes with dynamic camera and lighting,
        never relying on raw lyric string concatenation.
        """
        if palette is None:
            if visual_style in STYLE_PALETTES:
                palette = dict(STYLE_PALETTES[visual_style])
            else:
                palette = {
                    "colors": "0x14b8a6|0x06b6d4",
                    "bg": "0x0a0f1d",
                    "primary_color": (20, 184, 166),
                    "accent_color": (6, 182, 212),
                    "desc": style_desc,
                    "atmosphere": atmosphere_desc,
                    "negative": "blurry, low resolution, watermark, bad hands, distorted anatomy"
                }
        else:
            palette = dict(palette)

        if atmosphere_desc:
            palette["atmosphere"] = atmosphere_desc
        if style_desc:
            palette["desc"] = style_desc

        concept_title = f"{job.title or 'Track'} Cinematic Treatment"
        char_seed = character_desc or "Lead vocalist in contemporary cinematic attire"

        scenes: List[Dict[str, Any]] = []
        for meta in clips_meta:
            idx = meta["clip_index"]
            sec = meta["section"].lower()
            is_vocal = meta["is_vocal"]

            if "intro" in sec:
                scene_type = SceneType.ENVIRONMENTAL_BROLL.value
                energy = 2
                visual_action = f"Atmospheric establishing shot of {palette['desc']} environment with rising fog and ambient illumination"
                camera = "Slow wide establishing panoramic sweep"
                lighting = "Soft environmental silhouette with gentle rim glow"
                note = "Establishing world and sonic atmosphere"
            elif "chorus" in sec or "hook" in sec:
                scene_type = SceneType.VOCAL_PERFORMANCE.value if is_vocal else SceneType.METAPHORICAL_VISUAL.value
                energy = 5
                visual_action = (
                    f"Lead performer {char_seed} delivering explosive singing performance surrounded by dynamic particle pulses"
                    if is_vocal else
                    f"Surreal metaphorical light cascade refracting through architectural glass and water droplets"
                )
                camera = "360-degree orbital medium shot emphasizing peak musical intensity"
                lighting = "Dynamic rhythmic stroboscopic backlight matching percussion"
                note = "Climax chorale energy and peak emotional resonance"
            elif "bridge" in sec or "solo" in sec:
                scene_type = SceneType.INSTRUMENTAL_FOCUS.value
                energy = 4
                visual_action = "Intimate macro focus on instrument playing technique and vibrant acoustic reflections"
                camera = "Dutch angle low push-in with anamorphic rim flare"
                lighting = "Anamorphic cyan rim lighting with deep dramatic shadows"
                note = "Instrumental solo focus with closed mouth performance"
            elif "outro" in sec:
                scene_type = SceneType.ENVIRONMENTAL_BROLL.value
                energy = 1
                visual_action = "Slow receding vista as scene lights gently dissolve into dark reflective water"
                camera = "Slow cinematic tracking crane pull-back"
                lighting = "Golden hour twilight silhouette fading to dark violet"
                note = "Denouement and lingering emotional resolution"
            else:
                # Verse
                scene_type = SceneType.VOCAL_PERFORMANCE.value if is_vocal else SceneType.NARRATIVE_STORY.value
                energy = 3
                visual_action = (
                    f"Intimate portrait of {char_seed} singing with subtle, poignant expression"
                    if is_vocal else
                    "Cinematic character walking through architectural spaces reflecting on past events"
                )
                camera = "Intimate emotive close-up with shallow depth of field and bokeh"
                lighting = "Volumetric warm amber stage spotlights piercing atmospheric haze"
                note = "Lyrical exposition and narrative progression"

            if visual_style == "custom" and atmosphere_desc:
                lighting = f"{lighting} ({atmosphere_desc})"

            prompt = (
                f"{palette['desc']} cinema: {visual_action}. "
                f"{camera}, {lighting}. {palette['atmosphere']}."
            )

            s_m, s_s = int(meta["start_time"] // 60), int(meta["start_time"] % 60)
            e_m, e_s = int(meta["end_time"] // 60), int(meta["end_time"] % 60)
            time_label = f"{s_m}:{s_s:02d} - {e_m}:{e_s:02d}"

            scenes.append({
                "clip_index": idx,
                "start_time": meta["start_time"],
                "end_time": meta["end_time"],
                "duration": meta["duration"],
                "time_str": time_label,
                "is_vocal": is_vocal,
                "scene_type": scene_type,
                "section_label": meta["section"],
                "musical_energy": energy,
                "visual_action": visual_action,
                "prompt": prompt,
                "negative_prompt": palette.get("negative", ""),
                "camera": camera,
                "lighting": lighting,
                "directors_note": note,
                "lyrics": meta["lyrics"]
            })

        return VideoDirectorTreatment(
            job_id=str(job.id),
            concept_title=concept_title,
            logline=f"Section-matched visual narrative directed in the aesthetic of {palette['desc']}.",
            visual_metaphor=f"Dynamic light pulses embodying the musical journey of {job.title or 'the track'}.",
            color_palette_arc=palette["atmosphere"],
            character_profile=char_seed,
            scenes=scenes,
            llm_used=False,
            provider="deterministic_fallback",
            model="director_v2_engine"
        )

    def reimagine_scene(
        self,
        job: Job,
        clip_index: int,
        current_scene: Dict[str, Any],
        user_instruction: Optional[str] = None,
        visual_style: str = "neon-cyberpunk",
        character_desc: Optional[str] = None,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Request the AI Visual Director to re-conceive a single scene with alternate cinematography or user direction.
        """
        from app.services.llm_service import LLMService

        palette = STYLE_PALETTES.get(visual_style, STYLE_PALETTES["neon-cyberpunk"])
        should_use_llm = use_llm and os.environ.get("MILIMO_FAST_TEST_MODE") != "1"

        if should_use_llm:
            prompt = (
                f"You are a music video director. Re-imagine Scene #{clip_index} of the music video for '{job.title or 'Track'}'.\n"
                f"Genre: {job.tags or 'Modern'} | Visual Style: {palette['desc']}\n"
                f"Current Scene Type: {current_scene.get('scene_type')}\n"
                f"Current Visual: {current_scene.get('visual_action') or current_scene.get('prompt')}\n"
                f"Current Section: {current_scene.get('section_label', 'Verse')}\n"
                f"User Creative Direction: {user_instruction or 'Provide a fresh, innovative cinematic take'}\n"
                f"Character Seed: {character_desc or 'Lead performer'}\n\n"
                "Return a JSON object with: scene_type, musical_energy (1-5), visual_action, diffusion_prompt, camera_motion, lighting_and_atmosphere, directors_note."
            )

            try:
                resp, _, _ = LLMService.generate_text_via_active(prompt, options={"temperature": 0.8})
                if resp:
                    parsed = self._extract_json_treatment(resp)
                    if parsed:
                        return {
                            "clip_index": clip_index,
                            "scene_type": parsed.get("scene_type", current_scene.get("scene_type")),
                            "musical_energy": int(parsed.get("musical_energy", current_scene.get("musical_energy", 3))),
                            "visual_action": parsed.get("visual_action", current_scene.get("visual_action")),
                            "prompt": parsed.get("diffusion_prompt") or f"{parsed.get('visual_action')}. {parsed.get('camera_motion')}. {parsed.get('lighting_and_atmosphere')}.",
                            "camera": parsed.get("camera_motion", current_scene.get("camera")),
                            "lighting": parsed.get("lighting_and_atmosphere", current_scene.get("lighting")),
                            "directors_note": parsed.get("directors_note", "Re-imagined by AI Director")
                        }
            except Exception as e:
                logger.warning(f"Reimagine scene via LLM failed: {e}")

        # Fallback variation
        alt_camera = CAMERA_MOTIONS[(clip_index * 2) % len(CAMERA_MOTIONS)]
        alt_lighting = LIGHTING_DESIGNS[(clip_index * 2) % len(LIGHTING_DESIGNS)]
        new_prompt = f"{palette['desc']} cinema: Alternate dynamic take on scene {clip_index}. {alt_camera}, {alt_lighting}. {palette['atmosphere']}."
        return {
            "clip_index": clip_index,
            "scene_type": current_scene.get("scene_type", SceneType.CINEMATIC_BROLL.value),
            "musical_energy": current_scene.get("musical_energy", 3),
            "visual_action": f"Alternate take: {user_instruction or 'Enhanced dynamic motion'}",
            "prompt": new_prompt,
            "camera": alt_camera,
            "lighting": alt_lighting,
            "directors_note": "Re-imagined with alternate camera perspective"
        }

    def segment_song(
        self,
        job: Job,
        max_clip_duration: Optional[float] = None,
        model_name: Optional[str] = "wan_14b",
        bpm: Optional[float] = None,
        visual_style: str = "neon-cyberpunk",
        vocal_stem_path: Optional[str] = None,
        character_desc: Optional[str] = None,
        custom_style_prompt: Optional[str] = None,
        pacing_bias: int = 0,
        visible_cast: Optional[List[str]] = None,
        user_scenes: Optional[List[Dict[str, Any]]] = None
    ) -> VideoPlan:
        """
        Produce a production VideoPlan.
        If user_scenes are provided (e.g. customized or reviewed in UI), strictly honors them!
        Otherwise, runs generate_director_treatment to produce an intelligent AI script.
        """
        model_max = self.get_model_max_duration(model_name)
        effective_max = max(1.0, min(float(max_clip_duration or model_max), model_max))

        # If user passed custom pre-planned scenes, honor them directly
        if user_scenes and len(user_scenes) > 0:
            clips: List[SceneClip] = []
            for s in user_scenes:
                idx = int(s.get("clip_index") or len(clips) + 1)
                st = float(s.get("start_time") or 0.0)
                et = float(s.get("end_time") or (st + effective_max))
                dur = float(s.get("duration") or (et - st))
                is_voc = bool(s.get("is_vocal"))
                clips.append(SceneClip(
                    clip_index=idx,
                    start_time=round(st, 2),
                    end_time=round(et, 2),
                    duration=round(dur, 2),
                    time_str=s.get("time_str") or f"{int(st)}s - {int(et)}s",
                    is_vocal=is_voc,
                    scene_type=s.get("scene_type") or (SceneType.VOCAL_PERFORMANCE.value if is_voc else SceneType.CINEMATIC_BROLL.value),
                    lyrics=s.get("lyrics", ""),
                    prompt=s.get("prompt", ""),
                    negative_prompt=s.get("negative_prompt", ""),
                    camera=s.get("camera", "Cinematic camera movement"),
                    lighting=s.get("lighting", "Atmospheric studio lighting"),
                    section_label=s.get("section_label", "Verse"),
                    musical_energy=int(s.get("musical_energy", 3)),
                    visual_action=s.get("visual_action", ""),
                    directors_note=s.get("directors_note", "")
                ))
            vocal_count = sum(1 for c in clips if c.is_vocal)
            return VideoPlan(
                job_id=str(job.id),
                total_clips=len(clips),
                vocal_clips_count=vocal_count,
                broll_clips_count=len(clips) - vocal_count,
                max_clip_duration=effective_max,
                model_max_duration=model_max,
                model_name=model_name or "wan_14b",
                bpm=bpm or 120.0,
                visual_style=visual_style,
                clips=clips
            )

        # Generate fresh AI director treatment
        treatment = self.generate_director_treatment(
            job=job,
            model_name=model_name,
            max_clip_duration=max_clip_duration,
            visual_style=visual_style,
            custom_style_prompt=custom_style_prompt,
            pacing_bias=pacing_bias,
            visible_cast=visible_cast,
            character_desc=character_desc,
            vocal_stem_path=vocal_stem_path
        )

        clips = []
        for s in treatment.scenes:
            clips.append(SceneClip(
                clip_index=s["clip_index"],
                start_time=s["start_time"],
                end_time=s["end_time"],
                duration=s["duration"],
                time_str=s["time_str"],
                is_vocal=s["is_vocal"],
                scene_type=s["scene_type"],
                lyrics=s.get("lyrics", ""),
                prompt=s["prompt"],
                negative_prompt=s.get("negative_prompt", ""),
                camera=s["camera"],
                lighting=s["lighting"],
                section_label=s.get("section_label", "Verse"),
                musical_energy=s.get("musical_energy", 3),
                visual_action=s.get("visual_action", ""),
                directors_note=s.get("directors_note", "")
            ))

        vocal_count = sum(1 for c in clips if c.is_vocal)
        resolved_bpm = float(bpm) if bpm and bpm > 0 else self.detect_bpm(job.audio_path, default_bpm=120.0)

        return VideoPlan(
            job_id=str(job.id),
            total_clips=len(clips),
            vocal_clips_count=vocal_count,
            broll_clips_count=len(clips) - vocal_count,
            max_clip_duration=effective_max,
            model_max_duration=model_max,
            model_name=model_name or "wan_14b",
            bpm=round(resolved_bpm, 1),
            visual_style=visual_style,
            concept_title=treatment.concept_title,
            logline=treatment.logline,
            visual_metaphor=treatment.visual_metaphor,
            character_profile=treatment.character_profile,
            clips=clips
        )

    def generate_storyboard_scenes(
        self,
        job: Job,
        visual_style: str = "neon-cyberpunk",
        custom_style_prompt: Optional[str] = None,
        model_name: Optional[str] = "wan_14b"
    ) -> List[Dict[str, Any]]:
        """
        Generate rich scene storyboard cards for the frontend directing notes.
        """
        treatment = self.generate_director_treatment(
            job=job,
            model_name=model_name,
            visual_style=visual_style,
            custom_style_prompt=custom_style_prompt
        )
        return [
            {
                "clip_index": s["clip_index"],
                "time": s["time_str"],
                "prompt": s["prompt"],
                "camera": s["camera"],
                "lighting": s["lighting"],
                "scene_type": s["scene_type"],
                "section_label": s.get("section_label", "Verse"),
                "musical_energy": s.get("musical_energy", 3),
                "visual_action": s.get("visual_action", ""),
                "directors_note": s.get("directors_note", ""),
                "is_vocal": s["is_vocal"],
                "lyrics": s.get("lyrics", "")
            }
            for s in treatment.scenes
        ]


video_director = VideoDirector()
