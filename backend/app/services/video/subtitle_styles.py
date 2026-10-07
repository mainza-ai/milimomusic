"""
Production Subtitle & Karaoke Typography Engine for Milimo Music.
Generates Advanced SubStation Alpha (.ass) scripts with frame-accurate word-level
karaoke sweep tags, multi-genre typography presets, and social platform safe zones.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple

from app.transcription.karaoke import LyricSyncEngine

# Default directory for local fonts
ASSETS_FONTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets", "fonts")
)


def get_fonts_dir() -> Optional[str]:
    """Returns local font directory if it exists and contains fonts."""
    if os.path.isdir(ASSETS_FONTS_DIR) and os.listdir(ASSETS_FONTS_DIR):
        return ASSETS_FONTS_DIR
    return None


@dataclass(frozen=True)
class SubtitlePreset:
    id: str
    name: str
    font_name: str
    font_scale: float  # Fraction of video height
    bold: int
    italic: int
    primary_color_ass: str  # Sung highlight color (&HAABBGGRR)
    secondary_color_ass: str  # Unsung base color (&HAABBGGRR)
    outline_color_ass: str  # Border stroke (&HAABBGGRR)
    back_color_ass: str  # Drop shadow/glow (&HAABBGGRR)
    outline_width: float
    shadow_depth: float
    spacing: float
    alignment: int  # ASS numpad alignment (1-9)
    margin_v_ratio: float  # Fraction of video height
    animation_type: str  # 'smooth_sweep', 'stepped', 'fade', 'pop'
    description: str


SUBTITLE_PRESETS: Dict[str, SubtitlePreset] = {
    "neon": SubtitlePreset(
        id="neon",
        name="Cyberpunk Neon Glow",
        font_name="Arial",
        font_scale=0.052,
        bold=1,
        italic=0,
        primary_color_ass="&H00D4B606",  # Electric Cyan highlight when sung
        secondary_color_ass="&H00FFFFFF",  # Pure White unsung
        outline_color_ass="&H00050505",  # Deep Solid Black Border
        back_color_ass="&H80000000",  # Strong Drop Shadow
        outline_width=3.6,
        shadow_depth=2.4,
        spacing=0.8,
        alignment=2,  # Bottom Center
        margin_v_ratio=0.08,
        animation_type="smooth_sweep",
        description="High-contrast electric cyan text with magenta atmospheric neon glow",
    ),
    "spotify": SubtitlePreset(
        id="spotify",
        name="Spotify / Apple Music Studio",
        font_name="Arial",
        font_scale=0.050,
        bold=1,
        italic=0,
        primary_color_ass="&H00FFFFFF",  # 100% Pure White
        secondary_color_ass="&H60FFFFFF",  # Translucent White
        outline_color_ass="&H00050505",  # Deep Solid Black Stroke
        back_color_ass="&H90000000",  # Strong drop shadow
        outline_width=3.2,
        shadow_depth=2.0,
        spacing=0.5,
        alignment=2,  # Bottom Center
        margin_v_ratio=0.09,
        animation_type="smooth_sweep",
        description="Clean, modern streaming lyric style with smooth syllable fill",
    ),
    "kinetic_pop": SubtitlePreset(
        id="kinetic_pop",
        name="Kinetic Syllable Pop",
        font_name="Arial Black",
        font_scale=0.054,
        bold=1,
        italic=0,
        primary_color_ass="&H0000D4FF",  # Electric Canary Gold
        secondary_color_ass="&H00FFFFFF",  # Crisp White
        outline_color_ass="&H00000000",  # Heavy Black Outline
        back_color_ass="&H90000000",
        outline_width=4.0,
        shadow_depth=2.5,
        spacing=1.2,
        alignment=2,  # Bottom Center
        margin_v_ratio=0.085,
        animation_type="smooth_sweep",
        description="High-energy bold kinetic pop font with gold syllable accents",
    ),
    "cinematic": SubtitlePreset(
        id="cinematic",
        name="Cinematic Editorial",
        font_name="Georgia",
        font_scale=0.046,
        bold=1,
        italic=0,
        primary_color_ass="&H0080C2E6",  # Warm Vintage Gold
        secondary_color_ass="&H00F0F0F0",  # Crisp Light Silver
        outline_color_ass="&H00000000",  # Solid Black
        back_color_ass="&HB0000000",
        outline_width=2.8,
        shadow_depth=1.8,
        spacing=2.0,
        alignment=2,  # Bottom Center
        margin_v_ratio=0.075,
        animation_type="fade",
        description="Elegant editorial serif typography with gentle cinematic dissolve",
    ),
    "social_vertical": SubtitlePreset(
        id="social_vertical",
        name="Social Vertical (TikTok / Reels Safe)",
        font_name="Arial Black",
        font_scale=0.054,
        bold=1,
        italic=0,
        primary_color_ass="&H0000E6FF",  # High-Visibility Yellow
        secondary_color_ass="&H00FFFFFF",  # Bright White
        outline_color_ass="&H00000000",  # Heavy Black Stroke
        back_color_ass="&HA0000000",
        outline_width=4.5,
        shadow_depth=2.5,
        spacing=0.5,
        alignment=5,  # Middle Center
        margin_v_ratio=0.18,
        animation_type="smooth_sweep",
        description="Center-viewport placement optimized for 9:16 mobile platforms",
    ),
    "retro_vhs": SubtitlePreset(
        id="retro_vhs",
        name="80s Retro VHS Synthwave",
        font_name="Courier New",
        font_scale=0.044,
        bold=1,
        italic=0,
        primary_color_ass="&H009948EC",  # Hot Magenta
        secondary_color_ass="&H00F755A8",  # Retro Cyan
        outline_color_ass="&H001A051E",  # Dark Violet Stroke
        back_color_ass="&H60000000",
        outline_width=2.5,
        shadow_depth=3.0,
        spacing=1.0,
        alignment=2,
        margin_v_ratio=0.08,
        animation_type="stepped",
        description="Retro monospaced font with analog synthwave chromatic contrast",
    ),
}

# Alias mapping for legacy names
PRESET_ALIASES: Dict[str, str] = {
    "karaoke": "neon",
    "neon-cyberpunk": "neon",
    "bouncing": "kinetic_pop",
    "pop": "kinetic_pop",
    "film": "cinematic",
    "tiktok": "social_vertical",
    "reels": "social_vertical",
    "vertical": "social_vertical",
    "vhs": "retro_vhs",
}


def resolve_preset(preset_key: Optional[str]) -> SubtitlePreset:
    """Resolve a preset key or alias to a canonical SubtitlePreset."""
    if not preset_key:
        return SUBTITLE_PRESETS["neon"]
    key = preset_key.lower().strip()
    if key in SUBTITLE_PRESETS:
        return SUBTITLE_PRESETS[key]
    alias = PRESET_ALIASES.get(key)
    if alias and alias in SUBTITLE_PRESETS:
        return SUBTITLE_PRESETS[alias]
    return SUBTITLE_PRESETS["neon"]


def format_ass_timestamp(sec: float) -> str:
    """Format floating point seconds into ASS timestamp string H:MM:SS.cs."""
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs >= 100:
        s += 1
        cs -= 100
        if s >= 60:
            m += 1
            s -= 60
    h = m // 60
    m = m % 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_karaoke_ass_script(
    timed_lines: List[Dict[str, Any]],
    width: int = 1280,
    height: int = 720,
    visual_style: str = "neon-cyberpunk",
    subtitle_style: str = "neon",
    aspect_ratio: str = "16:9",
    font_family: Optional[str] = None,
    font_size_override: Optional[int] = None,
    palette_colors: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None,
) -> str:
    """
    Generate Advanced SubStation Alpha (.ass) subtitle file content
    with precision word-level karaoke sweep tags, genre presets, and safe zones.

    Args:
        timed_lines: List of TimedLine dicts containing text, start, end, words.
        width: Video canvas width in pixels.
        height: Video canvas height in pixels.
        visual_style: Overall production visual aesthetic (for palette colors).
        subtitle_style: Typography preset (neon, spotify, kinetic_pop, cinematic, social_vertical, retro_vhs).
        aspect_ratio: '16:9', '9:16', '1:1', or '21:9'.
        font_family: Optional font name override.
        font_size_override: Optional explicit font size in points.
        palette_colors: Optional ((r,g,b), (ar,ag,ab)) tuple to dynamically colorize primary/accent.

    Returns:
        Full UTF-8 .ass formatted subtitle string ready for FFmpeg.
    """
    preset = resolve_preset(subtitle_style)

    # 1. Resolve Colors
    if palette_colors:
        (pr, pg, pb), (ar, ag, ab) = palette_colors
        # ASS hex order: &HAABBGGRR
        # PrimaryColour is the active sung highlight; SecondaryColour is the unsung base text
        primary_ass = f"&H00{ab:02X}{ag:02X}{ar:02X}"
        accent_ass = "&H00FFFFFF"  # Always crisp white for unsung lyrics to ensure maximum legibility
    else:
        primary_ass = preset.primary_color_ass
        accent_ass = preset.secondary_color_ass

    # 2. Geometry & Safe Zones
    font_name = font_family or preset.font_name
    font_size = font_size_override or int(max(16, round(height * preset.font_scale)))

    is_vertical = aspect_ratio == "9:16" or (height > width)
    if is_vertical and subtitle_style not in ("social_vertical", "tiktok"):
        # For vertical videos, shift alignment to center (5) or apply 20% vertical margin
        # to guarantee subtitles are never clipped by mobile navigation bars or UI elements
        alignment = 5
        margin_v = int(height * 0.18)
    else:
        alignment = preset.alignment
        margin_v = int(max(20, round(height * preset.margin_v_ratio)))

    margin_lr = int(max(30, round(width * 0.06)))

    ass_lines: List[str] = [
        "[Script Info]",
        "Title: Milimo Music Synchronized Lyric Video",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_name},{font_size},{primary_ass},{accent_ass},{preset.outline_color_ass},"
        f"{preset.back_color_ass},{preset.bold},{preset.italic},0,0,100,100,{preset.spacing},0,1,"
        f"{preset.outline_width},{preset.shadow_depth},{alignment},{margin_lr},{margin_lr},{margin_v},1",
        f"Style: SectionHeader,{font_name},{int(font_size * 0.72)},{accent_ass},{primary_ass},"
        f"&H00090A10,&H80000000,1,1,0,0,100,100,2.0,0,1,2.0,1.5,8,{margin_lr},{margin_lr},"
        f"{int(height * 0.05)},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    # 3. Process Events
    tag_type = "kf" if preset.animation_type == "smooth_sweep" else "k"

    for line in timed_lines:
        raw_text = line.get("text", "").strip()
        if not raw_text:
            continue

        start_sec = max(0.0, float(line.get("start", 0.0)))
        end_sec = max(start_sec + 0.3, float(line.get("end", start_sec + 2.0)))

        start_t = format_ass_timestamp(start_sec)
        end_t = format_ass_timestamp(end_sec)

        is_section = bool(
            line.get("is_section")
            or (raw_text.startswith("[") and raw_text.endswith("]"))
        )

        if is_section:
            # Elegant Chapter Cue Banner at top of screen with smooth fade
            clean_hdr = re.sub(r"[\[\]]", "", raw_text).strip().upper()
            ass_lines.append(
                f"Dialogue: 0,{start_t},{end_t},SectionHeader,,0,0,0,,{{\\fad(250,250)}}— {clean_hdr} —"
            )
            continue

        # Sung lyric line
        words: List[Dict[str, Any]] = line.get("words", [])

        if preset.animation_type == "fade":
            # Cinematic Style: Clean text with subtle dissolve without wipe tags
            ass_lines.append(
                f"Dialogue: 0,{start_t},{end_t},Default,,0,0,0,,{{\\fad(200,200)}}{raw_text}"
            )
            continue

        # If word timestamps are missing, generate proportional syllable breakdowns on the fly
        if not words:
            word_tokens = raw_text.split()
            if word_tokens:
                syllable_weights = [
                    LyricSyncEngine._estimate_syllables(w) for w in word_tokens
                ]
                tot_syl = max(1, sum(syllable_weights))
                line_duration = end_sec - start_sec
                cur_w_start = start_sec
                words = []
                for w_tok, syl in zip(word_tokens, syllable_weights):
                    w_dur = line_duration * (syl / tot_syl)
                    words.append({
                        "word": w_tok,
                        "start": cur_w_start,
                        "end": cur_w_start + max(0.1, w_dur - 0.02),
                    })
                    cur_w_start += w_dur

        # Assemble Word-Level Karaoke String
        karaoke_parts: List[str] = []
        cur_cursor = start_sec

        for idx, w_info in enumerate(words):
            w_text = w_info.get("word", "").strip()
            if not w_text:
                continue

            w_start = float(w_info.get("start", cur_cursor))
            w_end = float(w_info.get("end", w_start + 0.3))

            # Account for vocal lead-in pause before this word
            if w_start > cur_cursor + 0.06:
                pause_cs = int(round((w_start - cur_cursor) * 100))
                if pause_cs > 0:
                    karaoke_parts.append(f"{{\\k{pause_cs}}}")

            # Word active duration in centiseconds
            word_dur_cs = max(4, int(round((w_end - w_start) * 100)))
            karaoke_parts.append(f"{{\\{tag_type}{word_dur_cs}}}{w_text}")

            cur_cursor = w_end
            if idx < len(words) - 1:
                karaoke_parts.append(" ")

        dialogue_text = "".join(karaoke_parts)
        if not dialogue_text:
            dialogue_text = raw_text

        ass_lines.append(f"Dialogue: 0,{start_t},{end_t},Default,,0,0,0,,{dialogue_text}")

    return "\n".join(ass_lines)


_DETECTED_ENCODER: Optional[Tuple[str, List[str]]] = None
_DETECTED_FFMPEG: Optional[str] = None
_HAS_SUBTITLES_FILTER: Optional[bool] = None


def find_ffmpeg_executable(force_refresh: bool = False) -> str:
    """
    Finds optimal ffmpeg executable on system, checking for ffmpeg-full or local builds.
    """
    global _DETECTED_FFMPEG
    if _DETECTED_FFMPEG and not force_refresh and "ffmpeg-full" in _DETECTED_FFMPEG:
        return _DETECTED_FFMPEG

    import shutil
    candidates = [
        "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
        "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
        shutil.which("ffmpeg") or "ffmpeg"
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            _DETECTED_FFMPEG = c
            return _DETECTED_FFMPEG
    _DETECTED_FFMPEG = "ffmpeg"
    return _DETECTED_FFMPEG


def has_subtitles_filter(ffmpeg_bin: Optional[str] = None) -> bool:
    """
    Checks whether the active ffmpeg binary supports the 'subtitles' filter (requires libass).
    """
    global _HAS_SUBTITLES_FILTER
    if _HAS_SUBTITLES_FILTER is not None and ffmpeg_bin is None:
        return _HAS_SUBTITLES_FILTER

    binary = ffmpeg_bin or find_ffmpeg_executable()
    import subprocess
    try:
        res = subprocess.run([binary, "-filters"], capture_output=True, text=True, timeout=2)
        supported = " subtitles " in (res.stdout or "")
        if ffmpeg_bin is None:
            _HAS_SUBTITLES_FILTER = supported
        return supported
    except Exception:
        return False


def detect_hardware_encoder(ffmpeg_bin: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Detect available hardware video encoder (VideoToolbox on macOS, NVENC on CUDA, or libx264 fallback).
    Caches result for ultra-fast invocation.
    """
    global _DETECTED_ENCODER
    if _DETECTED_ENCODER is not None and ffmpeg_bin is None:
        return _DETECTED_ENCODER

    binary = ffmpeg_bin or find_ffmpeg_executable()
    import subprocess
    try:
        res = subprocess.run([binary, "-encoders"], capture_output=True, text=True, timeout=2)
        stdout = res.stdout or ""
        if "h264_videotoolbox" in stdout:
            test_cmd = [
                binary, "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
                "-c:v", "h264_videotoolbox", "-b:v", "2M", "-f", "null", "-"
            ]
            t_res = subprocess.run(test_cmd, capture_output=True, text=True, timeout=2)
            if t_res.returncode == 0:
                detected = ("h264_videotoolbox", ["-b:v", "6M"])
                if ffmpeg_bin is None:
                    _DETECTED_ENCODER = detected
                return detected
        elif "h264_nvenc" in stdout:
            test_cmd = [
                binary, "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
                "-c:v", "h264_nvenc", "-b:v", "2M", "-f", "null", "-"
            ]
            t_res = subprocess.run(test_cmd, capture_output=True, text=True, timeout=2)
            if t_res.returncode == 0:
                detected = ("h264_nvenc", ["-preset", "p4", "-b:v", "6M"])
                if ffmpeg_bin is None:
                    _DETECTED_ENCODER = detected
                return detected
    except Exception:
        pass

    fallback = ("libx264", ["-preset", "veryfast", "-crf", "20"])
    if ffmpeg_bin is None:
        _DETECTED_ENCODER = fallback
    return fallback

