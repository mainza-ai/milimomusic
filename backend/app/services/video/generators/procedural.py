"""
Procedural & Ken Burns Visual Generator (Fallback / Preview mode).
"""

import os
import math
import asyncio
import logging
from typing import Optional, Dict, Any

from app.services.video.generators.base import BaseVideoGenerator

logger = logging.getLogger(__name__)


class ProceduralVideoGenerator(BaseVideoGenerator):
    @property
    def name(self) -> str:
        return "procedural"

    @property
    def is_available(self) -> bool:
        return True

    async def generate_clip(
        self,
        prompt: str,
        duration: float,
        out_path: str,
        width: int = 1280,
        height: int = 720,
        image_path: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        visual_style: str = "neon-cyberpunk",
        **kwargs
    ) -> bool:
        """
        Renders an atmospheric visual clip using high-end dynamic camera motion
        over scene artwork or procedural trigonometric synthesis.
        """
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

        if image_path and os.path.isfile(image_path):
            filter_str = (
                f"scale={int(width * 1.25)}:{int(height * 1.25)},"
                f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={total_d}:s={width}x{height},"
                f"eq=contrast=1.10:saturation=1.20:brightness=0.01"
            )
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", image_path,
                "-f", "lavfi", "-t", str(duration), "-i", "anullsrc=r=44100:cl=stereo",
                "-vf", filter_str,
                "-t", str(duration),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path
            ]
        else:
            filter_str = (
                f"nullsrc=s={width}x{height}:d={duration},"
                f"geq=r='20+35*sin(X/120+T*2.2)+45*cos(Y/140+T*1.8)':"
                f"g='15+30*cos(Y/130+T*2.0)+40*sin(X/150+T*2.4)':"
                f"b='40+60*sin((X+Y)/160+T*2.6)+30*cos(X/110+T*1.9)',"
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
        return proc.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0
