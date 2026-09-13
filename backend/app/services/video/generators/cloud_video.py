"""
Cloud Video Generator Provider — Fal.ai, Replicate, and MiniMax API.
"""

import os
import base64
import asyncio
import logging
from typing import Optional, Dict, Any

import httpx

from app.services.video.generators.base import BaseVideoGenerator

logger = logging.getLogger(__name__)


class CloudVideoGenerator(BaseVideoGenerator):
    def __init__(self, service: str = "fal", model: str = "wan-t2v"):
        self.service = service.lower()
        self.model = model

    @property
    def name(self) -> str:
        return f"cloud_{self.service}_{self.model}"

    @property
    def is_available(self) -> bool:
        if self.service == "fal":
            return bool(os.environ.get("FAL_KEY"))
        elif self.service == "replicate":
            return bool(os.environ.get("REPLICATE_API_TOKEN"))
        elif self.service == "minimax":
            return bool(os.environ.get("MINIMAX_API_KEY"))
        return False

    async def generate_clip(
        self,
        prompt: str,
        duration: float,
        out_path: str,
        width: int = 1280,
        height: int = 720,
        image_path: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Dispatches clip generation to Cloud GPU endpoints.
        """
        if not self.is_available:
            logger.warning(f"Cloud Video API key not configured for {self.service}.")
            return False

        try:
            if self.service == "fal":
                fal_key = os.environ.get("FAL_KEY")
                headers = {"Authorization": f"Key {fal_key}", "Content-Type": "application/json"}

                if image_path and os.path.isfile(image_path):
                    with open(image_path, "rb") as f:
                        img_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
                    endpoint = "https://queue.fal.run/fal-ai/wan-i2v"
                    payload = {
                        "prompt": prompt,
                        "image_url": img_b64,
                        "duration": min(5.0, duration)
                    }
                else:
                    endpoint = "https://queue.fal.run/fal-ai/wan-t2v"
                    payload = {
                        "prompt": prompt,
                        "duration": min(5.0, duration)
                    }

                async with httpx.AsyncClient(timeout=240.0) as client:
                    resp = await client.post(endpoint, json=payload, headers=headers)
                    if resp.status_code not in (200, 201):
                        logger.error(f"Fal.ai video error: {resp.text}")
                        return False

                    data = resp.json()
                    video_url = data.get("video", {}).get("url")
                    if video_url:
                        vid_resp = await client.get(video_url)
                        if vid_resp.status_code == 200:
                            with open(out_path, "wb") as f_out:
                                f_out.write(vid_resp.content)
                            return True

            elif self.service == "replicate":
                token = os.environ.get("REPLICATE_API_TOKEN")
                headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
                payload = {
                    "version": "wan-video/wan-2.1-14b",
                    "input": {"prompt": prompt}
                }
                async with httpx.AsyncClient(timeout=300.0) as client:
                    resp = await client.post("https://api.replicate.com/v1/predictions", json=payload, headers=headers)
                    if resp.status_code in (200, 201):
                        pred = resp.json()
                        pred_id = pred["id"]
                        # Poll for completion
                        for _ in range(60):
                            await asyncio.sleep(4)
                            poll_resp = await client.get(f"https://api.replicate.com/v1/predictions/{pred_id}", headers=headers)
                            poll_data = poll_resp.json()
                            if poll_data.get("status") == "succeeded":
                                out_url = poll_data.get("output")
                                if isinstance(out_url, list): out_url = out_url[0]
                                if out_url:
                                    vid_resp = await client.get(out_url)
                                    with open(out_path, "wb") as f_out:
                                        f_out.write(vid_resp.content)
                                    return True
                            elif poll_data.get("status") == "failed":
                                break

            return False

        except Exception as e:
            logger.error(f"CloudVideoGenerator error: {e}", exc_info=True)
            return False
