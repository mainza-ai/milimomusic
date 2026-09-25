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

    async def _poll_fal_queue(
        self,
        client: httpx.AsyncClient,
        endpoint_id: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        out_path: str,
        timeout: float = 300.0
    ) -> bool:
        """
        Submits request to Fal.ai queue and polls status_url until completion or timeout.
        """
        queue_url = f"https://queue.fal.run/{endpoint_id}"
        resp = await client.post(queue_url, json=payload, headers=headers)
        if resp.status_code not in (200, 201):
            logger.error(f"Fal.ai {endpoint_id} queue submission error ({resp.status_code}): {resp.text}")
            return False

        data = resp.json()
        # Direct response or fast-path cache hit
        video_url = data.get("video", {}).get("url")
        if video_url:
            vid_resp = await client.get(video_url)
            if vid_resp.status_code == 200:
                with open(out_path, "wb") as f_out:
                    f_out.write(vid_resp.content)
                return True

        request_id = data.get("request_id")
        if not request_id:
            logger.error(f"Fal.ai response missing request_id: {data}")
            return False

        status_url = data.get("status_url") or f"https://queue.fal.run/{endpoint_id}/requests/{request_id}/status"
        response_url = data.get("response_url") or f"https://queue.fal.run/{endpoint_id}/requests/{request_id}"

        # Poll loop with progressive backoff
        start_time = asyncio.get_event_loop().time()
        poll_interval = 2.0
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            await asyncio.sleep(poll_interval)
            poll_interval = min(5.0, poll_interval * 1.2)

            try:
                poll_resp = await client.get(status_url, headers=headers)
                if poll_resp.status_code != 200:
                    logger.debug(f"Fal.ai status check ({poll_resp.status_code}): {poll_resp.text}")
                    continue

                status_data = poll_resp.json()
                status = status_data.get("status")

                if status == "COMPLETED":
                    res_resp = await client.get(response_url, headers=headers)
                    if res_resp.status_code == 200:
                        res_data = res_resp.json()
                        v_url = res_data.get("video", {}).get("url")
                        if v_url:
                            vid_resp = await client.get(v_url)
                            if vid_resp.status_code == 200:
                                with open(out_path, "wb") as f_out:
                                    f_out.write(vid_resp.content)
                                return True
                    logger.error(f"Fal.ai completed but failed to fetch video output: {res_resp.text}")
                    return False
                elif status in ("FAILED", "ERROR"):
                    logger.error(f"Fal.ai request {request_id} failed: {status_data.get('error') or status_data}")
                    return False

            except Exception as poll_err:
                logger.warning(f"Error during Fal.ai poll ({request_id}): {poll_err}")

        logger.error(f"Fal.ai request {request_id} timed out after {timeout} seconds")
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

                payload: Dict[str, Any] = {
                    "prompt": prompt,
                    "duration": min(5.0, duration),
                }
                if negative_prompt:
                    payload["negative_prompt"] = negative_prompt

                if image_path and os.path.isfile(image_path):
                    with open(image_path, "rb") as f:
                        img_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
                    payload["image_url"] = img_b64
                    endpoint_id = "fal-ai/wan-i2v"
                else:
                    endpoint_id = "fal-ai/wan-t2v"

                async with httpx.AsyncClient(timeout=320.0) as client:
                    return await self._poll_fal_queue(
                        client=client,
                        endpoint_id=endpoint_id,
                        payload=payload,
                        headers=headers,
                        out_path=out_path,
                        timeout=300.0
                    )

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
