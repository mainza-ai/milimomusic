"""
Cloud Lip-Sync Provider — Fal.ai & Replicate API integration.
"""

import os
import base64
import uuid
import asyncio
import logging
from typing import Optional, Dict, Any

import httpx

from app.core.paths import get_data_dir
from app.services.video.lip_sync.base import BaseLipSyncProvider

logger = logging.getLogger(__name__)
TEMP_DIR = str(get_data_dir() / "video_cache")



async def _fal_upload(client, path: str, content_type: str):
    """Upload a local file to fal storage and return its hosted URL.

    Returns None when the file is missing or the upload endpoint is unavailable —
    callers must then skip the cloud path rather than sending a data URI.
    """
    import os

    if not path or not os.path.isfile(path):
        return None
    key = os.environ.get("FAL_KEY")
    if not key:
        return None
    try:
        with open(path, "rb") as fh:
            files = {"file": (os.path.basename(path), fh.read(), content_type)}
        async with httpx.AsyncClient(timeout=120.0) as uploader:
            resp = await uploader.post(
                "https://rest.alpha.fal.ai/storage/upload",
                headers={"Authorization": f"Key {key}"},
                files=files,
            )
        if resp.status_code not in (200, 201):
            logger.warning("fal upload failed (%s): %s", resp.status_code, resp.text[:200])
            return None
        return (resp.json() or {}).get("url")
    except Exception as exc:  # pragma: no cover - network path
        logger.warning("fal upload error: %s", exc)
        return None


class CloudLipSyncProvider(BaseLipSyncProvider):
    def __init__(self, service: str = "fal"):
        self.service = service.lower()

    @property
    def name(self) -> str:
        return f"cloud_{self.service}_lipsync"

    @property
    def is_available(self) -> bool:
        if self.service == "fal":
            return bool(os.environ.get("FAL_KEY"))
        # D4: Replicate is not supported (its API needs hosted URLs and a version
        # hash; the previous branch sent data URIs and could never succeed).
        return False

    async def render_lip_sync(self, *args, **kwargs) -> bool:
        ok = await self._render_impl(*args, **kwargs)
        self._set_outcome(
            f"cloud_{self.service}",
            False,
            None if ok else "Cloud lip-sync produced no video.",
        )
        return ok

    async def _render_impl(
        self,
        face_image_path: str,
        vocal_audio_path: str,
        start_time: float,
        duration: float,
        out_path: str,
        width: int = 1280,
        height: int = 720,
        **kwargs
    ) -> bool:
        """
        Submits singing performance generation to Cloud GPU endpoints.
        """
        if not self.is_available:
            logger.warning(f"Cloud Lip-Sync API key not configured for {self.service}.")
            return False

        slice_audio = os.path.join(TEMP_DIR, f"cloud_vocal_{uuid.uuid4().hex[:8]}.wav")
        cmd_cut = [
            "ffmpeg", "-y", "-ss", str(start_time), "-t", str(duration),
            "-i", vocal_audio_path, "-ar", "44100", "-ac", "1", slice_audio
        ]
        proc = await asyncio.create_subprocess_exec(*cmd_cut, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()

        try:
            # Base64 blobs are kept only for providers that accept inline data;
            # fal does not (see _fal_upload).
            with open(face_image_path, "rb") as f:
                img_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
            with open(slice_audio, "rb") as f:
                audio_b64 = "data:audio/wav;base64," + base64.b64encode(f.read()).decode("utf-8")

            if self.service == "fal":
                fal_key = os.environ.get("FAL_KEY")
                headers = {"Authorization": f"Key {fal_key}", "Content-Type": "application/json"}
                # fal requires *hosted* URLs. A base64 data URI 4xxs, so upload
                # first and fail honestly when hosting is unavailable.
                image_url = await _fal_upload(None, face_image_path, "image/jpeg")
                audio_url = await _fal_upload(None, slice_audio, "audio/wav")
                if not image_url or not audio_url:
                    logger.error(
                        "fal lip-sync skipped: could not host the inputs "
                        "(data URIs are rejected by the API)."
                    )
                    return False
                payload = {
                    "source_image_url": image_url,
                    "driving_audio_url": audio_url,
                }
                async with httpx.AsyncClient(timeout=300.0) as client:
                    resp = await client.post("https://queue.fal.run/fal-ai/live-portrait", json=payload, headers=headers)
                    if resp.status_code not in (200, 201):
                        logger.error(f"Fal.ai lip-sync queue error ({resp.status_code}): {resp.text}")
                        return False

                    data = resp.json()
                    video_url = data.get("video", {}).get("url")
                    if video_url:
                        vid_resp = await client.get(video_url)
                        if vid_resp.status_code == 200:
                            with open(out_path, "wb") as f_out:
                                f_out.write(vid_resp.content)
                            return True

                    request_id = data.get("request_id")
                    if not request_id:
                        logger.error(f"Fal.ai lip-sync response missing request_id: {data}")
                        return False

                    status_url = data.get("status_url") or f"https://queue.fal.run/fal-ai/live-portrait/requests/{request_id}/status"
                    response_url = data.get("response_url") or f"https://queue.fal.run/fal-ai/live-portrait/requests/{request_id}"

                    start_time = asyncio.get_event_loop().time()
                    poll_interval = 2.0
                    while (asyncio.get_event_loop().time() - start_time) < 240.0:
                        await asyncio.sleep(poll_interval)
                        poll_interval = min(5.0, poll_interval * 1.2)
                        try:
                            poll_resp = await client.get(status_url, headers=headers)
                            if poll_resp.status_code != 200:
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
                                return False
                            elif status in ("FAILED", "ERROR"):
                                logger.error(f"Fal.ai lip-sync failed: {status_data}")
                                return False
                        except Exception as poll_e:
                            logger.warning(f"Error polling Fal.ai lip-sync: {poll_e}")

                    logger.error(f"Fal.ai lip-sync request {request_id} timed out")
                    return False

        except Exception as e:
            logger.error(f"CloudLipSyncProvider request failed: {e}", exc_info=True)
            return False
        finally:
            if os.path.isfile(slice_audio):
                os.remove(slice_audio)
