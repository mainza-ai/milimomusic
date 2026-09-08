"""
Range-capable static file serving for Milimo Music.

The pinned Starlette (0.36.x) `StaticFiles`/`FileResponse` ignore HTTP Range
requests (verified: `Range: bytes=0-1023` returns 200 + full body). Without
ranges, seeking/scrubbing is broken everywhere and Safari (desktop + all iOS
browsers, which mandate ranges) cannot play served audio at all.

`RangedStaticFiles` is a drop-in `StaticFiles` subclass that honors single
suffix/offset byte ranges (206 + Content-Range + Accept-Ranges), answers 416
for unsatisfiable/multi ranges, and delegates everything else (404s, HEAD,
conditional requests, traversal protection) to the stock implementation.
"""

from __future__ import annotations

import mimetypes
import os
from email.utils import formatdate
from typing import AsyncIterator, Optional, Tuple

import anyio
import anyio.to_thread
from starlette.datastructures import Headers
from starlette.responses import FileResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send

_CHUNK_SIZE = 64 * 1024


def parse_range_header(range_header: str, size: int) -> Optional[Tuple[int, int]]:
    """Parse a single byte range. Returns (start, end) inclusive, or None.

    None means absent/invalid/multi-range (caller decides 416 vs ignore).
    """
    if not range_header.startswith("bytes="):
        return None
    spec = range_header[len("bytes="):].strip()
    if "," in spec:  # multipart ranges not supported
        return None
    start_s, sep, end_s = spec.partition("-")
    if not sep:
        return None
    try:
        if start_s == "":
            # suffix range: last N bytes
            suffix = int(end_s)
            if suffix <= 0:
                return None
            start = max(0, size - suffix)
            return (start, size - 1)
        start = int(start_s)
        end = int(end_s) if end_s != "" else size - 1
    except ValueError:
        return None
    if start >= size or end < start:
        return None
    return (start, min(end, size - 1))


async def _aiter_slice(path: str, offset: int, length: int) -> AsyncIterator[bytes]:
    def _read_chunk(f, n: int) -> bytes:
        return f.read(n)

    with open(path, "rb") as f:
        f.seek(offset)
        remaining = length
        while remaining > 0:
            data = await anyio.to_thread.run_sync(_read_chunk, f, min(_CHUNK_SIZE, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


class RangedFileResponse(Response):
    """206 partial-content response streaming [start, end] of a file."""

    media_type = "application/octet-stream"

    def __init__(self, path: str, start: int, end: int, size: int, stat_result: os.stat_result):
        self.path = path
        self.start = start
        self.end = end
        self.size = size
        content_type, _ = mimetypes.guess_type(path)
        headers = {
            "content-range": f"bytes {start}-{end}/{size}",
            "accept-ranges": "bytes",
            "content-length": str(end - start + 1),
            "last-modified": formatdate(stat_result.st_mtime, usegmt=True),
        }
        super().__init__(
            content=b"",
            status_code=206,
            headers=headers,
            media_type=content_type or self.media_type,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["method"] == "HEAD":
            await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return
        await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
        length = self.end - self.start + 1
        async for chunk in _aiter_slice(self.path, self.start, length):
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


class RangedStaticFiles(StaticFiles):
    """StaticFiles + single-range support + always-on Accept-Ranges."""

    def file_response(self, full_path, stat_result, scope, status_code: int = 200):  # type: ignore[override]
        # NB: sync by contract — StaticFiles.__call__ invokes this without await.
        request_headers = Headers(scope=scope)
        if scope.get("method") == "GET" and "range" in request_headers:
            size = stat_result.st_size
            parsed = parse_range_header(request_headers["range"], size)
            if parsed is None:
                return Response(
                    content=b"Requested range not satisfiable",
                    status_code=416,
                    headers={
                        "content-range": f"bytes */{size}",
                        "accept-ranges": "bytes",
                    },
                    media_type="text/plain",
                )
            start, end = parsed
            return RangedFileResponse(str(full_path), start, end, size, stat_result)
        response = FileResponse(full_path, status_code=status_code, stat_result=stat_result)
        response.headers["accept-ranges"] = "bytes"
        if self.is_not_modified(response.headers, request_headers):
            from starlette.staticfiles import NotModifiedResponse

            return NotModifiedResponse(response.headers)
        return response
