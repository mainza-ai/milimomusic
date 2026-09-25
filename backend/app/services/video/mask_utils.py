"""
Video Character Mask Utilities — Single-Frame Bounded Memory Processing.

Implements frame-by-frame bounded mask memory composition for Recast and Repaint
operations, eliminating whole-video 4D index and occupancy array memory spikes.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("milimo.mask_utils")


def compose_single_frame_mask(
    height: int,
    width: int,
    bounding_box: Optional[Tuple[int, int, int, int]] = None,
    feather_radius: int = 4,
    invert: bool = False,
) -> np.ndarray:
    """
    Generate a 2D float32 mask [0.0, 1.0] for a single frame within bounding box ROI.
    Avoids multi-gigabyte allocations by operating strictly on one frame at a time.
    """
    mask = np.zeros((height, width), dtype=np.float32)

    if bounding_box is None:
        # Full frame active
        mask.fill(1.0)
    else:
        x_min, y_min, x_max, y_max = bounding_box
        x_min = max(0, min(width - 1, int(x_min)))
        y_min = max(0, min(height - 1, int(y_min)))
        x_max = max(x_min + 1, min(width, int(x_max)))
        y_max = max(y_min + 1, min(height, int(y_max)))

        mask[y_min:y_max, x_min:x_max] = 1.0

        if feather_radius > 0:
            # Localized linear feathering around bounding edges without full 2D convolution
            f = min(feather_radius, (x_max - x_min) // 2, (y_max - y_min) // 2)
            if f > 0:
                for i in range(f):
                    alpha = (i + 1) / (f + 1)
                    # Horizontal edges
                    mask[y_min + i, x_min:x_max] = np.minimum(mask[y_min + i, x_min:x_max], alpha)
                    mask[y_max - 1 - i, x_min:x_max] = np.minimum(mask[y_max - 1 - i, x_min:x_max], alpha)
                    # Vertical edges
                    mask[y_min:y_max, x_min + i] = np.minimum(mask[y_min:y_max, x_min + i], alpha)
                    mask[y_min:y_max, x_max - 1 - i] = np.minimum(mask[y_min:y_max, x_max - 1 - i], alpha)

    if invert:
        mask = 1.0 - mask

    return mask


def compose_recast_character_masks(
    num_frames: int,
    height: int,
    width: int,
    per_frame_bboxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None,
    feather_radius: int = 4,
    invert: bool = False,
) -> Iterator[np.ndarray]:
    """
    Generator streaming 2D character masks one frame at a time.

    Guarantee: Memory usage stays O(H x W) instead of O(T x H x W), preventing
    RAM exhaustion during multi-character Recast operations.
    """
    for frame_idx in range(num_frames):
        bbox = None
        if per_frame_bboxes and frame_idx < len(per_frame_bboxes):
            bbox = per_frame_bboxes[frame_idx]

        frame_mask = compose_single_frame_mask(
            height=height,
            width=width,
            bounding_box=bbox,
            feather_radius=feather_radius,
            invert=invert,
        )
        yield frame_mask

        # Proactively release frame slice memory
        del frame_mask


def _compose_recast_character_masks(
    frames: List[Any],
    per_frame_bboxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None,
    feather_radius: int = 4,
    invert: bool = False,
) -> Iterator[np.ndarray]:
    """
    Adapter accepting video frame objects or shapes and streaming bounded masks.
    """
    num_frames = len(frames)
    if num_frames == 0:
        return

    # Derive height, width from first frame
    first = frames[0]
    if hasattr(first, "shape"):
        h, w = first.shape[:2]
    elif isinstance(first, (tuple, list)) and len(first) >= 2:
        h, w = first[0], first[1]
    else:
        h, w = 720, 1280

    yield from compose_recast_character_masks(
        num_frames=num_frames,
        height=h,
        width=w,
        per_frame_bboxes=per_frame_bboxes,
        feather_radius=feather_radius,
        invert=invert,
    )
