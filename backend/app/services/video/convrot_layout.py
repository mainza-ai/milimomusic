"""
ConvRot Weight Layout Parser & H3 Singularity Quantization Descriptor Utilities.

Parses ComfyUI quantization metadata (.comfy_quant) from safetensors checkpoints,
extracts ConvRot matrix layout, group sizes (256), and supports LightX2V 4-step Turbo presets.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("milimo.video.convrot_layout")


@dataclass
class ConvRotQuantConfig:
    """ConvRot quantization descriptors parsed from safetensors or config."""

    quant_format: str = "convrot_int8"
    group_size: int = 256
    has_grouped_qkv: bool = True
    scale_dtype: str = "float16"
    turbo_preset_steps: int = 4
    turbo_cfg: float = 1.0
    video_shift: float = 12.0
    audio_shift: float = 3.0


class ConvRotLayoutParser:
    """Safetensors header reader and ConvRot weight layout transformer."""

    @classmethod
    def read_safetensors_metadata(cls, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract the JSON header metadata from a safetensors file without loading tensor bytes.
        """
        p = Path(file_path)
        if not p.exists() or p.stat().st_size < 8:
            return {}

        try:
            with open(p, "rb") as f:
                header_bytes_len = struct.unpack("<Q", f.read(8))[0]
                if header_bytes_len > 100 * 1024 * 1024:  # Safety cap at 100MB
                    return {}
                header_json = f.read(header_bytes_len).decode("utf-8")
                data = json.loads(header_json)
                metadata = data.get("__metadata__", {})
                return metadata
        except Exception as e:
            logger.debug(f"Could not read safetensors metadata from {p}: {e}")
            return {}

    @classmethod
    def parse_quant_descriptor(
        cls,
        metadata: Dict[str, Any],
        fallback_group_size: int = 256,
    ) -> ConvRotQuantConfig:
        """
        Parse .comfy_quant or ComfyUI quantization descriptors.
        """
        quant_format = "convrot_int8"
        group_size = fallback_group_size
        has_grouped_qkv = True

        comfy_quant = metadata.get("comfy_quant") or metadata.get(".comfy_quant")
        if comfy_quant:
            if isinstance(comfy_quant, str):
                try:
                    comfy_quant = json.loads(comfy_quant)
                except Exception:
                    pass

            if isinstance(comfy_quant, dict):
                quant_format = comfy_quant.get("format", quant_format)
                group_size = comfy_quant.get("group_size", group_size)
                has_grouped_qkv = comfy_quant.get("grouped_qkv", True)

        return ConvRotQuantConfig(
            quant_format=quant_format,
            group_size=group_size,
            has_grouped_qkv=has_grouped_qkv,
            turbo_preset_steps=4,
            turbo_cfg=1.0,
            video_shift=12.0,
            audio_shift=3.0,
        )

    @classmethod
    def transform_convrot_qkv_layout(
        cls,
        qkv_weight: Any,
        num_heads: int = 32,
        head_dim: int = 64,
    ) -> Any:
        """
        Permute grouped QKV rows to align with ConvRot rotational matrix kernels.
        """
        if torch is None or not isinstance(qkv_weight, torch.Tensor):
            return qkv_weight

        # Layout transform on CPU to protect VRAM
        orig_device = qkv_weight.device
        w = qkv_weight.cpu()

        # If shape matches [3 * hidden_dim, in_features]
        if w.ndim == 2:
            out_features, in_features = w.shape
            hidden_dim = num_heads * head_dim
            if out_features == 3 * hidden_dim:
                # Reshape to [3, num_heads, head_dim, in_features]
                w_reshaped = w.view(3, num_heads, head_dim, in_features)
                # Permute for rotary stride: [num_heads, 3, head_dim, in_features]
                w_permuted = w_reshaped.permute(1, 0, 2, 3).contiguous()
                return w_permuted.view(out_features, in_features).to(orig_device)

        return qkv_weight
