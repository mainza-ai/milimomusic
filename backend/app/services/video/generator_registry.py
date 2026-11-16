"""
Video Generator Registry.
Dynamic resolver for video generators based on user model selection and provider type.
Guarantees that selected models and their specific parameters are strictly honored,
eliminating silent fallthroughs to unintended diffusion models.
"""

import logging
from typing import Dict, Optional, Any

from app.services.video.generators.base import BaseVideoGenerator
from app.services.video.generators.diffusers_wan import DiffusersWanGenerator
from app.services.video.generators.diffusers_ltx import DiffusersLTXGenerator
from app.services.video.generators.minimax_h3 import MiniMaxH3Generator
from app.services.video.generators.cloud_video import CloudVideoGenerator
from app.services.video.generators.procedural import ProceduralVideoGenerator
from app.services.video.model_specs import normalize_model_key, get_model_spec, VideoModelSpec

logger = logging.getLogger(__name__)


class VideoGeneratorRegistry:
    """Authoritative registry and factory for all video generation engines."""

    _instances: Dict[str, BaseVideoGenerator] = {}

    @classmethod
    def resolve(
        cls,
        model_name: Optional[str] = "wan_14b",
        provider_type: Optional[str] = "local"
    ) -> BaseVideoGenerator:
        """
        Resolve the appropriate BaseVideoGenerator for the requested model and provider.
        Never silently fall through to an unintended model.
        """
        canonical_key = normalize_model_key(model_name)
        p_type = (provider_type or "local").lower().strip()
        instance_key = f"{canonical_key}:{p_type}"

        if instance_key in cls._instances:
            return cls._instances[instance_key]

        generator: BaseVideoGenerator

        # 1. Cloud Providers
        if p_type == "cloud_fal":
            generator = CloudVideoGenerator(service="fal", model=canonical_key)
        elif p_type == "cloud_replicate":
            generator = CloudVideoGenerator(service="replicate", model=canonical_key)
        elif p_type == "cloud_minimax":
            generator = CloudVideoGenerator(service="minimax", model="video-01")

        # 2. Local Models
        elif canonical_key in ("hailuo_h3", "minimax_h3"):
            generator = MiniMaxH3Generator()

        elif canonical_key == "wan_1.3b":
            generator = DiffusersWanGenerator(model_size="1.3b")

        elif canonical_key == "wan_14b":
            generator = DiffusersWanGenerator(model_size="14b")

        elif canonical_key == "ltx_video":
            generator = DiffusersLTXGenerator()

        elif canonical_key == "audioreactive":
            generator = ProceduralVideoGenerator()

        else:
            spec = get_model_spec(canonical_key)
            if not getattr(spec, "implemented", True):
                # Refuse instead of substituting an animatic for a diffusion model.
                # The old path logged a line and rendered Ken Burns output while the
                # picker still said "CogVideoX"/"HunyuanVideo".
                raise NotImplementedError(
                    spec.unimplemented_reason
                    or f"Model '{canonical_key}' has no local pipeline in this build."
                )
            logger.info(f"Model '{canonical_key}' selected; procedural generator (declared procedural).")
            generator = ProceduralVideoGenerator()

        cls._instances[instance_key] = generator
        logger.info(f"VideoGeneratorRegistry: Resolved '{model_name}' (provider='{p_type}') -> {generator.name}")
        return generator

    @classmethod
    def get_spec(cls, model_name: Optional[str]) -> VideoModelSpec:
        """Get the authoritative specification contract for a model."""
        return get_model_spec(model_name)

    @classmethod
    def unload_all(cls) -> None:
        """Unload and flush memory for all instantiated local generators."""
        for name, gen in list(cls._instances.items()):
            try:
                if hasattr(gen, "unload"):
                    gen.unload()
            except Exception as e:
                logger.warning(f"Error unloading generator {name}: {e}")
        cls._instances.clear()
