"""
Global Hardware Coordinator — Unified Resource Controller for Milimo Music.

Combines device lock serialization, live VRAM/RAM telemetry, memory reclamation,
and empirical profile autotuning across CUDA, Apple Silicon MPS, and CPU.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.core.hardware_autotune import HardwareAutoTune, MemoryProfile
from app.core.hardware_lock import GlobalHardwareCoordinator

logger = logging.getLogger("milimo.hardware_coordinator")


class HardwareCoordinator(GlobalHardwareCoordinator):
    """
    Unified accelerator and memory coordinator.

    Inherits lock management from GlobalHardwareCoordinator while binding
    HardwareAutoTune profile constraints, safe memory limits, and reference caches.
    """

    @classmethod
    def get_active_profile(cls) -> MemoryProfile:
        """Retrieve active empirical memory profile (1-5)."""
        return HardwareAutoTune.resolve_profile()

    @classmethod
    def get_comprehensive_telemetry(cls) -> Dict[str, Any]:
        """Fetch unified telemetry including lock state, memory usage, and active profile."""
        base_telemetry = cls.get_telemetry()
        profile = cls.get_active_profile()
        hw_info = HardwareAutoTune.get_hardware_info()

        return {
            **base_telemetry,
            "profile_id": profile.profile_id,
            "profile_name": profile.name,
            "offload_strategy": profile.offload_strategy,
            "quantization_preference": profile.quantization_preference,
            "max_reference_images": profile.max_reference_images,
            "safe_vram_allowance_mb": HardwareAutoTune.get_safe_vram_allowance_mb(profile),
            "ram_total_gb": hw_info.get("ram_total_gb", 16.0),
            "ram_available_gb": hw_info.get("ram_available_gb", 8.0),
            "compute_capability": hw_info.get("compute_capability"),
        }


# Global alias for convenience
hardware_coordinator = HardwareCoordinator()
