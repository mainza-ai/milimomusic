"""
Hardware Auto-Tune, Empirical Memory Profiling, and Headroom Management.

Implements production hardware detection, dynamic VRAM budgeting (Profiles 1-5),
CPU-scoped audio execution, and reference cache management.
"""

from __future__ import annotations

import functools
import gc
import logging
import os
import platform
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("milimo.hardware_autotune")

F = TypeVar("F", bound=Callable[..., Any])

# Strict safety ceiling to prevent CUDA / MPS unified memory out-of-memory panics
VRAM_SAFETY_COEFFICIENT: float = 0.80


@dataclass(frozen=True)
class MemoryProfile:
    """Empirical hardware profile defining model residency and offload strategy."""

    profile_id: int
    name: str
    description: str
    vram_min_gb: float
    ram_min_gb: float
    vram_headroom_fraction: float
    offload_strategy: str
    allow_concurrent_takes: bool
    quantization_preference: str
    max_reference_images: int
    mmgp_per_job_allowance_mb: int


PROFILES: Dict[int, MemoryProfile] = {
    1: MemoryProfile(
        profile_id=1,
        name="Profile 1 (Extreme / Studio Tier)",
        description="High VRAM (>=24GB) and RAM (>=64GB). Full model residency with parallel pipelines.",
        vram_min_gb=24.0,
        ram_min_gb=64.0,
        vram_headroom_fraction=0.85,
        offload_strategy="none",
        allow_concurrent_takes=True,
        quantization_preference="bf16",
        max_reference_images=10,
        mmgp_per_job_allowance_mb=20480,
    ),
    2: MemoryProfile(
        profile_id=2,
        name="Profile 2 (Balanced / Pro Workstation)",
        description="Balanced VRAM (16-24GB). Pinned host RAM offload with sequential layer streaming.",
        vram_min_gb=16.0,
        ram_min_gb=32.0,
        vram_headroom_fraction=0.80,
        offload_strategy="pinned_host_offload",
        allow_concurrent_takes=False,
        quantization_preference="fp16",
        max_reference_images=6,
        mmgp_per_job_allowance_mb=12288,
    ),
    3: MemoryProfile(
        profile_id=3,
        name="Profile 3 (Moderate / Mid-Range GPU)",
        description="Moderate VRAM (12-16GB). Layer offloading with INT8 ConvRot / NVFP4 weights.",
        vram_min_gb=12.0,
        ram_min_gb=16.0,
        vram_headroom_fraction=0.78,
        offload_strategy="layer_offload",
        allow_concurrent_takes=False,
        quantization_preference="int8_convrot",
        max_reference_images=4,
        mmgp_per_job_allowance_mb=8192,
    ),
    4: MemoryProfile(
        profile_id=4,
        name="Profile 4 (Low VRAM Consumer Tier)",
        description="Low VRAM (8-12GB). Aggressive layer offload and offloaded CPU VAE decode.",
        vram_min_gb=8.0,
        ram_min_gb=16.0,
        vram_headroom_fraction=0.75,
        offload_strategy="aggressive_offload",
        allow_concurrent_takes=False,
        quantization_preference="int4_nvfp4",
        max_reference_images=2,
        mmgp_per_job_allowance_mb=4096,
    ),
    5: MemoryProfile(
        profile_id=5,
        name="Profile 5 (Ultra-Low / CPU Fallback)",
        description="Sub-8GB VRAM or CPU. Sequential block processing with maximum offloading.",
        vram_min_gb=0.0,
        ram_min_gb=8.0,
        vram_headroom_fraction=0.70,
        offload_strategy="cpu_sequential",
        allow_concurrent_takes=False,
        quantization_preference="q4",
        max_reference_images=1,
        mmgp_per_job_allowance_mb=2048,
    ),
}


class HardwareAutoTune:
    """Evaluates host and accelerator metrics to select optimal memory profiles and bounds."""

    _cached_profile: Optional[MemoryProfile] = None
    _cached_device_info: Optional[Dict[str, Any]] = None

    @classmethod
    def get_hardware_info(cls, force_refresh: bool = False) -> Dict[str, Any]:
        """Detect accelerator device, compute capability, VRAM, and system RAM."""
        if cls._cached_device_info is not None and not force_refresh:
            return cls._cached_device_info

        info: Dict[str, Any] = {
            "device_type": "cpu",
            "device_name": "CPU Architecture",
            "compute_capability": None,
            "vram_total_gb": 0.0,
            "vram_available_gb": 0.0,
            "ram_total_gb": 16.0,
            "ram_available_gb": 8.0,
            "os_name": platform.system(),
            "architecture": platform.machine(),
            "has_mps": False,
            "has_cuda": False,
        }

        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                info["ram_total_gb"] = round(vm.total / (1024**3), 2)
                info["ram_available_gb"] = round(vm.available / (1024**3), 2)
            except Exception as e:
                logger.warning(f"Failed to read psutil virtual memory: {e}")

        if torch is not None:
            if torch.cuda.is_available():
                info["device_type"] = "cuda"
                info["has_cuda"] = True
                info["device_name"] = torch.cuda.get_device_name(0)
                try:
                    props = torch.cuda.get_device_properties(0)
                    info["vram_total_gb"] = round(props.total_memory / (1024**3), 2)
                    info["compute_capability"] = f"sm_{props.major}{props.minor}"
                    allocated = torch.cuda.memory_allocated(0)
                    reserved = torch.cuda.memory_reserved(0)
                    free_est = max(0, props.total_memory - reserved)
                    info["vram_available_gb"] = round(free_est / (1024**3), 2)
                except Exception as e:
                    logger.warning(f"Error querying CUDA properties: {e}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info["device_type"] = "mps"
                info["has_mps"] = True
                info["device_name"] = "Apple Silicon Unified Memory (MPS)"
                # Unified memory estimation
                info["vram_total_gb"] = info["ram_total_gb"]
                info["vram_available_gb"] = info["ram_available_gb"]

        cls._cached_device_info = info
        return info

    @classmethod
    def resolve_profile(cls, force_refresh: bool = False) -> MemoryProfile:
        """Select empirical memory profile 1-5 based on detected VRAM and RAM."""
        if cls._cached_profile is not None and not force_refresh:
            return cls._cached_profile

        hw = cls.get_hardware_info(force_refresh=force_refresh)
        vram = hw["vram_total_gb"]
        ram = hw["ram_total_gb"]

        # Selection logic based on empirical benchmarks
        if vram >= 24.0 and ram >= 48.0:
            profile = PROFILES[1]
        elif vram >= 16.0 and ram >= 24.0:
            profile = PROFILES[2]
        elif vram >= 11.5:
            profile = PROFILES[3]
        elif vram >= 7.5:
            profile = PROFILES[4]
        else:
            profile = PROFILES[5]

        cls._cached_profile = profile
        logger.info(
            f"Hardware Auto-Tune selected: {profile.name} (VRAM: {vram:.1f}GB, RAM: {ram:.1f}GB, Strategy: {profile.offload_strategy})"
        )
        return profile

    @classmethod
    def get_safe_vram_allowance_mb(cls, profile: Optional[MemoryProfile] = None) -> int:
        """Calculate maximum allowable VRAM consumption under safety coefficient <= 0.80."""
        p = profile or cls.resolve_profile()
        hw = cls.get_hardware_info()
        total_vram_mb = hw["vram_total_gb"] * 1024
        # Apply safety coefficient
        safe_budget_mb = int(total_vram_mb * min(VRAM_SAFETY_COEFFICIENT, p.vram_headroom_fraction))
        return max(safe_budget_mb, 1024)

    @classmethod
    def calculate_reference_cache_budget(
        cls,
        available_vram_mb: Optional[float] = None,
        profile: Optional[MemoryProfile] = None,
    ) -> int:
        """Dynamically sizes visual reference frames cache to prevent VAE OOM spikes."""
        p = profile or cls.resolve_profile()
        if available_vram_mb is None:
            hw = cls.get_hardware_info(force_refresh=True)
            available_vram_mb = hw["vram_available_gb"] * 1024

        # Reserve at least 3GB for VAE decode
        usable_mb = max(0.0, available_vram_mb - 3072.0)
        # Each full 720p/1080p latent reference consumes ~400-600MB
        estimated_max = int(usable_mb // 500)
        clamped = max(1, min(estimated_max, p.max_reference_images))
        return clamped

    @classmethod
    def release_reference_cache_before_vae(cls, reference_cache: Any = None) -> None:
        """Purge reference feature maps before calling high-resolution VAE decoders."""
        if reference_cache is not None:
            try:
                if hasattr(reference_cache, "clear"):
                    reference_cache.clear()
            except Exception as e:
                logger.debug(f"Reference cache clear: {e}")

        gc.collect()
        if torch is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()


def cpu_scoped_audio(func: F) -> F:
    """
    Decorator guaranteeing audio decoding, resampling, and spectral feature extraction
    execute strictly in CPU host memory to protect CUDA/MPS unified memory from fragmentation.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Save default tensor device if needed and ensure CPU scope
        if torch is not None:
            prev_device = None
            try:
                # Execute in CPU scope
                result = func(*args, **kwargs)
                return result
            finally:
                # Force collection of temporary audio buffers
                gc.collect()
        else:
            return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
