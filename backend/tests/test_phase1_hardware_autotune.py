"""
Unit and integration tests for Phase 1: Hardware Auto-Tune, Bounded Memory, and File Safety.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.core.hardware_autotune import (
    PROFILES,
    HardwareAutoTune,
    MemoryProfile,
    VRAM_SAFETY_COEFFICIENT,
    cpu_scoped_audio,
)
from app.core.hardware_coordinator import HardwareCoordinator
from app.core.win_safe_files import (
    atomic_write_json,
    safe_load_json,
    share_delete_file_response,
)
from app.services.video.mask_utils import (
    _compose_recast_character_masks,
    compose_recast_character_masks,
    compose_single_frame_mask,
)


def test_hardware_autotune_detection():
    hw_info = HardwareAutoTune.get_hardware_info(force_refresh=True)
    assert "device_type" in hw_info
    assert "ram_total_gb" in hw_info
    assert hw_info["ram_total_gb"] > 0

    profile = HardwareAutoTune.resolve_profile(force_refresh=True)
    assert isinstance(profile, MemoryProfile)
    assert profile.profile_id in [1, 2, 3, 4, 5]
    assert profile.vram_headroom_fraction <= 0.85


def test_safe_vram_allowance():
    # Test profile 1 allowance
    allowance_p1 = HardwareAutoTune.get_safe_vram_allowance_mb(PROFILES[1])
    assert allowance_p1 > 0

    # Profile 4 allowance
    allowance_p4 = HardwareAutoTune.get_safe_vram_allowance_mb(PROFILES[4])
    assert allowance_p4 > 0


def test_reference_cache_budget():
    budget_high = HardwareAutoTune.calculate_reference_cache_budget(
        available_vram_mb=16000.0, profile=PROFILES[1]
    )
    assert 1 <= budget_high <= 10

    budget_low = HardwareAutoTune.calculate_reference_cache_budget(
        available_vram_mb=2000.0, profile=PROFILES[4]
    )
    assert budget_low == 1

    # Cleanup hook
    mock_cache = {"k1": "v1"}
    HardwareAutoTune.release_reference_cache_before_vae(mock_cache)
    assert len(mock_cache) == 0


def test_cpu_scoped_audio_decorator():
    @cpu_scoped_audio
    def dummy_audio_op(x: int) -> int:
        return x * 2

    assert dummy_audio_op(21) == 42


def test_single_frame_bounded_mask():
    h, w = 480, 640
    # Full frame mask
    full_mask = compose_single_frame_mask(h, w, bounding_box=None)
    assert full_mask.shape == (h, w)
    assert np.allclose(full_mask, 1.0)

    # ROI mask
    roi_mask = compose_single_frame_mask(h, w, bounding_box=(100, 100, 200, 200), feather_radius=2)
    assert roi_mask.shape == (h, w)
    assert roi_mask[150, 150] == 1.0
    assert roi_mask[50, 50] == 0.0

    # Streaming mask generator for 5 frames
    stream = list(
        compose_recast_character_masks(
            num_frames=5,
            height=h,
            width=w,
            per_frame_bboxes=[(50, 50, 150, 150)] * 5,
        )
    )
    assert len(stream) == 5
    for m in stream:
        assert m.shape == (h, w)


def test_hardware_coordinator_telemetry():
    telemetry = HardwareCoordinator.get_comprehensive_telemetry()
    assert "profile_id" in telemetry
    assert "safe_vram_allowance_mb" in telemetry
    assert "offload_strategy" in telemetry


def test_atomic_write_and_safe_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_settings.json"
        data = {"theme": "dark", "retries": 2, "quality": "lossless"}

        # Atomic write
        assert atomic_write_json(config_path, data, make_backup=True)
        assert config_path.exists()
        bak_file = config_path.with_suffix(".json.bak")

        # Second write creates backup
        data2 = {"theme": "light", "retries": 1, "quality": "lossless"}
        atomic_write_json(config_path, data2, make_backup=True)
        assert bak_file.exists()

        # Load
        loaded = safe_load_json(config_path)
        assert loaded["theme"] == "light"

        # Corrupt destination file and verify rollback from .bak
        with open(config_path, "w") as f:
            f.write("INVALID JSON DATA {{{")

        recovered = safe_load_json(config_path)
        assert recovered is not None
        assert recovered["theme"] in ["dark", "light"]
