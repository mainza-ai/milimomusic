"""
Test active model selection, disk persistence, and ImageService MLX diffusion integration.
"""
import os
import json
import pytest
from app.services.model_manager import model_manager, _get_active_models_path
from app.services.image_service import image_service


def test_active_model_persistence():
    """Verify active model selection persists to disk and reloads cleanly."""
    active_path = _get_active_models_path()
    assert active_path is not None

    # Verify custom MLX model exists and is marked active
    active_img = model_manager.get_active_model("image")
    assert active_img is not None
    assert "flux" in active_img["id"].lower() or "custom" in active_img["id"].lower()
    assert active_img["is_active"] is True

    # Test setting active model and reloading
    model_manager.set_active_model("custom_aitrader_flux2_klein_9b_mlx_4bit")
    with open(active_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("image") == "custom_aitrader_flux2_klein_9b_mlx_4bit"

    # ImageService default model must match active model
    default_model_id = image_service.get_default_image_model()
    assert default_model_id == "custom_aitrader_flux2_klein_9b_mlx_4bit"


def test_image_service_mlx_diffusion():
    """Verify ImageService executes real MLX diffusion with Flux2Klein."""
    # Ensure active image model is FLUX.2 Klein 9B MLX
    active_img = model_manager.get_active_model("image")
    assert active_img["is_installed"] is True
    assert active_img["local_path"] and os.path.isdir(active_img["local_path"])

    result = image_service.generate_cover(
        prompt="retro synth studio tape neon",
        style="synthwave cinematic",
        aspect_ratio="1:1",
        model_id=active_img["id"]
    )

    assert result is not None
    assert result.get("engine") == "mflux_flux2_mlx"
    assert result.get("diffusion_error") is None
    assert result.get("file_path") and os.path.isfile(result["file_path"])
    assert result.get("url") and result["url"].startswith("/covers/")
