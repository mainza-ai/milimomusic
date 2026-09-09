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

    # Resolve the active image model (catalog default or machine custom) — the
    # exact id is environment-specific, so we pin/verify whatever is active.
    active_img = model_manager.get_active_model("image")
    assert active_img is not None
    assert "flux" in active_img["id"].lower() or "custom" in active_img["id"].lower()
    assert active_img["is_active"] is True
    active_id = active_img["id"]

    # Test setting active model and reloading
    model_manager.set_active_model(active_id)
    with open(active_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("image") == active_id

    # ImageService default model must match active model
    default_model_id = image_service.get_default_image_model()
    assert default_model_id == active_id


def test_image_service_mlx_diffusion():
    """Verify ImageService executes real MLX diffusion with Flux2Klein."""
    pytest.importorskip("mflux", reason="Requires mflux MLX diffusion (Apple Silicon local test)")
    # Ensure active image model is installed (local weights present)
    active_img = model_manager.get_active_model("image")
    if not active_img.get("is_installed"):
        pytest.skip("No installed image model weights available for MLX diffusion")
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
