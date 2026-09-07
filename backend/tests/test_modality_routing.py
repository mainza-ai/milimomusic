"""
Regression tests for video->audio misrouting (MiniMax H3 case) and
registry repair semantics. Hermetic: no network, no real model dirs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.modality import infer_category, infer_modality, VALID_CATEGORIES


def test_h3_mlx_repo_is_video():
    assert infer_category("pipenetwork/MiniMax-H3-MLX-8bit") == "video"


def test_h3_official_repo_is_video():
    assert infer_category("MiniMaxAI/MiniMax-H3") == "video"


def test_h3_pipeline_tag_is_video():
    # H3's real HF tag, previously missing from every allowlist.
    assert infer_category("some-org/some-weights", pipeline_tag="image-text-to-video") == "video"
    assert infer_category("some-org/some-weights", pipeline_tag="text-to-video") == "video"


def test_music3_stays_audio():
    assert infer_category("mlx-community/MiniMax-Music3-mxfp4") == "audio"
    assert infer_category("mlx-community/MiniMax-Music3-8bit") == "audio"
    assert infer_category("MiniMaxAI/MiniMax-Music3") == "audio"


def test_bare_minimax_org_does_not_force_audio_for_video_tag():
    cat, reason = infer_modality("MiniMaxAI/MiniMax-H3", pipeline_tag="image-text-to-video")
    assert cat == "video"
    assert "pipeline_tag" in reason


def test_flux_stays_image_and_vae_filename_compat():
    assert infer_category("black-forest-labs/FLUX.1-schnell") == "image"
    assert infer_category("unknown/some-repo", filenames=["weights.vae.safetensors"]) == "image"


def test_wan_stays_video():
    assert infer_category("Wan-AI/Wan2.1-T2V-1.3B") == "video"


def test_taxonomy_has_no_custom_category():
    assert set(VALID_CATEGORIES) == {"audio", "image", "video"}


def test_update_custom_model_rejects_invalid_category(monkeypatch):
    from app.services.model_manager import model_manager
    # Nonexistent id must return None, never create or raise KeyError.
    assert model_manager.update_custom_model("nonexistent-id-xyz", {"category": "video"}) is None
    # Invalid taxonomy value must raise, even for an existing entry.
    monkeypatch.setattr(
        model_manager,
        "_load_custom_models",
        lambda: [{"id": "custom_x", "repo_id": "org/x", "category": "audio", "local_path": None}],
    )
    try:
        model_manager.update_custom_model("custom_x", {"category": "custom"})
    except ValueError:
        return
    raise AssertionError("expected ValueError for category='custom'")


def test_audio_provider_refuses_video_weights(tmp_path):
    import json
    from app.providers.hf_audio_provider import HuggingFaceAudioProvider
    d = tmp_path / "MiniMax-H3-MLX-8bit"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"_class_name": "MiniMaxH3DiTModel"}))
    p = HuggingFaceAudioProvider("pipenetwork/MiniMax-H3-MLX-8bit", local_path=str(d))
    assert p._looks_like_video_weights() is True
