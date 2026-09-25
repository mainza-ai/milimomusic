import pytest
import os
from unittest.mock import patch
from app.providers.minimax_provider import MiniMaxMusic3Provider

@pytest.mark.asyncio
async def test_minimax_strict_inference_raises_when_real_inference_fails(monkeypatch):
    monkeypatch.setenv("MILIMO_STRICT_INFERENCE", "1")
    provider = MiniMaxMusic3Provider()

    with patch("app.providers.minimax_provider.run_real_minimax_inference", side_effect=RuntimeError("VRAM Allocation Failed")):
        with pytest.raises(RuntimeError) as exc_info:
            await provider.generate(
                job_id="test_strict_job",
                prompt="Electronic synthwave beat",
                lyrics="",
                duration_ms=5000,
                seed=42,
            )

    assert "Strict Inference Error" in str(exc_info.value)
    assert "VRAM Allocation Failed" in str(exc_info.value)

@pytest.mark.asyncio
async def test_minimax_strict_inference_raises_when_no_weights(monkeypatch):
    monkeypatch.setenv("MILIMO_STRICT_INFERENCE", "1")
    monkeypatch.setenv("MINIMAX_MODEL_PATH", "")
    monkeypatch.setenv("MILIMO_MINIMAX_SNAPSHOT", "")
    provider = MiniMaxMusic3Provider()
    provider.snapshot_path = "/nonexistent/path/nowhere"

    with patch("app.providers.minimax_provider._MLX_AUDIO_AVAILABLE", False):
        with patch("app.services.model_manager.model_manager.get_active_model", return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                await provider.generate(
                    job_id="test_strict_job2",
                    prompt="Electronic synthwave beat",
                    lyrics="",
                    duration_ms=5000,
                    seed=42,
                )

    assert "Strict Inference Error" in str(exc_info.value)

