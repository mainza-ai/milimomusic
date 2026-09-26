"""
Comprehensive Test Suite for Cross-Modal Model Lifecycle, Eager Eviction, and Open Audio Alternatives.
Validates:
1. GlobalHardwareCoordinator multi-hook eviction bus, memory flushing, and dual policies (eager vs warm_ttl).
2. StableAudioProvider and MusicGenProvider contract, capabilities, and unload.
3. ProviderRegistry resolution for open audio models and unload_all.
4. ModelManager multi-modal catalog, active model sync, and auto-routing in generation.
5. ImageService, Separator, MuScriptor, and NeuralSVC unload contracts.
"""

import os
import pytest
import asyncio
from unittest.mock import MagicMock, patch

from app.core.hardware_lock import GlobalHardwareCoordinator
from app.providers.base import GenerationProvider, GenerationCapabilities
from app.providers.stable_audio_provider import StableAudioProvider
from app.providers.musicgen_provider import MusicGenProvider
from app.providers.registry import ProviderRegistry, provider_registry
from app.services.model_manager import model_manager
from app.services.image_service import image_service
from app.transcription.real_separator import unload_model as unload_separator
from app.transcription.muscriptor_provider import muscriptor_provider
from app.services.voice.neural_svc import neural_svc


def test_hardware_coordinator_multi_hook_eviction_bus():
    """Verify GlobalHardwareCoordinator supports multiple hooks per modality without overwriting."""
    calls = []

    def hook_a():
        calls.append("a")

    def hook_b():
        calls.append("b")

    GlobalHardwareCoordinator.register_eviction_hook("test_multi_mod", hook_a)
    GlobalHardwareCoordinator.register_eviction_hook("test_multi_mod", hook_b)

    assert len(GlobalHardwareCoordinator._eviction_hooks["test_multi_mod"]) == 2

    calls.clear()
    GlobalHardwareCoordinator.evict_modality("test_multi_mod")
    assert "a" in calls
    assert "b" in calls


def test_hardware_coordinator_memory_policies():
    """Verify GlobalHardwareCoordinator dual memory policy (eager vs warm_ttl)."""
    # Test setting policy to warm_ttl
    res = GlobalHardwareCoordinator.set_memory_policy("warm_ttl", ttl_seconds=60)
    assert res["policy"] == "warm_ttl"
    assert res["ttl_seconds"] == 60.0

    # Test setting policy back to eager
    res2 = GlobalHardwareCoordinator.set_memory_policy("eager")
    assert res2["policy"] == "eager"

    # Invalid policy throws
    with pytest.raises(ValueError):
        GlobalHardwareCoordinator.set_memory_policy("invalid_policy_name")


def test_hardware_coordinator_flush_memory():
    """Verify flush_memory executes without exceptions on the active platform."""
    res = GlobalHardwareCoordinator.flush_memory()
    assert isinstance(res, dict)
    assert "after_mb" in res


@pytest.mark.asyncio
async def test_hardware_coordinator_scoped_device_eviction():
    """Verify scoped_device context manager acquires lock and evicts foreign modalities."""
    evicted = []
    GlobalHardwareCoordinator.register_eviction_hook("eviction_test_modality", lambda: evicted.append(True))

    async with GlobalHardwareCoordinator.scoped_device("different_modality"):
        assert len(evicted) > 0
        assert GlobalHardwareCoordinator.get_active_consumer() == "different_modality"

    assert GlobalHardwareCoordinator.get_active_consumer() == "idle"


def test_stable_audio_provider_contract():
    """Verify StableAudioProvider capabilities, interface compliance, and unload behavior."""
    provider = StableAudioProvider()
    assert isinstance(provider, GenerationProvider)

    caps = provider.get_capabilities()
    assert isinstance(caps, GenerationCapabilities)
    assert caps.provider_id == "stable_audio_open_1_0"
    assert caps.default_sample_rate == 44100
    assert caps.max_duration_sec == 47
    assert "Stability AI" in caps.license_class

    # Test unload when pipeline is not initialized
    assert provider.unload() is False
    assert provider._is_loaded is False

    # Simulate loaded pipeline
    mock_pipe = MagicMock()
    mock_pipe.to.return_value = mock_pipe
    provider.pipeline = mock_pipe
    provider._is_loaded = True

    assert provider.unload() is True
    assert provider.pipeline is None
    assert provider._is_loaded is False


def test_musicgen_provider_contract():
    """Verify MusicGenProvider small and melody capabilities, interface, and unload behavior."""
    small_provider = MusicGenProvider("facebook/musicgen-small")
    assert isinstance(small_provider, GenerationProvider)
    caps_small = small_provider.get_capabilities()
    assert caps_small.provider_id == "musicgen"
    assert caps_small.default_sample_rate == 44100
    assert "Meta" in caps_small.description

    melody_provider = MusicGenProvider("facebook/musicgen-melody")
    caps_melody = melody_provider.get_capabilities()
    assert caps_melody.provider_id == "musicgen_melody"

    # Test unload
    mock_model = MagicMock()
    mock_processor = MagicMock()
    melody_provider.model = mock_model
    melody_provider.processor = mock_processor
    melody_provider._is_loaded = True

    assert melody_provider.unload() is True
    assert melody_provider.model is None
    assert melody_provider.processor is None
    assert melody_provider._is_loaded is False


def test_provider_registry_resolution_and_unload_all():
    """Verify ProviderRegistry resolves StableAudio and MusicGen, and unload_all functions."""
    registry = ProviderRegistry()

    # Stable Audio resolution
    p_sa = registry.get_provider("stable_audio_open_1_0")
    assert isinstance(p_sa, StableAudioProvider)

    p_sa_alias = registry.get_provider("stabilityai/stable-audio-open-1.0")
    assert isinstance(p_sa_alias, StableAudioProvider)

    p_sa_hf = registry.get_provider("hf:stabilityai/stable-audio-open-1.0")
    assert isinstance(p_sa_hf, StableAudioProvider)

    # MusicGen resolution
    p_mg_small = registry.get_provider("musicgen_small")
    assert isinstance(p_mg_small, MusicGenProvider)

    p_mg_repo = registry.get_provider("facebook/musicgen-small")
    assert isinstance(p_mg_repo, MusicGenProvider)

    p_mg_melody = registry.get_provider("facebook/musicgen-melody")
    assert isinstance(p_mg_melody, MusicGenProvider)
    assert p_mg_melody.get_capabilities().provider_id == "musicgen_melody"

    # Test unload_all
    unloaded_count = registry.unload_all()
    assert isinstance(unloaded_count, int)


def test_model_manager_catalog_and_active_sync():
    """Verify ModelManager catalog contains open audio alternatives and set_active_model syncs."""
    tree = model_manager.get_model_tree()
    ids = [m["id"] for m in tree]

    assert "stable_audio_open_1_0" in ids
    assert "musicgen_small" in ids
    assert "musicgen_melody" in ids
    assert "minimax_music3_mxfp4" in ids

    # Check hardware profile recommendations
    sa_model = next(m for m in tree if m["id"] == "stable_audio_open_1_0")
    assert sa_model["category"] == "audio"
    assert sa_model["size_gb"] > 0

    mg_small_model = next(m for m in tree if m["id"] == "musicgen_small")
    assert mg_small_model["category"] == "audio"
    assert mg_small_model["size_gb"] <= 2.0

    # Test set_active_model sync
    active = model_manager.set_active_model("musicgen_small")
    assert active["id"] == "musicgen_small"
    assert provider_registry.get_active_provider_id() == "musicgen_small"

    # Reset active back to default MiniMax
    model_manager.set_active_model("minimax_music3_mxfp4")
    provider_registry.set_active_provider("minimax_music3")


def test_image_service_unload_models():
    """Verify image_service.unload_models cleans loaded pipelines and runs GC/Metal flush."""
    # Test when uninitialized
    assert image_service.unload_models() is False

    # Simulate loaded diffusers pipeline
    mock_pipe = MagicMock()
    image_service._loaded_diffusers_pipeline = mock_pipe
    image_service._loaded_diffusers_model_id = "test_model"

    assert image_service.unload_models() is True
    assert image_service._loaded_diffusers_pipeline is None
    assert image_service._loaded_diffusers_model_id is None


def test_neural_svc_and_muscriptor_unload():
    """Verify NeuralSVCService and MuScriptorProvider unload cleanly."""
    assert neural_svc.unload() is True
    assert muscriptor_provider.unload() is True


def test_llm_service_unload_omlx_and_providers():
    """Verify LLMService.unload_local_model triggers OMLX, Ollama, and LM Studio unloads."""
    from app.services.llm_service import LLMService

    llm = LLMService()

    # 1. Test oMLX unload with mocked requests
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        unloaded = llm.unload_local_model(provider_name="omlx", model_name="Qwen3.6-35B")
        assert len(unloaded["unloaded"]) == 1
        assert "omlx:Qwen3.6-35B" in unloaded["unloaded"]
        mock_post.assert_called_once()
        assert "unload" in mock_post.call_args[0][0]

    # 2. Test Ollama keep_alive: 0 unload
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        unloaded = llm.unload_local_model(provider_name="ollama", model_name="llama3.1")
        assert len(unloaded["unloaded"]) == 1
        assert "ollama:llama3.1" in unloaded["unloaded"]
        mock_post.assert_called_once()
        assert mock_post.call_args[1]["json"]["keep_alive"] == 0

    # 3. Test LM Studio unload
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        unloaded = llm.unload_local_model(provider_name="lmstudio", model_name="local-model")
        assert len(unloaded["unloaded"]) == 1
        assert "lmstudio:local-model" in unloaded["unloaded"]
        mock_post.assert_called_once()

    # 4. Graceful handling of network error
    with patch("requests.post", side_effect=Exception("Connection refused")):
        unloaded = llm.unload_local_model(provider_name="omlx", model_name="Qwen3.6-35B")
        assert len(unloaded["errors"]) > 0


def test_global_hardware_coordinator_llm_eviction():
    """Verify GlobalHardwareCoordinator registers and triggers llm modality eviction."""
    from app.services.llm_service import LLMService

    with patch.object(LLMService, "unload_local_model", return_value={"unloaded": ["omlx:model"], "errors": []}) as mock_unload:
        # Trigger eviction of LLM modality
        GlobalHardwareCoordinator.evict_modality("llm")
        mock_unload.assert_called_once()


def test_flux2_adaptive_diffusion_parameters():
    """Verify ImageService selects base model 24 steps & 3.5 guidance vs distilled 4 steps & 1.0 guidance."""
    # Test base model detection
    base_res = None
    with patch.object(image_service, "_run_mlx_diffusion") as mock_mlx:
        # Mock executor to call synchronously
        with patch("app.services.image_service._mlx_executor.submit") as mock_submit:
            mock_future = MagicMock()
            mock_future.result.return_value = None
            mock_submit.return_value = mock_future

            # Render with base model repo_id
            image_service._render_diffusion_image(
                full_prompt="photorealistic scene",
                width=1280,
                height=720,
                chosen_model_id="flux2_klein_9b",
                local_path=None,
                repo_id="black-forest-labs/FLUX.2-klein-base-9B",
                is_installed=True,
                dest_path="/tmp/test_base.png",
            )
            mock_submit.assert_called_once()
            kwargs = mock_submit.call_args[1]
            assert kwargs["is_base"] is True
            assert kwargs["steps"] == 24
            assert kwargs["guidance"] == 3.5

    # Test turbo / distilled model detection
    with patch("app.services.image_service._mlx_executor.submit") as mock_submit:
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_submit.return_value = mock_future

        image_service._render_diffusion_image(
            full_prompt="photorealistic scene",
            width=1280,
            height=720,
            chosen_model_id="flux2_klein_turbo_4step",
            local_path=None,
            repo_id="black-forest-labs/FLUX.2-klein-4step",
            is_installed=True,
            dest_path="/tmp/test_turbo.png",
        )
        mock_submit.assert_called_once()
        kwargs = mock_submit.call_args[1]
        assert kwargs["is_base"] is False
        assert kwargs["steps"] == 4
        assert kwargs["guidance"] == 1.0

