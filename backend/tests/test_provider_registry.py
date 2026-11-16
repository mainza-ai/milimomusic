import pytest
from app.providers.registry import ProviderNotFoundError, ProviderRegistry
from app.providers.minimax_provider import MiniMaxMusic3Provider
from app.providers.heartmula_provider import HeartMuLaProvider


def test_provider_registry_defaults():
    registry = ProviderRegistry()
    
    # Verify default active provider belongs to MiniMax Music 3 family
    assert registry.get_active_provider_id() in ["minimax_music3", "minimax_music3_mxfp4"]
    
    # Verify MiniMax Music 3 provider
    minimax = registry.get_provider("minimax_music3")
    assert isinstance(minimax, MiniMaxMusic3Provider)
    caps = minimax.get_capabilities()
    assert caps.provider_id == "minimax_music3"
    assert caps.supports_structured_caption is True

    # Verify HeartMuLa provider registration
    heartmula = registry.get_provider("heartmula")
    assert isinstance(heartmula, HeartMuLaProvider)
    hm_caps = heartmula.get_capabilities()
    assert hm_caps.provider_id == "heartmula"
    # Decision D3: HeartMuLa is kept as a legacy engine with honest capabilities.
    # It has no LoRA loader, no continuation and no in-painting — the old flags
    # advertised all three.
    assert hm_caps.legacy is True
    assert hm_caps.supports_lora is False
    assert hm_caps.supports_track_extension is False
    assert hm_caps.supports_segment_repair is False

    # Verify HeartMuLa alias
    heartmula_alias = registry.get_provider("heartmula_3b")
    assert isinstance(heartmula_alias, HeartMuLaProvider)


def test_provider_registry_unknown_engine_raises():
    """Silently substituting MiniMax made a missing model look installed."""
    registry = ProviderRegistry()
    with pytest.raises(ProviderNotFoundError) as exc:
        registry.get_provider("non_existent_engine_xyz")
    assert "not available" in str(exc.value)
