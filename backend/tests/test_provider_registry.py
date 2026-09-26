import pytest
from app.providers.registry import ProviderRegistry
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
    assert hm_caps.supports_lora is True

    # Verify HeartMuLa alias
    heartmula_alias = registry.get_provider("heartmula_3b")
    assert isinstance(heartmula_alias, HeartMuLaProvider)


def test_provider_registry_fallback():
    registry = ProviderRegistry()
    # Non-existent provider should cleanly fall back without throwing
    fallback = registry.get_provider("non_existent_engine_xyz")
    assert fallback is not None
    assert fallback.get_capabilities().provider_id in ["minimax_music3", "heartmula"]
