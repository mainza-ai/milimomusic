"""Capability manifests must be earned, not asserted (plan task 1.6).

Every provider used to advertise features it did not implement:
`supports_segment_repair=True` on four providers whose `repair_segment` returned
the input file, `supports_track_extension=True` on providers that generated a new
song, and `supports_lora=True` on MiniMax which has no LoRA code at all.

This test is the contract: a flag may only be `True` when the corresponding
method exists, is not inherited from the base "refuse honestly" default, and is
not one of the known no-op shapes.
"""

from __future__ import annotations

import inspect

import pytest

from app.providers.base import GenerationProvider
from app.providers.registry import provider_registry

# Distinct provider classes, one instance each (aliases share the same instance).
_PROVIDERS = {}
for _pid in list(provider_registry.providers):
    _inst = provider_registry.get_provider(_pid)
    _PROVIDERS.setdefault(type(_inst).__name__, _inst)


def _overrides(provider, method: str) -> bool:
    """True when `provider`'s class defines `method` itself (not the base default)."""
    return getattr(type(provider), method, None) is not getattr(GenerationProvider, method, None)


def _body_source(provider, method: str) -> str:
    try:
        return inspect.getsource(getattr(type(provider), method))
    except (OSError, TypeError):  # pragma: no cover - dynamically defined
        return ""


@pytest.mark.parametrize("name", sorted(_PROVIDERS))
def test_track_extension_claim_is_earned(name):
    provider = _PROVIDERS[name]
    caps = provider.get_capabilities()
    if not caps.supports_track_extension:
        # Claiming "no" is always allowed, but it must actually refuse.
        assert not _overrides(provider, "extend") or "NotImplementedError" in _body_source(provider, "extend"), (
            f"{name} advertises no extension support but implements extend(); "
            "either fix the flag or the method"
        )
        return

    assert _overrides(provider, "extend"), f"{name} claims track extension but inherits the refusing default"
    source = _body_source(provider, "extend")
    assert "NotImplementedError" not in source, f"{name} claims track extension but its extend() refuses"


@pytest.mark.parametrize("name", sorted(_PROVIDERS))
def test_segment_repair_claim_is_earned(name):
    provider = _PROVIDERS[name]
    caps = provider.get_capabilities()
    if not caps.supports_segment_repair:
        return
    assert _overrides(provider, "repair_segment"), (
        f"{name} claims segment repair but inherits the refusing default"
    )
    source = _body_source(provider, "repair_segment")
    assert "NotImplementedError" not in source
    # The specific lie the audit found: returning the input untouched.
    assert "audio_path=audio_path" not in source, (
        f"{name} claims segment repair but returns its input unchanged"
    )


@pytest.mark.parametrize("name", sorted(_PROVIDERS))
def test_lora_claim_is_earned(name):
    provider = _PROVIDERS[name]
    caps = provider.get_capabilities()
    if not caps.supports_lora:
        return
    assert hasattr(provider, "load_lora_checkpoint"), (
        f"{name} claims LoRA support but exposes no load_lora_checkpoint()"
    )


@pytest.mark.parametrize("name", sorted(_PROVIDERS))
def test_voice_conversion_claim_is_earned(name):
    """Voice conversion is a pipeline service; a provider that claims it must do it."""
    provider = _PROVIDERS[name]
    caps = provider.get_capabilities()
    if not caps.supports_voice_conversion:
        return
    assert hasattr(provider, "convert_voice"), (
        f"{name} claims voice conversion but exposes no convert_voice(); "
        "the pipeline owns this step"
    )


def test_legacy_providers_are_marked():
    """D3: HeartMuLa stays registered for reproducibility but is flagged legacy."""
    caps = provider_registry.get_provider("heartmula").get_capabilities()
    assert caps.legacy is True
    assert caps.supports_track_extension is False
    assert caps.supports_segment_repair is False


def test_yue2_is_gone():
    """D1: the sine-synth provider that falsified used_fallback_synth was deleted."""
    from app.providers.registry import ProviderNotFoundError

    with pytest.raises(ProviderNotFoundError):
        provider_registry.get_provider("yue2")


def test_unknown_provider_raises_instead_of_substituting():
    """1.7: picking a missing model must not silently return MiniMax audio."""
    from app.providers.registry import ProviderNotFoundError

    with pytest.raises(ProviderNotFoundError) as exc:
        provider_registry.get_provider("definitely-not-a-real-engine")
    assert "not available" in str(exc.value)
