"""Lip-sync engine registry (plan 7.27).

The render request carried `lip_sync_engine` and the orchestrator ignored it, so the
inspector offered a choice that changed nothing. These tests pin the selection and,
more importantly, that an unavailable engine is replaced with a *stated* reason
rather than silently swapped.
"""

from __future__ import annotations

import pytest

from app.services.video.lip_sync import registry as reg


def test_jaw_warp_is_always_available():
    info = reg.get_engine("jaw_warp")
    assert info is not None and info.available is True
    assert info.neural is False


def test_the_legacy_client_alias_resolves():
    """The inspector shipped `fallback` before the registry existed."""
    provider, reason = reg.resolve_local_provider("fallback")
    assert type(provider).__name__ == "SmoothVisemeFallbackProvider"
    assert reason is None


def test_liveportrait_without_weights_resolves_to_jaw_warp_with_a_reason(monkeypatch):
    monkeypatch.delenv("MILIMO_LIVEPORTRAIT_AUDIO_SCRIPT", raising=False)
    provider, reason = reg.resolve_local_provider("live_portrait")
    assert type(provider).__name__ == "SmoothVisemeFallbackProvider"
    assert reason and "unavailable" in reason.lower()
    assert "jaw-warp" in reason.lower()


def test_unknown_engine_is_not_silently_accepted():
    provider, reason = reg.resolve_local_provider("definitely-not-an-engine")
    assert type(provider).__name__ == "SmoothVisemeFallbackProvider"
    assert "unknown" in reason.lower()


def test_a_cloud_engine_cannot_be_selected_as_a_local_engine():
    provider, reason = reg.resolve_local_provider("cloud_fal")
    assert type(provider).__name__ == "SmoothVisemeFallbackProvider"
    assert "cloud engine" in reason.lower()


def test_cloud_provider_resolution_follows_the_render_provider():
    assert reg.resolve_cloud_provider("cloud_fal").service == "fal"
    assert reg.resolve_cloud_provider("local") is None


def test_catalogue_reports_capability_and_licence_for_every_engine():
    engines = reg.engine_catalogue()
    ids = {e.id for e in engines}
    assert {"jaw_warp", "live_portrait", "cloud_fal", "cloud_replicate"} <= ids
    for engine in engines:
        # Every engine states why it cannot run; licences are recorded, not implied.
        assert engine.licence
        if not engine.available:
            assert engine.reason
    replicate = reg.get_engine("cloud_replicate")
    assert replicate.available is False
    assert "D4" in (replicate.reason or "")


def test_cloud_availability_tracks_the_key(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    assert reg.get_engine("cloud_fal").available is False
    monkeypatch.setenv("FAL_KEY", "test-key")
    assert reg.get_engine("cloud_fal").available is True


def test_engine_endpoint_lists_availability(client):
    res = client.get("/videos/lipsync-engines")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    ids = {e["id"] for e in body["engines"]}
    assert "jaw_warp" in ids
    for engine in body["engines"]:
        assert "available" in engine
        assert "neural" in engine
