"""Lip-sync engine registry (plan 7.27).

The render request carried `lip_sync_engine` from the first day of the studio and
the orchestrator **ignored it**, always instantiating LivePortrait (which itself
delegates to the jaw-warp animatic when weights are absent). The UI therefore
offered a choice that had no effect.

This registry is the single table the API, the renderer and the inspector read:
what engines exist, whether each can actually run here, and why not when it
cannot. Requesting an unavailable engine falls back to the jaw-warp animatic
**with a recorded reason** instead of silently doing something else.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("milimo.video.lipsync.registry")

JAW_WARP = "jaw_warp"
#: Legacy client value for the same engine (the inspector shipped `fallback` first).
ALIASES = {"fallback": JAW_WARP}
LIVE_PORTRAIT = "live_portrait"
CLOUD_FAL = "cloud_fal"
CLOUD_REPLICATE = "cloud_replicate"


@dataclass(frozen=True)
class LipSyncEngineInfo:
    id: str
    name: str
    kind: str                     # "local" | "cloud"
    available: bool
    reason: Optional[str] = None  # why it cannot run, when available is False
    requires: Optional[str] = None
    licence: Optional[str] = None
    neural: bool = False
    #: Regions whose licence excludes the engine (from the provider spec).
    territory_blocked: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _live_portrait_reason() -> Optional[str]:
    """None when the neural path can run, else the reason it cannot."""
    try:
        from app.services.video.lip_sync.live_portrait import LivePortraitProvider

        provider = LivePortraitProvider()
        if provider._get_weights_path() is None:
            return "LivePortrait weights are not installed (models/video/liveportrait)."
        if provider._audio_entry_point() is None:
            return (
                "No audio-driven entry point found. Upstream LivePortrait is "
                "video-driven; install an audio wrapper (inference_audio.py) or set "
                "MILIMO_LIVEPORTRAIT_AUDIO_SCRIPT."
            )
        if not provider.is_available:
            return "No MPS/CUDA device is available for neural inference."
        return None
    except Exception as exc:  # pragma: no cover - defensive
        return f"LivePortrait could not be inspected: {exc}"


def engine_catalogue() -> List[LipSyncEngineInfo]:
    """Every engine the studio knows about, with real availability."""
    has_fal = bool(os.environ.get("FAL_KEY"))
    lp_reason = _live_portrait_reason()

    engines = [
        LipSyncEngineInfo(
            id=JAW_WARP,
            name="Procedural Jaw-Warp Animatic",
            kind="local",
            available=True,
            reason=None,
            requires=None,
            licence="Milimo (in-house, procedural)",
            neural=False,
        ),
        LipSyncEngineInfo(
            id=LIVE_PORTRAIT,
            name="LivePortrait (neural, audio-driven)",
            kind="local",
            available=lp_reason is None,
            reason=lp_reason,
            requires="models/video/liveportrait + audio entry point",
            licence="MIT code/weights; bundled InsightFace detectors are research-only",
            neural=True,
        ),
        LipSyncEngineInfo(
            id=CLOUD_FAL,
            name="fal.ai hosted lip sync",
            kind="cloud",
            available=has_fal,
            reason=None if has_fal else "FAL_KEY is not set.",
            requires="FAL_KEY",
            licence="Provider terms",
            neural=True,
        ),
        LipSyncEngineInfo(
            id=CLOUD_REPLICATE,
            name="Replicate hosted lip sync",
            kind="cloud",
            available=False,
            reason="Replicate was never wired (decision D4: cloud is fal only).",
            requires="—",
            licence="Provider terms",
            neural=True,
        ),
    ]

    # Plan 8.4: the external tiers are real engines with a declared contract.
    # Reporting them here is what makes "install the weights and it works" true.
    try:
        from app.services.video.lip_sync.cli_provider import (
            AVATAR_CLI_SPECS,
            spec_available,
            spec_reason,
        )

        for spec in AVATAR_CLI_SPECS.values():
            engines.append(LipSyncEngineInfo(
                id=spec.id,
                name=spec.name,
                kind="local",
                available=spec_available(spec),
                reason=spec_reason(spec),
                requires=f"{spec.env_var} → {spec.script}",
                licence=spec.licence,
                neural=True,
                territory_blocked=list(spec.territory_blocked) or None,
            ))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("External avatar engines not enumerated: %s", exc)

    return engines


def get_engine(engine_id: Optional[str]) -> Optional[LipSyncEngineInfo]:
    wanted = (engine_id or "").strip().lower()
    wanted = ALIASES.get(wanted, wanted)
    for info in engine_catalogue():
        if info.id == wanted:
            return info
    return None


def resolve_local_provider(preferred: Optional[str]) -> Tuple[Any, Optional[str]]:
    """Return `(provider, fallback_reason)` for a local render.

    `fallback_reason` is None when the requested engine will actually run; it is a
    sentence when the jaw-warp animatic is used instead, so the task status can say
    so before the first clip is rendered.
    """
    from app.services.video.lip_sync.fallback import SmoothVisemeFallbackProvider

    wanted = (preferred or LIVE_PORTRAIT).strip().lower()
    wanted = ALIASES.get(wanted, wanted)
    if wanted == JAW_WARP:
        return SmoothVisemeFallbackProvider(), None

    info = get_engine(wanted)
    if info is None:
        return (
            SmoothVisemeFallbackProvider(),
            f"Unknown lip-sync engine '{preferred}'; used the jaw-warp animatic.",
        )
    if info.kind == "cloud":
        # Cloud engines are chosen by provider, not by this field.
        return (
            SmoothVisemeFallbackProvider(),
            f"'{info.name}' is a cloud engine; select the cloud provider to use it. "
            f"Used the jaw-warp animatic.",
        )
    if not info.available:
        return (
            SmoothVisemeFallbackProvider(),
            f"{info.name} unavailable: {info.reason} Used the jaw-warp animatic.",
        )

    # External tiers run through their own CLI contract (plan 8.4).
    from app.services.video.lip_sync.cli_provider import provider_for

    external = provider_for(wanted)
    if external is not None:
        return external, None

    from app.services.video.lip_sync.live_portrait import LivePortraitProvider

    return LivePortraitProvider(), None


def resolve_cloud_provider(provider_type: Optional[str]):
    """The hosted lip-sync provider matching the render's provider, if any."""
    from app.services.video.lip_sync.cloud_lipsync import CloudLipSyncProvider

    if provider_type == "cloud_fal":
        return CloudLipSyncProvider("fal")
    if provider_type == "cloud_replicate":
        return CloudLipSyncProvider("replicate")
    return None
