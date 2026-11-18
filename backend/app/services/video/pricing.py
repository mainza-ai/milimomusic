"""Cloud video cost estimation (plan 7.7).

A hosted render bills **per generated second**. The studio previously let a user
queue 40 scenes with no indication of spend (audit E3), which is not acceptable
for a private deployment where the user owns the bill.

Rates are conservative list prices captured 2026-11-18 and are *estimates*: the
provider's own pricelist is authoritative. Any engine we do not price returns
``usd: None`` with a note rather than a fabricated number.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

PRICING_AS_OF = "2026-11-18"
CURRENCY = "USD"

# usd per second of generated video, keyed by provider then model family.
_RATES: Dict[str, Dict[str, float]] = {
    # fal.ai (the only cloud provider wired in this build — decision D4)
    "fal": {
        "wan_14b": 0.05,
        "wan_1.3b": 0.02,
        "ltx_video": 0.04,
        "hailuo_h3": 0.10,
        "minimax_h3": 0.10,
        "kling": 0.16,
    },
    "minimax": {
        "video-01": 0.10,
    },
}

_DEFAULT_MODEL = {
    "fal": "wan_14b",
    "minimax": "video-01",
}

# Rendered video usually exceeds the delivered cut (lattice rounding, transitions).
DEFAULT_OVERHEAD = 1.10


def estimate_cloud_cost(
    service: str,
    clip_count: float,
    clip_seconds: float,
    model: Optional[str] = None,
    overhead: float = DEFAULT_OVERHEAD,
) -> Dict[str, Any]:
    """Estimate the cost of a cloud render.

    Args:
        service: provider key (``fal``, ``minimax``).
        clip_count: number of generated clips.
        clip_seconds: planned duration of each clip, in seconds.
        model: model family; defaults to the provider's standard model.
        overhead: multiplier covering lattice over-render and retries.
    """
    svc = (service or "").lower().strip()
    rates = _RATES.get(svc)
    seconds = max(0.0, float(clip_count) * float(clip_seconds)) * max(1.0, float(overhead))

    if not rates:
        return {
            "service": svc,
            "seconds": round(seconds, 2),
            "usd": None,
            "currency": CURRENCY,
            "note": f"No price table for '{svc}' — engine is not priced in this build.",
            "as_of": PRICING_AS_OF,
        }

    key = (model or _DEFAULT_MODEL.get(svc) or "").lower().strip()
    rate = rates.get(key)
    if rate is None:
        return {
            "service": svc,
            "model": key,
            "seconds": round(seconds, 2),
            "usd": None,
            "currency": CURRENCY,
            "note": f"'{key}' is not priced for {svc} — estimate not available.",
            "as_of": PRICING_AS_OF,
        }

    return {
        "service": svc,
        "model": key,
        "seconds": round(seconds, 2),
        "usd": round(seconds * rate, 2),
        "usd_per_second": rate,
        "currency": CURRENCY,
        "overhead": overhead,
        "note": "Estimate from list prices; the provider's invoice is authoritative.",
        "as_of": PRICING_AS_OF,
    }


def requires_confirmation(estimate: Dict[str, Any], threshold_usd: float = 5.0) -> bool:
    """True when a render should be confirmed before spending."""
    usd = estimate.get("usd")
    return bool(usd is not None and usd >= threshold_usd)


CONFIRMATION_THRESHOLD_USD = 5.0
