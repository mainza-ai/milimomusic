"""Licence classification and audit (audit finding G8).

The model catalogue has carried a `license` string per model for months, and nothing
ever read it. Cross-checking the current catalogue against Hugging Face metadata
found the product's *effective* default image model
(`flux_2_klein_9b`) under the **FLUX.2 Non-Commercial License**, while the 4B
sibling is Apache-2.0. That is exactly the failure mode this module exists to make
impossible to ship silently.

It classifies licences into what a deployer actually needs to know, and reports
which non-permissive models are reachable as a default.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("milimo.licensing")

PERMISSIVE = "permissive"
CONDITIONAL = "conditional"
NON_COMMERCIAL = "non_commercial"
UNKNOWN = "unknown"

#: Order matters: the first match wins, so the strictest signals are tested first.
_RULES = (
    (NON_COMMERCIAL, (
        r"non[-\s]?commercial", r"\bnc\b", r"cc[-\s]?by[-\s]?nc", r"research only",
        r"research-only", r"noncommercial",
    )),
    (PERMISSIVE, (
        r"apache[- ]2", r"\bmit\b", r"bsd", r"cc0", r"public domain",
    )),
    (CONDITIONAL, (
        r"community", r"openrail", r"cc[-\s]?by", r"stability ai", r"custom",
        r"open weights", r"llama", r"gemma", r"falcon",
    )),
)


def classify(licence: Optional[str]) -> str:
    """Classify a licence string as it is written in the catalogue.

    Unknown strings are `unknown`, never `permissive`: an unclassified licence is a
    reason to look, not a licence to ship.
    """
    text = (licence or "").strip().lower()
    if not text:
        return UNKNOWN
    for label, patterns in _RULES:
        for pattern in patterns:
            if re.search(pattern, text):
                return label
    return UNKNOWN


def is_commercially_usable(licence: Optional[str]) -> bool:
    """Conservative: only explicitly permissive or clearly conditional terms pass."""
    return classify(licence) in (PERMISSIVE, CONDITIONAL)


def _entry_lookup(entries: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(e.get("id")): e for e in entries if e.get("id")}


def audit_catalogue(
    entries: Iterable[Dict[str, Any]],
    *,
    defaults: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Audit a model catalogue.

    Args:
        entries: catalogue rows (`id`, `category`, `license`, `is_default`).
        defaults: `{category: model_id}` of the models the app actually resolves as
            defaults — which is not always what the catalogue marks `is_default`.

    Returns a report with per-model classes and a `blockers` list: models that are
    non-commercial or unknown **and** reachable as a default.
    """
    rows = list(entries)
    by_id = _entry_lookup(rows)
    resolved = dict(defaults or {})

    # The catalogue's own `is_default` flag counts as reachable too.
    for row in rows:
        if row.get("is_default") and row.get("category"):
            resolved.setdefault(str(row["category"]), str(row["id"]))

    models: List[Dict[str, Any]] = []
    blockers: List[Dict[str, Any]] = []
    for row in rows:
        licence = row.get("license")
        klass = classify(licence)
        model_id = str(row.get("id"))
        reachable = model_id in resolved.values()
        record = {
            "id": model_id,
            "category": row.get("category"),
            "license": licence,
            "class": klass,
            "reachable_as_default": reachable,
            "commercially_usable": is_commercially_usable(licence),
        }
        models.append(record)
        if reachable and klass in (NON_COMMERCIAL, UNKNOWN):
            blockers.append(record)

    counts: Dict[str, int] = {}
    for record in models:
        counts[record["class"]] = counts.get(record["class"], 0) + 1

    # Milimo is a non-commercial, locally-run project, so a non-permissive default is
    # a *notice*, not a failure — provided the deployment says so deliberately.
    acknowledged = os.environ.get("MILIMO_NONCOMMERCIAL_OK") == "1"

    return {
        "models": models,
        "defaults": resolved,
        "blockers": blockers,
        "counts": counts,
        "ok": (not blockers) or acknowledged,
        "acknowledged": acknowledged,
        "note": (
            "A blocker is a model that is non-commercial or unclassified AND reachable "
            "as a default. Replace it, or set MILIMO_NONCOMMERCIAL_OK=1 for a "
            "deliberately non-commercial deployment (which is this project's stated "
            "purpose)."
        ),
    }


def collect_catalogue() -> List[Dict[str, Any]]:
    """Read the live model catalogue without instantiating heavy services."""
    try:
        from app.services.model_manager import model_manager

        return list(model_manager.get_model_tree())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not read the model catalogue for the licence audit: %s", exc)
        return []


def resolve_effective_defaults() -> Dict[str, str]:
    """Best-effort: which model each category would actually use right now.

    The catalogue's `is_default` flag is not the whole story — `image_service`
    resolves its own default (which is how `flux_2_klein_9b` became the effective
    default while the catalogue flagged nothing).
    """
    defaults: Dict[str, str] = {}

    try:  # image: mirrors `ImageService._resolve_image_model`
        from app.services.image_service import image_service

        resolved = image_service._resolve_image_model(None)  # type: ignore[attr-defined]
        if isinstance(resolved, tuple) and resolved:
            defaults["image"] = str(resolved[0])
    except Exception as exc:
        logger.debug("Image default unresolved for licence audit: %s", exc)

    try:  # video: the active engine
        from app.services.video_service import video_service

        active = video_service.get_active_video_engine()
        model_id = (active or {}).get("model_id") or (active or {}).get("engine")
        if model_id:
            defaults["video"] = str(model_id)
    except Exception as exc:
        logger.debug("Video default unresolved for licence audit: %s", exc)

    return defaults
