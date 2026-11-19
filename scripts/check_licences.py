#!/usr/bin/env python3
"""Licence gate for CI (plan 8.9b).

Milimo is a non-commercial, locally-run project, so a non-commercial default is
acceptable — but it must be **deliberate and visible**, not discovered later. This
gate fails when:

* a reachable default has an *unknown* licence (nobody has looked), or
* a reachable default is non-commercial and the build has not acknowledged it.

Acknowledging is explicit: `MILIMO_NONCOMMERCIAL_OK=1`. CI sets it for this project
but still prints every finding, so a new non-commercial default appears in the log.

Exit codes: 0 clean/acknowledged, 1 blocked.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def run() -> int:
    from app.core.licensing import (
        NON_COMMERCIAL,
        UNKNOWN,
        audit_catalogue,
        collect_catalogue,
        resolve_effective_defaults,
    )

    catalogue = collect_catalogue()
    if not catalogue:
        print("licence-gate: catalogue unavailable; skipping (no data to judge)")
        return 0

    report = audit_catalogue(catalogue, defaults=resolve_effective_defaults())
    counts = report["counts"]
    print("licence-gate: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"licence-gate: defaults={report['defaults']}")

    acknowledged = report.get("acknowledged", False)
    hard: list[dict] = []
    soft: list[dict] = []
    for entry in report["blockers"]:
        (soft if entry["class"] == NON_COMMERCIAL else hard).append(entry)

    for entry in report["blockers"]:
        marker = "ACK" if entry["class"] == NON_COMMERCIAL and acknowledged else "FAIL"
        print(
            f"  [{marker}] {entry['id']} ({entry['category']}): {entry['license']} "
            f"-> {entry['class']}, reachable as default"
        )

    if hard:
        print("licence-gate: FAILED — unclassified default licence(s); classify them in the catalogue.")
        return 1
    if soft and not acknowledged:
        print(
            "licence-gate: FAILED — non-commercial default. Replace it, or set "
            "MILIMO_NONCOMMERCIAL_OK=1 if this deployment is deliberately non-commercial."
        )
        return 1

    try:
        from app.core.licensing import audit_dependencies

        deps = audit_dependencies()
        for entry in deps["flagged"]:
            print(f"  [DEP] {entry['package']} {entry['version']}: {entry['reason']}")
        if deps["flagged"]:
            print(f"licence-gate: {len(deps['flagged'])} restricted package(s) installed (notice only)")
    except Exception as _dep_err:  # pragma: no cover - defensive
        print(f"licence-gate: dependency audit skipped ({_dep_err})")

    print("licence-gate: OK" + (" (non-commercial deployment acknowledged)" if soft else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
