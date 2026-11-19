#!/usr/bin/env python3
"""Endpoint + bundle benchmark harness (plan 9.9).

Performance work that isn't measured regresses silently. This harness records the
numbers the audit cared about and can compare them against a stored baseline.

Usage:
    python scripts/bench_api.py                      # print the table
    python scripts/bench_api.py --write-baseline     # store today's numbers
    python scripts/bench_api.py --compare            # fail on >50% regression

Design notes:
  * Absolute thresholds are not portable (CI runners are slower than a dev laptop),
    so regression *comparison* is opt-in and ratio-based.
  * The 200 and 304 paths are measured separately: a 304 must be the cheap one.
  * The bundle size is read from `frontend/dist` when it exists, so a JS regression
    shows up next to the API numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Repo-relative import roots, so the harness works from a plain checkout without
#: the caller having to remember the PYTHONPATH the CI job exports.
for _rel in ("backend", "muscriptor", os.path.join("mulacover", "src")):
    _abs = os.path.join(ROOT, _rel)
    if os.path.isdir(_abs):
        sys.path.insert(0, _abs)

BASELINE_PATH = os.path.join(ROOT, "scripts", "bench_baseline.json")

ROUTES = [
    "/health",
    "/models/tree",
    "/videos/models",
    "/videos/presets",
    "/videos/capability-matrix",
    "/system/licences",
    "/characters",
]


def _measure(client, route: str, repeats: int = 5) -> Dict[str, Any]:
    """Median 200 time and median 304 time for one route."""
    times: List[float] = []
    etag = None
    for _ in range(repeats):
        start = time.perf_counter()
        resp = client.get(route)
        times.append((time.perf_counter() - start) * 1000)
        etag = etag or resp.headers.get("etag")

    not_modified: List[float] = []
    if etag:
        for _ in range(repeats):
            start = time.perf_counter()
            client.get(route, headers={"If-None-Match": etag})
            not_modified.append((time.perf_counter() - start) * 1000)

    times.sort()
    not_modified.sort()
    return {
        "status": resp.status_code,
        "median_ms": round(times[len(times) // 2], 2),
        "p95_ms": round(times[min(len(times) - 1, int(len(times) * 0.95))], 2),
        "304_ms": round(not_modified[len(not_modified) // 2], 2) if not_modified else None,
        "kb": round(len(resp.content) / 1024, 1),
    }


def _bundle_kb() -> float | None:
    dist = os.path.join(ROOT, "frontend", "dist", "assets")
    if not os.path.isdir(dist):
        return None
    total = sum(
        os.path.getsize(os.path.join(dist, name))
        for name in os.listdir(dist)
        if name.endswith(".js")
    )
    return round(total / 1024, 1)


def _preflight() -> Optional[str]:
    """The one failure here that is not a performance number: a missing submodule.

    `app.main` imports the MuScriptor provider at module load, so an uninitialised
    submodule surfaces as a bare `ModuleNotFoundError` that says nothing about the
    fix. Name the cause and the command instead (CI failure 2026-11-19).
    """
    if not os.path.isfile(os.path.join(ROOT, "muscriptor", "muscriptor", "__init__.py")):
        return (
            "bench: the `muscriptor` submodule is not checked out, so `app.main` "
            "cannot be imported.\n"
            "  local: git submodule update --init --recursive\n"
            "  CI:    actions/checkout with `submodules: recursive`"
        )
    return None


def run() -> Dict[str, Any]:
    os.environ.setdefault("MILIMO_DB_NAME", "bench_probe.db")
    from fastapi.testclient import TestClient

    from app.main import app, create_db_and_tables
    from app.services.model_manager import model_manager

    create_db_and_tables()
    model_manager.get_model_tree()          # what the boot warm-up does

    client = TestClient(app)
    results = {route: _measure(client, route) for route in ROUTES}
    return {"routes": results, "bundle_kb": _bundle_kb(), "at": time.time()}


def _print(report: Dict[str, Any]) -> None:
    print(f"{'route':28s} {'status':>6s} {'median':>8s} {'p95':>8s} {'304':>7s} {'size':>8s}")
    for route, m in report["routes"].items():
        nm = "-" if m["304_ms"] is None else f"{m['304_ms']:.1f}"
        print(f"{route:28s} {m['status']:>6d} {m['median_ms']:>7.1f}ms "
              f"{m['p95_ms']:>7.1f}ms {nm:>6s}ms {m['kb']:>7.1f}KB")
    if report.get("bundle_kb") is not None:
        print(f"{'frontend JS (dist)':28s} {'':>6s} {'':>8s} {'':>8s} {'':>7s} {report['bundle_kb']:>7.1f}KB")


def _compare(report: Dict[str, Any], max_regression: float) -> int:
    if not os.path.isfile(BASELINE_PATH):
        print(f"bench: no baseline at {BASELINE_PATH}; skipping comparison")
        return 0
    with open(BASELINE_PATH, encoding="utf-8") as fh:
        baseline = json.load(fh)

    failures = []
    for route, current in report["routes"].items():
        previous = (baseline.get("routes") or {}).get(route)
        if not previous or not previous.get("median_ms"):
            continue
        ratio = current["median_ms"] / previous["median_ms"]
        if ratio > 1 + max_regression:
            failures.append(
                f"{route}: {previous['median_ms']:.1f}ms -> {current['median_ms']:.1f}ms "
                f"({ratio:.2f}x)"
            )

    if failures:
        print("bench: REGRESSION")
        for line in failures:
            print(f"  {line}")
        return 1
    print("bench: no regressions beyond tolerance")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--max-regression", type=float, default=0.5)
    args = parser.parse_args()

    problem = _preflight()
    if problem:
        print(problem, file=sys.stderr)
        return 2

    report = run()
    _print(report)

    if args.write_baseline:
        with open(BASELINE_PATH, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"bench: baseline written to {BASELINE_PATH}")
        return 0
    if args.compare:
        return _compare(report, args.max_regression)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
