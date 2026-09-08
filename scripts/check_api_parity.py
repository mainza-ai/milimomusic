#!/usr/bin/env python3
"""API <-> UI parity gate for Milimo Music.

Fails (exit 1) when:
  1. a backend route has no frontend caller (axios, fetch, EventSource,
     window.open/href/src URL builders), unless explicitly exempted, or
  2. a frontend call targets a route the backend does not define
     (guaranteed runtime 404, e.g. the Training Studio gaps of 2026-09).

Run:  python3 scripts/check_api_parity.py  (from repo root)
Exemptions live in EXEMPT_ROUTES with a reason — adding one requires intent,
not accident.
"""

from __future__ import annotations

import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MAIN_PY = REPO_ROOT / "backend" / "app" / "main.py"
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"

# (method, normalized path) -> reason. Ops-only or intentionally UI-less routes.
EXEMPT_ROUTES: dict[tuple[str, str], str] = {
    ("DELETE", "/agents/runs"): "ops retention prune (MILIMO_RUN_RETENTION_DAYS); intentionally UI-less",
}

_AXIOS_RE = re.compile(r"axios\.(get|post|put|patch|delete)(?:<[^>]*>)?\(\s*[`'\"]([^`'\"]+)[`'\"]")
_FETCH_RE = re.compile(r"(?:fetch|EventSource)\(\s*[`'\"]([^`'\"]+)[`'\"]")
# Plain template URL builders (window.open / href / src / returned URLs).
# Applied per-line, skipping axios.* lines (those carry their real method).
# The middle alternative covers a slash glued to an interpolation (`/${x}`).
_URL_LITERAL_RE = re.compile(r"\$\{API_BASE_URL\}((?:/\$\{[^}]+\}|/[A-Za-z0-9_.~%/\-]+|\$\{[^}]+\})+)")
# VITE-based base URLs resolve to the same API; rewrite so one matcher fits.
_VITE_ENV_RE = re.compile(r"\$\{import\.meta\.env\.VITE_API_URL[^}]*\}")

# Served by static mounts (RangedStaticFiles), not @app routes — client paths
# under these prefixes are covered by definition.
STATIC_PREFIXES = ("/audio/", "/covers/")


def _norm(path: str) -> str:
    path = re.sub(r"\$\{[^}]+\}", "{id}", path)
    path = re.sub(r"\{[^}]+\}", "{id}", path)
    return path


def backend_routes() -> set[tuple[str, str]]:
    text = MAIN_PY.read_text()
    out = set()
    for m in re.finditer(r"@app\.(get|post|put|patch|delete|websocket)\(\s*\"([^\"]+)\"", text):
        out.add((m.group(1).upper(), _norm(m.group(2))))
    return out


def frontend_calls() -> set[tuple[str, str]]:
    out = set()
    for f in list(FRONTEND_SRC.rglob("*.ts")) + list(FRONTEND_SRC.rglob("*.tsx")):
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        for m in _AXIOS_RE.finditer(text):
            url = m.group(2).replace("${API_BASE_URL}", "")
            out.add((m.group(1).upper(), _norm(url)))
        for m in _FETCH_RE.finditer(text):
            url = m.group(1).replace("${API_BASE_URL}", "")
            if url.startswith("/") or url.startswith("http"):
                out.add(("GET", _norm(url)))
        # Plain template URL builders (window.open / href / src / returns).
        # Skip axios.* lines: those templates carry their real HTTP method.
        for line in text.splitlines():
            if "axios." in line:
                continue
            line = _VITE_ENV_RE.sub("${API_BASE_URL}", line)
            for m in _URL_LITERAL_RE.finditer(line):
                out.add(("GET", _norm(m.group(1))))
    return out


def _template_covers(route: tuple[str, str], call: tuple[str, str]) -> bool:
    """True when a client call resolves to a backend route.

    `{id}` segments on either side match any single concrete segment, so
    `/transcribe/export/{id}/midi` (format filled in by the caller) matches
    the parameterized `/transcribe/export/{id}/{id}` route.
    """
    if route[0] != call[0]:
        return False
    rsegs = route[1].strip("/").split("/")
    csegs = call[1].strip("/").split("/")
    if len(rsegs) != len(csegs):
        return False
    return all(r == "{id}" or c == "{id}" or r == c for r, c in zip(rsegs, csegs))


def check() -> tuple[list, list]:
    routes = backend_routes()
    calls = frontend_calls()
    orphan_routes = sorted(
        r for r in routes
        if r not in EXEMPT_ROUTES and not any(_template_covers(r, c) for c in calls)
    )
    dangling_calls = sorted(
        c for c in calls
        if not c[1].startswith(STATIC_PREFIXES)
        and not any(_template_covers(r, c) for r in routes)
    )
    return orphan_routes, dangling_calls


def main() -> int:
    orphan_routes, dangling_calls = check()
    if not orphan_routes and not dangling_calls:
        print(f"parOK: {len(backend_routes())} routes, all called; "
              f"{len(frontend_calls())} client calls, all resolve.")
        return 0
    if orphan_routes:
        print("ORPHAN BACKEND ROUTES (no frontend caller, not exempt):")
        for method, path in orphan_routes:
            print(f"  {method:6} {path}")
    if dangling_calls:
        print("DANGLING CLIENT CALLS (no backend route -> runtime 404):")
        for method, path in dangling_calls:
            print(f"  {method:6} {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
