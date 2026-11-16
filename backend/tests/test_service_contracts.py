"""Static contract test: every attribute call on our own service singletons must exist.

The audit found three call sites invoking methods that were never implemented —
`provider_registry.get(...)`, `task_queue.fail_task(...)`, `video_service.update_task(...)`.
Each failed *inside* an error or cancel handler, so the failure was swallowed and the
user-visible symptom was "the feature silently does nothing / the job never finishes".

Rather than pin three individual fixes, this test parses the source tree and asserts
that any `x.method(` where `x` is a known singleton resolves to a real attribute.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"

# Singleton names → the object they resolve to, and the module path that defines it.
SINGLETONS: dict[str, str] = {
    "provider_registry": "app.providers.registry:provider_registry",
    "task_queue": "app.core.task_queue:DurableTaskQueue",
    "music_service": "app.services.music_service:music_service",
    "image_service": "app.services.image_service:image_service",
    "video_service": "app.services.video_service:video_service",
    "video_orchestrator": "app.services.video.video_orchestrator:video_orchestrator",
    "lyric_sync_engine": "app.transcription.karaoke:lyric_sync_engine",
    "muscriptor_provider": "app.transcription.muscriptor_provider:muscriptor_provider",
}

# Objects that are legitimately dynamic (pydantic models, dicts) and excluded.
SKIP_ATTRS = {"get", "keys", "items", "values", "copy", "update", "pop", "setdefault", "append"}


def _resolve(spec: str):
    import importlib

    module_path, _, attr = spec.partition(":")
    module = importlib.import_module(module_path)
    obj = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _iter_calls():
    """Yield (file, lineno, singleton_name, method_name) for every `singleton.method(`."""
    for path in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # a broken file should fail loudly elsewhere
            pytest.fail(f"{path} does not parse: {exc}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if not isinstance(func.value, ast.Name):
                continue
            owner = func.value.id
            if owner not in SINGLETONS:
                continue
            if func.attr in ("__", ) or func.attr.startswith("__"):
                continue
            yield path, node.lineno, owner, func.attr


@pytest.mark.parametrize("singleton,spec", sorted(SINGLETONS.items()))
def test_singleton_methods_exist(singleton: str, spec: str):
    target = _resolve(spec)
    missing = []
    for path, lineno, owner, attr in _iter_calls():
        if owner != singleton:
            continue
        if not hasattr(target, attr):
            missing.append(f"{path.relative_to(APP_DIR.parent)}:{lineno} calls {singleton}.{attr}()")
    assert not missing, "Calls to non-existent methods:\n  " + "\n  ".join(missing)


def test_provider_registry_has_get_provider_not_get():
    """Regression guard for the in-paint bug: repair called `.get()`, which never existed."""
    from app.providers.registry import provider_registry

    assert hasattr(provider_registry, "get_provider")
    assert not hasattr(provider_registry, "get")


def test_durable_task_queue_public_surface():
    """The queue's public methods are what the orchestrators call; keep them real."""
    from app.core.task_queue import DurableTaskQueue

    for name in (
        "enqueue_task",
        "get_task",
        "update_task",
        "fail_task",
        "recover_interrupted_tasks",
        "create_task_workspace",
    ):
        assert callable(getattr(DurableTaskQueue, name, None)), f"DurableTaskQueue.{name} is missing"
