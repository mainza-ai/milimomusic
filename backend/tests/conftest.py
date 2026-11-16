import os
import sys
from pathlib import Path
import pytest

# Ensure backend, muscriptor, and mulacover are in python path
backend_dir = Path(__file__).parent.parent
muscriptor_dir = backend_dir.parent / "muscriptor"
mulacover_dir = backend_dir.parent / "mulacover" / "src"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(muscriptor_dir) not in sys.path:
    sys.path.insert(0, str(muscriptor_dir))
if str(mulacover_dir) not in sys.path:
    sys.path.insert(0, str(mulacover_dir))


# Test Database Isolation: ensure tests NEVER pollute the production jobs.db!
os.environ.setdefault("MILIMO_DB_NAME", "test_jobs.db")
# Fast Test Mode: bypass slow live 35B LLM models and heavy unmocked diffusion during tests
os.environ.setdefault("MILIMO_FAST_TEST_MODE", "1")


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Ensure all SQLModel tables and migrations exist in test_jobs.db before running tests."""
    from app.main import create_db_and_tables, engine
    create_db_and_tables()
    yield
    # Clean up test database after test session
    engine.dispose()
    test_db = os.environ.get("MILIMO_DB_NAME", "test_jobs.db")
    for f in [test_db, f"{test_db}-wal", f"{test_db}-shm"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


@pytest.fixture(autouse=True)
def restore_model_registry_state():
    """Snapshot/restore the on-disk model registry around every test.

    Several tests activate, register and delete models through the real endpoints,
    which writes `data/models/active_models.json` and `custom_models.json`. Without
    this, one run leaves e.g. `video: custom_wan_ai_...` behind and the next run's
    engine-resolution assertions fail — tests must not mutate developer state.
    """
    from app.services import model_manager as mm

    paths = []
    for resolver in ("_get_custom_models_path", "_get_active_models_path"):
        fn = getattr(mm, resolver, None)
        if not callable(fn):
            continue
        try:
            paths.append(fn())
        except Exception:
            pass

    snapshots = {}
    for p in paths:
        try:
            with open(p, "rb") as fh:
                snapshots[p] = fh.read()
        except OSError:
            snapshots[p] = None

    yield

    for p, blob in snapshots.items():
        try:
            if blob is None:
                if os.path.exists(p):
                    os.remove(p)
            else:
                with open(p, "wb") as fh:
                    fh.write(blob)
        except OSError:
            pass


@pytest.fixture()
def client():
    """Shared ASGI test client.

    Several modules had grown their own copy; this is the same shape so tests can
    be added anywhere without re-declaring it.
    """
    import asyncio

    import httpx

    from app.main import app

    class _SyncClient:
        def __init__(self, asgi_app):
            self._asyncio = asyncio
            self.transport = httpx.ASGITransport(app=asgi_app)

        def _run(self, coro):
            loop = self._asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        def request(self, method, url, **kw):
            async def _go():
                async with httpx.AsyncClient(transport=self.transport, base_url="http://test") as c:
                    return await c.request(method, url, **kw)

            return self._run(_go())

        def get(self, url, **kw):
            return self.request("GET", url, **kw)

        def post(self, url, **kw):
            return self.request("POST", url, **kw)

        def put(self, url, **kw):
            return self.request("PUT", url, **kw)

        def patch(self, url, **kw):
            return self.request("PATCH", url, **kw)

        def delete(self, url, **kw):
            return self.request("DELETE", url, **kw)

    return _SyncClient(app)
