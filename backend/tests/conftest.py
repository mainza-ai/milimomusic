import os
import sys
from pathlib import Path
import pytest

# Ensure backend and muscriptor are in python path
backend_dir = Path(__file__).parent.parent
muscriptor_dir = backend_dir.parent / "muscriptor"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(muscriptor_dir) not in sys.path:
    sys.path.insert(0, str(muscriptor_dir))


# Test Database Isolation: ensure tests NEVER pollute the production jobs.db!
os.environ.setdefault("MILIMO_DB_NAME", "test_jobs.db")


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
