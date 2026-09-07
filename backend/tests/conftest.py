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


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Ensure all SQLModel tables and migrations exist before running tests."""
    from app.main import create_db_and_tables
    create_db_and_tables()
