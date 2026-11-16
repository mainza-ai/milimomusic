"""Alembic environment for Milimo Music.

The database URL always comes from the application engine, so `alembic upgrade`
and the running API can never point at different files. SQLite needs batch mode
for ALTER-heavy migrations, which is enabled per-operation in the revisions.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# `backend/` on the path so `app.*` imports resolve when invoked from the repo root.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

config = context.config

# Do NOT call `fileConfig` here: it reconfigures the *root* logger, which tears
# down the application's own handlers when migrations run inside the app lifespan
# (the symptom was the boot log stopping at "Will assume non-transactional DDL").
# Alembic's own loggers inherit from the app configuration instead.


def _database_url() -> str:
    """Resolve the URL the *application* uses."""
    try:
        from app.main import engine  # noqa: WPS433 (runtime import is the point)

        return str(engine.url)
    except Exception:
        db_name = os.environ.get("MILIMO_DB_NAME", "jobs.db")
        return f"sqlite:///{db_name}"


def _target_metadata():
    try:
        from sqlmodel import SQLModel

        import app.models  # noqa: F401  (registers the tables)

        return SQLModel.metadata
    except Exception:
        return None


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=_target_metadata(),
        literal_binds=True,
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=_target_metadata(),
            render_as_batch=True,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
