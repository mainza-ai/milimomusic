"""
Schema migration runner with automatic backup.

The application no longer patches its own schema with `ALTER TABLE` on every
boot: `alembic upgrade head` is the only way the schema changes, and it is
preceded by a timestamped SQLite backup so a bad migration is always recoverable.

`ensure_schema()` is called once from the app lifespan.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("milimo.migrations")

BACKEND_DIR = Path(__file__).resolve().parents[2]
ALEMBIC_INI = BACKEND_DIR / "alembic.ini"
BACKUP_DIR_NAME = "db_backups"
DEFAULT_BACKUP_ROTATIONS = 5


def _db_path_from_url(url: str) -> Optional[Path]:
    try:
        if not str(url).startswith("sqlite"):
            return None
        raw = str(url).split("///", 1)[1]
        return Path(raw).resolve()
    except Exception:
        return None


def backup_database(db_path: Optional[Path], rotations: int = DEFAULT_BACKUP_ROTATIONS) -> Optional[Path]:
    """Copy the SQLite file with the backup API (safe while the DB is open)."""
    if not db_path or not db_path.exists() or db_path.stat().st_size == 0:
        return None

    backup_dir = db_path.parent / BACKUP_DIR_NAME
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = backup_dir / f"{db_path.stem}_{stamp}{db_path.suffix}"

    try:
        with sqlite3.connect(str(db_path)) as src, sqlite3.connect(str(dest)) as dst:
            src.backup(dst)
    except Exception as exc:
        logger.warning("Database backup failed (%s); continuing with migration.", exc)
        return None

    # Rotate: keep the newest N backups.
    backups = sorted(backup_dir.glob(f"{db_path.stem}_*{db_path.suffix}"), key=lambda p: p.stat().st_mtime)
    for old in backups[:-rotations] if rotations > 0 else []:
        try:
            old.unlink()
        except OSError:
            pass
    logger.info("Database backed up to %s", dest)
    return dest


def _alembic_config():
    from alembic.config import Config

    if not ALEMBIC_INI.exists():
        return None
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    return cfg


def _stamp_baseline_if_needed(cfg, db_path: Optional[Path]) -> None:
    """Adopt a pre-Alembic database by recording the baseline revision.

    An existing install has all the tables but no `alembic_version` row, so
    `upgrade head` would try to re-run 0001 (harmless but noisy). Stamping 0001
    first states the truth: "this schema is the baseline".
    """
    from alembic import command
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import create_engine, inspect

    if db_path is None:
        return
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            current = ctx.get_current_revision()
            has_tables = inspect(conn).has_table("job")
        if current is None and has_tables:
            command.stamp(cfg, "0001")
            logger.info("Adopted existing database as Alembic baseline (0001).")
    finally:
        engine.dispose()


def run_migrations(db_url: str) -> bool:
    """Apply pending migrations. Returns True when the schema is up to date."""
    cfg = _alembic_config()
    if cfg is None:
        logger.warning("alembic.ini not found; skipping migrations.")
        return False

    from alembic import command

    db_path = _db_path_from_url(db_url)
    backup_database(db_path)
    _stamp_baseline_if_needed(cfg, db_path)

    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")
    return True


def ensure_schema(engine, fallback_create_all=None) -> None:
    """Boot-time entry point: migrate, or fall back loudly."""
    try:
        run_migrations(str(engine.url))
        logger.info("Database schema is at head.")
    except Exception as exc:
        logger.error(
            "Alembic migration failed (%s). Falling back to create_all + the legacy "
            "column patch so the app can still start; FIX the migration before "
            "relying on this install.",
            exc,
            exc_info=True,
        )
        if fallback_create_all is not None:
            fallback_create_all()
