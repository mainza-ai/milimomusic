"""indexes and uniqueness constraints the hot paths need

Created 2026-11-16.

`/history` sorts by `created_at` and filters on `status` with no index, the boot
reconciler scans `status`, project/session listings scan on unindexed columns that
were declared `index=True` but never materialised on existing tables, and
`playlisttrack` / `agentassignment` accepted duplicate rows.

Every statement is idempotent (`IF NOT EXISTS`) because this migration runs
against databases created by `create_all`, which already made some of them.
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_job_status_created ON job (status, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS ix_job_created_at ON job (created_at DESC)",
    "CREATE INDEX IF NOT EXISTS ix_job_project_id ON job (project_id)",
    "CREATE INDEX IF NOT EXISTS ix_job_session_id ON job (session_id)",
    "CREATE INDEX IF NOT EXISTS ix_job_release_id ON job (release_id)",
    "CREATE INDEX IF NOT EXISTS ix_job_artist_profile_id ON job (artist_profile_id)",
    "CREATE INDEX IF NOT EXISTS ix_agentrun_parent_run_id ON agentrun (parent_run_id)",
    "CREATE INDEX IF NOT EXISTS ix_agentrun_status ON agentrun (status)",
]

# Duplicates must be removed before the unique index can be created.
DEDUPE = [
    """DELETE FROM playlisttrack WHERE rowid NOT IN (
           SELECT MIN(rowid) FROM playlisttrack GROUP BY playlist_id, job_id)""",
    """DELETE FROM agentassignment WHERE rowid NOT IN (
           SELECT MIN(rowid) FROM agentassignment GROUP BY profile_id, role)""",
]

UNIQUE_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_playlist_track ON playlisttrack (playlist_id, job_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_agent_assignment_role ON agentassignment (profile_id, role)",
]


def _table_exists(bind, name: str) -> bool:
    from sqlalchemy import inspect

    return inspect(bind).has_table(name)


def upgrade() -> None:
    bind = op.get_bind()
    for stmt in INDEXES:
        op.execute(stmt)

    for table, stmt in zip(("playlisttrack", "agentassignment"), DEDUPE):
        if _table_exists(bind, table):
            op.execute(stmt)

    for table, stmt in zip(("playlisttrack", "agentassignment"), UNIQUE_INDEXES):
        if _table_exists(bind, table):
            op.execute(stmt)


def downgrade() -> None:
    for stmt in (
        "DROP INDEX IF EXISTS ux_agent_assignment_role",
        "DROP INDEX IF EXISTS ux_playlist_track",
        "DROP INDEX IF EXISTS ix_agentrun_status",
        "DROP INDEX IF EXISTS ix_agentrun_parent_run_id",
        "DROP INDEX IF EXISTS ix_job_artist_profile_id",
        "DROP INDEX IF EXISTS ix_job_release_id",
        "DROP INDEX IF EXISTS ix_job_session_id",
        "DROP INDEX IF EXISTS ix_job_project_id",
        "DROP INDEX IF EXISTS ix_job_created_at",
        "DROP INDEX IF EXISTS ix_job_status_created",
    ):
        op.execute(stmt)
