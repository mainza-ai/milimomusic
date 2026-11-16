"""job.deleted_at — soft delete (plan 2.11)

Created 2026-11-16. A mistaken DELETE used to remove the row *and* the audio file;
`deleted_at` hides a track while `?purge=true` performs the irreversible removal.
"""

from alembic import op
import sqlalchemy as sa

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def _has_column(bind, table: str, column: str) -> bool:
    return column in {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table("job") and not _has_column(bind, "job", "deleted_at"):
        with op.batch_alter_table("job") as batch:
            batch.add_column(sa.Column("deleted_at", sa.DateTime(), nullable=True))
    op.execute("CREATE INDEX IF NOT EXISTS ix_job_deleted_at ON job (deleted_at)")


def downgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table("job") and _has_column(bind, "job", "deleted_at"):
        with op.batch_alter_table("job") as batch:
            batch.drop_column("deleted_at")
