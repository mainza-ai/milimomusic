"""job.owner_id — single-user, multi-tenant-ready (decision D2)

Created 2026-11-16. Existing rows are stamped 'local'.
"""

from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def _has_column(bind, table: str, column: str) -> bool:
    return column in {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table("job"):
        return
    if not _has_column(bind, "job", "owner_id"):
        with op.batch_alter_table("job") as batch:
            batch.add_column(sa.Column("owner_id", sa.String(), nullable=True))
    op.execute("UPDATE job SET owner_id = 'local' WHERE owner_id IS NULL")
    op.execute("CREATE INDEX IF NOT EXISTS ix_job_owner_id ON job (owner_id)")


def downgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table("job") and _has_column(bind, "job", "owner_id"):
        with op.batch_alter_table("job") as batch:
            batch.drop_column("owner_id")
