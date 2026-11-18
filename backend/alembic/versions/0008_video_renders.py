"""videorender — one row per video render take (plan 7.4)

Created 2026-11-18. `job.video_path` was a single string, so every re-render
overwrote the previous master; takes keep each attempt's config, artifacts and QC
verdict so the UI can switch between them.

Guarded like the other migrations: safe on a database that already has the table
(legacy `create_all` bootstrap) and on one that does not.
"""

from alembic import op
import sqlalchemy as sa

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table("videorender"):
        return
    op.create_table(
        "videorender",
        sa.Column("id", sa.CHAR(32), primary_key=True),
        sa.Column("job_id", sa.CHAR(32), sa.ForeignKey("job.id", ondelete="CASCADE"), nullable=False),
        sa.Column("task_id", sa.String(), nullable=True),
        sa.Column("kind", sa.String(), nullable=False, server_default="advanced"),
        sa.Column("status", sa.String(), nullable=False, server_default="rendering"),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column("video_path", sa.String(), nullable=True),
        sa.Column("file_path", sa.String(), nullable=True),
        sa.Column("poster_path", sa.String(), nullable=True),
        sa.Column("proxy_path", sa.String(), nullable=True),
        sa.Column("qc_json", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("is_selected", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_videorender_job_id ON videorender (job_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_videorender_task_id ON videorender (task_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_videorender_kind ON videorender (kind)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_videorender_status ON videorender (status)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_videorender_is_selected ON videorender (is_selected)")


def downgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table("videorender"):
        op.drop_table("videorender")
