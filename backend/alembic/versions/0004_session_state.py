"""job.session_state_json — server-side DAW session recall

Created 2026-11-16. Mix/loop/tempo/automation persisted per track so they survive a
cache clear or a different browser (they lived only in localStorage).
"""

from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def _has_column(bind, table: str, column: str) -> bool:
    return column in {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table("job") and not _has_column(bind, "job", "session_state_json"):
        with op.batch_alter_table("job") as batch:
            batch.add_column(sa.Column("session_state_json", sa.Text(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table("job") and _has_column(bind, "job", "session_state_json"):
        with op.batch_alter_table("job") as batch:
            batch.drop_column("session_state_json")
