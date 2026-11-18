"""characters — reusable performer identity (plan 8.3)

Created 2026-11-18. Consistency was per-render config; a character is the
description plus a reference plate that survives across songs.

Guarded like the other migrations: safe on a database that already has the table
(legacy `create_all` bootstrap) and on one that does not.
"""

from alembic import op
import sqlalchemy as sa

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table("character"):
        return
    op.create_table(
        "character",
        sa.Column("id", sa.CHAR(32), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("plate_path", sa.String(), nullable=True),
        sa.Column("plate_sha256", sa.String(), nullable=True),
        sa.Column("lora_path", sa.String(), nullable=True),
        sa.Column("owner_id", sa.String(), nullable=False, server_default="local"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_character_name ON character (name)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_character_owner_id ON character (owner_id)")


def downgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table("character"):
        op.drop_table("character")
