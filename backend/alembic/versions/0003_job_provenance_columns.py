"""job provenance / heartbeat / retry columns + visualasset.engine

Created 2026-11-16.

`0001_baseline` materialises the SQLModel metadata with `create_all`, which does
**not** add columns to a table that already exists — so an adopted database (one
created by the old boot-time `ALTER TABLE` loop) needs an explicit additive
migration. Every step is guarded by an inspector check so the revision is safe to
run against fresh and existing databases alike.
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

JOB_COLUMNS = [
    ("provenance_json", sa.Text()),
    ("quality", sa.String()),
    ("last_heartbeat_at", sa.DateTime()),
    ("attempts", sa.Integer()),
    ("max_attempts", sa.Integer()),
]


def _columns(bind, table: str) -> set:
    return {c["name"] for c in sa.inspect(bind).get_columns(table)}


def _has_table(bind, table: str) -> bool:
    return sa.inspect(bind).has_table(table)


def upgrade() -> None:
    bind = op.get_bind()

    if _has_table(bind, "job"):
        existing = _columns(bind, "job")
        for name, coltype in JOB_COLUMNS:
            if name in existing:
                continue
            with op.batch_alter_table("job") as batch:
                batch.add_column(sa.Column(name, coltype, nullable=True))

    if _has_table(bind, "visualasset"):
        if "engine" not in _columns(bind, "visualasset"):
            with op.batch_alter_table("visualasset") as batch:
                batch.add_column(sa.Column("engine", sa.String(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    if _has_table(bind, "job"):
        existing = _columns(bind, "job")
        for name, _ in JOB_COLUMNS:
            if name not in existing:
                continue
            with op.batch_alter_table("job") as batch:
                batch.drop_column(name)
    if _has_table(bind, "visualasset"):
        if "engine" in _columns(bind, "visualasset"):
            with op.batch_alter_table("visualasset") as batch:
                batch.drop_column("engine")
