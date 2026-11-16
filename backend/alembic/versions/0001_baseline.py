"""baseline — the schema as it stood when Alembic was introduced

Created 2026-11-16. This revision is deliberately declarative rather than a
transcription of the old boot-time `ALTER TABLE` loop: it materialises the
SQLModel metadata, and it is a **no-op on an existing database** (every table is
already present, so `create_all(checkfirst=True)` skips them). Fresh databases
get the full schema from here.

Adopting Alembic on a live install:
    alembic stamp 0001     # record the current schema as the baseline
    alembic upgrade head   # apply anything newer
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    from sqlmodel import SQLModel

    import app.models  # noqa: F401  (registers every table on the metadata)

    bind = op.get_bind()
    SQLModel.metadata.create_all(bind=bind, checkfirst=True)


def downgrade() -> None:
    """The baseline cannot be reversed safely (it may have adopted a live DB)."""
    raise RuntimeError(
        "0001_baseline is irreversible by design: it may represent a pre-existing "
        "database. Restore from a backup instead."
    )
