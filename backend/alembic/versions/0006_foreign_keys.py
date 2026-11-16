"""foreign keys on the association columns + orphan repair

Created 2026-11-16.

Only `SessionMessage.session_id` had a real foreign key; every other association was
a bare string, so referential integrity depended on ~40 lines of hand-written
cascade code and the live database already held orphans (6 `job.session_id`,
2 `job.release_id`, 3 `release.profile_id`, 2 `playlisttrack`).

This revision:
  1. nulls the orphans (a dangling pointer is worse than an explicit absence),
  2. rebuilds the affected tables via SQLite batch mode with the FKs and
     `ON DELETE SET NULL` / `CASCADE` semantics the code was emulating by hand.

SQLite cannot ALTER-in a constraint, so batch mode recreates each table. The
pre-migration backup taken by `app.core.migrations` is what makes that safe.
"""

from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def upgrade() -> None:
    bind = op.get_bind()

    # 1. Repair: a pointer to a row that no longer exists must become NULL, not lie.
    if _has_table(bind, "job"):
        op.execute("""
            UPDATE job SET session_id = NULL
            WHERE session_id IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM session s WHERE s.id = job.session_id)
        """)
        op.execute("""
            UPDATE job SET project_id = NULL
            WHERE project_id IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM project p WHERE p.id = job.project_id)
        """)
        op.execute("""
            UPDATE job SET release_id = NULL
            WHERE release_id IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM release r WHERE r.id = job.release_id)
        """)

    if _has_table(bind, "release"):
        # `release.profile_id` is NOT NULL, so an orphan cannot be nulled — and
        # deleting a user's album container to satisfy a constraint would be
        # destructive. Re-home them on a clearly-labelled recovery profile instead.
        orphan_count = bind.execute(sa.text("""
            SELECT COUNT(*) FROM release r
            WHERE r.profile_id IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM artistprofile p WHERE p.id = r.profile_id)
        """)).scalar() or 0
        if orphan_count:
            # `artistprofile` has several NOT NULL text columns.
            op.execute("""
                INSERT INTO artistprofile
                    (id, name, bio, lore_json, tags, created_at, updated_at)
                SELECT 'recovered-profiles', 'Recovered releases',
                       'Album containers whose artist profile was deleted before foreign keys existed.',
                       '{}', '[]', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                WHERE NOT EXISTS (SELECT 1 FROM artistprofile WHERE id = 'recovered-profiles')
            """)
            op.execute("""
                UPDATE release SET profile_id = 'recovered-profiles'
                WHERE profile_id IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM artistprofile p WHERE p.id = release.profile_id)
            """)

    if _has_table(bind, "playlisttrack"):
        op.execute("""
            DELETE FROM playlisttrack
            WHERE NOT EXISTS (SELECT 1 FROM job j WHERE j.id = playlisttrack.job_id)
        """)

    # 2. Rebuild with real constraints. Batch mode recreates the table, copies the
    #    rows, and recreates indexes.
    if _has_table(bind, "job"):
        with op.batch_alter_table("job", recreate="always") as batch:
            batch.create_foreign_key(
                "fk_job_project", "project", ["project_id"], ["id"], ondelete="SET NULL"
            )
            batch.create_foreign_key(
                "fk_job_session", "session", ["session_id"], ["id"], ondelete="SET NULL"
            )

    if _has_table(bind, "release"):
        with op.batch_alter_table("release", recreate="always") as batch:
            batch.create_foreign_key(
                "fk_release_profile", "artistprofile", ["profile_id"], ["id"], ondelete="CASCADE"
            )

    if _has_table(bind, "playlisttrack"):
        with op.batch_alter_table("playlisttrack", recreate="always") as batch:
            batch.create_foreign_key(
                "fk_playlist_track_job", "job", ["job_id"], ["id"], ondelete="CASCADE"
            )


def downgrade() -> None:
    bind = op.get_bind()
    if _has_table(bind, "job"):
        with op.batch_alter_table("job", recreate="always") as batch:
            batch.drop_constraint("fk_job_session", type_="foreignkey")
            batch.drop_constraint("fk_job_project", type_="foreignkey")
    if _has_table(bind, "playlisttrack"):
        with op.batch_alter_table("playlisttrack", recreate="always") as batch:
            batch.drop_constraint("fk_playlist_track_job", type_="foreignkey")
