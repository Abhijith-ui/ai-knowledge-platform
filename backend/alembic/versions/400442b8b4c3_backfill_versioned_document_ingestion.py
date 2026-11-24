"""Backfill documents into versioned ingestion.

Revision ID: 400442b8b4c3
Revises: 230b4a4d1e68
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "400442b8b4c3"
down_revision: Union[str, Sequence[str], None] = "230b4a4d1e68"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

MIGRATION_MARKER = "400442b8b4c3"


def upgrade() -> None:
    """Migrate extracted content to run-scoped storage."""

    connection = op.get_bind()

    # 1. Add the composite unique constraint required by the
    # documents.active_ingestion_run_id foreign key.
    op.create_unique_constraint(
        "uq_ingestion_runs_id_document_id",
        "ingestion_runs",
        ["id", "document_id"],
    )

    # 2. Add the nullable active run reference.
    op.add_column(
        "documents",
        sa.Column("active_ingestion_run_id", sa.Uuid(), nullable=True),
    )

    # 3. Build an exact mapping for the historical runs created here.
    op.execute(sa.text("""
            CREATE TEMPORARY TABLE _migration_run_map (
                document_id UUID PRIMARY KEY,
                ingestion_run_id UUID NOT NULL UNIQUE
            ) ON COMMIT DROP
        """))

    op.execute(sa.text("""
            INSERT INTO _migration_run_map (
                document_id,
                ingestion_run_id
            )
            SELECT
                d.id,
                gen_random_uuid()
            FROM documents AS d
            INNER JOIN document_contents AS dc
                ON dc.document_id = d.id
        """))

    # 4. Create one historical PARTIAL run for each document with content.
    op.execute(sa.text("""
            INSERT INTO ingestion_runs (
                id,
                document_id,
                run_number,
                status,
                trigger_type,
                config_snapshot,
                started_at,
                completed_at,
                created_at,
                updated_at
            )
            SELECT
                mapping.ingestion_run_id,
                d.id,
                COALESCE(
                    (
                        SELECT MAX(existing.run_number)
                        FROM ingestion_runs AS existing
                        WHERE existing.document_id = d.id
                    ),
                    0
                ) + 1,
                'partial',
                'migration',
                jsonb_build_object(
                    'migration_revision',
                    :migration_marker
                ),
                COALESCE(
                    d.processing_started_at,
                    d.processed_at,
                    dc.created_at
                ),
                COALESCE(d.processed_at, dc.updated_at),
                d.created_at,
                d.updated_at
            FROM _migration_run_map AS mapping
            INNER JOIN documents AS d
                ON d.id = mapping.document_id
            INNER JOIN document_contents AS dc
                ON dc.document_id = d.id
        """).bindparams(migration_marker=MIGRATION_MARKER))

    # 5. Create five historical stage records for each mapped run.
    op.execute(sa.text("""
            INSERT INTO ingestion_stage_runs (
                id,
                ingestion_run_id,
                stage,
                status,
                attempt_count,
                started_at,
                completed_at,
                metrics,
                created_at,
                updated_at
            )
            SELECT
                gen_random_uuid(),
                mapping.ingestion_run_id,
                stages.stage::ingestion_stage,
                CASE
                    WHEN stages.stage = 'extraction'
                        THEN 'succeeded'
                    ELSE 'skipped'
                END::ingestion_stage_run_status,
                CASE
                    WHEN stages.stage = 'extraction'
                        THEN GREATEST(COALESCE(d.processing_attempts, 0), 1)
                    ELSE 0
                END,
                CASE
                    WHEN stages.stage = 'extraction'
                        THEN ir.started_at
                    ELSE NULL
                END,
                CASE
                    WHEN stages.stage = 'extraction'
                        THEN ir.completed_at
                    ELSE NULL
                END,
                '{}'::jsonb,
                ir.created_at,
                ir.updated_at
            FROM _migration_run_map AS mapping
            INNER JOIN ingestion_runs AS ir
                ON ir.id = mapping.ingestion_run_id
            INNER JOIN documents AS d
                ON d.id = mapping.document_id
            CROSS JOIN (
                VALUES
                    ('extraction'),
                    ('structure_analysis'),
                    ('chunking'),
                    ('embedding'),
                    ('vector_indexing')
            ) AS stages(stage)
        """))

    # 6. Add new content identity and run reference.
    op.add_column(
        "document_contents",
        sa.Column("id", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "document_contents",
        sa.Column("ingestion_run_id", sa.Uuid(), nullable=True),
    )

    # 7. Associate every legacy content record with its exact mapped run.
    op.execute(sa.text("""
            UPDATE document_contents AS dc
            SET
                id = gen_random_uuid(),
                ingestion_run_id = mapping.ingestion_run_id
            FROM _migration_run_map AS mapping
            WHERE mapping.document_id = dc.document_id
        """))

    # 8. Preserve the original mapping independently.
    # Intentionally no foreign keys: the ledger must survive cascades.
    op.create_table(
        "document_ingestion_migration_ledger",
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("ingestion_run_id", sa.Uuid(), nullable=False),
        sa.Column("content_id", sa.Uuid(), nullable=False),
        sa.Column("migration_revision", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint(
            "document_id",
            name="pk_document_ingestion_migration_ledger",
        ),
        sa.UniqueConstraint(
            "ingestion_run_id",
            name="uq_document_ingestion_migration_ledger_run",
        ),
        sa.UniqueConstraint(
            "content_id",
            name="uq_document_ingestion_migration_ledger_content",
        ),
    )

    op.execute(sa.text("""
            INSERT INTO document_ingestion_migration_ledger (
                document_id,
                ingestion_run_id,
                content_id,
                migration_revision
            )
            SELECT
                mapping.document_id,
                mapping.ingestion_run_id,
                dc.id,
                :migration_marker
            FROM _migration_run_map AS mapping
            INNER JOIN document_contents AS dc
                ON dc.ingestion_run_id = mapping.ingestion_run_id
        """).bindparams(migration_marker=MIGRATION_MARKER))

    # 9. Validate the migration before replacing the old relationship.
    unmigrated_content = connection.execute(sa.text("""
            SELECT COUNT(*)
            FROM document_contents
            WHERE id IS NULL OR ingestion_run_id IS NULL
        """)).scalar_one()

    if unmigrated_content:
        raise RuntimeError(
            f"Migration aborted: {unmigrated_content} document content "
            "record(s) could not be associated with an ingestion run."
        )

    invalid_run_count = connection.execute(sa.text("""
            SELECT COUNT(*)
            FROM _migration_run_map AS mapping
            LEFT JOIN ingestion_runs AS ir
                ON ir.id = mapping.ingestion_run_id
            WHERE ir.id IS NULL
        """)).scalar_one()

    if invalid_run_count:
        raise RuntimeError(
            "Migration aborted: one or more historical ingestion runs "
            "were not created."
        )

    invalid_stage_count = connection.execute(sa.text("""
            SELECT COUNT(*)
            FROM _migration_run_map AS mapping
            WHERE
                (
                    SELECT COUNT(*)
                    FROM ingestion_stage_runs AS stage_run
                    WHERE stage_run.ingestion_run_id =
                        mapping.ingestion_run_id
                ) <> 5
                OR NOT EXISTS (
                    SELECT 1
                    FROM ingestion_stage_runs AS stage_run
                    WHERE stage_run.ingestion_run_id =
                        mapping.ingestion_run_id
                      AND stage_run.stage = 'extraction'
                      AND stage_run.status = 'succeeded'
                )
                OR (
                    SELECT COUNT(*)
                    FROM ingestion_stage_runs AS stage_run
                    WHERE stage_run.ingestion_run_id =
                        mapping.ingestion_run_id
                      AND stage_run.status = 'skipped'
                ) <> 4
        """)).scalar_one()

    if invalid_stage_count:
        raise RuntimeError(
            "Migration aborted: historical ingestion stage validation failed."
        )

    # Verify the ledger contains exactly one entry for every mapped document.
    invalid_ledger_count = connection.execute(sa.text("""
            SELECT COUNT(*)
            FROM _migration_run_map AS mapping
            LEFT JOIN document_ingestion_migration_ledger AS ledger
                ON ledger.document_id = mapping.document_id
               AND ledger.ingestion_run_id = mapping.ingestion_run_id
            WHERE ledger.document_id IS NULL
        """)).scalar_one()

    if invalid_ledger_count:
        raise RuntimeError("Migration aborted: migration ledger validation failed.")

    active_historical_runs = connection.execute(sa.text("""
            SELECT COUNT(*)
            FROM documents AS d
            INNER JOIN _migration_run_map AS mapping
                ON mapping.document_id = d.id
            WHERE d.active_ingestion_run_id = mapping.ingestion_run_id
        """)).scalar_one()

    if active_historical_runs:
        raise RuntimeError(
            "Migration aborted: historical partial runs must not be active."
        )

    # 10. Replace the old document-based primary key.
    op.drop_constraint(
        "document_contents_pkey",
        "document_contents",
        type_="primary",
    )

    op.drop_constraint(
        "document_contents_document_id_fkey",
        "document_contents",
        type_="foreignkey",
    )

    op.drop_column("document_contents", "document_id")

    # 11. Enforce the new content identity.
    op.alter_column(
        "document_contents",
        "id",
        existing_type=sa.Uuid(),
        nullable=False,
    )
    op.alter_column(
        "document_contents",
        "ingestion_run_id",
        existing_type=sa.Uuid(),
        nullable=False,
    )

    op.create_primary_key(
        "document_contents_pkey",
        "document_contents",
        ["id"],
    )

    op.create_unique_constraint(
        "uq_document_contents_ingestion_run_id",
        "document_contents",
        ["ingestion_run_id"],
    )

    op.create_foreign_key(
        "document_contents_ingestion_run_id_fkey",
        "document_contents",
        "ingestion_runs",
        ["ingestion_run_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # 12. Add the deferred composite active-run foreign key.
    op.create_foreign_key(
        "fk_documents_active_ingestion_run",
        "documents",
        "ingestion_runs",
        ["active_ingestion_run_id", "id"],
        ["id", "document_id"],
        ondelete="NO ACTION",
        deferrable=True,
        initially="DEFERRED",
    )


def downgrade() -> None:
    """Restore document-scoped content and remove migration-created runs."""

    connection = op.get_bind()

    # 1. Refuse downgrade if application-created ingestion runs exist.
    non_migration_runs = connection.execute(sa.text("""
            SELECT COUNT(*)
            FROM ingestion_runs
            WHERE config_snapshot->>'migration_revision'
                IS DISTINCT FROM :migration_marker
        """).bindparams(migration_marker=MIGRATION_MARKER)).scalar_one()

    if non_migration_runs:
        raise RuntimeError(
            "Cannot downgrade: application-created ingestion runs exist."
        )

    # 2. Validate the independent migration ledger.
    #
    # Every ledger row must still have its original document, run, and content.
    # Their relationships and migration provenance must also remain unchanged.
    invalid_ledger_entries = connection.execute(sa.text("""
            SELECT COUNT(*)
            FROM document_ingestion_migration_ledger AS ledger
            LEFT JOIN documents AS d
                ON d.id = ledger.document_id
            LEFT JOIN ingestion_runs AS ir
                ON ir.id = ledger.ingestion_run_id
            LEFT JOIN document_contents AS dc
                ON dc.id = ledger.content_id
            WHERE ledger.migration_revision = :migration_marker
              AND (
                    d.id IS NULL
                    OR ir.id IS NULL
                    OR dc.id IS NULL
                    OR ir.document_id IS DISTINCT FROM ledger.document_id
                    OR dc.ingestion_run_id IS DISTINCT FROM ledger.ingestion_run_id
                    OR ir.config_snapshot->>'migration_revision'
                        IS DISTINCT FROM :migration_marker
                    OR ir.trigger_type IS DISTINCT FROM 'migration'
              )
        """).bindparams(migration_marker=MIGRATION_MARKER)).scalar_one()

    if invalid_ledger_entries:
        raise RuntimeError(
            "Cannot downgrade: migration ledger references missing or "
            "modified historical data."
        )

    # Every migration-created run must have a ledger entry.
    unrecorded_runs = connection.execute(sa.text("""
            SELECT COUNT(*)
            FROM ingestion_runs AS ir
            LEFT JOIN document_ingestion_migration_ledger AS ledger
                ON ledger.ingestion_run_id = ir.id
               AND ledger.migration_revision = :migration_marker
            WHERE ir.config_snapshot->>'migration_revision' = :migration_marker
              AND ledger.ingestion_run_id IS NULL
        """).bindparams(migration_marker=MIGRATION_MARKER)).scalar_one()

    if unrecorded_runs:
        raise RuntimeError(
            "Cannot downgrade: migration-created ingestion runs are missing "
            "from the migration ledger."
        )

    # 3. Verify migration-created runs retain their original state.
    invalid_migration_runs = connection.execute(sa.text("""
            SELECT COUNT(*)
            FROM ingestion_runs AS ir
            WHERE ir.config_snapshot->>'migration_revision' = :migration_marker
              AND (
                    ir.status <> 'partial'
                    OR ir.trigger_type <> 'migration'
                    OR (
                        SELECT COUNT(*)
                        FROM document_contents AS dc
                        WHERE dc.ingestion_run_id = ir.id
                    ) <> 1
                    OR (
                        SELECT COUNT(*)
                        FROM ingestion_stage_runs AS sr
                        WHERE sr.ingestion_run_id = ir.id
                    ) <> 5
                    OR (
                        SELECT COUNT(*)
                        FROM ingestion_stage_runs AS sr
                        WHERE sr.ingestion_run_id = ir.id
                          AND (
                                (
                                    sr.stage = 'extraction'
                                    AND sr.status = 'succeeded'
                                )
                                OR (
                                    sr.stage IN (
                                        'structure_analysis',
                                        'chunking',
                                        'embedding',
                                        'vector_indexing'
                                    )
                                    AND sr.status = 'skipped'
                                )
                          )
                    ) <> 5
              )
        """).bindparams(migration_marker=MIGRATION_MARKER)).scalar_one()

    if invalid_migration_runs:
        raise RuntimeError(
            "Cannot downgrade: migration-created ingestion data " "has been modified."
        )

    # 4. Refuse downgrade if multiple content versions exist
    # for the same document.
    duplicate_versions = connection.execute(sa.text("""
            SELECT ir.document_id
            FROM document_contents AS dc
            INNER JOIN ingestion_runs AS ir
                ON ir.id = dc.ingestion_run_id
            GROUP BY ir.document_id
            HAVING COUNT(*) > 1
            LIMIT 1
        """)).first()

    if duplicate_versions is not None:
        raise RuntimeError(
            "Cannot downgrade: multiple content versions exist for a document."
        )

    # 5. Do not remove a run that has been published as active.
    active_migration_runs = connection.execute(sa.text("""
            SELECT COUNT(*)
            FROM documents AS d
            INNER JOIN ingestion_runs AS ir
                ON ir.id = d.active_ingestion_run_id
            WHERE ir.config_snapshot->>'migration_revision' = :migration_marker
        """).bindparams(migration_marker=MIGRATION_MARKER)).scalar_one()

    if active_migration_runs:
        raise RuntimeError(
            "Cannot downgrade: a migration-created run is currently active."
        )

    # 6. Restore document_id using the independent ledger.
    op.drop_constraint(
        "fk_documents_active_ingestion_run",
        "documents",
        type_="foreignkey",
    )

    op.drop_column("documents", "active_ingestion_run_id")

    op.drop_constraint(
        "document_contents_ingestion_run_id_fkey",
        "document_contents",
        type_="foreignkey",
    )

    op.drop_constraint(
        "uq_document_contents_ingestion_run_id",
        "document_contents",
        type_="unique",
    )

    op.drop_constraint(
        "document_contents_pkey",
        "document_contents",
        type_="primary",
    )

    op.add_column(
        "document_contents",
        sa.Column("document_id", sa.Uuid(), nullable=True),
    )

    op.execute(sa.text("""
            UPDATE document_contents AS dc
            SET document_id = ledger.document_id
            FROM document_ingestion_migration_ledger AS ledger
            WHERE ledger.content_id = dc.id
              AND ledger.migration_revision = :migration_marker
        """).bindparams(migration_marker=MIGRATION_MARKER))

    missing_document_ids = connection.execute(sa.text("""
            SELECT COUNT(*)
            FROM document_contents
            WHERE document_id IS NULL
        """)).scalar_one()

    if missing_document_ids:
        raise RuntimeError(
            "Cannot downgrade: some content records have no document mapping."
        )

    op.alter_column(
        "document_contents",
        "document_id",
        existing_type=sa.Uuid(),
        nullable=False,
    )

    op.create_primary_key(
        "document_contents_pkey",
        "document_contents",
        ["document_id"],
    )

    op.create_foreign_key(
        "document_contents_document_id_fkey",
        "document_contents",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.drop_column("document_contents", "ingestion_run_id")
    op.drop_column("document_contents", "id")

    # 7. Remove only records created by this specific migration.
    connection.execute(sa.text("""
            DELETE FROM ingestion_stage_runs AS stage_run
            USING ingestion_runs AS ir
            WHERE stage_run.ingestion_run_id = ir.id
              AND ir.config_snapshot->>'migration_revision' = :migration_marker
        """).bindparams(migration_marker=MIGRATION_MARKER))

    connection.execute(sa.text("""
            DELETE FROM ingestion_runs
            WHERE config_snapshot->>'migration_revision' = :migration_marker
        """).bindparams(migration_marker=MIGRATION_MARKER))

    # 8. Remove the ledger after successful restoration and cleanup.
    op.drop_table("document_ingestion_migration_ledger")

    # 9. Remove the composite unique constraint.
    op.drop_constraint(
        "uq_ingestion_runs_id_document_id",
        "ingestion_runs",
        type_="unique",
    )
