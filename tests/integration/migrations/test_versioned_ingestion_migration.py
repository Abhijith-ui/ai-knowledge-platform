from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import Connection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALEMBIC_CONFIG_PATH = PROJECT_ROOT / "backend" / "alembic.ini"

CURRENT_REVISION = "400442b8b4c3"
PREVIOUS_REVISION = "230b4a4d1e68"
MIGRATION_MARKER = "400442b8b4c3"

DOCUMENT_ID = "22222222-2222-4222-8222-222222222222"

# This URL must never be used. It detects accidental fallback to
# Alembic's independently created database connection.
INVALID_DATABASE_URL = (
    "postgresql+psycopg://invalid:invalid@127.0.0.1:1/should_not_connect"
)


def create_alembic_config() -> Config:
    """Create an Alembic configuration with an invalid fallback URL."""
    config = Config(str(ALEMBIC_CONFIG_PATH))
    config.set_main_option(
        "sqlalchemy.url",
        INVALID_DATABASE_URL.replace("%", "%%"),
    )
    return config


def downgrade_using_connection(connection: Connection) -> None:
    """Execute downgrade using the supplied test connection."""
    config = create_alembic_config()
    config.attributes["connection"] = connection

    command.downgrade(config, PREVIOUS_REVISION)


def test_alembic_connection_injection(
    migration_connection: Connection,
) -> None:
    """Verify Alembic uses the exact externally supplied connection."""
    connection = migration_connection

    revision = connection.execute(
        text("SELECT version_num FROM alembic_version")
    ).scalar_one()

    assert revision == CURRENT_REVISION

    transaction = connection.get_transaction()
    assert transaction is not None
    assert transaction.is_active

    config = create_alembic_config()
    config.attributes["connection"] = connection

    command.upgrade(config, "head")

    assert transaction.is_active
    assert not connection.closed

    revision_after = connection.execute(
        text("SELECT version_num FROM alembic_version")
    ).scalar_one()

    assert revision_after == CURRENT_REVISION


def test_downgrade_rejects_modified_historical_run(
    migration_connection: Connection,
) -> None:
    """Reject downgrade when migration-created ingestion data is modified."""
    connection = migration_connection

    result = connection.execute(
        text("""
            UPDATE ingestion_runs
            SET status = 'succeeded'
            WHERE config_snapshot->>'migration_revision' = :marker
              AND status = 'partial'
              AND trigger_type = 'migration'
            RETURNING id
        """),
        {"marker": MIGRATION_MARKER},
    )

    assert len(result.fetchall()) == 1

    with pytest.raises(
        RuntimeError,
        match="migration-created ingestion data has been modified",
    ):
        downgrade_using_connection(connection)


def test_downgrade_rejects_application_created_runs(
    migration_connection: Connection,
) -> None:
    """Reject downgrade when application-created ingestion runs exist."""
    connection = migration_connection

    result = connection.execute(
        text("""
            UPDATE ingestion_runs
            SET config_snapshot = '{}'::jsonb,
                trigger_type = 'upload'
            WHERE config_snapshot->>'migration_revision' = :marker
            RETURNING id
        """),
        {"marker": MIGRATION_MARKER},
    )

    assert len(result.fetchall()) == 1

    with pytest.raises(
        RuntimeError,
        match="application-created ingestion runs exist",
    ):
        downgrade_using_connection(connection)


def test_downgrade_rejects_active_historical_run(
    migration_connection: Connection,
) -> None:
    """Reject downgrade when a migration-created run is active."""
    connection = migration_connection

    result = connection.execute(
        text("""
            UPDATE documents AS d
            SET active_ingestion_run_id = ir.id
            FROM ingestion_runs AS ir
            WHERE ir.document_id = d.id
              AND ir.config_snapshot->>'migration_revision' = :marker
              AND d.active_ingestion_run_id IS NULL
            RETURNING d.id
        """),
        {"marker": MIGRATION_MARKER},
    )

    assert len(result.fetchall()) == 1

    with pytest.raises(
        RuntimeError,
        match="a migration-created run is currently active",
    ):
        downgrade_using_connection(connection)


def test_downgrade_rejects_invalid_historical_stage_configuration(
    migration_connection: Connection,
) -> None:
    """Reject downgrade when a historical stage has an unexpected status."""
    connection = migration_connection

    result = connection.execute(
        text("""
            UPDATE ingestion_stage_runs AS sr
            SET status = 'succeeded'
            FROM ingestion_runs AS ir
            WHERE sr.ingestion_run_id = ir.id
              AND ir.config_snapshot->>'migration_revision' = :marker
              AND sr.stage = 'vector_indexing'
              AND sr.status = 'skipped'
            RETURNING sr.id
        """),
        {"marker": MIGRATION_MARKER},
    )

    assert len(result.fetchall()) == 1

    with pytest.raises(
        RuntimeError,
        match="migration-created ingestion data has been modified",
    ):
        downgrade_using_connection(connection)


def test_downgrade_rejects_unrecorded_migration_created_run(
    migration_connection: Connection,
) -> None:
    """Reject downgrade when a migration-created run has no ledger entry."""
    connection = migration_connection

    duplicate_run_id = uuid4()

    # Create another migration-marked run for the same document.
    # It deliberately has no corresponding ledger entry.
    connection.execute(
        text("""
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
                :duplicate_run_id,
                document_id,
                (
                    SELECT MAX(existing.run_number) + 1
                    FROM ingestion_runs AS existing
                    WHERE existing.document_id = original.document_id
                ),
                status,
                trigger_type,
                config_snapshot,
                started_at,
                completed_at,
                created_at,
                updated_at
            FROM ingestion_runs AS original
            WHERE original.config_snapshot->>'migration_revision' = :marker
            LIMIT 1
        """),
        {
            "duplicate_run_id": duplicate_run_id,
            "marker": MIGRATION_MARKER,
        },
    )

    with pytest.raises(
        RuntimeError,
        match="migration-created ingestion runs are missing from the migration ledger",
    ):
        downgrade_using_connection(connection)


def test_downgrade_rejects_deleted_historical_run(
    migration_connection: Connection,
) -> None:
    """Reject downgrade when a ledger-referenced ingestion run is deleted."""
    connection = migration_connection

    result = connection.execute(
        text("""
            DELETE FROM ingestion_runs
            WHERE id = (
                SELECT ingestion_run_id
                FROM document_ingestion_migration_ledger
                WHERE migration_revision = :marker
            )
            RETURNING id
        """),
        {"marker": MIGRATION_MARKER},
    )

    assert len(result.fetchall()) == 1

    with pytest.raises(
        RuntimeError,
        match="migration ledger references missing or modified historical data",
    ):
        downgrade_using_connection(connection)


def test_downgrade_rejects_deleted_historical_content(
    migration_connection: Connection,
) -> None:
    """Reject downgrade when ledger-referenced extracted content is deleted."""
    connection = migration_connection

    result = connection.execute(
        text("""
            DELETE FROM document_contents
            WHERE id = (
                SELECT content_id
                FROM document_ingestion_migration_ledger
                WHERE migration_revision = :marker
            )
            RETURNING id
        """),
        {"marker": MIGRATION_MARKER},
    )

    assert len(result.fetchall()) == 1

    with pytest.raises(
        RuntimeError,
        match="migration ledger references missing or modified historical data",
    ):
        downgrade_using_connection(connection)


def test_downgrade_rejects_deleted_historical_document(
    migration_connection: Connection,
) -> None:
    """Reject downgrade when the original document has been deleted."""
    connection = migration_connection

    result = connection.execute(
        text("""
            DELETE FROM documents
            WHERE id = :document_id
            RETURNING id
        """),
        {"document_id": DOCUMENT_ID},
    )

    assert len(result.fetchall()) == 1

    # The document deletion cascades to its ingestion run, stage records,
    # and content. The independent migration ledger remains.
    with pytest.raises(
        RuntimeError,
        match="migration ledger references missing or modified historical data",
    ):
        downgrade_using_connection(connection)
