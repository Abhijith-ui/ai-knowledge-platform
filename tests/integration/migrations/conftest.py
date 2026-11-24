from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import Connection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALEMBIC_CONFIG_PATH = PROJECT_ROOT / "backend" / "alembic.ini"

LEGACY_REVISION = "230b4a4d1e68"
CURRENT_REVISION = "400442b8b4c3"

USER_ID = "11111111-1111-4111-8111-111111111111"
DOCUMENT_ID = "22222222-2222-4222-8222-222222222222"

LEGACY_CONTENT = "Synthetic legacy content for automated migration testing."

load_dotenv(PROJECT_ROOT / ".env")

from tests.support.database import (  # noqa: E402
    get_migration_test_database_url,
)


def _create_alembic_config(database_url: str) -> Config:
    """Create Alembic configuration for the dedicated migration database."""
    config = Config(str(ALEMBIC_CONFIG_PATH))
    config.set_main_option(
        "sqlalchemy.url",
        database_url.replace("%", "%%"),
    )
    return config


def _reset_public_schema(engine: Engine) -> None:
    """Reset only the public schema of the dedicated migration database."""
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public AUTHORIZATION CURRENT_USER"))


def _seed_legacy_data(engine: Engine) -> None:
    """Insert deterministic data using the pre-versioned document schema."""
    with engine.begin() as connection:
        connection.execute(
            text("""
                INSERT INTO users (id, email)
                VALUES (:user_id, 'migration-ci@example.test')
            """),
            {"user_id": USER_ID},
        )

        connection.execute(
            text("""
                INSERT INTO documents (
                    id,
                    user_id,
                    filename,
                    content_type,
                    size_bytes,
                    checksum_sha256,
                    storage_key,
                    status,
                    created_at,
                    updated_at,
                    processing_attempts,
                    processing_started_at,
                    processed_at
                )
                VALUES (
                    :document_id,
                    :user_id,
                    'migration-ci.txt',
                    'text/plain',
                    100,
                    :checksum,
                    'migration-ci/fixture/document.txt',
                    'ready',
                    NOW() - INTERVAL '10 minutes',
                    NOW(),
                    1,
                    NOW() - INTERVAL '5 minutes',
                    NOW()
                )
            """),
            {
                "document_id": DOCUMENT_ID,
                "user_id": USER_ID,
                "checksum": "a" * 64,
            },
        )

        connection.execute(
            text("""
                INSERT INTO document_contents (
                    document_id,
                    extracted_text,
                    character_count,
                    page_count,
                    created_at,
                    updated_at
                )
                VALUES (
                    :document_id,
                    :extracted_text,
                    LENGTH(:extracted_text),
                    1,
                    NOW() - INTERVAL '5 minutes',
                    NOW()
                )
            """),
            {
                "document_id": DOCUMENT_ID,
                "extracted_text": LEGACY_CONTENT,
            },
        )


def _verify_migrated_baseline(engine: Engine) -> None:
    """Verify the migration produced the expected historical baseline."""
    with engine.connect() as connection:
        revision = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()

        if revision != CURRENT_REVISION:
            raise RuntimeError(
                f"Expected migration revision {CURRENT_REVISION}, " f"found {revision}."
            )

        result = (
            connection.execute(
                text("""
                SELECT
                    ir.status,
                    ir.trigger_type,
                    ir.config_snapshot->>'migration_revision'
                        AS migration_revision,
                    COUNT(DISTINCT sr.id) AS stage_count,
                    COUNT(DISTINCT dc.id) AS content_count,
                    d.active_ingestion_run_id,
                    dc.extracted_text,
                    dc.character_count
                FROM documents AS d
                JOIN ingestion_runs AS ir
                    ON ir.document_id = d.id
                JOIN ingestion_stage_runs AS sr
                    ON sr.ingestion_run_id = ir.id
                JOIN document_contents AS dc
                    ON dc.ingestion_run_id = ir.id
                JOIN document_ingestion_migration_ledger AS ledger
                    ON ledger.document_id = d.id
                   AND ledger.ingestion_run_id = ir.id
                   AND ledger.content_id = dc.id
                WHERE d.id = :document_id
                GROUP BY
                    ir.id,
                    d.active_ingestion_run_id,
                    dc.id
            """),
                {"document_id": DOCUMENT_ID},
            )
            .mappings()
            .one()
        )

        expected = {
            "status": "partial",
            "trigger_type": "migration",
            "migration_revision": CURRENT_REVISION,
            "stage_count": 5,
            "content_count": 1,
            "active_ingestion_run_id": None,
            "extracted_text": LEGACY_CONTENT,
            "character_count": len(LEGACY_CONTENT),
        }

        for field, expected_value in expected.items():
            if result[field] != expected_value:
                raise RuntimeError(
                    f"Migration baseline validation failed for {field}: "
                    f"expected {expected_value!r}, found {result[field]!r}."
                )


@pytest.fixture(scope="session")
def migration_db_engine() -> Iterator[Engine]:
    """Prepare and expose the isolated migration-test database."""
    database_url = get_migration_test_database_url()
    engine = create_engine(database_url, pool_pre_ping=True)

    try:
        _reset_public_schema(engine)

        config = _create_alembic_config(database_url)
        command.upgrade(config, LEGACY_REVISION)

        _seed_legacy_data(engine)

        command.upgrade(config, CURRENT_REVISION)

        _verify_migrated_baseline(engine)

        yield engine

    finally:
        engine.dispose()


@pytest.fixture
def migration_connection(
    migration_db_engine: Engine,
) -> Iterator[Connection]:
    """Provide a connection whose test mutations are always rolled back."""
    with migration_db_engine.connect() as connection:
        transaction = connection.begin()

        try:
            yield connection
        finally:
            if transaction.is_active:
                transaction.rollback()
