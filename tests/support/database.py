import os

from sqlalchemy.engine import make_url

EXPECTED_TEST_DATABASE = "ai_knowledge_platform_test"
EXPECTED_MIGRATION_TEST_DATABASE = "ai_knowledge_platform_migration_ci_test"


def _get_validated_database_url(
    environment_variable: str,
    expected_database: str,
) -> str:
    """Return a database URL after validating its driver and database name."""
    database_url = os.environ.get(environment_variable)

    if not database_url:
        raise RuntimeError(
            f"{environment_variable} must be set to the dedicated database."
        )

    try:
        parsed_url = make_url(database_url)
    except Exception as exc:
        raise RuntimeError(f"{environment_variable} is invalid.") from exc

    if parsed_url.drivername != "postgresql+psycopg":
        raise RuntimeError(
            f"{environment_variable} must use the postgresql+psycopg driver."
        )

    if parsed_url.database != expected_database:
        raise RuntimeError(f"Tests may only use the '{expected_database}' database.")

    return database_url


def get_test_database_url() -> str:
    """Return a validated URL for the dedicated application test database."""
    return _get_validated_database_url(
        "TEST_DATABASE_URL",
        EXPECTED_TEST_DATABASE,
    )


def get_migration_test_database_url() -> str:
    """Return a validated URL for the isolated migration test database."""
    return _get_validated_database_url(
        "MIGRATION_TEST_DATABASE_URL",
        EXPECTED_MIGRATION_TEST_DATABASE,
    )
