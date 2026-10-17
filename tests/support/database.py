import os

from sqlalchemy.engine import make_url

EXPECTED_TEST_DATABASE = "ai_knowledge_platform_test"


def get_test_database_url() -> str:
    """Return a validated URL for the dedicated test database."""
    database_url = os.environ.get("TEST_DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "TEST_DATABASE_URL must be set to the dedicated test database."
        )

    try:
        parsed_url = make_url(database_url)
    except Exception as exc:
        raise RuntimeError("TEST_DATABASE_URL is invalid.") from exc

    if parsed_url.drivername != "postgresql+psycopg":
        raise RuntimeError("TEST_DATABASE_URL must use the postgresql+psycopg driver.")

    if parsed_url.database != EXPECTED_TEST_DATABASE:
        raise RuntimeError(
            f"Tests may only use the '{EXPECTED_TEST_DATABASE}' database."
        )

    return database_url
