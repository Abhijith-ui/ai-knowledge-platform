import pytest

from tests.support.database import get_test_database_url


def test_test_database_url_requires_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)

    with pytest.raises(
        RuntimeError,
        match="TEST_DATABASE_URL must be set",
    ):
        get_test_database_url()


def test_test_database_url_rejects_development_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TEST_DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/"
        "ai_knowledge_platform",
    )

    with pytest.raises(
        RuntimeError,
        match="Tests may only use",
    ):
        get_test_database_url()


def test_test_database_url_rejects_unsupported_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TEST_DATABASE_URL",
        "sqlite:///ai_knowledge_platform_test",
    )

    with pytest.raises(
        RuntimeError,
        match="must use the postgresql\\+psycopg driver",
    ):
        get_test_database_url()


def test_test_database_url_accepts_dedicated_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/"
        "ai_knowledge_platform_test"
    )
    monkeypatch.setenv("TEST_DATABASE_URL", database_url)

    assert get_test_database_url() == database_url
