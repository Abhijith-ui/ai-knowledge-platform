from pathlib import Path

import pytest
from ai_knowledge_platform.models.document import Document
from ai_knowledge_platform.models.document_content import DocumentContent
from ai_knowledge_platform.models.ingestion_run import IngestionRun
from ai_knowledge_platform.models.ingestion_stage_run import IngestionStageRun
from ai_knowledge_platform.models.refresh_session import RefreshSession
from ai_knowledge_platform.models.user import User
from ai_knowledge_platform.models.user_credential import UserCredential
from alembic import command
from alembic.config import Config
from dotenv import load_dotenv
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from tests.support.database import get_test_database_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@pytest.fixture(scope="session")
def migrated_test_schema() -> None:
    """Ensure the dedicated test database is migrated to the current head."""
    database_url = get_test_database_url()
    alembic_config_path = (
        Path(__file__).resolve().parents[1] / "backend" / "alembic.ini"
    )
    config = Config(str(alembic_config_path))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))

    command.upgrade(config, "head")


@pytest.fixture
def db(migrated_test_schema: None) -> Session:
    """Provide a PostgreSQL session with isolated test data."""
    database_url = get_test_database_url()
    engine = create_engine(database_url, pool_pre_ping=True)

    try:
        with Session(engine) as session:
            session.execute(delete(IngestionStageRun))
            session.execute(delete(DocumentContent))
            session.execute(delete(IngestionRun))
            session.execute(delete(Document))
            session.execute(delete(RefreshSession))
            session.execute(delete(UserCredential))
            session.execute(delete(User))
            session.commit()

            yield session

            session.rollback()
    finally:
        engine.dispose()
