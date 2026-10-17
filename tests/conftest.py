import pytest
from ai_knowledge_platform.db.base import Base
from ai_knowledge_platform.models.document import Document
from ai_knowledge_platform.models.refresh_session import RefreshSession
from ai_knowledge_platform.models.user import User
from ai_knowledge_platform.models.user_credential import UserCredential
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from tests.support.database import get_test_database_url


@pytest.fixture
def db() -> Session:
    """Provide a PostgreSQL session with isolated test data."""
    database_url = get_test_database_url()
    engine = create_engine(database_url, pool_pre_ping=True)

    try:
        Base.metadata.create_all(engine)

        with Session(engine) as session:
            session.execute(delete(Document))
            session.execute(delete(RefreshSession))
            session.execute(delete(UserCredential))
            session.execute(delete(User))
            session.commit()

            yield session

            session.rollback()
    finally:
        engine.dispose()
