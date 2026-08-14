import pytest
from ai_knowledge_platform.db.base import Base
from ai_knowledge_platform.models.user import User
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session


@pytest.fixture
def db() -> Session:
    engine = create_engine(
        "postgresql+psycopg://postgres:postgres@localhost:5432/ai_knowledge_platform"
    )

    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.execute(delete(User))
        session.commit()

        yield session

        session.rollback()

    engine.dispose()
