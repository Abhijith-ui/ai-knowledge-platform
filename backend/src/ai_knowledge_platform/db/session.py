from collections.abc import Generator

from sqlalchemy.orm import Session, sessionmaker

from ai_knowledge_platform.db.engine import engine

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
