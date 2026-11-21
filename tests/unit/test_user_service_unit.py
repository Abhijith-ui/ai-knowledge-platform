from ai_knowledge_platform.models.user import User
from ai_knowledge_platform.services.user import UserService
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def test_create_user() -> None:
    engine = create_engine("sqlite:///:memory:")

    User.__table__.create(engine)

    with Session(engine) as db:
        service = UserService(db)

        user = service.create_user("test@example.com")

        db.commit()

        assert user.email == "test@example.com"
        assert user.id is not None

    engine.dispose()
