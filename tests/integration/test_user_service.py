import pytest
from ai_knowledge_platform.services.user import UserService
from sqlalchemy.orm import Session


def test_create_user(db: Session) -> None:
    service = UserService(db)

    user = service.create_user("integration-user-001@example.com")

    assert user.email == "integration-user-001@example.com"
    assert user.id is not None


def test_create_duplicate_user(db: Session) -> None:
    service = UserService(db)

    service.create_user("duplicate-test@example.com")
    db.flush()

    with pytest.raises(ValueError):
        service.create_user("duplicate-test@example.com")
