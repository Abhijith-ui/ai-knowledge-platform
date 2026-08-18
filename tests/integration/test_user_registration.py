import pytest
from ai_knowledge_platform.repositories.user_credential import (
    UserCredentialRepository,
)
from ai_knowledge_platform.security.password import verify_password
from ai_knowledge_platform.services.user import UserService
from sqlalchemy.orm import Session


def test_register_user_creates_user_and_credentials(db: Session) -> None:
    service = UserService(db)

    user = service.register_user(
        "registration@example.com",
        "StrongPassword123!",
    )

    db.flush()

    repository = UserCredentialRepository(db)
    credential = repository.get_by_user_id(user.id)

    assert user.email == "registration@example.com"
    assert user.id is not None
    assert credential is not None
    assert credential.user_id == user.id


def test_register_user_stores_hashed_password(db: Session) -> None:
    password = "StrongPassword123!"

    service = UserService(db)

    user = service.register_user(
        "password-test@example.com",
        password,
    )

    db.flush()

    credential = UserCredentialRepository(db).get_by_user_id(user.id)

    assert credential is not None
    assert credential.password_hash != password
    assert verify_password(password, credential.password_hash)


def test_register_user_rejects_duplicate_email(db: Session) -> None:
    service = UserService(db)

    service.register_user(
        "duplicate-registration@example.com",
        "StrongPassword123!",
    )

    with pytest.raises(ValueError):
        service.register_user(
            "duplicate-registration@example.com",
            "AnotherPassword123!",
        )
