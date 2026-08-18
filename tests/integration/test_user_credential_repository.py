from uuid import uuid4

from ai_knowledge_platform.models.user import User
from ai_knowledge_platform.repositories.user_credential import (
    UserCredentialRepository,
)
from sqlalchemy.orm import Session


def test_create_user_credential(db: Session) -> None:
    user = User(email="credential-test@example.com")
    db.add(user)
    db.flush()

    repository = UserCredentialRepository(db)

    credential = repository.create(
        user_id=user.id,
        password_hash="hashed-password-value",
    )

    assert credential.user_id == user.id
    assert credential.password_hash == "hashed-password-value"
    assert credential.created_at is not None
    assert credential.updated_at is not None


def test_get_user_credential_by_user_id(db: Session) -> None:
    user = User(email="credential-get@example.com")
    db.add(user)
    db.flush()

    repository = UserCredentialRepository(db)

    created = repository.create(
        user_id=user.id,
        password_hash="hashed-password-value",
    )

    found = repository.get_by_user_id(user.id)

    assert found is not None
    assert found.user_id == created.user_id
    assert found.password_hash == "hashed-password-value"


def test_get_user_credential_returns_none_for_unknown_user(
    db: Session,
) -> None:
    repository = UserCredentialRepository(db)

    credential = repository.get_by_user_id(uuid4())

    assert credential is None
