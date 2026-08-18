from sqlalchemy.orm import Session

from ai_knowledge_platform.models.user import User
from ai_knowledge_platform.repositories.user import UserRepository
from ai_knowledge_platform.repositories.user_credential import (
    UserCredentialRepository,
)
from ai_knowledge_platform.security.password import hash_password


class UserService:
    def __init__(self, db: Session) -> None:
        self.repository = UserRepository(db)
        self.credential_repository = UserCredentialRepository(db)

    def create_user(self, email: str) -> User:
        existing_user = self.repository.get_by_email(email)

        if existing_user:
            raise ValueError(f"User with email {email} already exists.")

        return self.repository.create(email)

    def register_user(self, email: str, password: str) -> User:
        existing_user = self.repository.get_by_email(email)

        if existing_user:
            raise ValueError(f"User with email {email} already exists.")

        password_hash = hash_password(password)

        user = self.repository.create(email)

        self.credential_repository.create(
            user_id=user.id,
            password_hash=password_hash,
        )

        return user
