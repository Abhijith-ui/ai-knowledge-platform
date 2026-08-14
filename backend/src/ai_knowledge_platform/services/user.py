from sqlalchemy.orm import Session

from ai_knowledge_platform.models.user import User
from ai_knowledge_platform.repositories.user import UserRepository


class UserService:
    def __init__(self, db: Session) -> None:
        self.repository = UserRepository(db)

    def create_user(self, email: str) -> User:
        existing_user = self.repository.get_by_email(email)

        if existing_user:
            raise ValueError(f"User with email {email} already exists.")

        return self.repository.create(email)
