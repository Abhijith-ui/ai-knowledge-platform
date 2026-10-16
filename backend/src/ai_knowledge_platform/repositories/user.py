from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_knowledge_platform.models.user import User


class UserRepository:
    """Provide database access to user records."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_id(self, user_id: UUID) -> User | None:
        """Retrieve a user by their primary key."""
        statement = select(User).where(User.id == user_id)
        return self.db.scalar(statement)

    def get_by_email(self, email: str) -> User | None:
        """Retrieve a user by email address."""
        statement = select(User).where(User.email == email)
        return self.db.scalar(statement)

    def create(self, email: str) -> User:
        """Create a user and flush it within the current transaction."""
        user = User(email=email)
        self.db.add(user)
        self.db.flush()
        return user
