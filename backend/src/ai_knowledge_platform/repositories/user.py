from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_knowledge_platform.models.user import User


class UserRepository:

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_email(self, email: str) -> User | None:
        statement = select(User).where(User.email == email)

        return self.db.scalar(statement)

    def create(self, email: str) -> User:
        user = User(email=email)

        self.db.add(user)
        self.db.flush()

        return user
