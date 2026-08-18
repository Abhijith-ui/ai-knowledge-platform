from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_knowledge_platform.models.user_credential import UserCredential


class UserCredentialRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_user_id(self, user_id: UUID) -> UserCredential | None:
        statement = select(UserCredential).where(UserCredential.user_id == user_id)

        return self.db.scalar(statement)

    def create(
        self,
        user_id: UUID,
        password_hash: str,
    ) -> UserCredential:
        now = datetime.now(timezone.utc)

        credential = UserCredential(
            user_id=user_id,
            password_hash=password_hash,
            created_at=now,
            updated_at=now,
        )

        self.db.add(credential)
        self.db.flush()

        return credential
