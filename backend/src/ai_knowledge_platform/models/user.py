from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ai_knowledge_platform.db.base import Base

if TYPE_CHECKING:
    from ai_knowledge_platform.models.user_credential import UserCredential


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    credential: Mapped[UserCredential] = relationship(
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
