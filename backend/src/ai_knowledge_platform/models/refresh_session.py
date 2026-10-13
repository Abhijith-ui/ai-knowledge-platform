from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    String,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from ai_knowledge_platform.db.base import Base

if TYPE_CHECKING:
    pass


class RefreshSession(Base):
    """Persist refresh-token state without storing raw tokens."""

    __tablename__ = "refresh_sessions"
    __table_args__ = (
        CheckConstraint(
            "expires_at > created_at",
            name="ck_refresh_sessions_expiry_after_creation",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # SHA-256 hex digest of the opaque refresh token, never the raw token.
    token_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )

    # Tokens issued by one rotation chain share this identifier.
    family_id: Mapped[UUID] = mapped_column(
        Uuid,
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Non-null means this token was rotated into a successor.
    replaced_by_id: Mapped[UUID | None] = mapped_column(
        Uuid,
        ForeignKey("refresh_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
