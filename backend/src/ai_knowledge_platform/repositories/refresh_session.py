from datetime import datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from ai_knowledge_platform.models.refresh_session import RefreshSession


class RefreshSessionRepository:
    """Database operations for refresh-token sessions.

    The calling service owns transaction boundaries.
    This repository never commits or rolls back transactions.
    """

    @staticmethod
    def create(
        db: Session,
        *,
        user_id: UUID,
        token_hash: str,
        family_id: UUID,
        created_at: datetime,
        expires_at: datetime,
    ) -> RefreshSession:
        """Create and flush a refresh-session record."""
        session = RefreshSession(
            user_id=user_id,
            token_hash=token_hash,
            family_id=family_id,
            created_at=created_at,
            expires_at=expires_at,
        )
        db.add(session)
        db.flush()
        return session

    @staticmethod
    def get_by_token_hash(
        db: Session,
        token_hash: str,
        *,
        for_update: bool = False,
    ) -> RefreshSession | None:
        """Find a session by token hash, optionally locking its row."""
        statement = select(RefreshSession).where(
            RefreshSession.token_hash == token_hash
        )

        if for_update:
            statement = statement.with_for_update()

        return db.scalar(statement)

    @staticmethod
    def mark_rotated(
        db: Session,
        *,
        session_id: UUID,
        successor_id: UUID,
        revoked_at: datetime,
    ) -> bool:
        """Atomically consume an active token exactly once."""
        result = cast(
            CursorResult[Any],
            db.execute(
                update(RefreshSession)
                .where(
                    RefreshSession.id == session_id,
                    RefreshSession.revoked_at.is_(None),
                    RefreshSession.replaced_by_id.is_(None),
                )
                .values(
                    revoked_at=revoked_at,
                    replaced_by_id=successor_id,
                )
            ),
        )
        return result.rowcount == 1

    @staticmethod
    def revoke_session(
        db: Session,
        *,
        session_id: UUID,
        revoked_at: datetime,
    ) -> bool:
        """Revoke one active refresh session."""
        result = cast(
            CursorResult[Any],
            db.execute(
                update(RefreshSession)
                .where(
                    RefreshSession.id == session_id,
                    RefreshSession.revoked_at.is_(None),
                )
                .values(revoked_at=revoked_at)
            ),
        )
        return result.rowcount == 1

    @staticmethod
    def revoke_family(
        db: Session,
        *,
        family_id: UUID,
        revoked_at: datetime,
    ) -> int:
        """Revoke all active sessions in a refresh-token family."""
        result = cast(
            CursorResult[Any],
            db.execute(
                update(RefreshSession)
                .where(
                    RefreshSession.family_id == family_id,
                    RefreshSession.revoked_at.is_(None),
                )
                .values(revoked_at=revoked_at)
            ),
        )
        return result.rowcount or 0

    @staticmethod
    def revoke_all_for_user(
        db: Session,
        *,
        user_id: UUID,
        revoked_at: datetime,
    ) -> int:
        """Revoke all active refresh sessions belonging to a user."""
        result = cast(
            CursorResult[Any],
            db.execute(
                update(RefreshSession)
                .where(
                    RefreshSession.user_id == user_id,
                    RefreshSession.revoked_at.is_(None),
                )
                .values(revoked_at=revoked_at)
            ),
        )
        return result.rowcount or 0
