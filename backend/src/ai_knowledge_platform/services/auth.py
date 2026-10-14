from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from ai_knowledge_platform.core.config import get_settings
from ai_knowledge_platform.repositories.refresh_session import (
    RefreshSessionRepository,
)
from ai_knowledge_platform.repositories.user import UserRepository
from ai_knowledge_platform.security.jwt import (
    JWTService,
    TokenType,
    TokenValidationError,
)
from ai_knowledge_platform.security.password import verify_password


class AuthenticationError(Exception):
    """Raised when authentication cannot be completed."""


class RefreshTokenReuseError(AuthenticationError):
    """Raised when a previously rotated refresh token is reused."""


class AuthService:
    """Coordinate credential verification and refresh-token lifecycle."""

    def __init__(
        self,
        db: Session,
        jwt_service: JWTService,
    ) -> None:
        self.db = db
        self.jwt_service = jwt_service
        self.settings = get_settings()
        self.users = UserRepository(db)
        self.refresh_sessions = RefreshSessionRepository()

    @staticmethod
    def _hash_refresh_token(token: str) -> str:
        """Hash a high-entropy refresh token before storing it."""
        return sha256(token.encode("utf-8")).hexdigest()

    def login(
        self,
        *,
        email: str,
        password: str,
    ) -> dict[str, str | int]:
        """Authenticate a user and create a refresh session."""
        user = self.users.get_by_email(email)

        if user is None or user.credential is None:
            raise AuthenticationError("Invalid email or password.")

        if not verify_password(
            password,
            user.credential.password_hash,
        ):
            raise AuthenticationError("Invalid email or password.")

        now = datetime.now(UTC)
        refresh_lifetime = timedelta(days=self.settings.jwt_refresh_token_expire_days)

        access_token = self.jwt_service.create_access_token(user.id)
        refresh_token = self.jwt_service.create_refresh_token(user.id)

        self.refresh_sessions.create(
            self.db,
            user_id=user.id,
            token_hash=self._hash_refresh_token(refresh_token),
            family_id=uuid4(),
            created_at=now,
            expires_at=now + refresh_lifetime,
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": (self.settings.jwt_access_token_expire_minutes * 60),
        }

    def refresh(
        self,
        *,
        refresh_token: str,
    ) -> dict[str, str | int]:
        """Rotate a refresh token and detect reuse of consumed tokens."""
        try:
            claims = self.jwt_service.decode_token(
                refresh_token,
                expected_type=TokenType.REFRESH,
            )
            user_id = UUID(claims["sub"])
        except (TokenValidationError, KeyError, ValueError, TypeError) as exc:
            raise AuthenticationError("Invalid refresh token.") from exc

        now = datetime.now(UTC)
        token_hash = self._hash_refresh_token(refresh_token)

        session = self.refresh_sessions.get_by_token_hash(
            self.db,
            token_hash,
            for_update=True,
        )

        if session is None or session.user_id != user_id:
            raise AuthenticationError("Invalid refresh token.")

        if session.revoked_at is not None:
            if session.replaced_by_id is not None:
                self.refresh_sessions.revoke_family(
                    self.db,
                    family_id=session.family_id,
                    revoked_at=now,
                )
                raise RefreshTokenReuseError(
                    "Previously rotated refresh token was reused."
                )

            raise AuthenticationError("Invalid refresh token.")

        if session.expires_at <= now:
            raise AuthenticationError("Invalid refresh token.")

        new_refresh_token = self.jwt_service.create_refresh_token(user_id)
        new_access_token = self.jwt_service.create_access_token(user_id)

        successor = self.refresh_sessions.create(
            self.db,
            user_id=user_id,
            token_hash=self._hash_refresh_token(new_refresh_token),
            family_id=session.family_id,
            created_at=now,
            expires_at=now
            + timedelta(days=self.settings.jwt_refresh_token_expire_days),
        )

        rotated = self.refresh_sessions.mark_rotated(
            self.db,
            session_id=session.id,
            successor_id=successor.id,
            revoked_at=now,
        )

        if not rotated:
            raise AuthenticationError("Invalid refresh token.")

        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": (self.settings.jwt_access_token_expire_minutes * 60),
        }

    def logout(self, *, refresh_token: str) -> None:
        """Revoke the session associated with a refresh token."""
        try:
            claims = self.jwt_service.decode_token(
                refresh_token,
                expected_type=TokenType.REFRESH,
            )
        except TokenValidationError:
            return

        token_hash = self._hash_refresh_token(refresh_token)
        session = self.refresh_sessions.get_by_token_hash(
            self.db,
            token_hash,
        )

        if session is None or str(session.user_id) != claims["sub"]:
            return

        self.refresh_sessions.revoke_session(
            self.db,
            session_id=session.id,
            revoked_at=datetime.now(UTC),
        )

    def logout_all(self, *, user_id: UUID) -> int:
        """Revoke every active refresh session belonging to a user."""
        return self.refresh_sessions.revoke_all_for_user(
            self.db,
            user_id=user_id,
            revoked_at=datetime.now(UTC),
        )
