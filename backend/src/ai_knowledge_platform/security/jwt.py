from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateKey,
    RSAPublicKey,
)
from jwt.exceptions import InvalidTokenError


class TokenType(StrEnum):
    """Supported JWT token types."""

    ACCESS = "access"
    REFRESH = "refresh"


class TokenValidationError(ValueError):
    """Raised when a JWT is invalid or used for the wrong purpose."""


class JWTService:
    """Create and validate RS256-signed access and refresh tokens."""

    def __init__(
        self,
        private_key: RSAPrivateKey,
        public_key: RSAPublicKey,
        access_token_lifetime: timedelta,
        refresh_token_lifetime: timedelta,
    ) -> None:
        if private_key.key_size < 2048 or public_key.key_size < 2048:
            raise ValueError("RSA keys must be at least 2048 bits.")

        if private_key.public_key().public_numbers() != public_key.public_numbers():
            raise ValueError("RSA private and public keys do not match.")

        if access_token_lifetime <= timedelta(0):
            raise ValueError("Access token lifetime must be positive.")

        if refresh_token_lifetime <= timedelta(0):
            raise ValueError("Refresh token lifetime must be positive.")

        self._private_key = private_key
        self._public_key = public_key
        self._access_token_lifetime = access_token_lifetime
        self._refresh_token_lifetime = refresh_token_lifetime

    def create_access_token(self, subject: str | UUID) -> str:
        """Create a short-lived token for accessing protected resources."""
        return self._create_token(
            subject=subject,
            token_type=TokenType.ACCESS,
            lifetime=self._access_token_lifetime,
        )

    def create_refresh_token(self, subject: str | UUID) -> str:
        """Create a longer-lived token for obtaining new tokens."""
        return self._create_token(
            subject=subject,
            token_type=TokenType.REFRESH,
            lifetime=self._refresh_token_lifetime,
        )

    def _create_token(
        self,
        subject: str | UUID,
        token_type: TokenType,
        lifetime: timedelta,
    ) -> str:
        subject_value = str(subject)

        if not subject_value.strip():
            raise ValueError("Token subject must not be empty.")

        now = datetime.now(timezone.utc)

        payload: dict[str, Any] = {
            "sub": subject_value,
            "type": token_type.value,
            "iat": now,
            "exp": now + lifetime,
            "jti": str(uuid4()),
        }

        return jwt.encode(
            payload,
            self._private_key,
            algorithm="RS256",
        )

    def decode_token(
        self,
        token: str,
        expected_type: TokenType,
    ) -> dict[str, Any]:
        """Verify a token's signature, required claims, and intended type."""
        if not isinstance(token, str) or not token.strip():
            raise TokenValidationError("Invalid token.")

        try:
            payload = jwt.decode(
                token,
                self._public_key,
                algorithms=["RS256"],
                options={
                    "require": ["sub", "type", "iat", "exp", "jti"],
                },
            )
        except (InvalidTokenError, TypeError, ValueError) as exc:
            raise TokenValidationError("Invalid token.") from exc

        if payload.get("type") != expected_type.value:
            raise TokenValidationError("Token type does not match.")

        return payload
