from typing import cast
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ai_knowledge_platform.security.jwt import (
    JWTService,
    TokenType,
    TokenValidationError,
)

bearer_scheme = HTTPBearer(auto_error=False)


def get_jwt_service(request: Request) -> JWTService:
    """Retrieve the JWT service initialized during application startup."""
    return cast(JWTService, request.app.state.jwt_service)


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    jwt_service: JWTService = Depends(get_jwt_service),
) -> UUID:
    """Validate an access token and return its authenticated user ID."""
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise unauthorized

    try:
        claims = jwt_service.decode_token(
            credentials.credentials,
            expected_type=TokenType.ACCESS,
        )
        return UUID(claims["sub"])
    except (TokenValidationError, KeyError, ValueError, TypeError) as exc:
        raise unauthorized from exc
