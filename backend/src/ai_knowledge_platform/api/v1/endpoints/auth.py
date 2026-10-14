from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from ai_knowledge_platform.db.session import get_db
from ai_knowledge_platform.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    MessageResponse,
    RefreshRequest,
    TokenResponse,
)
from ai_knowledge_platform.security.jwt import JWTService
from ai_knowledge_platform.services.auth import (
    AuthenticationError,
    AuthService,
    RefreshTokenReuseError,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


def get_jwt_service(request: Request) -> JWTService:
    """Retrieve the JWT service initialized during application startup."""
    return cast(JWTService, request.app.state.jwt_service)


@router.post("/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
    db: Session = Depends(get_db),
    jwt_service: JWTService = Depends(get_jwt_service),
) -> dict[str, str | int]:
    """Authenticate a user and issue access and refresh tokens."""
    service = AuthService(db, jwt_service)

    try:
        tokens = service.login(
            email=payload.email,
            password=payload.password,
        )
        db.commit()
        return tokens
    except AuthenticationError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except Exception:
        db.rollback()
        raise


@router.post("/refresh", response_model=TokenResponse)
def refresh(
    payload: RefreshRequest,
    db: Session = Depends(get_db),
    jwt_service: JWTService = Depends(get_jwt_service),
) -> dict[str, str | int]:
    """Rotate a refresh token and detect refresh-token reuse."""
    service = AuthService(db, jwt_service)

    try:
        tokens = service.refresh(refresh_token=payload.refresh_token)
        db.commit()
        return tokens

    except RefreshTokenReuseError as exc:
        # Persist the security action even though the request is rejected.
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    except AuthenticationError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    except Exception:
        db.rollback()
        raise


@router.post("/logout", response_model=MessageResponse)
def logout(
    payload: LogoutRequest,
    db: Session = Depends(get_db),
    jwt_service: JWTService = Depends(get_jwt_service),
) -> dict[str, str]:
    """Revoke the supplied refresh session."""
    service = AuthService(db, jwt_service)

    try:
        service.logout(refresh_token=payload.refresh_token)
        db.commit()
        return {"message": "Logged out successfully."}
    except Exception:
        db.rollback()
        raise
