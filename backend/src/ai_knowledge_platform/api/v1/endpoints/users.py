from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ai_knowledge_platform.db.session import get_db
from ai_knowledge_platform.schemas.user import UserCreate, UserRegister
from ai_knowledge_platform.services.user import UserService

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_user(
    payload: UserCreate,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    service = UserService(db)

    try:
        user = service.create_user(str(payload.email))
        db.commit()
    except ValueError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return {
        "id": str(user.id),
        "email": user.email,
    }


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(
    payload: UserRegister,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    service = UserService(db)

    try:
        user = service.register_user(
            email=str(payload.email),
            password=payload.password,
        )
        db.commit()
    except ValueError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return {
        "id": str(user.id),
        "email": user.email,
    }
