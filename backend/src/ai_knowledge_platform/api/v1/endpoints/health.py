from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from ai_knowledge_platform.db.session import get_db

router = APIRouter()


@router.get("/", summary="Health Check")
async def health_check(db: Session = Depends(get_db)) -> dict[str, str]:

    db.execute(text("SELECT 1"))

    return {"status": "healthy", "database": "connected"}
