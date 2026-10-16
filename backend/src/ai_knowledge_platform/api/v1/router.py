from fastapi import APIRouter

from ai_knowledge_platform.api.v1.endpoints.auth import router as auth_router
from ai_knowledge_platform.api.v1.endpoints.documents import (
    router as documents_router,
)
from ai_knowledge_platform.api.v1.endpoints.health import router as health_router
from ai_knowledge_platform.api.v1.endpoints.users import router as users_router

v1_router = APIRouter(prefix="/v1")

v1_router.include_router(health_router)
v1_router.include_router(users_router)
v1_router.include_router(auth_router)
v1_router.include_router(documents_router)
