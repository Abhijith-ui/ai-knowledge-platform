from fastapi import APIRouter

from ai_knowledge_platform.api.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
