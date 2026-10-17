from fastapi import FastAPI

from ai_knowledge_platform.api.router import api_router
from ai_knowledge_platform.core.config import get_settings
from ai_knowledge_platform.core.lifespan import lifespan
from ai_knowledge_platform.core.request_limits import (
    DocumentRequestLimitMiddleware,
)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(DocumentRequestLimitMiddleware)
app.include_router(api_router)
