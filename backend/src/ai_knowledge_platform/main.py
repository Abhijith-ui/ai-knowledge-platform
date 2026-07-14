from fastapi import FastAPI

from ai_knowledge_platform.api.router import api_router
from ai_knowledge_platform.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="AI Knowledge Platform",
    description="Production-grade AI Knowledge Platform API",
    version="0.1.0",
)


app.include_router(api_router)
