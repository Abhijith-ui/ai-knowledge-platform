from fastapi import FastAPI

from ai_knowledge_platform.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="AI Knowledge Platform",
    description="Production-grade AI Knowledge Platform API",
    version="0.1.0",
)


@app.get("/", tags=["Health"])
async def health_check() -> dict[str, str]:
    """
    Basic health check endpoint.

    Returns:
        A simple message indicating that the API is running.
    """
    return {"message": "AI Knowledge Platform API is running"}
