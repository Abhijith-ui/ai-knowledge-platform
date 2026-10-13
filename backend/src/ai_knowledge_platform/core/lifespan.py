import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import FastAPI

from ai_knowledge_platform.core.config import get_settings
from ai_knowledge_platform.core.logging import configure_logging
from ai_knowledge_platform.security.jwt import JWTService
from ai_knowledge_platform.security.keys import load_rsa_keys

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and release application-level resources."""

    configure_logging()
    logger.info("Starting AI Knowledge Platform...")

    settings = get_settings()

    try:
        private_key, public_key = load_rsa_keys(
            private_key_path=settings.jwt_private_key_path,
            public_key_path=settings.jwt_public_key_path,
        )

        app.state.jwt_service = JWTService(
            private_key=private_key,
            public_key=public_key,
            access_token_lifetime=timedelta(
                minutes=settings.jwt_access_token_expire_minutes
            ),
            refresh_token_lifetime=timedelta(
                days=settings.jwt_refresh_token_expire_days
            ),
        )
    except Exception:
        logger.exception("Failed to initialize JWT service.")
        raise

    logger.info("JWT service initialized successfully.")

    try:
        yield
    finally:
        if hasattr(app.state, "jwt_service"):
            del app.state.jwt_service

        logger.info("Stopping AI Knowledge Platform...")
