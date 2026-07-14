import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_knowledge_platform.core.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage the application lifecycle.
    """

    configure_logging()

    logger.info("Starting AI Knowledge Platform...")

    yield

    logger.info("Stopping AI Knowledge Platform...")
