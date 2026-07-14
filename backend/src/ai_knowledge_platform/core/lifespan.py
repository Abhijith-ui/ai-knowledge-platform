from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:

    print("Starting AI Knowledge Platform...")

    yield

    print("Stopping AI Knowledge Platform...")
