"""FastAPI application factory module."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routes import api, web


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
settings = get_settings()
settings.ensure_directories()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings.ensure_directories()
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
app.include_router(web.router)
app.include_router(api.router)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    """Serve the lightweight project favicon without a noisy browser 404."""

    return FileResponse(settings.static_dir / "favicon.svg", media_type="image/svg+xml")
