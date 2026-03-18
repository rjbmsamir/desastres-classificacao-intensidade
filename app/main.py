"""Aplicação FastAPI mínima preparada para expansão."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.errors import register_exception_handlers
from app.api.router import router
from app.web.routes import router as web_router

STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"


def create_app() -> FastAPI:
    """Constrói a aplicação HTTP."""
    app = FastAPI(
        title="Projeto Desastres API",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.include_router(router)
    app.include_router(web_router)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    register_exception_handlers(app)
    return app


app = create_app()
