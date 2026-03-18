"""Aplicação FastAPI mínima preparada para expansão."""
from __future__ import annotations

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.router import router


def create_app() -> FastAPI:
    """Constrói a aplicação HTTP."""
    app = FastAPI(title="Projeto Desastres API", version="0.1.0")
    app.include_router(router)
    register_exception_handlers(app)
    return app


app = create_app()
