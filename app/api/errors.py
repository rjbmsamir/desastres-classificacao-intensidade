"""Tratamento centralizado de erros da API."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.model import ModelNotFoundError


def register_exception_handlers(app: FastAPI) -> None:
    """Registra handlers HTTP para erros de domínio conhecidos."""

    @app.exception_handler(ModelNotFoundError)
    async def handle_model_not_found(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
