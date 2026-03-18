"""Tratamento centralizado de erros da API."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.features import MissingColumnsError
from app.core.model import ModelNotFoundError
from app.utils.dataframe_io import EmptyFileError, FileParsingError, UnsupportedFileFormatError


def register_exception_handlers(app: FastAPI) -> None:
    """Registra handlers HTTP para erros de domínio conhecidos."""

    @app.exception_handler(ModelNotFoundError)
    async def handle_model_not_found(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(MissingColumnsError)
    async def handle_missing_columns(request: Request, exc: MissingColumnsError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(exc),
                "missing_columns": exc.missing_columns,
            },
        )

    @app.exception_handler(UnsupportedFileFormatError)
    async def handle_unsupported_format(request: Request, exc: UnsupportedFileFormatError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(EmptyFileError)
    async def handle_empty_file(request: Request, exc: EmptyFileError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(FileParsingError)
    async def handle_file_parsing_error(request: Request, exc: FileParsingError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Erro interno inesperado: {exc.__class__.__name__}: {exc}"},
        )
