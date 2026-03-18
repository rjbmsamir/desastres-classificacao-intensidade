"""Rotas HTTP base da aplicação."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.features import CLASS_LABELS
from app.schemas.prediction import HealthResponse, ModelMetadataResponse
from .prediction_routes import router as prediction_router

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def healthcheck() -> HealthResponse:
    """Endpoint simples de saúde da aplicação."""
    return HealthResponse(status="ok")


@router.get("/model/labels", response_model=ModelMetadataResponse, tags=["model"])
def model_labels() -> ModelMetadataResponse:
    """Expõe metadados estáticos do classificador."""
    return ModelMetadataResponse(class_labels=CLASS_LABELS)


router.include_router(prediction_router)
