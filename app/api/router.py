"""Rotas HTTP base da aplicação."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.features import CLASS_LABELS
from app.schemas.prediction import HealthResponse, ModelInfoResponse, ModelMetadataResponse
from app.services.prediction_service import get_model_info
from .predict_file_routes import router as predict_file_router
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


@router.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    """Expõe metadados detalhados do artefato do modelo."""
    return ModelInfoResponse(**get_model_info())


router.include_router(prediction_router)
router.include_router(predict_file_router)
