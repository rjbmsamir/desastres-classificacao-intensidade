"""Schemas de entrada e saída para a camada HTTP."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Resposta simples de healthcheck."""

    status: str


class ModelMetadataResponse(BaseModel):
    """Metadados mínimos do modelo expostos pela API."""

    class_labels: List[str]


class PredictionRequest(BaseModel):
    """Payload de inferência em lote baseado em registros JSON."""

    records: List[Dict[str, Any]] = Field(..., min_length=1)
    source: str = "api"

    @field_validator("records")
    @classmethod
    def validate_records(cls, value: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Garante que o payload contenha ao menos um objeto JSON."""
        if any(not isinstance(item, dict) or not item for item in value):
            raise ValueError("Cada item de 'records' deve ser um objeto JSON não vazio.")
        return value


class PredictionItemResponse(BaseModel):
    """Representa uma linha prevista pela API."""

    y_pred: str
    y_pred_display: str
    y_true: Optional[str] = None
    y_true_display: Optional[str] = None
    probabilities: Dict[str, Optional[float]]


class PredictionResponse(BaseModel):
    """Resposta da inferência em lote."""

    predictions: List[PredictionItemResponse]
