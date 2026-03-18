"""Schemas de resposta para upload e predição em lote por arquivo."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel

from app.schemas.prediction import PredictionItemResponse


class PredictionFileResponse(BaseModel):
    """Resposta estruturada de predição a partir de arquivo."""

    filename: str
    total_rows: int
    total_processed: int
    columns_received: List[str]
    required_columns: List[str]
    predictions: List[PredictionItemResponse]
