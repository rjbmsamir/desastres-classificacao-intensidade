"""Rotas HTTP de inferência."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends

from app.schemas.prediction import PredictionItemResponse, PredictionRequest, PredictionResponse
from app.services.prediction_service import build_prediction_items, predict_from_dataframe

from .dependencies import get_model_path

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
@router.post("/predict-json", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    model_path: Path = Depends(get_model_path),
) -> PredictionResponse:
    """Executa inferência síncrona em lote a partir de JSON."""
    df = pd.DataFrame(payload.records)
    result = predict_from_dataframe(
        df,
        source=payload.source,
        model_path=model_path,
        tracker=None,
    )
    predictions = [PredictionItemResponse(**item) for item in build_prediction_items(result, input_columns=list(df.columns))]
    return PredictionResponse(
        total_records=len(df),
        total_processed=len(predictions),
        predictions=predictions,
    )
