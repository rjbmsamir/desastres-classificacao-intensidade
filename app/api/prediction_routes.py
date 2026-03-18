"""Rotas HTTP de inferência."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends

from app.schemas.prediction import PredictionItemResponse, PredictionRequest, PredictionResponse
from app.services.prediction_service import predict_from_dataframe

from .dependencies import get_model_path

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictionResponse)
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

    probability_columns = [col for col in result.columns if col.startswith("proba_")]
    predictions = []
    for _, row in result.iterrows():
        probabilities = {col.removeprefix("proba_"): row[col] for col in probability_columns}
        predictions.append(
            PredictionItemResponse(
                y_pred=row["y_pred"],
                y_pred_display=row["y_pred_display"],
                y_true=row.get("y_true"),
                y_true_display=row.get("y_true_display"),
                probabilities=probabilities,
            )
        )
    return PredictionResponse(predictions=predictions)
