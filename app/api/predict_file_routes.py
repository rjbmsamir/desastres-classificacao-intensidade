"""Rotas HTTP para upload de arquivo e predição em lote."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.config import SHEET_NAME
from app.core.features import DIRECT_INPUT_FEATURES
from app.schemas.prediction import PredictionItemResponse
from app.schemas.prediction_file import PredictionFileResponse
from app.services.prediction_service import build_prediction_items, predict_from_dataframe
from app.utils.dataframe_io import load_uploaded_dataframe

from .dependencies import get_model_path

router = APIRouter(tags=["prediction"])


@router.post("/predict-file", response_model=PredictionFileResponse)
async def predict_file(
    file: UploadFile = File(...),
    model_path: Path = Depends(get_model_path),
) -> PredictionFileResponse:
    """Executa inferência em lote a partir de arquivo enviado por upload."""
    content = await file.read()
    df = load_uploaded_dataframe(file.filename or "", content, sheet_name=SHEET_NAME)
    result = predict_from_dataframe(
        df,
        source="api_file_upload",
        model_path=model_path,
        tracker=None,
    )
    predictions = [PredictionItemResponse(**item) for item in build_prediction_items(result, input_columns=list(df.columns))]
    return PredictionFileResponse(
        filename=file.filename or "",
        total_rows=len(df),
        total_processed=len(predictions),
        columns_received=list(df.columns),
        required_columns=list(DIRECT_INPUT_FEATURES),
        predictions=predictions,
    )
