"""Compatibilidade para carregamento do modelo e geração de predições."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .core.config import MODEL_PATH, SHEET_NAME
from .core.model import ModelNotFoundError, load_model_artifact
from .services.prediction_service import predict_from_dataframe, predict_from_file as service_predict_from_file
from .services.tracking import SQLiteExperimentTracker


def load_model(model_path: Path = MODEL_PATH):
    """Carrega o artefato serializado do modelo."""
    artifact = load_model_artifact(model_path)
    return artifact.pipeline, artifact.features, artifact.run_id, artifact.class_labels, artifact.metadata


def predict_from_df(
    df: pd.DataFrame,
    *,
    source: str = "inference",
    model_path: Path = MODEL_PATH,
    run_id: Optional[int] = None,
    log_predictions: bool = True,
) -> pd.DataFrame:
    """Mantém compatibilidade com a API antiga baseada em DataFrame."""
    tracker = SQLiteExperimentTracker() if log_predictions else None
    return predict_from_dataframe(
        df,
        source=source,
        model_path=model_path,
        run_id=run_id,
        tracker=tracker,
    )


def predict_from_file(
    path: Path,
    *,
    sheet_name: str = SHEET_NAME,
    source: str = "batch",
    model_path: Path = MODEL_PATH,
    run_id: Optional[int] = None,
    log_predictions: bool = True,
) -> pd.DataFrame:
    """Mantém compatibilidade com a API antiga baseada em arquivo."""
    tracker = SQLiteExperimentTracker() if log_predictions else None
    return service_predict_from_file(
        path,
        sheet_name=sheet_name,
        source=source,
        model_path=model_path,
        run_id=run_id,
        tracker=tracker,
    )
