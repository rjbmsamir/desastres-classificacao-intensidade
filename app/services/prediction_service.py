"""Serviços de inferência reutilizáveis por diferentes interfaces."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from app.core.config import MODEL_PATH, SHEET_NAME, TARGET_COLUMN
from app.core.features import encode_target, ensure_valid_features, to_display_label
from app.core.model import ModelArtifact, extract_class_probabilities, load_model_artifact
from app.services.tracking import ExperimentTracker, NullExperimentTracker
from app.utils.dataframe_io import load_dataframe
from app.utils.hashing import make_row_hash

CANONICAL_MAP = {
    "nivel i": "nivel_I",
    "nível i": "nivel_I",
    "nivel_i": "nivel_I",
    "nivel ii": "nivel_II",
    "nível ii": "nivel_II",
    "nivel_ii": "nivel_II",
    "nivel iii": "nivel_III",
    "nível iii": "nivel_III",
    "nivel_iii": "nivel_III",
    "raro": "raro",
    "rara": "raro",
}


def _normalize_response_value(value: Any) -> Any:
    """Normaliza valores para serialização JSON segura."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return value
    return value


def load_trained_model(model_path: Path = MODEL_PATH) -> ModelArtifact:
    """Carrega o artefato treinado do modelo."""
    return load_model_artifact(model_path)


def get_model_info(model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    """Retorna metadados resumidos do artefato do modelo."""
    artifact = load_model_artifact(model_path)
    algorithm = None
    if hasattr(artifact.pipeline, "named_steps"):
        estimator = artifact.pipeline.named_steps.get("clf")
        if estimator is not None:
            algorithm = type(estimator).__name__
    return {
        "model_loaded": True,
        "artifact_path": str(model_path),
        "run_id": artifact.run_id,
        "class_labels": artifact.class_labels,
        "features": artifact.features,
        "metadata_keys": sorted(artifact.metadata.keys()),
        "algorithm": algorithm,
    }


def _resolve_true_labels(df: pd.DataFrame) -> List[Optional[str]]:
    if TARGET_COLUMN not in df.columns:
        return [None] * len(df)
    try:
        return encode_target(df[TARGET_COLUMN])
    except Exception:
        normalized = []
        for value in df[TARGET_COLUMN]:
            if pd.isna(value):
                normalized.append(None)
                continue
            text = str(value).strip().lower()
            normalized.append(CANONICAL_MAP.get(text, str(value)))
        return normalized


def build_prediction_records(
    *,
    run_id: Optional[int],
    source: str,
    df: pd.DataFrame,
    predictions: Iterable[str],
    probabilities: Iterable[Dict[str, Optional[float]]],
    y_true: Iterable[Optional[str]],
):
    """Monta o payload de persistência das predições."""
    for idx, (pred, proba_map, true_label) in enumerate(zip(predictions, probabilities, y_true)):
        payload = df.iloc[idx].to_dict()
        yield {
            "run_id": run_id,
            "source": source,
            "row_hash": make_row_hash(payload),
            "y_true": true_label,
            "y_pred": pred,
            "probabilities": proba_map,
            "payload": payload,
        }


def predict_from_dataframe(
    df: pd.DataFrame,
    *,
    source: str = "inference",
    model_path: Path = MODEL_PATH,
    run_id: Optional[int] = None,
    tracker: ExperimentTracker | None = None,
) -> pd.DataFrame:
    """Executa inferência sobre um DataFrame já carregado."""
    artifact = load_model_artifact(model_path)
    ensure_valid_features(df)
    X = df[artifact.features].copy()
    preds = artifact.pipeline.predict(X)
    probas = extract_class_probabilities(artifact.pipeline, X, artifact.class_labels)
    y_true = _resolve_true_labels(df)

    active_tracker = tracker or NullExperimentTracker()
    active_tracker.log_predictions(
        build_prediction_records(
            run_id=run_id or artifact.run_id,
            source=source,
            df=df,
            predictions=preds,
            probabilities=probas,
            y_true=y_true,
        )
    )

    result = df.copy()
    result["y_pred"] = preds
    for label in artifact.class_labels:
        result[f"proba_{label}"] = [proba_map.get(label) if proba_map else None for proba_map in probas]
    if any(label is not None for label in y_true):
        result["y_true"] = y_true
        result["y_true_display"] = [to_display_label(label) if label is not None else None for label in y_true]
    result["y_pred_display"] = [to_display_label(label) for label in preds]
    return result


def predict_from_file(
    path: Path,
    *,
    sheet_name: str = SHEET_NAME,
    source: str = "batch",
    model_path: Path = MODEL_PATH,
    run_id: Optional[int] = None,
    tracker: ExperimentTracker | None = None,
) -> pd.DataFrame:
    """Carrega um arquivo tabular e executa a inferência."""
    df = load_dataframe(path, sheet_name=sheet_name)
    return predict_from_dataframe(
        df,
        source=source,
        model_path=model_path,
        run_id=run_id,
        tracker=tracker,
    )


def build_prediction_items(result: pd.DataFrame, *, input_columns: List[str]) -> List[Dict[str, Any]]:
    """Converte o DataFrame de saída em itens serializáveis para a API."""
    probability_columns = [col for col in result.columns if col.startswith("proba_")]
    items: List[Dict[str, Any]] = []
    for idx, row in result.iterrows():
        items.append(
            {
                "row_index": int(idx),
                "input_data": {
                    column: _normalize_response_value(row[column]) for column in input_columns if column in row.index
                },
                "y_pred": _normalize_response_value(row["y_pred"]),
                "y_pred_display": _normalize_response_value(row["y_pred_display"]),
                "y_true": _normalize_response_value(row.get("y_true")),
                "y_true_display": _normalize_response_value(row.get("y_true_display")),
                "probabilities": {
                    col.removeprefix("proba_"): _normalize_response_value(row[col]) for col in probability_columns
                },
            }
        )
    return items
