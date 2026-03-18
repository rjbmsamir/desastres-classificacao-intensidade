"""Serviços de treinamento reutilizáveis pela CLI e pela API."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import CV_REPEATS, CV_SPLITS, MODEL_PATH, RANDOM_STATE, TARGET_COLUMN
from app.core.features import CLASS_LABELS, FEATURES, encode_target, select_feature_frame
from app.core.model import ModelArtifact, extract_class_probabilities, save_model_artifact
from app.services.prediction_service import build_prediction_records
from app.services.tracking import ExperimentTracker, NullExperimentTracker
from app.utils.dataframe_io import load_dataframe


@dataclass(slots=True)
class TrainingResult:
    """Resultado consolidado do treinamento."""

    run_id: Optional[int]
    model_path: Path
    metrics_summary: Dict[str, float]
    n_samples: int
    n_features: int


def build_pipeline() -> Pipeline:
    """Monta o pipeline supervisionado usado no projeto."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def compute_fold_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calcula métricas por dobra da validação cruzada."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    metrics: Dict[str, float] = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    for label, p_val, r_val, f_val in zip(CLASS_LABELS, precision, recall, f1):
        metrics[f"precision_{label}"] = float(p_val)
        metrics[f"recall_{label}"] = float(r_val)
        metrics[f"f1_{label}"] = float(f_val)
    return metrics


def summarize_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Agrega média e desvio padrão das métricas por dobra."""
    aggregated: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            aggregated[key].append(float(value))

    summary: Dict[str, float] = {}
    for key, values in aggregated.items():
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return summary


def _log_training_predictions(
    *,
    tracker: ExperimentTracker,
    run_id: Optional[int],
    pipeline: Pipeline,
    X: pd.DataFrame,
    y_true: List[str],
    source: str,
) -> None:
    predictions = pipeline.predict(X)
    probas = extract_class_probabilities(pipeline, X, CLASS_LABELS)
    df_payload = X.copy()
    df_payload[TARGET_COLUMN] = y_true
    tracker.log_predictions(
        build_prediction_records(
            run_id=run_id,
            source=source,
            df=df_payload.reset_index(drop=True),
            predictions=predictions,
            probabilities=probas,
            y_true=y_true,
        )
    )


def train_model(
    *,
    data_path: Path,
    sheet_name: str,
    model_path: Path = MODEL_PATH,
    tracker: ExperimentTracker | None = None,
) -> TrainingResult:
    """Executa o fluxo completo de treino, avaliação e persistência do modelo."""
    df = load_dataframe(data_path, sheet_name=sheet_name)
    X = select_feature_frame(df)
    y = encode_target(df[TARGET_COLUMN])
    base_pipeline = build_pipeline()
    active_tracker = tracker or NullExperimentTracker()

    run_id = active_tracker.start_run(
        algorithm=type(base_pipeline.named_steps["clf"]).__name__,
        params=base_pipeline.get_params(deep=True),
        data_path=data_path.resolve(),
        n_samples=len(X),
        n_features=len(FEATURES),
        target_positive=CLASS_LABELS[-1],
    )

    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    metrics_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        pipe = clone(base_pipeline)
        pipe.fit(X.iloc[train_idx], [y[i] for i in train_idx])
        preds = pipe.predict(X.iloc[test_idx])
        fold_metrics = compute_fold_metrics([y[i] for i in test_idx], preds)
        fold_metrics["fold"] = fold_idx
        metrics_rows.append(fold_metrics)

    active_tracker.log_cv_metrics(run_id, metrics_rows)
    summary = summarize_metrics([{k: v for k, v in row.items() if k != "fold"} for row in metrics_rows])

    trained_pipeline = base_pipeline.fit(X, y)
    save_model_artifact(
        ModelArtifact(
            pipeline=trained_pipeline,
            features=list(FEATURES),
            class_labels=list(CLASS_LABELS),
            run_id=run_id,
            metadata={},
        ),
        model_path=model_path,
    )
    _log_training_predictions(
        tracker=active_tracker,
        run_id=run_id,
        pipeline=trained_pipeline,
        X=X.reset_index(drop=True),
        y_true=y,
        source="train",
    )
    active_tracker.finalize_run(run_id, summary)

    return TrainingResult(
        run_id=run_id,
        model_path=model_path,
        metrics_summary=summary,
        n_samples=len(X),
        n_features=len(FEATURES),
    )
