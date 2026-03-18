"""Operações de serialização e acesso ao artefato do modelo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from joblib import dump, load

from .config import MODEL_PATH
from .features import CLASS_LABELS, FEATURES


class ModelNotFoundError(FileNotFoundError):
    """Disparado quando o arquivo do modelo não é encontrado."""


@dataclass(slots=True)
class ModelArtifact:
    """Representa o pipeline persistido e seus metadados essenciais."""

    pipeline: object
    features: List[str]
    run_id: Optional[int]
    class_labels: List[str]
    metadata: Dict[str, object]


def load_model_artifact(model_path: Path = MODEL_PATH) -> ModelArtifact:
    """Carrega o artefato persistido do modelo."""
    if not model_path.exists():
        raise ModelNotFoundError(f"Modelo não encontrado em {model_path}.")
    obj = load(model_path)
    return ModelArtifact(
        pipeline=obj.get("pipeline"),
        features=list(obj.get("features", FEATURES)),
        run_id=obj.get("run_id"),
        class_labels=list(obj.get("class_labels", CLASS_LABELS)),
        metadata=obj,
    )


def save_model_artifact(artifact: ModelArtifact, model_path: Path = MODEL_PATH) -> None:
    """Persiste o artefato do modelo em disco."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            **artifact.metadata,
            "pipeline": artifact.pipeline,
            "features": artifact.features,
            "class_labels": artifact.class_labels,
            "run_id": artifact.run_id,
        },
        model_path,
    )


def extract_class_probabilities(
    pipeline,
    X: pd.DataFrame,
    class_labels: Iterable[str],
) -> List[Dict[str, Optional[float]]]:
    """Extrai probabilidades por classe, quando suportado pelo pipeline."""
    class_labels_list = list(class_labels)
    if not hasattr(pipeline, "predict_proba"):
        return [{label: None for label in class_labels_list} for _ in range(len(X))]

    proba = pipeline.predict_proba(X)
    estimator = pipeline.named_steps.get("clf") if hasattr(pipeline, "named_steps") else None
    classes = None
    if estimator is not None and hasattr(estimator, "classes_"):
        classes = list(estimator.classes_)
    elif hasattr(pipeline, "classes_"):
        classes = list(pipeline.classes_)
    if not classes:
        return [{label: None for label in class_labels_list} for _ in range(len(X))]

    idx_map = {label: classes.index(label) for label in classes if label in class_labels_list}
    results: List[Dict[str, Optional[float]]] = []
    for row in proba:
        row_probs: Dict[str, Optional[float]] = {}
        for label in class_labels_list:
            idx = idx_map.get(label)
            row_probs[label] = float(row[idx]) if idx is not None else None
        results.append(row_probs)
    return results
