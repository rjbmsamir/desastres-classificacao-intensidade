"""Abstrações de persistência de execuções e predições."""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Protocol, Sequence

from app.utils.sqlite_store import (
    create_run,
    finalize_run,
    init_db,
    insert_cv_metrics,
    insert_predictions,
)


class ExperimentTracker(Protocol):
    """Contrato mínimo para rastreamento de execuções."""

    def start_run(
        self,
        *,
        algorithm: str,
        params: Mapping[str, object],
        data_path,
        n_samples: int,
        n_features: int,
        target_positive: str,
    ) -> Optional[int]:
        """Inicia uma nova execução e retorna seu identificador."""

    def log_cv_metrics(self, run_id: Optional[int], rows: Sequence[Mapping[str, object]]) -> None:
        """Registra métricas de validação cruzada."""

    def log_predictions(self, rows: Iterable[Mapping[str, object]]) -> None:
        """Registra predições produzidas pela aplicação."""

    def finalize_run(self, run_id: Optional[int], metrics: Mapping[str, object]) -> None:
        """Finaliza a execução persistindo o resumo agregado."""


class NullExperimentTracker:
    """Implementação nula para cenários sem persistência."""

    def start_run(self, **kwargs) -> Optional[int]:  # noqa: ANN003
        return None

    def log_cv_metrics(self, run_id: Optional[int], rows: Sequence[Mapping[str, object]]) -> None:
        return None

    def log_predictions(self, rows: Iterable[Mapping[str, object]]) -> None:
        return None

    def finalize_run(self, run_id: Optional[int], metrics: Mapping[str, object]) -> None:
        return None


class SQLiteExperimentTracker:
    """Adaptador SQLite para rastreamento opcional de execuções."""

    def __init__(self) -> None:
        init_db()

    def start_run(
        self,
        *,
        algorithm: str,
        params: Mapping[str, object],
        data_path,
        n_samples: int,
        n_features: int,
        target_positive: str,
    ) -> int:
        return create_run(
            algorithm=algorithm,
            params=params,
            data_path=data_path,
            n_samples=n_samples,
            n_features=n_features,
            target_positive=target_positive,
        )

    def log_cv_metrics(self, run_id: Optional[int], rows: Sequence[Mapping[str, object]]) -> None:
        if run_id is None:
            return
        insert_cv_metrics(run_id, rows)

    def log_predictions(self, rows: Iterable[Mapping[str, object]]) -> None:
        insert_predictions(rows)

    def finalize_run(self, run_id: Optional[int], metrics: Mapping[str, object]) -> None:
        if run_id is None:
            return
        finalize_run(run_id, metrics)
