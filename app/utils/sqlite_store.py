"""Persistência SQLite opcional para métricas e predições."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from app.core.config import DB_PATH
from app.core.features import CLASS_LABELS


def get_conn() -> sqlite3.Connection:
    """Abre conexão SQLite com `row_factory` configurado."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Inicializa o schema SQLite e migra colunas opcionais."""
    schema_file = Path(__file__).resolve().parents[1] / "schema.sql"
    with get_conn() as conn, open(schema_file, "r", encoding="utf-8") as handle:
        conn.executescript(handle.read())
        _ensure_additional_columns(conn)


@contextmanager
def db_session():
    """Abre uma sessão transacional simples do SQLite."""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_run(
    *,
    algorithm: str,
    params: Mapping[str, object],
    data_path: Path,
    n_samples: int,
    n_features: int,
    target_positive: str,
) -> int:
    """Cria um registro de execução de treinamento."""
    started_at = datetime.utcnow().isoformat()
    payload = json.dumps(params, default=str)
    with db_session() as conn:
        cur = conn.execute(
            (
                "INSERT INTO runs (started_at, algorithm, params_json, data_path, "
                "n_samples, n_features, target_positive) VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                started_at,
                algorithm,
                payload,
                str(data_path),
                n_samples,
                n_features,
                target_positive,
            ),
        )
        return int(cur.lastrowid)


def finalize_run(run_id: int, metrics: Mapping[str, object]) -> None:
    """Finaliza a execução persistindo o resumo agregado."""
    finished_at = datetime.utcnow().isoformat()
    metrics_json = json.dumps(metrics, default=str)
    with db_session() as conn:
        conn.execute(
            "UPDATE runs SET finished_at = ?, metrics_summary_json = ? WHERE id = ?",
            (finished_at, metrics_json, run_id),
        )


def insert_cv_metrics(run_id: int, rows: Sequence[Mapping[str, object]]) -> None:
    """Persiste as métricas por dobra de validação cruzada."""
    if not rows:
        return
    payload = []
    for row in rows:
        entry = [
            run_id,
            int(row["fold"]),
            float(row["f1_macro"]),
            float(row["balanced_accuracy"]),
        ]
        for label in CLASS_LABELS:
            entry.extend(
                [
                    float(row.get(f"precision_{label}", 0.0)),
                    float(row.get(f"recall_{label}", 0.0)),
                    float(row.get(f"f1_{label}", 0.0)),
                ]
            )
        payload.append(tuple(entry))
    with db_session() as conn:
        conn.executemany(
            (
                "INSERT INTO cv_metrics (run_id, fold, f1_macro, balanced_accuracy, "
                "precision_nivel_I, recall_nivel_I, f1_nivel_I, "
                "precision_nivel_II, recall_nivel_II, f1_nivel_II, "
                "precision_nivel_III, recall_nivel_III, f1_nivel_III, "
                "precision_raro, recall_raro, f1_raro) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            payload,
        )


def insert_predictions(rows: Iterable[Mapping[str, object]]) -> None:
    """Persiste predições e payloads de entrada para auditoria."""
    payload = [
        (
            row.get("run_id"),
            row.get("source"),
            row.get("row_hash"),
            row.get("y_true"),
            row.get("y_pred"),
            _prob_raro(row.get("probabilities")),
            _prob_comum(row.get("probabilities")),
            json.dumps(row.get("probabilities"), ensure_ascii=False, default=str),
            json.dumps(row.get("payload"), ensure_ascii=False, default=str),
        )
        for row in rows
    ]
    if not payload:
        return
    with db_session() as conn:
        conn.executemany(
            (
                "INSERT OR REPLACE INTO predictions (run_id, source, row_hash, y_true, "
                "y_pred, proba_rara, proba_comum, probabilities_json, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            payload,
        )


def fetch_recent_runs(limit: int = 10):
    """Consulta as execuções recentes."""
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM runs ORDER BY COALESCE(finished_at, started_at) DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()


def _prob_raro(probabilities: Mapping[str, object] | None) -> float | None:
    if not probabilities:
        return None
    value = probabilities.get("raro")
    return float(value) if value is not None else None


def _prob_comum(probabilities: Mapping[str, object] | None) -> float | None:
    if not probabilities:
        return None
    total = 0.0
    found = False
    for label in CLASS_LABELS:
        if label == "raro":
            continue
        value = probabilities.get(label)
        if value is not None:
            total += float(value)
            found = True
    return total if found else None


def _ensure_additional_columns(conn: sqlite3.Connection) -> None:
    table_defs = {
        "predictions": {
            "probabilities_json": "TEXT",
        },
        "cv_metrics": {
            "precision_nivel_I": "REAL",
            "recall_nivel_I": "REAL",
            "f1_nivel_I": "REAL",
            "precision_nivel_II": "REAL",
            "recall_nivel_II": "REAL",
            "f1_nivel_II": "REAL",
            "precision_nivel_III": "REAL",
            "recall_nivel_III": "REAL",
            "f1_nivel_III": "REAL",
            "precision_raro": "REAL",
            "recall_raro": "REAL",
            "f1_raro": "REAL",
        },
    }

    for table, columns in table_defs.items():
        existing_cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, col_type in columns.items():
            if column not in existing_cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
