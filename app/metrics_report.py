"""Ferramentas de inspeção das métricas armazenadas no SQLite."""
from __future__ import annotations

import argparse
import json

from .db import fetch_recent_runs, get_conn, init_db
from .features import CLASS_LABELS, to_display_label


def show_runs(limit: int = 10) -> None:
    init_db()
    rows = fetch_recent_runs(limit)
    if not rows:
        print("Nenhuma execução registrada.")
        return
    for row in rows:
        metrics_json = row["metrics_summary_json"] or "{}"
        metrics = json.loads(metrics_json)
        print("-" * 60)
        print(f"Run #{row['id']} - algoritmo: {row['algorithm']}")
        print(f"Período: {row['started_at']} -> {row['finished_at']}")
        print(f"Amostras: {row['n_samples']} | Features: {row['n_features']}")
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.4f}")


def show_predictions(limit: int = 20) -> None:
    init_db()
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT run_id, source, y_true, y_pred, probabilities_json, payload_json "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
    if not rows:
        print("Nenhuma predição registrada.")
        return
    for row in rows:
        payload = json.loads(row["payload_json"])
        print("-" * 60)
        probs = json.loads(row["probabilities_json"] or "{}")
        print(
            f"run_id={row['run_id']} | source={row['source']} | "
            f"y_true={to_display_label(row['y_true'])} | y_pred={to_display_label(row['y_pred'])}"
        )
        for label in CLASS_LABELS:
            print(f"  proba_{label}: {probs.get(label)}")
        print(f"payload: {payload}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta relatórios do banco de métricas.")
    parser.add_argument("--limit", type=int, default=10, help="Quantidade de execuções a listar.")
    parser.add_argument(
        "--show-predictions",
        action="store_true",
        help="Lista também as últimas predições registradas.",
    )
    args = parser.parse_args()
    show_runs(limit=args.limit)
    if args.show_predictions:
        show_predictions(limit=args.limit)


if __name__ == "__main__":
    main()
