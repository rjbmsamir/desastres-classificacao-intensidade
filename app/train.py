"""Rotina de treinamento do modelo."""
from __future__ import annotations

import argparse
from pathlib import Path

from .core.config import DATA_PATH, MODEL_PATH, SHEET_NAME
from .services.tracking import NullExperimentTracker, SQLiteExperimentTracker
from .services.training_service import TrainingResult, build_pipeline, train_model
from .utils.dataframe_io import load_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo de classificação de desastres.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Caminho para a base de treinamento (xlsx ou csv).",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        default=SHEET_NAME,
        help="Nome da aba quando a base for Excel.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Caminho de saída para o modelo treinado.",
    )
    parser.add_argument(
        "--disable-tracking",
        action="store_true",
        help="Executa o treinamento sem persistir métricas e predições no SQLite.",
    )
    return parser.parse_args()


def load_dataset(path: Path, sheet_name: str):
    """Mantém compatibilidade com o carregamento histórico de datasets."""
    return load_dataframe(path, sheet_name=sheet_name)


def main() -> None:
    args = parse_args()
    tracker = NullExperimentTracker() if args.disable_tracking else SQLiteExperimentTracker()
    result = train_model(
        data_path=args.data_path,
        sheet_name=args.sheet_name,
        model_path=args.model_path,
        tracker=tracker,
    )

    run_label = f"Run #{result.run_id}" if result.run_id is not None else "Execução sem persistência"
    print(f"{run_label} finalizada.")
    for key, value in sorted(result.metrics_summary.items()):
        print(f"{key}: {value:.4f}")
    print(f"Modelo salvo em {result.model_path}")


if __name__ == "__main__":
    main()
