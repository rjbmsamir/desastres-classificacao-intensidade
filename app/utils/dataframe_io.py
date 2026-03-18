"""Leitura de arquivos tabulares de entrada."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataframe(path: Path, *, sheet_name: str) -> pd.DataFrame:
    """Carrega DataFrame a partir de CSV, Excel ou Parquet."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(path, sep=";", engine="python")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Formato de arquivo não suportado: {path.suffix}")
