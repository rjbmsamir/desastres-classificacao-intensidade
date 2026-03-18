"""Leitura de arquivos tabulares de entrada."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd


class UnsupportedFileFormatError(ValueError):
    """Disparado quando o tipo de arquivo não é suportado."""


class EmptyFileError(ValueError):
    """Disparado quando o arquivo não possui conteúdo útil."""


class FileParsingError(ValueError):
    """Disparado quando o arquivo não pode ser convertido em DataFrame."""


def _ensure_non_empty_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Valida que o DataFrame possui pelo menos uma linha."""
    if df.empty:
        raise EmptyFileError("Arquivo vazio ou sem linhas processáveis.")
    return df


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Lê CSV a partir de bytes com fallback simples de UTF-8."""
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            text = content.decode(encoding)
            return pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise FileParsingError(f"Falha ao interpretar CSV: {exc}") from exc
    raise FileParsingError("Falha ao decodificar CSV em UTF-8.")


def _load_dataframe_from_suffix(suffix: str, source, *, sheet_name: str) -> pd.DataFrame:
    """Lê um DataFrame com base na extensão informada."""
    if suffix in {".xlsx", ".xls"}:
        try:
            return _ensure_non_empty_dataframe(pd.read_excel(source, sheet_name=sheet_name))
        except EmptyFileError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise FileParsingError(f"Falha ao interpretar planilha Excel: {exc}") from exc
    if suffix == ".csv":
        if isinstance(source, bytes):
            return _ensure_non_empty_dataframe(_read_csv_bytes(source))
        try:
            return _ensure_non_empty_dataframe(pd.read_csv(source, sep=None, engine="python"))
        except EmptyFileError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise FileParsingError(f"Falha ao interpretar CSV: {exc}") from exc
    if suffix == ".parquet":
        try:
            return _ensure_non_empty_dataframe(pd.read_parquet(source))
        except EmptyFileError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise FileParsingError(f"Falha ao interpretar Parquet: {exc}") from exc
    raise UnsupportedFileFormatError(f"Formato de arquivo não suportado: {suffix}")


def load_dataframe(path: Path, *, sheet_name: str) -> pd.DataFrame:
    """Carrega DataFrame a partir de CSV, Excel ou Parquet."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {path}")
    return _load_dataframe_from_suffix(path.suffix.lower(), path, sheet_name=sheet_name)


def load_uploaded_dataframe(filename: str, content: bytes, *, sheet_name: str) -> pd.DataFrame:
    """Carrega DataFrame a partir de bytes enviados por upload."""
    if not filename:
        raise UnsupportedFileFormatError("Nome de arquivo ausente no upload.")
    if not content.strip():
        raise EmptyFileError("Arquivo vazio.")
    suffix = Path(filename).suffix.lower()
    source = io.BytesIO(content) if suffix in {".xlsx", ".xls"} else content
    return _load_dataframe_from_suffix(suffix, source, sheet_name=sheet_name)
