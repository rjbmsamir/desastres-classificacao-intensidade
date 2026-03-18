"""Dependências compartilhadas da camada FastAPI."""
from __future__ import annotations

from pathlib import Path

from app.core.config import MODEL_PATH


def get_model_path() -> Path:
    """Fornece o caminho padrão do artefato do modelo."""
    return MODEL_PATH
