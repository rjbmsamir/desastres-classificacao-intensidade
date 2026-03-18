"""Configurações centrais do projeto."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

DATA_PATH = Path(os.getenv("DATA_PATH", BASE_DIR / "data" / "desastres.xlsx"))
SHEET_NAME = os.getenv("SHEET_NAME", "desastres")
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "projeto-desastres.db"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "app" / "model.joblib"))
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "cluster_H")
POSITIVE_CLASS = os.getenv("POSITIVE_CLASS", "raro")
NEGATIVE_CLASS = os.getenv("NEGATIVE_CLASS", "comum")
CV_SPLITS = int(os.getenv("CV_SPLITS", "5"))
CV_REPEATS = int(os.getenv("CV_REPEATS", "5"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
