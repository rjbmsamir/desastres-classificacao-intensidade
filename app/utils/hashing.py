"""Funções utilitárias de hashing para deduplicação."""
from __future__ import annotations

import json
from hashlib import sha256
from typing import Mapping


def make_row_hash(payload: Mapping[str, object]) -> str:
    """Gera hash estável para o payload de uma linha."""
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return sha256(serialized.encode("utf-8")).hexdigest()
