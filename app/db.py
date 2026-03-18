"""Compatibilidade para persistência SQLite legada."""
from app.utils.hashing import make_row_hash
from app.utils.sqlite_store import *  # noqa: F401,F403
