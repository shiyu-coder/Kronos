"""
db/connection.py
────────────────
SQLite connection factory for the Kronos server.

The database file path is resolved as follows (first match wins):
  1. KRONOS_DB_PATH environment variable
  2. <server-root>/db/kronos.db   (default)

All files related to the database (schema migrations, the .db file itself)
live in the same `db/` folder so the persistence layer is self-contained.
"""

import os
import sqlite3
from pathlib import Path

# ── Default DB path: web/server/db/kronos.db ────────────────────────────────
_DEFAULT_DB_PATH = Path(__file__).parent / "kronos.db"
DB_PATH: Path = Path(os.environ.get("KRONOS_DB_PATH", str(_DEFAULT_DB_PATH)))


def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row-factory set to Row."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrent read performance
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn
