"""
db/migrations.py
─────────────────
Schema bootstrap / migration runner for the Kronos SQLite database.

Call `run_migrations()` once at application startup (via FastAPI lifespan).
Each migration is idempotent — safe to run on every server restart.
"""

import logging

from .connection import get_connection

logger = logging.getLogger(__name__)

# ── Migration list (append-only, never edit existing entries) ────────────────
_MIGRATIONS: list[str] = [
    # v1 — initial schema
    """
    CREATE TABLE IF NOT EXISTS saved_results (
        id         TEXT PRIMARY KEY,
        type       TEXT NOT NULL,
        label      TEXT,
        data       TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    # v2 — index for fast listing / filtering by type
    """
    CREATE INDEX IF NOT EXISTS idx_saved_results_type
    ON saved_results (type);
    """,
    # v3 — index for chronological listing
    """
    CREATE INDEX IF NOT EXISTS idx_saved_results_created_at
    ON saved_results (created_at DESC);
    """,
]


def run_migrations() -> None:
    """Execute all pending DDL migrations against the database."""
    with get_connection() as conn:
        for sql in _MIGRATIONS:
            conn.execute(sql)
        conn.commit()
    logger.info("DB migrations applied successfully (db: %s)", _get_db_path())


def _get_db_path() -> str:
    from .connection import DB_PATH

    return str(DB_PATH)
