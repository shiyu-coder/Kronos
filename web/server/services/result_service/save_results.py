"""
services/saved_result_service.py
──────────────────────────────────
CRUD service for saved prediction results.

`data` is always a typed PricePredictionResponse (single) or
list[PricePredictionResponse] (batch). The service serialises to JSON for
SQLite storage and deserialises back to the correct typed shape on read,
using the `type` column to decide which Pydantic model to use.
"""

import json
import logging
import uuid
from datetime import datetime, timezone

from db.connection import get_connection
from schemas.prediction import PricePredictionResponse
from schemas.saved_result import (PredictionData, ResultType,
                                  SavedResultDetail, SavedResultMeta)

logger = logging.getLogger(__name__)


def _deserialize_data(result_type: ResultType, raw_json: str) -> PredictionData:
    """Parse stored JSON back to the correct typed Python object."""
    payload = json.loads(raw_json)
    if result_type == "predict/price/batch":
        return [PricePredictionResponse.model_validate(item) for item in payload]
    return PricePredictionResponse.model_validate(payload)


class SavedResultService:

    # ── Write ────────────────────────────────────────────────────────────────
    def save(
        self,
        result_type: ResultType,
        data: PredictionData,
        label: str | None = None,
    ) -> SavedResultDetail:
        """Persist a prediction result and return the full saved record."""
        record_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        # Serialise: handle both single response and list of responses
        if isinstance(data, list):
            data_json = json.dumps([item.model_dump() for item in data])
        else:
            data_json = data.model_dump_json()

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO saved_results (id, type, label, data, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (record_id, result_type, label, data_json, created_at),
            )
            conn.commit()

        logger.info("Saved result id=%s type=%s", record_id, result_type)

        return SavedResultDetail(
            id=record_id,
            type=result_type,
            label=label,
            data=data,
            created_at=created_at,
        )

    # ── Read ─────────────────────────────────────────────────────────────────
    def list_all(self) -> list[SavedResultMeta]:
        """Return metadata for all saved results, newest first."""
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT id, type, label, created_at
                FROM saved_results
                ORDER BY created_at DESC
                """).fetchall()

        return [
            SavedResultMeta(
                id=row["id"],
                type=row["type"],
                label=row["label"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_by_id(self, record_id: str) -> SavedResultDetail | None:
        """Return the full saved result for *record_id*, or None if missing."""
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, type, label, data, created_at
                FROM saved_results
                WHERE id = ?
                """,
                (record_id,),
            ).fetchone()

        if row is None:
            return None

        return SavedResultDetail(
            id=row["id"],
            type=row["type"],
            label=row["label"],
            data=_deserialize_data(row["type"], row["data"]),
            created_at=row["created_at"],
        )

    # ── Delete ───────────────────────────────────────────────────────────────
    def delete(self, record_id: str) -> bool:
        """Delete a saved result. Returns True if a row was removed."""
        with get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM saved_results WHERE id = ?",
                (record_id,),
            )
            conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted saved result id=%s", record_id)
        return deleted
