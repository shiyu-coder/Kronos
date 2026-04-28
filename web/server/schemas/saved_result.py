"""
schemas/saved_result.py
────────────────────────
Pydantic models for the Save / List / Get saved-result APIs.

The `data` field in both SaveResultRequest and SavedResultDetail is typed
as the actual prediction response payload:
  - PricePredictionResponse        for type="predict/price"
  - list[PricePredictionResponse]  for type="predict/price/batch"

This mirrors exactly what /prediction/price and /prediction/price/batch return,
so the frontend can pass the response straight through without any reshaping.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from schemas.prediction import PricePredictionResponse

# ── Type discriminator ────────────────────────────────────────────────────────

ResultType = Literal["predict/price", "predict/price/batch"]

# Union: single response OR list of responses (batch)
PredictionData = PricePredictionResponse | list[PricePredictionResponse]


# ── Request ───────────────────────────────────────────────────────────────────


class SaveResultRequest(BaseModel):
    """Body sent by the client when saving a prediction result.

    `data` must be the exact JSON object returned by either:
      - POST /prediction/price        → PricePredictionResponse
      - POST /prediction/price/batch  → list[PricePredictionResponse]
    """

    type: ResultType = Field(
        ...,
        description="Logical result type. Must match the data shape.",
    )
    label: str | None = Field(
        default=None,
        description="Optional human-readable label (e.g. 'ETHUSDT 1h').",
    )
    data: PredictionData = Field(
        ...,
        description="The full prediction response payload.",
    )


# ── Responses ─────────────────────────────────────────────────────────────────


class SavedResultMeta(BaseModel):
    """Metadata-only view of a saved result (used in list responses)."""

    id: str
    type: ResultType
    label: str | None
    created_at: str


class SavedResultDetail(BaseModel):
    """Full view of a saved result including the typed data payload."""

    id: str
    type: ResultType
    label: str | None
    data: PredictionData
    created_at: str
