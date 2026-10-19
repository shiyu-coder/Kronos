from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..edge.features import extract_edge_features
from ..edge.outcomes import resolve_plan_target_pct, resolve_trade_risk_pct, walk_triple_barrier
from ..strategy.empty_space import score_empty_space
from ..strategy.potter_box import detect_potter_box, score_potter_research_candidate
from ..strategy.potter_doctrine import score_potter_doctrine_v2

if TYPE_CHECKING:
    # Import-for-typing only: keeps retrieval.py free of the model-loading
    # adapter at runtime when the historical build doesn't opt into Kronos.
    from ..models.kronos_adapter import KronosAdapter


@dataclass
class EdgeRecord:
    ticker: str
    timestamp: str
    direction: str
    features: dict[str, Any]
    outcome_return_pct: float
    outcome_label: str
    r_multiple: float
    mae_pct: float
    mfe_pct: float
    # Defaults keep older index files loadable.
    exit_reason: str = "horizon"
    risk_pct_used: float = 0.0
    outcome_method: str = "close_horizon"
    target_pct_used: float = 0.0
    target_mode: str = "nearest_empty_space"


# Setup-shape features only: scale-free, present in both historical and live
# records, and free of price-level or provider-dependent dimensions. Price
# levels (latest_close, box_top, atr_value) made analogs cluster by share
# price, and options/kronos/data fields are populated live but zero/absent in
# history, which distorted live-vs-historical distances.
ANALOG_FEATURE_KEYS = frozenset(
    {
        "potter_passed",
        "empty_space_passed",
        "box_width_pct",
        "close_position_in_box",
        "breakout_distance_pct",
        "abs_breakout_distance_pct",
        "breakout_strength_pct",
        "range_compression_ratio",
        "no_trend_score",
        "top_touches",
        "bottom_touches",
        "volume_expansion",
        "volume_percentile",
        "realized_volatility_pct",
        "recent_return_pct",
        "empty_space_score",
        "rr_ratio",
        "distance_to_target_pct",
        "risk_pct",
        "doctrine_v2_passed",
        "doctrine_v2_score",
        "doctrine_v2_box_stack_score",
        "doctrine_v2_punchback_reclaim",
        "doctrine_v2_failed_reentry",
        "research_score",
        "research_passed",
    }
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _parse_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def _numeric_feature_keys(query_features: dict, candidate_features: dict) -> list[str]:
    keys = []
    for key in sorted(ANALOG_FEATURE_KEYS.intersection(query_features).intersection(candidate_features)):
        qv = query_features.get(key)
        cv = candidate_features.get(key)
        if isinstance(qv, bool) or isinstance(cv, bool):
            continue
        if isinstance(qv, (int, float)) and isinstance(cv, (int, float)):
            if math.isfinite(float(qv)) and math.isfinite(float(cv)):
                keys.append(key)
    return keys


def _distance(query_features: dict, candidate_features: dict) -> float:
    keys = _numeric_feature_keys(query_features, candidate_features)
    if not keys:
        return float("inf")
    pieces = []
    for key in keys:
        qv = float(query_features[key])
        cv = float(candidate_features[key])
        scale = max(abs(qv), abs(cv), 1.0)
        pieces.append(((qv - cv) / scale) ** 2)
    return float(np.sqrt(np.mean(pieces)))


def _is_numeric_feature(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _record_payload(record: EdgeRecord) -> dict[str, Any]:
    return asdict(record)


def _future_outcome(
    bars: pd.DataFrame,
    idx: int,
    horizon: int,
    direction: str,
    entry: float,
    risk_pct: float,
    target_pct: float = 0.0,
    atr_value: float = 0.0,
    next_target_pct: float = 0.0,
) -> dict[str, Any]:
    """Triple-barrier outcome for one historical candidate.

    The old label was the sign of the close exactly `horizon` bars out, which
    called a stopped-out trade a "win" if price later drifted back green.
    """
    future = bars.iloc[idx + 1 : idx + 1 + horizon]
    if future.empty or entry <= 0:
        return {
            "return_pct": 0.0,
            "label": "loss",
            "r_multiple": 0.0,
            "mae_pct": 0.0,
            "mfe_pct": 0.0,
            "exit_reason": "no_data",
            "risk_pct_used": 0.0,
            "method": "triple_barrier",
            "target_pct_used": 0.0,
            "target_mode": "no_data",
        }
    risk = resolve_trade_risk_pct(risk_pct, atr_value, entry)
    plan = resolve_plan_target_pct(
        _finite_float(target_pct),
        _finite_float(next_target_pct),
        _finite_float(atr_value),
        entry,
        risk,
    )
    outcome = walk_triple_barrier(future, direction, entry, risk, plan["target_pct"])
    # None target (no-target plan) is stored as 0.0; target_mode disambiguates.
    outcome["target_pct_used"] = plan["target_pct"] if plan["target_pct"] is not None else 0.0
    outcome["target_mode"] = plan["target_mode"]
    return outcome


def build_edge_records_from_bars(
    ticker: str,
    bars: pd.DataFrame,
    horizon: int = 5,
    min_history: int = 35,
    kronos: "KronosAdapter | None" = None,
    kronos_direction_filter: str | None = None,
) -> list[EdgeRecord]:
    """Rebuild the walk-forward edge index from a ticker's daily bars.

    ``kronos`` is opt-in and off by default (``None``), so the default call
    path is byte-identical to the pre-Kronos behavior: no adapter, no model
    load, no ``kronos_*`` fields populated. When an adapter is supplied, each
    decision window is evaluated through it the same way the live pipeline
    does (``kronos.evaluate(ticker, window, direction)``), and the result is
    handed to ``extract_edge_features`` unprefixed, exactly like
    ``KronosAdapter.evaluate``'s live callers already do.

    ``kronos_direction_filter`` (default ``None``) restricts which decision
    windows are actually forwarded to the adapter. ``None`` preserves today's
    behavior: every built record with a Kronos adapter present is evaluated.
    Set it to e.g. ``"bullish"`` and only windows whose ``direction`` matches
    are evaluated; every other record is still created identically, just with
    ``kr=None`` (same as "Kronos never consulted"). This is a pure
    cost-control lever - the records it skips are exactly the ones a
    bullish-only ranking test would never read - and it changes nothing about
    the records it does evaluate. It has no effect when ``kronos is None``.
    """
    if bars is None or len(bars) <= min_history + horizon:
        return []
    clean = bars.sort_index().copy()
    records: list[EdgeRecord] = []

    for idx in range(min_history, len(clean) - horizon):
        window = clean.iloc[: idx + 1]
        pb = detect_potter_box(ticker, window)
        direction = pb.direction
        research = score_potter_research_candidate(pb, window)
        if direction not in {"bullish", "bearish"}:
            direction = research.get("direction")
        if direction not in {"bullish", "bearish"}:
            continue

        entry = _finite_float(pb.breakout_close)
        if entry <= 0:
            continue
        es = score_empty_space(window, direction, entry, pb.cost_basis or entry)
        doctrine = score_potter_doctrine_v2(ticker, window, pb, es)

        kr = None
        if kronos is not None and (kronos_direction_filter is None or direction == kronos_direction_filter):
            # Leakage invariant: Kronos may only ever see bars at or before
            # the decision timestamp. `window` is already a `.iloc[:idx+1]`
            # prefix of `clean`, but this guard makes that a hard, checked
            # property rather than an implicit one a future refactor could
            # silently break (a leak here manufactures a false edge).
            decision_ts = pd.Timestamp(clean.index[idx])
            window_max_ts = pd.Timestamp(window.index.max())
            if window_max_ts > decision_ts:
                raise RuntimeError(
                    f"kronos leakage guard tripped for {ticker} idx={idx}: "
                    f"window max timestamp {window_max_ts} is after decision "
                    f"timestamp {decision_ts}"
                )
            kr_result = kronos.evaluate(ticker, window, direction)
            # Same handling as the live research path (_kronos_research_fields
            # in main.py): a None directional_agreement means Kronos failed to
            # load or infer, not that it disagreed. Passing that result through
            # would make extract_edge_features' `if kr else None` guard see a
            # truthy-but-empty dict and silently default to 0.0, which reads
            # downstream as "maximum disagreement" - a fabricated signal. Leave
            # kr as None (same as "Kronos never consulted") on that path.
            if kr_result.directional_agreement is not None:
                kr = kr_result

        features = extract_edge_features(ticker, window, pb, es, doctrine_v2=doctrine, kronos=kr)
        features["direction"] = direction
        features["research_score"] = _finite_float(research.get("score"))
        features["research_passed"] = 1.0 if research.get("passed") else 0.0
        outcome = _future_outcome(
            clean,
            idx,
            horizon,
            direction,
            entry,
            risk_pct=_finite_float(features.get("risk_pct")),
            target_pct=_finite_float(features.get("distance_to_target_pct")),
            atr_value=_finite_float(features.get("atr_value")),
            next_target_pct=_finite_float(es.diagnostics.get("distance_to_next_target_pct")),
        )
        records.append(
            EdgeRecord(
                ticker=ticker,
                timestamp=str(features.get("timestamp") or pd.Timestamp(clean.index[idx]).isoformat()),
                direction=direction,
                features=features,
                outcome_return_pct=outcome["return_pct"],
                outcome_label=outcome["label"],
                r_multiple=outcome["r_multiple"],
                mae_pct=outcome["mae_pct"],
                mfe_pct=outcome["mfe_pct"],
                exit_reason=outcome["exit_reason"],
                risk_pct_used=outcome["risk_pct_used"],
                outcome_method=outcome["method"],
                target_pct_used=outcome["target_pct_used"],
                target_mode=outcome["target_mode"],
            )
        )
    return records


def find_analogs(
    query_features: dict,
    records: Iterable[EdgeRecord | dict],
    k: int = 7,
    embargo_days: int = 5,
    allow_future: bool = True,
    direction_match: bool = False,
    cross_ticker_embargo_days: int = 0,
) -> list[dict[str, Any]]:
    if isinstance(records, EdgeAnalogIndex):
        return records.find_analogs(
            query_features,
            k=k,
            embargo_days=embargo_days,
            allow_future=allow_future,
            direction_match=direction_match,
            cross_ticker_embargo_days=cross_ticker_embargo_days,
        )

    query_ticker = str(query_features.get("ticker", "")).upper()
    query_ts = _parse_ts(query_features.get("timestamp"))
    query_direction = str(query_features.get("direction", ""))
    scored = []

    for raw in records:
        record = raw if isinstance(raw, EdgeRecord) else EdgeRecord(**raw)
        candidate_ts = _parse_ts(record.timestamp)
        if (
            direction_match
            and query_direction in {"bullish", "bearish"}
            and record.direction in {"bullish", "bearish"}
            and record.direction != query_direction
        ):
            continue
        if (
            query_ticker
            and record.ticker.upper() == query_ticker
            and query_ts is not None
            and candidate_ts is not None
            and abs((query_ts - candidate_ts).days) < embargo_days
        ):
            continue
        if (
            cross_ticker_embargo_days > 0
            and query_ts is not None
            and candidate_ts is not None
            and abs((query_ts - candidate_ts).days) < cross_ticker_embargo_days
        ):
            continue
        if not allow_future and query_ts is not None and candidate_ts is not None and candidate_ts >= query_ts:
            continue
        dist = _distance(query_features, record.features)
        if not math.isfinite(dist):
            continue
        payload = _record_payload(record)
        payload["distance"] = dist
        scored.append(payload)

    scored.sort(key=lambda row: row["distance"])
    return scored[:k]


def select_recent_records(records: list[EdgeRecord], limit: int) -> list[EdgeRecord]:
    """Most recent records by timestamp across all tickers.

    The index file is grouped ticker-by-ticker, so a plain tail slice
    validates only the last few watchlist names instead of a
    cross-sectional sample.
    """
    if limit <= 0 or len(records) <= limit:
        return list(records)

    def sort_key(record: EdgeRecord) -> tuple[int, pd.Timestamp]:
        ts = _parse_ts(record.timestamp)
        if ts is None:
            return (0, pd.Timestamp(0, tz="UTC"))
        return (1, ts)

    ordered = sorted(records, key=sort_key)
    return ordered[-limit:]


class EdgeAnalogIndex:
    """Vectorized in-memory analog search over EdgeRecord feature dictionaries."""

    def __init__(self, records: Iterable[EdgeRecord | dict]):
        self.records = [raw if isinstance(raw, EdgeRecord) else EdgeRecord(**raw) for raw in records]
        self._tickers = [record.ticker.upper() for record in self.records]
        self._directions = [str(record.direction) for record in self.records]
        self._timestamps = [_parse_ts(record.timestamp) for record in self.records]
        self._payloads: list[dict[str, Any] | None] = [None] * len(self.records)
        self._keys = self._feature_keys(self.records)
        self._matrix = self._feature_matrix(self.records, self._keys)

    def find_analogs(
        self,
        query_features: dict,
        k: int = 7,
        embargo_days: int = 5,
        allow_future: bool = True,
        direction_match: bool = False,
        cross_ticker_embargo_days: int = 0,
    ) -> list[dict[str, Any]]:
        if not self.records or k <= 0:
            return []

        distances = self._distances(query_features)
        if direction_match:
            query_direction = str(query_features.get("direction", ""))
            if query_direction in {"bullish", "bearish"}:
                for idx, direction in enumerate(self._directions):
                    if direction in {"bullish", "bearish"} and direction != query_direction:
                        distances[idx] = np.inf
        self._apply_time_filters(distances, query_features, embargo_days, allow_future, cross_ticker_embargo_days)
        finite_idx = np.flatnonzero(np.isfinite(distances))
        if finite_idx.size == 0:
            return []

        limit = min(k, finite_idx.size)
        if finite_idx.size > limit:
            candidate_idx = finite_idx[np.argpartition(distances[finite_idx], limit - 1)[:limit]]
        else:
            candidate_idx = finite_idx
        ordered_idx = candidate_idx[np.argsort(distances[candidate_idx], kind="stable")]
        return [self._payload_with_distance(int(idx), float(distances[idx])) for idx in ordered_idx]

    def _distances(self, query_features: dict) -> np.ndarray:
        distances = np.full(len(self.records), np.inf, dtype=float)
        if not self._keys:
            return distances

        query_vector = np.full(len(self._keys), np.nan, dtype=float)
        for idx, key in enumerate(self._keys):
            value = query_features.get(key)
            if _is_numeric_feature(value):
                query_vector[idx] = float(value)
        if not np.isfinite(query_vector).any():
            return distances

        valid = np.isfinite(self._matrix) & np.isfinite(query_vector)
        counts = valid.sum(axis=1)
        usable = counts > 0
        if not usable.any():
            return distances

        scale = np.maximum.reduce([np.abs(self._matrix), np.broadcast_to(np.abs(query_vector), self._matrix.shape), np.ones_like(self._matrix)])
        pieces = np.where(valid, ((self._matrix - query_vector) / scale) ** 2, 0.0)
        distances[usable] = np.sqrt(pieces[usable].sum(axis=1) / counts[usable])
        return distances

    def _apply_time_filters(
        self,
        distances: np.ndarray,
        query_features: dict,
        embargo_days: int,
        allow_future: bool,
        cross_ticker_embargo_days: int = 0,
    ) -> None:
        query_ticker = str(query_features.get("ticker", "")).upper()
        query_ts = _parse_ts(query_features.get("timestamp"))
        if query_ts is None:
            return
        for idx, (ticker, candidate_ts) in enumerate(zip(self._tickers, self._timestamps, strict=False)):
            if candidate_ts is None:
                continue
            if not allow_future and candidate_ts >= query_ts:
                distances[idx] = np.inf
                continue
            gap_days = abs((query_ts - candidate_ts).days)
            if query_ticker and ticker == query_ticker and gap_days < embargo_days:
                distances[idx] = np.inf
                continue
            if cross_ticker_embargo_days > 0 and gap_days < cross_ticker_embargo_days:
                distances[idx] = np.inf

    def _payload_with_distance(self, idx: int, distance: float) -> dict[str, Any]:
        payload = self._payloads[idx]
        if payload is None:
            payload = _record_payload(self.records[idx])
            self._payloads[idx] = payload
        out = dict(payload)
        out["distance"] = distance
        return out

    @classmethod
    def _feature_keys(cls, records: list[EdgeRecord]) -> list[str]:
        keys: set[str] = set()
        for record in records:
            for key, value in record.features.items():
                if key in ANALOG_FEATURE_KEYS and _is_numeric_feature(value):
                    keys.add(key)
        return sorted(keys)

    @staticmethod
    def _feature_matrix(records: list[EdgeRecord], keys: list[str]) -> np.ndarray:
        matrix = np.full((len(records), len(keys)), np.nan, dtype=float)
        for row_idx, record in enumerate(records):
            for col_idx, key in enumerate(keys):
                value = record.features.get(key)
                if _is_numeric_feature(value):
                    matrix[row_idx, col_idx] = float(value)
        return matrix


def save_edge_index(records: Iterable[EdgeRecord], path: str | Path) -> Path:
    # Atomic replace: this is a ~30MB single-file store rewritten wholesale
    # each lab run; a crash mid-write must never corrupt the whole evidence
    # base.
    from ..utils.atomic_io import atomic_write_text

    output_path = Path(path)
    atomic_write_text(output_path, json.dumps([asdict(record) for record in records], indent=2))
    return output_path


_EDGE_RECORD_FIELDS = {f.name for f in fields(EdgeRecord)}


def load_edge_index(path: str | Path) -> list[EdgeRecord]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"edge index at {input_path} is corrupt (invalid JSON at line {exc.lineno}); "
            "rebuild it with --mode build_retrieval_index"
        ) from exc
    if not isinstance(payload, list):
        return []
    # Ignore unknown keys so an index written by a newer schema (or one
    # hand-edited with extra fields) degrades gracefully instead of killing
    # every downstream lab stage with a TypeError.
    return [
        EdgeRecord(**{key: value for key, value in row.items() if key in _EDGE_RECORD_FIELDS})
        for row in payload
        if isinstance(row, dict)
    ]
