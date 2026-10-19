from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import (
    OUTCOME_EXPIRY_DAYS,
    OUTCOME_MIN_AGE_DAYS,
    OUTCOME_REVIEW_MAX_RECORDS,
    PRED_DAYS,
    REPORT_DIR,
    TIMEZONE,
)
from ..data.bar_contract import check_ohlcv_contract
from ..data.market_data import drop_vendor_placeholder_bars, fetch_intraday_bars
from ..data.synthetic_sessions import build_synthetic_sessions
from ..edge.outcomes import resolve_plan_target_pct, resolve_trade_risk_pct, walk_triple_barrier

# Recorded entry prices come from the decision-time session close, so on a
# consistent price basis entry/close is ~1. The smallest corporate action the
# split-adjusted refetch can introduce is 3:2; anything outside these bounds
# means history was re-scaled under the decision and the outcome would be
# walked on the wrong scale.
ENTRY_SCALE_MIN = 0.75
ENTRY_SCALE_MAX = 1.33
OUTCOME_SESSION_ALIGNMENT_VERSION = 2
_OUTCOME_RESULT_FIELDS = (
    "outcome_label",
    "outcome_ret_5bar_pct",
    "outcome_method",
    "outcome_target_mode",
    "outcome_return_pct",
    "outcome_r_multiple",
    "outcome_exit_reason",
    "outcome_risk_pct_used",
    "outcome_mae_pct",
    "outcome_mfe_pct",
    "outcome_resolved_at",
    "outcome_error",
    "outcome_session_alignment_version",
)


def _evaluate_result(direction: str, entry: float, future_close: float) -> tuple[str, float]:
    if direction == "bullish":
        ret = ((future_close - entry) / entry) * 100.0
    else:
        ret = ((entry - future_close) / entry) * 100.0
    label = "win" if ret > 0 else "loss"
    return label, float(ret)


def _session_atr_pct(synthetic: pd.DataFrame, position: int, entry: float, lookback: int = 14) -> float:
    if entry <= 0 or not {"High", "Low"}.issubset(synthetic.columns):
        return 0.0
    window = synthetic.iloc[max(0, position - lookback + 1) : position + 1]
    if window.empty:
        return 0.0
    spans = (window["High"] - window["Low"]).dropna()
    if spans.empty:
        return 0.0
    return float(spans.mean() / entry * 100.0)


def _triple_barrier_fields(
    synthetic: pd.DataFrame,
    start_pos: int,
    target_pos: int,
    direction: str,
    entry: float,
) -> dict | None:
    """Same outcome definition as the edge lab, applied to journal decisions.

    Labeling by the sign of the close at horizon called a stopped-out trade a
    win if price drifted back green - and those labels drive the adaptive
    policy. Requires OHLC sessions; callers fall back to close-at-horizon
    when High/Low are unavailable.
    """
    if entry <= 0 or not {"High", "Low", "Close"}.issubset(synthetic.columns):
        return None
    risk = resolve_trade_risk_pct(_session_atr_pct(synthetic, start_pos, entry), 0.0, entry)
    # Target comes from the SAME exit-geometry config the lab evidence uses
    # (shipped: no target). A hardcoded 2R here made the adaptive policy
    # learn from an exit geometry that is no longer traded. Journal records
    # carry no empty-space levels, so level-based sweep modes resolve to
    # their documented 2R fallback - identical to the old behaviour.
    plan = resolve_plan_target_pct(0.0, 0.0, 0.0, entry, risk)
    outcome = walk_triple_barrier(
        synthetic.iloc[start_pos + 1 : target_pos + 1],
        direction,
        entry,
        risk,
        plan["target_pct"],
    )
    if outcome["exit_reason"] == "no_data":
        return None
    outcome["target_mode"] = plan["target_mode"]
    return outcome


def _record_decision_timestamp(record: dict) -> pd.Timestamp:
    value = record.get("decision_ts") or record.get("entry_timestamp") or record.get("created_at")
    created = pd.Timestamp(value)
    if pd.isna(created):
        raise ValueError("decision timestamp is missing")
    if created.tzinfo is None:
        return created.tz_localize(TIMEZONE)
    return created.tz_convert(TIMEZONE)


def _now_timestamp() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TIMEZONE)


def _decision_session_position(index: pd.Index, created: pd.Timestamp) -> int:
    sessions = pd.DatetimeIndex(index)
    if sessions.empty:
        return -1
    decision_ts = created
    if sessions.tz is not None:
        decision_ts = decision_ts.tz_convert(sessions.tz)
    else:
        decision_ts = decision_ts.tz_localize(None)
    return int(sessions.searchsorted(decision_ts, side="right")) - 1


def _record_session_position(index: pd.Index, record: dict, created: pd.Timestamp) -> int:
    """Locate the synthetic session that actually produced the decision.

    Synthetic-session indices are start labels, not source-bar timestamps. A
    Monday 14:00 bar under the default 20:00 anchor is therefore labelled
    Sunday 00:00. Comparing the Monday decision timestamp directly with those
    labels selects Monday 00:00 -- the session built from Tuesday's bars --
    and shifts every outcome forward by one trading session.

    New journal rows retain the last source-bar timestamp, so map that timestamp
    through the same anchor rule used by ``build_synthetic_sessions`` and
    require the exact session to be present. Legacy rows fall back to the old
    decision-time lookup because their true source session cannot be recovered
    safely from the journal alone.
    """
    source_value = record.get("latest_source_timestamp")
    if not source_value:
        return _decision_session_position(index, created)

    source_ts = pd.Timestamp(source_value)
    if pd.isna(source_ts):
        return _decision_session_position(index, created)

    sessions = pd.DatetimeIndex(index)
    if sessions.empty:
        return -1
    if sessions.tz is not None:
        source_ts = source_ts.tz_localize(sessions.tz) if source_ts.tzinfo is None else source_ts.tz_convert(sessions.tz)
    elif source_ts.tzinfo is not None:
        source_ts = source_ts.tz_localize(None)

    anchor_minutes = int(record.get("anchor_hour", 20)) * 60 + int(record.get("anchor_minute", 0))
    source_minutes = source_ts.hour * 60 + source_ts.minute
    session_label = source_ts.normalize()
    if source_minutes < anchor_minutes:
        session_label -= pd.Timedelta(days=1)

    matches = sessions.get_indexer([session_label])
    return int(matches[0]) if len(matches) and matches[0] >= 0 else -1


def _synthetic_session_complete(
    session_label: pd.Timestamp,
    *,
    anchor_hour: int,
    anchor_minute: int,
    now: pd.Timestamp,
    data_delay_minutes: int = 0,
) -> bool:
    """Return whether a synthetic session has reached its closing anchor."""
    label = pd.Timestamp(session_label)
    now_ts = pd.Timestamp(now)
    if label.tzinfo is not None:
        now_ts = now_ts.tz_localize(label.tz) if now_ts.tzinfo is None else now_ts.tz_convert(label.tz)
        next_day = label.tz_localize(None).normalize() + pd.Timedelta(days=1)
        completion = (next_day + pd.Timedelta(hours=anchor_hour, minutes=anchor_minute)).tz_localize(label.tz)
    else:
        now_ts = now_ts.tz_localize(None) if now_ts.tzinfo is not None else now_ts
        completion = label.normalize() + pd.Timedelta(days=1, hours=anchor_hour, minutes=anchor_minute)
    completion += pd.Timedelta(minutes=max(int(data_delay_minutes or 0), 0))
    return bool(now_ts >= completion)


def _needs_session_alignment_repair(record: dict) -> bool:
    if record.get("outcome_status") != "resolved" or not record.get("latest_source_timestamp"):
        return False
    try:
        version = int(record.get("outcome_session_alignment_version", 0) or 0)
    except (TypeError, ValueError):
        version = 0
    return version < OUTCOME_SESSION_ALIGNMENT_VERSION


def _reset_resolved_outcome(record: dict) -> None:
    for field in _OUTCOME_RESULT_FIELDS:
        record.pop(field, None)
    record["outcome_status"] = "pending"


def review_pending_outcomes(records: list[dict], logger) -> tuple[list[dict], dict]:
    now = _now_timestamp()
    updated = 0
    skipped = 0
    expired = 0
    resolved_live = 0
    resolved_counterfactual = 0
    quarantined = 0
    contract_blocked = 0
    alignment_repairs_resolved = 0

    alignment_repairs = [r for r in records if _needs_session_alignment_repair(r)]
    pending = [r for r in records if r.get("outcome_status") == "pending"]
    reviewed_repairs = alignment_repairs[:OUTCOME_REVIEW_MAX_RECORDS]
    remaining_slots = max(OUTCOME_REVIEW_MAX_RECORDS - len(reviewed_repairs), 0)
    worklist = reviewed_repairs + pending[:remaining_slots]
    alignment_repair_ids = {id(record) for record in reviewed_repairs}

    for rec in worklist:
        try:
            repairing_alignment = id(rec) in alignment_repair_ids
            if repairing_alignment:
                # Once the alignment defect is known, its old label must not
                # remain eligible for tuning if the corrective fetch cannot
                # complete today. Fail closed as pending until recomputation.
                _reset_resolved_outcome(rec)
            if not rec.get("direction") or rec.get("entry_price") is None:
                rec["outcome_status"] = "not_applicable"
                skipped += 1
                continue

            created = _record_decision_timestamp(rec)
            too_old_to_ever_resolve = (now - created).days > OUTCOME_EXPIRY_DAYS

            if (now - created).days < OUTCOME_MIN_AGE_DAYS:
                continue

            # Outcomes are the ground truth the tuning loops learn from, and
            # they resolve days after the decision - so the free 16-min-
            # delayed SIP feed applies, and consolidated High/Low (not the
            # ~3% IEX slice) decides whether stop/target were really touched.
            intraday = fetch_intraday_bars(rec["ticker"], research=True)
            # A halted session arrives as a zero-volume forward-filled
            # placeholder that would block resolution forever ("retry later"
            # never gets cleaner for a historical halt); drop it so the walk
            # sees only real prints.
            intraday = drop_vendor_placeholder_bars(intraday)
            if intraday is not None and not intraday.empty:
                violations, _warnings = check_ohlcv_contract(intraday, profile="intraday")
                if violations:
                    # Corrupt bars must not become win/loss labels; leave the
                    # record pending and retry on the next (cleaner) fetch.
                    rec["outcome_error"] = f"bar contract violation: {'; '.join(violations)}"
                    contract_blocked += 1
                    continue
            synthetic, _ = build_synthetic_sessions(intraday, rec.get("anchor_hour", 20), rec.get("anchor_minute", 0), "30m", True)
            if synthetic.empty:
                continue

            start_pos = _record_session_position(synthetic.index, rec, created)
            if start_pos < 0:
                # The decision predates the available intraday window; that
                # window only slides forward, so this can never resolve.
                if too_old_to_ever_resolve:
                    rec["outcome_status"] = "not_applicable"
                    rec["outcome_error"] = "expired_before_resolution_window"
                    expired += 1
                continue
            target_pos = start_pos + PRED_DAYS
            if target_pos >= len(synthetic):
                continue
            if not _synthetic_session_complete(
                synthetic.index[target_pos],
                anchor_hour=int(rec.get("anchor_hour", 20)),
                anchor_minute=int(rec.get("anchor_minute", 0)),
                now=now,
                data_delay_minutes=int(synthetic.attrs.get("data_delay_minutes", 0) or 0),
            ):
                # A market-hours run can already see the fifth synthetic bar
                # while it is still forming. Wait for the anchor (and feed
                # delay) so afternoon stop/target touches and the final close
                # cannot change a label after it reaches autonomous tuning.
                continue

            entry_price = float(rec["entry_price"])
            decision_close = float(synthetic.iloc[start_pos]["Close"])
            if decision_close > 0:
                scale = entry_price / decision_close
                if not (ENTRY_SCALE_MIN <= scale <= ENTRY_SCALE_MAX):
                    # A split (or symbol re-scaling) between decision and
                    # review moved the bars onto a different price basis than
                    # the recorded entry; any barrier walk would be fiction.
                    rec["outcome_status"] = "not_applicable"
                    rec["outcome_error"] = (
                        f"entry price scale mismatch (entry={entry_price:.4f}, "
                        f"decision session close={decision_close:.4f}); likely corporate action"
                    )
                    quarantined += 1
                    continue
            future_close = float(synthetic.iloc[target_pos]["Close"])
            close_label, ret5 = _evaluate_result(rec["direction"], entry_price, future_close)

            rec["outcome_status"] = "resolved"
            rec["outcome_ret_5bar_pct"] = ret5
            barrier = _triple_barrier_fields(synthetic, start_pos, target_pos, rec["direction"], entry_price)
            if barrier is not None:
                rec["outcome_label"] = barrier["label"]
                rec["outcome_method"] = barrier["method"]
                rec["outcome_target_mode"] = barrier.get("target_mode")
                rec["outcome_return_pct"] = barrier["return_pct"]
                rec["outcome_r_multiple"] = barrier["r_multiple"]
                rec["outcome_exit_reason"] = barrier["exit_reason"]
                rec["outcome_risk_pct_used"] = barrier["risk_pct_used"]
                rec["outcome_mae_pct"] = barrier["mae_pct"]
                rec["outcome_mfe_pct"] = barrier["mfe_pct"]
            else:
                rec["outcome_label"] = close_label
                rec["outcome_method"] = "close_horizon"
                rec["outcome_return_pct"] = ret5
            rec["outcome_resolved_at"] = pd.Timestamp.utcnow().isoformat()
            rec["outcome_session_alignment_version"] = OUTCOME_SESSION_ALIGNMENT_VERSION
            updated += 1
            if repairing_alignment:
                alignment_repairs_resolved += 1
            if rec.get("counterfactual"):
                resolved_counterfactual += 1
            else:
                resolved_live += 1
        except Exception as exc:
            rec["outcome_error"] = str(exc)

    summary = {
        "pending_reviewed": len(worklist),
        "resolved_now": updated,
        "resolved_live": resolved_live,
        "resolved_counterfactual": resolved_counterfactual,
        "marked_not_applicable": skipped,
        "expired_unresolvable": expired,
        "quarantined_scale_mismatch": quarantined,
        "blocked_by_bar_contract": contract_blocked,
        "alignment_repairs_reviewed": len(reviewed_repairs),
        "alignment_repairs_resolved": alignment_repairs_resolved,
        "alignment_repairs_pending": sum(
            1 for record in reviewed_repairs if record.get("outcome_status") == "pending"
        ),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "outcome_review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("OUTCOME_REVIEW_SUMMARY: %s", json.dumps(summary))
    return records, summary
