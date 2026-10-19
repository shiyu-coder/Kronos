from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import REPORT_DIR
from ..utils.atomic_io import atomic_write_text

DECISIONS_PATH = REPORT_DIR / "scan_decisions.jsonl"
QUARANTINE_PATH = REPORT_DIR / "scan_decisions.quarantine.jsonl"

logger = logging.getLogger("scanner.outcome_store")


def decision_fingerprint(record: dict) -> str:
    decision_ts = str(record.get("decision_ts") or record.get("created_at") or "")
    decision_day = decision_ts[:10]
    session_day = _record_session_day(record, decision_day)
    parts = (
        str(record.get("ticker") or "").upper(),
        str(record.get("mode") or ""),
        str(record.get("direction") or ""),
        _record_sample_class(record),
        str(record.get("stage_failed") or "passed_all"),
        session_day,
    )
    return "|".join(parts)


def _record_sample_class(record: dict) -> str:
    status = str(record.get("outcome_status") or "")
    if status in {"pending", "resolved"}:
        return "tracked_sample"
    return "not_applicable"


def _record_session_day(record: dict, decision_day: str) -> str:
    explicit_day = record.get("source_session_date")
    if explicit_day:
        return str(explicit_day)[:10]

    source_ts = record.get("latest_source_timestamp")
    if source_ts:
        try:
            return pd.Timestamp(source_ts).date().isoformat()
        except Exception:
            logger.warning("JOURNAL_BAD_SOURCE_TIMESTAMP: %s", source_ts)

    return _weekend_adjusted_decision_day(decision_day)


def _weekend_adjusted_decision_day(decision_day: str) -> str:
    """Collapse old weekend rows onto Friday when source-session metadata is absent."""
    try:
        ts = pd.Timestamp(decision_day)
    except Exception:
        return decision_day
    if ts.dayofweek == 5:
        return (ts - pd.Timedelta(days=1)).date().isoformat()
    if ts.dayofweek == 6:
        return (ts - pd.Timedelta(days=2)).date().isoformat()
    return decision_day


def _record_rank(record: dict) -> tuple[int, int]:
    status_rank = {"resolved": 3, "pending": 2, "not_applicable": 1}
    populated = sum(value is not None and value != "" for value in record.values())
    return status_rank.get(str(record.get("outcome_status")), 0), populated


def _is_missing(value) -> bool:
    return value is None or value == "" or value == {} or value == []


def _merge_enrichment(existing: dict, incoming: dict) -> tuple[dict, bool]:
    merged = dict(existing)
    changed = False
    for key, value in incoming.items():
        if key in {"created_at", "decision_ts"}:
            continue
        if _is_missing(value):
            continue
        if key not in merged or _is_missing(merged.get(key)):
            merged[key] = value
            changed = True
            continue
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            nested = dict(merged[key])
            for nested_key, nested_value in value.items():
                if not _is_missing(nested_value) and (nested_key not in nested or _is_missing(nested.get(nested_key))):
                    nested[nested_key] = nested_value
                    changed = True
            merged[key] = nested
    return merged, changed


def deduplicate_decisions(records: Iterable[dict]) -> tuple[list[dict], dict]:
    rows = list(records)
    selected: dict[str, tuple[int, dict]] = {}
    for index, record in enumerate(rows):
        payload = dict(record)
        fingerprint = decision_fingerprint(payload)
        existing = selected.get(fingerprint)
        if existing is None:
            selected[fingerprint] = (index, payload)
            continue
        existing_index, existing_payload = existing
        if _record_rank(payload) > _record_rank(existing_payload):
            merged, _changed = _merge_enrichment(payload, existing_payload)
            selected[fingerprint] = (index, merged)
        else:
            merged, _changed = _merge_enrichment(existing_payload, payload)
            selected[fingerprint] = (existing_index, merged)
    clean = [payload for _index, payload in sorted(selected.values(), key=lambda item: item[0])]
    return clean, {
        "input_records": len(rows),
        "unique_records": len(clean),
        "duplicates_removed": max(0, len(rows) - len(clean)),
    }


def append_decision(record: dict) -> bool:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("created_at", pd.Timestamp.utcnow().isoformat())
    fingerprint = decision_fingerprint(payload)
    rows = load_decisions()
    for idx, existing in enumerate(rows):
        if decision_fingerprint(existing) != fingerprint:
            continue
        merged, changed = _merge_enrichment(existing, payload)
        if changed:
            rows[idx] = merged
            save_decisions(rows)
        return False
    with DECISIONS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return True


def load_decisions() -> list[dict]:
    """Load the journal, tolerating exactly one torn FINAL line.

    A crash mid-append can only tear the last line; that line is quarantined
    (not silently deleted - the journal is the system of record behind every
    expectancy statistic). An unparseable line anywhere else means real
    corruption, and the load fails closed instead of silently shrinking the
    evidence.
    """
    if not DECISIONS_PATH.exists():
        return []
    raw_lines = DECISIONS_PATH.read_text(encoding="utf-8").splitlines()
    numbered = [(idx + 1, line.strip()) for idx, line in enumerate(raw_lines) if line.strip()]
    rows = []
    for position, (line_no, line) in enumerate(numbered):
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            if position == len(numbered) - 1:
                _quarantine_torn_line(line_no, line)
                logger.warning(
                    "JOURNAL_TORN_LINE: quarantined unparseable final line %d of %s", line_no, DECISIONS_PATH
                )
                break
            raise RuntimeError(
                f"journal {DECISIONS_PATH} is corrupt at line {line_no} (not a torn append); "
                f"refusing to silently drop evidence rows: {exc}"
            ) from exc
    return rows


def _quarantine_torn_line(line_no: int, content: str) -> None:
    try:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "quarantined_at": pd.Timestamp.utcnow().isoformat(),
            "source_line": line_no,
            "content": content,
        }
        with QUARANTINE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.warning("JOURNAL_QUARANTINE_WRITE_FAILED: line %d could not be preserved", line_no)


def save_decisions(records: Iterable[dict]):
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    atomic_write_text(DECISIONS_PATH, text)
