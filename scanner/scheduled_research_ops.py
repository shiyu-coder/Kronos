"""Idempotent launcher for the scheduled research-operations cycle.

Task Scheduler does not apply RestartCount when a process is terminated by a
console Ctrl+C event (0xC000013A).  Recovery triggers therefore invoke this
module again later in the same market session.  A completed report for the
current New York trading date makes those later invocations cheap no-ops.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from .utils.atomic_io import atomic_write_json


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "scanner" / "reports" / "research_ops_report.json"
STATUS_PATH = REPO_ROOT / "scanner" / "reports" / "scheduled_research_ops_status.json"
MARKET_TIMEZONE = ZoneInfo("America/New_York")


def _aware_utc(now: datetime | None = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _parse_report_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _market_session_window(market_date: date) -> tuple[datetime, datetime] | None:
    """Return the regular NYSE session in UTC, or None when the exchange is closed."""

    import pandas_market_calendars as mcal

    schedule = mcal.get_calendar("NYSE").schedule(
        start_date=market_date,
        end_date=market_date,
    )
    if schedule.empty:
        return None

    market_open = schedule.iloc[0]["market_open"].to_pydatetime()
    market_close = schedule.iloc[0]["market_close"].to_pydatetime()
    return _aware_utc(market_open), _aware_utc(market_close)


def completed_for_market_date(report_path: Path, market_date: date) -> bool:
    """Return True only for a parseable, complete report from market_date."""

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False

    completed_at = _parse_report_timestamp(report.get("completed_at"))
    return bool(
        report.get("mode") == "research_ops"
        and completed_at is not None
        and completed_at.astimezone(MARKET_TIMEZONE).date() == market_date
        and report.get("edge_run_id")
        and isinstance(report.get("daily_brief"), dict)
    )


def _default_invoke() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "scanner.main", "--mode", "research_ops"],
        cwd=REPO_ROOT,
        check=False,
    )
    return int(completed.returncode)


def run_scheduled_research_ops(
    *,
    now: datetime | None = None,
    report_path: Path = REPORT_PATH,
    status_path: Path = STATUS_PATH,
    invoke: Callable[[], int] = _default_invoke,
    session_window_lookup: Callable[[date], tuple[datetime, datetime] | None] = _market_session_window,
) -> int:
    """Run once during an NYSE session unless today's cycle already exists."""

    started_at = _aware_utc(now)
    market_date = started_at.astimezone(MARKET_TIMEZONE).date()
    base_status = {
        "mode": "scheduled_research_ops",
        "market_date": market_date.isoformat(),
        "checked_at": started_at.isoformat(),
        "report_path": str(report_path.resolve()),
    }

    try:
        session_window = session_window_lookup(market_date)
    except Exception as exc:
        status = {
            **base_status,
            "status": "failed_market_calendar_check",
            "exit_code": 1,
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_write_json(status_path, status)
        print(f"Unable to verify the NYSE session for {market_date}; research ops blocked.")
        return 1

    if session_window is None:
        status = {**base_status, "status": "skipped_market_closed", "exit_code": 0}
        atomic_write_json(status_path, status)
        print(f"NYSE is closed on {market_date}; research ops skipped.")
        return 0

    market_open, market_close = (_aware_utc(value) for value in session_window)
    session_status = {
        "market_open_at": market_open.isoformat(),
        "market_close_at": market_close.isoformat(),
    }
    if market_close <= market_open:
        status = {
            **base_status,
            **session_status,
            "status": "failed_market_calendar_check",
            "exit_code": 1,
            "error": "NYSE session close must be after open",
        }
        atomic_write_json(status_path, status)
        print(f"Invalid NYSE session window for {market_date}; research ops blocked.")
        return 1
    if completed_for_market_date(report_path, market_date):
        status = {
            **base_status,
            **session_status,
            "status": "skipped_already_complete",
            "exit_code": 0,
        }
        atomic_write_json(status_path, status)
        print(f"Research ops already complete for {market_date}; recovery launch skipped.")
        return 0
    if not market_open <= started_at < market_close:
        status = {
            **base_status,
            **session_status,
            "status": "skipped_outside_market_hours",
            "exit_code": 0,
        }
        atomic_write_json(status_path, status)
        print(f"NYSE is outside regular hours; research ops skipped for {market_date}.")
        return 0

    atomic_write_json(status_path, {**base_status, **session_status, "status": "running"})
    try:
        exit_code = int(invoke())
    except KeyboardInterrupt:
        atomic_write_json(
            status_path,
            {
                **base_status,
                **session_status,
                "status": "interrupted",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": 130,
            },
        )
        return 130

    report_complete = completed_for_market_date(report_path, market_date)
    if exit_code == 0 and not report_complete:
        exit_code = 1
        final_status = "failed_missing_completion_report"
    elif exit_code == 0:
        final_status = "completed"
    else:
        final_status = "failed"

    atomic_write_json(
        status_path,
        {
            **base_status,
            **session_status,
            "status": final_status,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "exit_code": exit_code,
            "report_complete": report_complete,
        },
    )
    return exit_code


def main() -> int:
    return run_scheduled_research_ops()


if __name__ == "__main__":
    raise SystemExit(main())
