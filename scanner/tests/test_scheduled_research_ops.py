import json
from datetime import datetime, timezone
from pathlib import Path

from scanner.scheduled_research_ops import (
    completed_for_market_date,
    run_scheduled_research_ops,
)


def test_scheduled_research_ops_batch_propagates_python_exit_code():
    script = Path("scanner/run_research_ops_scheduled.bat").read_text(encoding="utf-8")
    lowered = script.lower()

    assert 'if exist "%~dp0retired.json"' in lowered
    assert "scanner is retired" in lowered
    assert "set \"research_ops_exit=%errorlevel%\"" in lowered
    assert "exit /b %research_ops_exit%" in lowered


def test_scheduled_task_registration_retries_interrupted_runs():
    script = Path("scanner/register_research_ops_task.ps1").read_text(encoding="utf-8")
    lowered = script.lower()

    assert "-restartcount 3" in lowered
    assert "-restartinterval (new-timespan -minutes 15)" in lowered
    assert "-startwhenavailable" in lowered
    assert "-allowstartifonbatteries" in lowered
    assert "-dontstopifgoingonbatteries" in lowered
    assert "-multipleinstances ignorenew" in lowered
    assert "addminutes(30)" in lowered
    assert "addminutes(60)" in lowered
    assert "(get-date) -gt $startat.addminutes(60)" in lowered
    assert "$startat = $startat.adddays(1)" in lowered


def test_scheduled_runner_skips_recovery_after_current_market_day_completed(tmp_path):
    report_path = tmp_path / "research_ops_report.json"
    status_path = tmp_path / "scheduled_research_ops_status.json"
    report_path.write_text(
        json.dumps(
            {
                "mode": "research_ops",
                "completed_at": "2026-08-05T18:38:17+00:00",
                "edge_run_id": "current-run",
                "daily_brief": {"mode": "brief"},
            }
        ),
        encoding="utf-8",
    )
    calls = []

    exit_code = run_scheduled_research_ops(
        now=datetime(2026, 8, 5, 19, 0, tzinfo=timezone.utc),
        report_path=report_path,
        status_path=status_path,
        invoke=lambda: calls.append("run") or 0,
    )

    assert exit_code == 0
    assert calls == []
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "skipped_already_complete"
    assert status["market_open_at"] == "2026-08-05T13:30:00+00:00"
    assert status["market_close_at"] == "2026-08-05T20:00:00+00:00"


def test_scheduled_runner_retries_stale_report_and_requires_completion(tmp_path):
    report_path = tmp_path / "research_ops_report.json"
    status_path = tmp_path / "scheduled_research_ops_status.json"
    report_path.write_text(
        json.dumps(
            {
                "mode": "research_ops",
                "completed_at": "2026-08-04T18:38:17+00:00",
                "edge_run_id": "stale-run",
                "daily_brief": {"mode": "brief"},
            }
        ),
        encoding="utf-8",
    )

    def complete_run():
        report_path.write_text(
            json.dumps(
                {
                    "mode": "research_ops",
                    "completed_at": "2026-08-05T19:05:00+00:00",
                    "edge_run_id": "recovery-run",
                    "daily_brief": {"mode": "brief"},
                }
            ),
            encoding="utf-8",
        )
        return 0

    exit_code = run_scheduled_research_ops(
        now=datetime(2026, 8, 5, 19, 0, tzinfo=timezone.utc),
        report_path=report_path,
        status_path=status_path,
        invoke=complete_run,
    )

    assert exit_code == 0
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "completed"
    assert completed_for_market_date(report_path, datetime(2026, 8, 5).date())


def test_scheduled_runner_fails_closed_when_child_omits_completion_report(tmp_path):
    exit_code = run_scheduled_research_ops(
        now=datetime(2026, 8, 5, 19, 0, tzinfo=timezone.utc),
        report_path=tmp_path / "missing.json",
        status_path=tmp_path / "status.json",
        invoke=lambda: 0,
    )

    assert exit_code == 1
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed_missing_completion_report"


def test_scheduled_runner_skips_nyse_holiday(tmp_path):
    calls = []

    exit_code = run_scheduled_research_ops(
        now=datetime(2026, 7, 3, 18, 30, tzinfo=timezone.utc),
        report_path=tmp_path / "missing.json",
        status_path=tmp_path / "status.json",
        invoke=lambda: calls.append("run") or 0,
    )

    assert exit_code == 0
    assert calls == []
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "skipped_market_closed"


def test_scheduled_runner_skips_after_early_close(tmp_path):
    calls = []

    exit_code = run_scheduled_research_ops(
        # 2026-11-27 closes at 13:00 ET; the usual 14:30 ET launch is too late.
        now=datetime(2026, 11, 27, 19, 30, tzinfo=timezone.utc),
        report_path=tmp_path / "missing.json",
        status_path=tmp_path / "status.json",
        invoke=lambda: calls.append("run") or 0,
    )

    assert exit_code == 0
    assert calls == []
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "skipped_outside_market_hours"
    assert status["market_close_at"] == "2026-11-27T18:00:00+00:00"
