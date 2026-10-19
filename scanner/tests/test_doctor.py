import scanner.main as scanner_main
from scanner.doctor import run_doctor


def test_doctor_reports_no_secret_values_and_ignored_runtime_artifacts():
    report = run_doctor()
    rendered = str(report)

    assert report["mode"] == "doctor"
    assert "checks" in report
    assert "runtime_artifacts_ignored" in report["checks"]
    assert report["checks"]["runtime_artifacts_ignored"]["passed"] is True
    assert "ALPACA_SECRET_KEY=" not in rendered
    assert "TELEGRAM_BOT_TOKEN=" not in rendered
    assert "MINIMAX_API_KEY=" not in rendered


def test_parse_args_accepts_doctor_mode(monkeypatch):
    monkeypatch.setattr("sys.argv", ["scanner.main", "--mode", "doctor"])

    args = scanner_main.parse_args()

    assert args.mode == "doctor"
