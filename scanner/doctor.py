from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "requests",
    "dotenv",
    "yfinance",
    # yfinance price repair imports SciPy at runtime for some bad-bar paths.
    "scipy",
    # Exchange-calendar session-completeness checks in the bar contract.
    "pandas_market_calendars",
)

RUNTIME_ARTIFACTS = (
    "scanner/.env",
    "scanner/logs",
    "scanner/reports/edge_audit_report.json",
    "scanner/reports/evidence/example/manifest.json",
    "scanner/tuning/overrides.json",
    "webui/prediction_results/prediction_example.json",
)


def _check(name: str, passed: bool, detail: str, value=None) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "detail": detail,
        "value": value,
    }


def _missing_modules(module_names: Iterable[str]) -> list[str]:
    missing = []
    for module_name in module_names:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def _git_ignored(root: Path, path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.returncode == 0


def run_doctor(root: str | Path | None = None) -> dict:
    """Return a no-secrets local health report for project review."""
    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    project_root = project_root.resolve()

    missing = _missing_modules(REQUIRED_MODULES)
    ignored = {path: _git_ignored(project_root, path) for path in RUNTIME_ARTIFACTS}
    checks = {
        "python_version": _check(
            "python_version",
            sys.version_info >= (3, 10),
            "Python 3.10+ is required.",
            ".".join(str(part) for part in sys.version_info[:3]),
        ),
        "required_modules": _check(
            "required_modules",
            not missing,
            "Core runtime modules must be importable.",
            {"missing": missing},
        ),
        "runtime_artifacts_ignored": _check(
            "runtime_artifacts_ignored",
            all(ignored.values()),
            "Secrets, logs, reports, tuning, and web prediction outputs should stay out of git.",
            ignored,
        ),
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {
        "mode": "doctor",
        "status": "ok" if not failed else "attention_required",
        "root": str(project_root),
        "checks": checks,
        "failed_checks": failed,
    }
