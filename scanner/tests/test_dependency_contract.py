from pathlib import Path

from scanner.doctor import REQUIRED_MODULES


def _text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").lower()


def test_yfinance_repair_dependencies_are_declared_and_doctored():
    root_requirements = _text("requirements.txt")
    scanner_requirements = _text("scanner/requirements-scanner.txt")
    package_metadata = _text("pyproject.toml")

    for dependency in ("scipy==1.18.0", "yfinance==1.5.1"):
        assert dependency in root_requirements
        assert dependency in scanner_requirements
        assert dependency in package_metadata

    assert "scipy" in REQUIRED_MODULES
