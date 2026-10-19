"""Deterministic provenance for the production scanner runtime."""

from __future__ import annotations

import hashlib
from pathlib import Path


def edge_runtime_fingerprint(source_root: str | Path | None = None) -> str:
    """Hash production Python source that can affect Edge evidence or execution.

    Tests, generated reports/logs, and one-off experiment harnesses are excluded:
    changing any of those must not invalidate a production evidence run. Runtime
    modules are read from the working tree, so committed and uncommitted source
    edits both change the fingerprint.
    """
    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[1]
    source_files: list[tuple[str, Path]] = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        parts = relative.parts
        if not parts or parts[0] in {"tests", "reports", "logs"}:
            continue
        if "__pycache__" in parts:
            continue
        if len(parts) >= 2 and parts[:2] == ("research", "experiments"):
            continue
        source_files.append((relative.as_posix(), path))

    digest = hashlib.sha256()
    for relative, path in sorted(source_files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"
