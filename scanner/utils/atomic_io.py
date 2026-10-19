"""Crash-safe file writes for evidence stores and reports.

A plain write_text truncates the target before writing: a crash mid-write
leaves a corrupt journal/index and the next run either dies or silently
rebuilds evidence from nothing. The universal fix is write-temp + fsync +
os.replace (atomic on NTFS and POSIX when the temp file shares the target's
directory - same filesystem is required for an atomic rename).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def atomic_write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=indent, ensure_ascii=False, default=str))
