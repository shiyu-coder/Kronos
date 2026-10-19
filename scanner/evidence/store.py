from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvidenceRun:
    mode: str
    root_dir: Path
    params: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    started_at: str = field(default_factory=_utc_now)
    _rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return self.root_dir / self.run_id

    def record_rows(self, kind: str, rows: Iterable[dict[str, Any]]) -> None:
        clean_rows = [self._with_run_fields(kind, row) for row in rows]
        if not clean_rows:
            return
        self._rows.setdefault(kind, []).extend(clean_rows)

    def record_metrics(self, namespace: str, metrics: dict[str, Any]) -> None:
        rows = []
        for metric, value in metrics.items():
            rows.append(
                {
                    "namespace": namespace,
                    "metric": metric,
                    "value": value,
                }
            )
        self.record_rows("metrics", rows)

    def log_artifact(self, source: str | Path, artifact_name: str | None = None) -> None:
        source_path = Path(source)
        if not source_path.exists() or not source_path.is_file():
            return
        output_name = artifact_name or source_path.name
        artifact_dir = self.run_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        destination = artifact_dir / output_name
        shutil.copy2(source_path, destination)
        self._artifacts[output_name] = {
            "path": str(destination.relative_to(self.run_dir)).replace("\\", "/"),
            "bytes": destination.stat().st_size,
        }

    def flush(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for kind, rows in sorted(self._rows.items()):
            jsonl_path = self.run_dir / f"{kind}.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
            artifact = {
                "path": jsonl_path.name,
                "rows": len(rows),
                "format": "jsonl",
            }
            parquet_path = self._try_write_parquet(kind, rows)
            if parquet_path is not None:
                artifact["parquet_path"] = parquet_path.name
            self._artifacts[kind] = artifact

        manifest = {
            "run_id": self.run_id,
            "mode": self.mode,
            "started_at": self.started_at,
            "finished_at": _utc_now(),
            "params": _json_safe(self.params),
            "tags": _json_safe(self.tags),
            "artifacts": self._artifacts,
        }
        manifest_path = self.run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path

    def _with_run_fields(self, kind: str, row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("mode", self.mode)
        payload.setdefault("row_kind", kind)
        payload.setdefault("recorded_at", _utc_now())
        return payload

    def _try_write_parquet(self, kind: str, rows: list[dict[str, Any]]) -> Path | None:
        parquet_path = self.run_dir / f"{kind}.parquet"
        try:
            pd.DataFrame([_json_safe(row) for row in rows]).to_parquet(parquet_path, index=False)
        except Exception:
            return None
        return parquet_path


def start_evidence_run(
    mode: str,
    root_dir: str | Path,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
) -> EvidenceRun:
    return EvidenceRun(
        mode=mode,
        root_dir=Path(root_dir),
        params=params or {},
        tags=tags or {},
    )
