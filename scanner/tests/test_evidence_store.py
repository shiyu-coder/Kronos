import json

from scanner.evidence.store import start_evidence_run


def test_evidence_run_writes_manifest_rows_and_metrics(tmp_path):
    run = start_evidence_run(
        mode="run_edge_lab",
        root_dir=tmp_path,
        params={"watchlist_count": 2, "thresholds": [45, 55, 65]},
        tags={"git_commit": "abc123"},
    )

    run.record_rows(
        "candidates",
        [
            {"ticker": "SOFI", "edge_score": 52.5, "recommendation": "research"},
            {"ticker": "PLTR", "edge_score": 66.0, "recommendation": "promote"},
        ],
    )
    run.record_metrics("validation", {"samples": 20, "precision_at_55": 0.75})
    run.flush()

    run_dir = tmp_path / run.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    candidate_lines = (run_dir / "candidates.jsonl").read_text(encoding="utf-8").strip().splitlines()
    metric_lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert manifest["run_id"] == run.run_id
    assert manifest["mode"] == "run_edge_lab"
    assert manifest["params"]["watchlist_count"] == 2
    assert manifest["tags"]["git_commit"] == "abc123"
    assert manifest["artifacts"]["candidates"]["rows"] == 2
    assert manifest["artifacts"]["metrics"]["rows"] == 2
    assert json.loads(candidate_lines[0])["ticker"] == "SOFI"
    assert {json.loads(line)["metric"] for line in metric_lines} == {"samples", "precision_at_55"}


def test_evidence_run_logs_existing_artifact(tmp_path):
    source = tmp_path / "edge_validation_report.json"
    source.write_text('{"samples": 10}', encoding="utf-8")

    run = start_evidence_run(mode="validate_edge", root_dir=tmp_path)
    run.log_artifact(source)
    run.flush()

    copied = tmp_path / run.run_id / "artifacts" / "edge_validation_report.json"
    manifest = json.loads((tmp_path / run.run_id / "manifest.json").read_text(encoding="utf-8"))

    assert copied.read_text(encoding="utf-8") == '{"samples": 10}'
    assert manifest["artifacts"]["edge_validation_report.json"]["path"] == "artifacts/edge_validation_report.json"
