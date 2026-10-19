from scanner.evidence.provenance import edge_runtime_fingerprint


def test_runtime_fingerprint_ignores_tests_but_tracks_runtime_source(tmp_path):
    (tmp_path / "edge").mkdir()
    (tmp_path / "tests").mkdir()
    runtime_file = tmp_path / "edge" / "scoring.py"
    test_file = tmp_path / "tests" / "test_scoring.py"
    runtime_file.write_text("VALUE = 1\n", encoding="utf-8")
    test_file.write_text("def test_value(): pass\n", encoding="utf-8")

    baseline = edge_runtime_fingerprint(tmp_path)
    test_file.write_text("def test_value(): assert True\n", encoding="utf-8")
    assert edge_runtime_fingerprint(tmp_path) == baseline

    runtime_file.write_text("VALUE = 2\n", encoding="utf-8")
    assert edge_runtime_fingerprint(tmp_path) != baseline
