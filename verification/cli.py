#!/usr/bin/env python3
"""Deterministic verification CLI for Kronos agents."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verification" / "out"
SEED_LOCK = OUT / "seed.lock"
FLOW_DIR = OUT / "flows"
SHOT_DIR = OUT / "screenshots"
BASELINES = ROOT / "verification" / "baselines"
FIXTURES = [
    ROOT / "tests" / "data" / "regression_input.csv",
    ROOT / "tests" / "data" / "regression_output_256.csv",
    ROOT / "tests" / "data" / "regression_output_512.csv",
]
SEED = 123


def die(msg: str, code: int = 1) -> None:
    print(f"verify: FAIL: {msg}", file=sys.stderr)
    raise SystemExit(code)


def ok(msg: str) -> None:
    print(f"verify: {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_seed() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in FIXTURES if not p.is_file()]
    if missing:
        die("missing fixtures: " + ", ".join(missing))
    lock = {
        "seed": SEED,
        "fixtures": {str(p.relative_to(ROOT)): sha256_file(p) for p in FIXTURES},
    }
    SEED_LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    ok(f"seed wrote {SEED_LOCK.relative_to(ROOT)} (seed={SEED})")
    for rel, digest in lock["fixtures"].items():
        ok(f"  {rel} sha256={digest[:12]}…")


def load_seed_lock() -> dict:
    if not SEED_LOCK.is_file():
        die("no seed.lock — run ./verify seed first")
    return json.loads(SEED_LOCK.read_text())


def assert_fixtures_match_lock(lock: dict) -> None:
    for rel, expected in lock["fixtures"].items():
        path = ROOT / rel
        if not path.is_file():
            die(f"fixture missing: {rel}")
        actual = sha256_file(path)
        if actual != expected:
            die(f"fixture drift {rel}: expected {expected[:12]}… got {actual[:12]}…")


def write_flow_marker(name: str, status: str, detail: dict) -> None:
    FLOW_DIR.mkdir(parents=True, exist_ok=True)
    path = FLOW_DIR / f"{name}.json"
    payload = {"flow": name, "status": status, "detail": detail}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ok(f"flow marker {path.relative_to(ROOT)} status={status}")


def cmd_run(flow: str) -> None:
    lock = load_seed_lock()
    assert_fixtures_match_lock(lock)
    if flow == "smoke":
        # Deterministic local checks — no network / no weights.
        import numpy as np

        np.random.seed(SEED)
        x = np.arange(8, dtype=np.float64)
        y = (x * 2.0).sum()
        if y != 56.0:
            write_flow_marker(flow, "fail", {"y": y})
            die(f"smoke invariant failed: y={y}")
        # Import local package surface without loading HF weights.
        sys.path.insert(0, str(ROOT))
        import model  # noqa: F401

        write_flow_marker(
            flow,
            "ok",
            {"y": y, "seed": SEED, "fixtures": sorted(lock["fixtures"].keys())},
        )
        ok("run smoke passed")
        return
    if flow == "regression":
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(SEED)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_kronos_regression.py",
            "-q",
            "--tb=line",
        ]
        ok("run regression: " + " ".join(cmd))
        proc = subprocess.run(cmd, cwd=ROOT, env=env)
        status = "ok" if proc.returncode == 0 else "fail"
        write_flow_marker(flow, status, {"returncode": proc.returncode})
        if proc.returncode != 0:
            die(f"regression pytest exited {proc.returncode}")
        ok("run regression passed")
        return
    die(f"unknown flow: {flow} (known: smoke, regression)")


def cmd_screenshot(page: str) -> None:
    lock = load_seed_lock()
    assert_fixtures_match_lock(lock)
    if page != "chart":
        die(f"unknown page: {page} (known: chart)")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(ROOT / "tests/data/regression_input.csv", parse_dates=["timestamps"])
    series = df["close"].iloc[:64].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=100)
    ax.plot(series, color="#1f77b4", linewidth=1.5)
    ax.set_title("Kronos verify chart (seed fixture)")
    ax.set_xlabel("i")
    ax.set_ylabel("close")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    out = SHOT_DIR / "chart.png"
    fig.savefig(out, format="png")
    plt.close(fig)
    digest = sha256_file(out)
    meta = SHOT_DIR / "chart.json"
    meta.write_text(
        json.dumps(
            {"page": "chart", "path": str(out.relative_to(ROOT)), "sha256": digest, "seed": SEED},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    ok(f"screenshot chart -> {out.relative_to(ROOT)} sha256={digest}")


def cmd_check() -> None:
    rules_path = ROOT / "verification" / "rules.yaml"
    if not rules_path.is_file():
        die("missing verification/rules.yaml")
    # Minimal YAML subset reader (no PyYAML required): we also keep a JSON twin? 
    # Prefer stdlib-only: encode rules as JSON for check, keep YAML as docs.
    # Actually parse the small YAML manually for our fixed schema.
    text = rules_path.read_text()
    require_files = []
    require_flow_ok = []
    baselines: dict[str, str] = {}
    require_seed = False
    section = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("require_seed_lock:"):
            require_seed = line.split(":", 1)[1].strip().lower() == "true"
            section = None
            continue
        if line.startswith("require_files:"):
            section = "files"
            continue
        if line.startswith("require_flow_ok:"):
            section = "flows"
            continue
        if line.startswith("baselines:"):
            section = "baselines"
            continue
        if line.startswith("version:"):
            section = None
            continue
        if section == "files" and line.strip().startswith("- "):
            require_files.append(line.strip()[2:].strip())
        elif section == "flows" and line.strip().startswith("- "):
            require_flow_ok.append(line.strip()[2:].strip())
        elif section == "baselines" and ":" in line:
            k, v = line.split(":", 1)
            baselines[k.strip()] = v.strip()

    failures: list[str] = []

    if require_seed:
        if not SEED_LOCK.is_file():
            failures.append("seed.lock missing — run ./verify seed")
        else:
            lock = json.loads(SEED_LOCK.read_text())
            for rel, expected in lock["fixtures"].items():
                path = ROOT / rel
                if not path.is_file():
                    failures.append(f"fixture missing: {rel}")
                elif sha256_file(path) != expected:
                    failures.append(f"fixture drift: {rel}")

    for rel in require_files:
        if not (ROOT / rel).is_file():
            failures.append(f"required file missing: {rel}")

    for flow in require_flow_ok:
        marker = FLOW_DIR / f"{flow}.json"
        if not marker.is_file():
            failures.append(f"flow not run: {flow} — ./verify run {flow}")
            continue
        data = json.loads(marker.read_text())
        if data.get("status") != "ok":
            failures.append(f"flow not ok: {flow} status={data.get('status')}")

    for page, baseline_rel in baselines.items():
        baseline = ROOT / baseline_rel
        shot = SHOT_DIR / f"{page}.png"
        if not baseline.is_file():
            failures.append(f"baseline missing: {baseline_rel}")
            continue
        expected = baseline.read_text().strip().split()[0]
        if not shot.is_file():
            failures.append(f"screenshot missing for {page} — ./verify screenshot {page}")
            continue
        actual = sha256_file(shot)
        if actual != expected:
            failures.append(
                f"screenshot mismatch {page}: baseline {expected[:12]}… got {actual[:12]}…"
            )

    if failures:
        for f in failures:
            print(f"verify: FAIL: {f}", file=sys.stderr)
        raise SystemExit(1)
    ok("check passed (all rules ok)")


def main(argv: list[str]) -> None:
    if len(argv) < 2 or argv[1] in {"-h", "--help", "help"}:
        print(
            "Usage:\n"
            "  ./verify seed\n"
            "  ./verify run <flow>        # smoke | regression\n"
            "  ./verify screenshot <page> # chart\n"
            "  ./verify check\n"
        )
        raise SystemExit(0 if len(argv) > 1 else 2)
    cmd = argv[1]
    if cmd == "seed":
        cmd_seed()
    elif cmd == "run":
        if len(argv) < 3:
            die("usage: ./verify run <flow>")
        cmd_run(argv[2])
    elif cmd == "screenshot":
        if len(argv) < 3:
            die("usage: ./verify screenshot <page>")
        cmd_screenshot(argv[2])
    elif cmd == "check":
        cmd_check()
    else:
        die(f"unknown command: {cmd}")


if __name__ == "__main__":
    main(sys.argv)
