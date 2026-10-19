# Repo Map — three products, one repo

Start here. This repository is a fork of the upstream [Kronos](https://github.com/shiyu-coder/Kronos) research model with two local products layered on top. No single README describes everything — this map says which doc is authoritative for what.

| Product | What it is | Entry point | Docs |
|---|---|---|---|
| Kronos foundation model (upstream) | Financial K-line time-series model this fork builds on | `model/` | [README.md](README.md) |
| Desktop forecasting app | Local one-click Streamlit chart forecaster | `kronos_app.py` via `launch_kronos.bat` | [README_JAKE.md](README_JAKE.md) |
| Potter Box scanner / evidence lab | **Retired trading strategy**; preserved fail-closed research archive | `python -m scanner.main` via `scanner/run_scanner.bat` | [scanner/README.md](scanner/README.md) and [retirement record](docs/RETIREMENT-2026-08-20.md) |
| Potter v3 research path | **Moved 2026-10-19** to its own project, `projects\Potter Box Strategy` (package `potter_box`), along with the source transcripts and the 2026-09-05 pre-registration/results | — | that project's README.md |

## Environment truth

- One virtualenv: `venv/` at repo root (Python 3.12, **PyTorch 2.7.0+cpu — CPU-only**). Every batch script hard-requires `venv\Scripts\python.exe`. Do not create a `.venv/`.
- Any CUDA/RTX claim in older docs is stale: no CUDA torch build is installed. `kronos_app.py` auto-detects the device and runs on CPU in the current environment.
- Setup: `install_deps.bat` (creates `venv/` if missing, then installs the pinned requirements). Dependency pins live in `requirements.txt` + `scanner/requirements-scanner.txt` and are kept in sync with the installed venv.
- Upstream research folders (`examples/`, `figures/`, `finetune/`, `finetune_csv/`, `webui/`) are fork baggage — not part of either local product's runtime.

## For agents — guardrails that outlive any session

- The scanner strategy is **retired and fail-closed**. Never remove `scanner/RETIRED.json`, re-enable its schedules, loosen thresholds, weaken gates, or force alerts unless every revival criterion in [the retirement record](docs/RETIREMENT-2026-08-20.md) has been met and a human explicitly authorizes revival.
- No profit claims from toy validation. Live alerting requires `--mode live` + valid Telegram credentials + `LIVE_MODE_ENABLED=true` + a passing readiness audit — do not shortcut any of them.
- Data-quality tiers are enforced by the scanner's gates; don't bypass or spoof them.
- New Potter Box method work goes in `projects\Potter Box Strategy`, not here. Removing `scanner/potter_v3/` on 2026-10-19 changed the runtime fingerprint (`scanner/evidence/provenance.py`) of any older edge audit report; that is expected.
- Verify before claiming anything works:

  ```powershell
  .\venv\Scripts\python.exe -m pytest -q
  .\venv\Scripts\python.exe -m scanner.main --mode doctor
  ```

- Point-in-time codebase analysis lives in `.planning/codebase/` (dated snapshots); day-to-day operating truth is `scanner/README.md` + `docs/daily-notes/`.

---

*This file replaced `LLM_PROJECT_MEMORY.md` (deleted 2026-07-02 — its 2026-05-24 contents predated research_ops, adaptive policy, and Potter Doctrine v2 and had gone stale).*
