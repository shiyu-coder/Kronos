# Technology Stack

**Analysis Date:** 2026-07-02

## Languages

**Primary:**
- Python 3.12.10 (venv interpreter; `pyproject.toml` declares `requires-python = ">=3.10"`) - Everything: scanner (`scanner/`), Kronos model (`model/`), Streamlit app (`kronos_app.py`), Flask webui (`webui/`), finetune tooling (`finetune/`, `finetune_csv/`)

**Secondary:**
- Windows Batch - Launchers and installers: `launch_kronos.bat`, `install_deps.bat`, `scanner/run_scanner.bat`, `scanner/setup_dependencies.bat`
- HTML/CSS/JS - Single Flask template `webui/templates/index.html`; inline CSS in `kronos_app.py` via `st.markdown`

## Runtime

**Environment:**
- CPython 3.12.10 on Windows 11 (all `.bat` scripts hardcode `venv\Scripts\python.exe`)
- Two virtualenvs exist at repo root: `venv/` (primary — referenced by every batch script and `scanner/README.md`) and `.venv/` (also Python 3.12.10). Use `venv/` for anything scripted.
- GPU is optional. `kronos_app.py:224` auto-detects CUDA (`torch.cuda.is_available()`), but the installed torch is `2.7.0+cpu` — CPU-only. `install_deps.bat` and `README_JAKE.md` reference CUDA builds (cu128, RTX 5060 Ti), which does NOT match the current venv. Treat CPU inference as the working state.

**Package Manager:**
- pip (no poetry/uv)
- Lockfile: missing. Pins live in `requirements.txt` (root), `scanner/requirements-scanner.txt`, `webui/requirements.txt`, and `pyproject.toml`. Installed venv versions have drifted above several pins (see Key Dependencies).

**Packaging:**
- `pyproject.toml` defines package `kronos-predictor-local` v0.1.0, setuptools build backend, packages `model*` and `scanner*` (tests excluded). Installable via `pip install -e .`; optional extras: `test` (pytest), `evidence` (pyarrow).

## Frameworks

**Core:**
- PyTorch 2.7.0+cpu (`torch==2.7.0` in `pyproject.toml`; `torch==2.7.0+cpu` via `--extra-index-url https://download.pytorch.org/whl/cpu` in `requirements.txt`) - Kronos model inference in `model/kronos.py`, `model/module.py`
- huggingface_hub 0.33.1 - Model/tokenizer loading via `PyTorchModelHubMixin.from_pretrained` in `model/kronos.py:13`
- pandas 2.2.2 + numpy (installed 2.4.6) - All market data, bars, backtests, edge features
- Streamlit (installed 1.56.0, unpinned) - One-click forecasting app `kronos_app.py`, launched by `launch_kronos.bat` on port 8501
- Flask 2.3.3 + flask-cors (pinned 4.0.0, installed 6.0.2) - Legacy upstream webui `webui/app.py`, default `127.0.0.1:7070`

**Testing:**
- pytest (pinned `>=8.2.0`, installed 9.0.3) - Config in `pytest.ini`; testpaths are `tests/` (model regression/sampling safety, webui security) and `scanner/tests/` (27 test modules covering edge engine, learning loop, data adapters, Telegram, doctor). Run: `.\venv\Scripts\python.exe -m pytest -q`

**Build/Dev:**
- setuptools >=69 + wheel - Build backend (`pyproject.toml`)
- No linter or formatter configured (no ruff/flake8/black/mypy config anywhere)
- No CI (no `.github/` directory); verification is the local runbook in `scanner/README.md` ("Quick Validation Runbook")

## Key Dependencies

**Critical:**
- yfinance (pinned `>=0.2.54`, installed 1.2.1) - Fallback bars, ticker validation, options chains, earnings/dividend calendar (`scanner/data/market_data.py`, `scanner/data/options_data.py`, `scanner/data/events.py`, `kronos_app.py`)
- requests (pinned `>=2.32.0`, installed 2.33.1) - All raw HTTP: Alpaca, Telegram, MiniMax (no vendor SDKs anywhere in `scanner/`)
- python-dotenv (pinned `>=1.0.1`, installed 1.2.2) - `.env` loading via `dotenv_values` in `scanner/main.py:185` (`_load_project_env_files`)
- einops 0.8.1, safetensors 0.6.2 - Kronos model internals (`model/module.py`, HF weight loading)
- pytz (pinned `>=2024.1`, `scanner/requirements-scanner.txt`) - Timezone handling; scanner is pinned to `America/New_York` (`scanner/config.py:51`)

**Infrastructure:**
- plotly (pinned 5.17.0, installed 6.7.0) - Charts in `kronos_app.py` and `webui/app.py`
- matplotlib 3.9.3, tqdm 4.67.1 - Upstream examples/finetune plotting and progress
- pyarrow (optional `evidence` extra, `>=16.0.0`, installed 23.0.1) - Parquet sidecars for evidence runs in `scanner/evidence/store.py:121` (`_try_write_parquet`); JSONL remains canonical fallback

**Declared/legacy but NOT wired into the scanner (upstream fork residue):**
- qlib + comet_ml - Only in `finetune/` (`finetune/qlib_data_preprocess.py`, `finetune/train_tokenizer.py:15`); not installed in venv
- akshare + tkinter - Only in `examples/` (Chinese-market demo scripts); not installed
- ccxt - Listed in `install_deps.bat:20` but never imported anywhere; dead reference
- torch.distributed (RANK/WORLD_SIZE/LOCAL_RANK) - Only in `finetune/utils/training_utils.py` and `finetune_csv/`

**Version drift warning:** installed venv versions exceed pins for flask-cors (6.0.2 vs 4.0.0), plotly (6.7.0 vs 5.17.0), numpy (2.4.6), yfinance (1.2.1), pytest (9.0.3). `requirements.txt` pins are not an accurate snapshot of the running environment.

## Configuration

**Environment:**
- Secrets via `.env` only. `scanner/main.py:79` (`ENV_PATHS`) checks repo root `.env` (not present) then `scanner/.env` (present — never read/commit it; gitignored via `.gitignore:64`). Template: `scanner/.env.example`.
- Existing OS env vars win over `.env` values (`scanner/main.py:192`).
- Non-secret runtime config is code: `scanner/config.py` (thresholds, model names, provider defaults, watchlist).
- Self-tuning overrides: `scanner/tuning/overrides.json` merged over `scanner/config.py` constants at import via `_apply_overrides()` (`scanner/config.py:102`); refresh with `reload_overrides()`.
- Webui env vars (all optional, safe local defaults): `KRONOS_WEBUI_DATA_DIR`, `KRONOS_WEBUI_PORT` (7070), `KRONOS_WEBUI_HOST` (127.0.0.1), `KRONOS_WEBUI_DEBUG`, `KRONOS_WEBUI_CORS_ORIGINS` (`webui/app.py:19-58`).
- Finetune config: `finetune/config.py` `Config` class (qlib paths, Comet ML keys as placeholders) — upstream demo, not used by scanner.

**Key env vars consumed by the scanner** (read in `scanner/main.py:197` `_load_env` and per-module `os.getenv` calls):
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `MARKET_DATA_PROVIDER` (auto|alpaca|yfinance), `ALPACA_FEED` (iex|sip), `ALPACA_OPTIONS_FEED` (indicative default)
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `LIVE_MODE_ENABLED`, `HEARTBEAT_ENABLED`
- `MINIMAX_ENABLED`, `MINIMAX_API_KEY`, `MINIMAX_BASE_URL`, `MINIMAX_MODEL`, `MINIMAX_TEMPERATURE`, `MINIMAX_MAX_OUTPUT_TOKENS`, `MINIMAX_TIMEOUT_SECONDS`

**Build:**
- `pyproject.toml` - Package metadata, deps, extras
- `pytest.ini` - Test discovery (`tests`, `scanner/tests`)
- `install_deps.bat` - Legacy CUDA-oriented installer (out of sync with CPU venv)
- `scanner/setup_dependencies.bat` - Canonical scanner setup: upgrades pip, installs `requirements.txt` + `scanner/requirements-scanner.txt` into `venv/`

## Platform Requirements

**Development:**
- Windows (paths, `.bat` launchers, `C:\Users\Jacob Higgins\projects\kronos-predictor` hardcoded in `install_deps.bat` and `launch_kronos.bat`)
- Python 3.10+ (enforced by `scanner/doctor.py:64` doctor check; venv is 3.12.10)
- ~500MB disk for HF model cache at `C:\Users\Jacob Higgins\.cache\huggingface\hub` (first-run download)
- Internet access for yfinance/Alpaca/HF Hub; Alpaca+Telegram+MiniMax keys optional (scanner degrades gracefully/fail-closed)

**Production:**
- None. This is a local desktop tool: Streamlit app via `launch_kronos.bat` (port 8501), scanner via `scanner/run_scanner.bat` / `python -m scanner.main`, webui via `webui/app.py` (localhost only). No hosting, no deployment pipeline.
- Health check before runs: `.\venv\Scripts\python.exe -m scanner.main --mode doctor` (`scanner/doctor.py` — verifies Python version, core imports, and that secret/artifact paths are gitignored)

---

*Stack analysis: 2026-07-02*
