# Kronos Financial Predictor (Desktop App)

**One-click AI market forecasting on your desktop.**
Model: NeoQuasar/Kronos-base (102.3M params) | Runs CPU-only (PyTorch 2.7.0+cpu) — a GPU is used automatically only if a CUDA torch build is installed

---

## What's in this repo

Three products share this repo — this file covers only the desktop forecasting app:

| Product | Docs |
|---------|------|
| Upstream Kronos model (`model/`) | [README.md](README.md) |
| **Desktop forecasting app** (`kronos_app.py`) | this file |
| Potter Box scanner / evidence lab (`scanner/`) — retired strategy archive | [retirement record](docs/RETIREMENT-2026-08-20.md) |

Full orientation: [REPO_MAP.md](REPO_MAP.md)

---

## How to Launch
Double-click **"Kronos Predictor"** on your Desktop.
The browser opens automatically at http://localhost:8501

---

## Supported Assets (just type the ticker)
| Type      | Examples                          |
|-----------|-----------------------------------|
| US Stocks | AAPL, TSLA, NVDA, MSFT, AMZN     |
| ETFs      | SPY, QQQ, IWM, GLD, TLT          |
| Crypto    | BTC-USD, ETH-USD, SOL-USD        |
| Forex     | EUR=X, GBP=X, JPY=X              |
| Indices   | ^GSPC, ^IXIC, ^DJI               |

---

## Settings Guide
- **Timeframe**: 1D is most reliable. 1H/4H needs 60 days of hourly data from Yahoo.
- **Lookback**: 400 bars = max context for Kronos-base (512 tokens)
- **Prediction bars**: 60-120 is the sweet spot
- **Temperature**: 1.0 is default. Higher = wider range predictions
- **Sample paths**: Average multiple Monte Carlo paths for smoother forecasts

---

## Project Location
`C:\Users\Jacob Higgins\projects\kronos-predictor\`

- `kronos_app.py` — main Streamlit app
- `launch_kronos.bat` — launcher (what the shortcut calls)
- `model/` — Kronos model code
- `venv/` — isolated Python environment (Python 3.12, PyTorch 2.7.0+cpu — CPU-only)

---

## Model Cache
Models download automatically on first run (~500MB) and are cached at:
`C:\Users\Jacob Higgins\.cache\huggingface\hub\`

---

## Troubleshooting
- **App won't start**: Run `launch_kronos.bat` directly to see error output
- **Slow forecasts**: Inference is CPU-only in the current venv — reduce sample paths or prediction bars. (The old "CUDA out of memory" advice no longer applies; no CUDA torch is installed.)
- **Ticker not found**: Use Yahoo Finance ticker format (e.g. BTC-USD not BTCUSDT)
- **Stale data**: Yahoo Finance delays some data 15 minutes for free tier
