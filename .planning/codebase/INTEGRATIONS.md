# External Integrations

**Analysis Date:** 2026-07-02

## APIs & External Services

**Market Data (stocks/bars):**
- Alpaca Market Data API - Primary bars provider when credentials present
  - SDK/Client: none — raw `requests` in `scanner/data/market_data.py` (`_alpaca_get` with retry/backoff on 429/5xx and `X-Request-ID` tracing)
  - Endpoints: `https://data.alpaca.markets/v2/stocks/{ticker}/bars` (paginated bars, `scanner/data/market_data.py:104`), `https://data.alpaca.markets/v2/stocks/{ticker}/quotes/latest` (validation quotes, `:180`), `https://paper-api.alpaca.markets/v2/assets/{ticker}` (tradability check, `:176`)
  - Auth: env vars `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` sent as `APCA-API-KEY-ID`/`APCA-API-SECRET-KEY` headers
  - Feeds: `ALPACA_FEED=iex` (free default) for scans; `sip` with a hardcoded 16-minute delay for research mode (`fetch_intraday_bars`/`fetch_daily_bars`, `research=True`)
  - Provider routing: `MARKET_DATA_PROVIDER` = `auto` (Alpaca then yfinance fallback) | `alpaca` (hard fail) | `yfinance` (`scanner/data/market_data.py:65`, defaults in `scanner/config.py:46`)
  - Provenance: every bars DataFrame carries `attrs` `data_provider`/`data_feed`/`data_delay_minutes`

- Yahoo Finance via `yfinance` - Fallback bars + everything Alpaca free tier lacks
  - SDK/Client: `yfinance` package
  - Used in: `scanner/data/market_data.py` (`yf.download` fallback, `yf.Ticker.info`/`fast_info` for validation), `kronos_app.py:79` (`fetch_ohlcv` for the Streamlit app — yfinance only, supports stocks/ETF/crypto/forex/index tickers)
  - Auth: none (free tier, ~15-min delayed)

**Options Data:**
- Alpaca Options Snapshots (beta) - Quotes, IV, Greeks availability, daily volume
  - Endpoint: `https://data.alpaca.markets/v1beta1/options/snapshots/{ticker}` (`scanner/data/options_data.py:32`)
  - Feed: env `ALPACA_OPTIONS_FEED` (default `indicative`; `opra` requires subscription). Indicative quotes get a quality penalty in `_options_quality` (`scanner/data/options_data.py:79`) and can never produce a `promote` recommendation.
- yfinance option chains - Expirations, strikes, open interest (OI always comes from yfinance)
  - `yf.Ticker(ticker).options` / `.option_chain(exp)` in `scanner/data/options_data.py:97`
  - Contract selection (`select_options_contract`) joins Alpaca quotes over yfinance chain rows by `contractSymbol`; result records `data_provider` = `alpaca+yfinance` or `yfinance`

**Event/Calendar Data:**
- yfinance calendar - Earnings dates and ex-dividend dates for event-risk gating (`scanner/data/events.py:43` `assess_event_risk`; fail-closed when earnings unknown per `BLOCK_ON_UNKNOWN_EARNINGS` in `scanner/config.py:23`)

**LLM:**
- MiniMax (OpenAI-compatible chat completions) - Optional second-opinion scoring of trade candidates
  - Client: raw `requests` POST to `{MINIMAX_BASE_URL}/chat/completions` (`scanner/ai/minimax_adapter.py:99`), default base `https://api.minimax.io/v1`, model `MiniMax-M2.7-highspeed` (`scanner/config.py:58-63`)
  - Auth: `Authorization: Bearer $MINIMAX_API_KEY`; hard-disabled unless `MINIMAX_ENABLED=true`
  - Contract: strict-JSON response (`response_format: json_object`) parsed into `{score_band A|B|C|REJECT, confidence 0-1, rationale, red_flags}` with regex fallback parsing, `<think>` tag redaction, 3x retry, and safe skip defaults on any failure
  - Test mode: `python -m scanner.main --mode test_minimax`

**Alerting:**
- Telegram Bot API - The scanner's only outbound alert channel
  - Endpoint: `https://api.telegram.org/bot{token}/sendMessage` via `requests.post` (`scanner/alerts/telegram.py:80` `send_telegram_message`, 3x retry on 429/5xx)
  - Auth: env `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`
  - Gating (fail-closed): live sends require `--mode live` AND `LIVE_MODE_ENABLED=true` AND a current edge readiness audit with `readiness=paper_trade_only` (`scanner/README.md` Safety Defaults); `HEARTBEAT_ENABLED` controls no-result heartbeats
  - Message body built by `render_alert_message` (`scanner/alerts/telegram.py:26`) including optional MiniMax block
  - Test mode: `python -m scanner.main --mode test_telegram`

**Model Hub:**
- Hugging Face Hub - Pretrained Kronos weights, downloaded on first use, no auth (public repos)
  - Loading: `PyTorchModelHubMixin.from_pretrained` on `KronosTokenizer`/`Kronos` (`model/kronos.py:13`)
  - Models: `NeoQuasar/Kronos-mini` (4.1M, tokenizer `NeoQuasar/Kronos-Tokenizer-2k`, ctx 2048), `NeoQuasar/Kronos-small` (24.7M, ctx 512), `NeoQuasar/Kronos-base` (102.3M, ctx 512) — map in `kronos_app.py:66` and `webui/app.py:70`
  - Scanner uses `NeoQuasar/Kronos-small` + `Kronos-Tokenizer-base` (`scanner/config.py:53-54`), lazy-loaded once in `scanner/models/kronos_adapter.py:21` (`_load_once`)
  - Cache: `C:\Users\Jacob Higgins\.cache\huggingface\hub` (~500MB first run, per `README_JAKE.md`)

**Upstream/legacy (present in fork, NOT part of the scanner product):**
- akshare (Chinese A-share data) - `examples/prediction_cn_markets_day.py`, `examples/prediction_new_GUI.py`; package not installed
- Qlib data + Comet ML experiment tracking - `finetune/qlib_data_preprocess.py`, `finetune/train_tokenizer.py:238`, config placeholders in `finetune/config.py:76`; packages not installed
- ccxt - named in `install_deps.bat:20` only; zero imports, dead reference

## Data Storage

**Databases:**
- None. No SQL/NoSQL anywhere; all persistence is local files.

**File Storage (local filesystem, all under repo):**
- Decision journal (append-only JSONL): `scanner/reports/scan_decisions.jsonl` via `scanner/learning/outcome_store.py` (`DECISIONS_PATH`, `append_decision`, dedup on ticker/setup/day)
- Reports (JSON): `scanner/reports/*.json` — edge scan/validation/diagnostic/audit reports (paths defined in `scanner/config.py:12-16`), plus `outcome_review_summary.json`, `calibration_summary.json`, `zero_result_diagnostic.json`, `adaptive_policy_report.json`, `research_ops_report.json`
- Evidence lab runs: `scanner/reports/evidence/<run_id>/manifest.json` + JSONL row artifacts, with optional Parquet sidecars when pyarrow is installed (`scanner/evidence/store.py:121`)
- Edge retrieval index: `scanner/reports/edge_retrieval_index.json` (`scanner/config.py:12`)
- Tuning overrides: `scanner/tuning/overrides.json` (written by autotune/adaptive-policy modes, read at config import)
- Webui outputs: `webui/prediction_results/`
- All runtime artifacts are gitignored; `scanner/doctor.py:18` (`RUNTIME_ARTIFACTS`) verifies ignore status via `git check-ignore`

**Caching:**
- Hugging Face model cache (user-level, `~/.cache/huggingface/hub`)
- Streamlit `@st.cache_resource` for the loaded predictor (`kronos_app.py:61`)
- No app-level data cache; bars are re-fetched per run

## Authentication & Identity

**Auth Provider:**
- None — local single-user tool. No login, no user accounts.
- Service credentials are API keys in `scanner/.env` (present, gitignored; template `scanner/.env.example`), loaded by `_load_project_env_files()` in `scanner/main.py:185` with OS-env precedence.

## Monitoring & Observability

**Error Tracking:**
- None external (no Sentry etc.)

**Logs:**
- Rotating file log `scanner/logs/scanner.log` via `scanner/utils/logging_setup.py`; per-ticker PASS/SKIP/ERROR lines and `STAGE_START`/`STAGE_DONE` timing from `_run_timed_stage` (`scanner/main.py:98`)
- API request tracing: `scanner/logs/request_ids.log` — one line per Alpaca call with status + `X-Request-ID` (`scanner/data/market_data.py:69` `_persist_request_id`); Telegram request-ids logged when present
- Health report: `--mode doctor` (`scanner/doctor.py`) — Python version, importability of core modules, gitignore hygiene; no secrets in output

## CI/CD & Deployment

**Hosting:**
- None — everything runs locally on the Windows desktop. Entry points: `launch_kronos.bat` (Streamlit, port 8501), `scanner/run_scanner.bat --mode <mode>` (CLI), `webui/app.py` (Flask dev server, 127.0.0.1:7070, CORS restricted to localhost origins by `get_cors_origins()` in `webui/app.py:49`)

**CI Pipeline:**
- None (no `.github/`). Verification is manual: `.\venv\Scripts\python.exe -m pytest -q` plus the runbook modes in `scanner/README.md`
- Git remotes: `fork` = `jakehiggins89/Kronos`, `origin` = `shiyu-coder/Kronos` (upstream open-source Kronos)

## Environment Configuration

**Required env vars (by feature — everything is optional-with-fallback except where noted):**
- Alpaca data: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` (absent → yfinance fallback under `MARKET_DATA_PROVIDER=auto`); `ALPACA_FEED`, `ALPACA_OPTIONS_FEED`
- Telegram alerts (required for live/test_telegram): `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`; gates `LIVE_MODE_ENABLED`, `HEARTBEAT_ENABLED`
- MiniMax (required only if enabled): `MINIMAX_ENABLED`, `MINIMAX_API_KEY`; optional `MINIMAX_BASE_URL`, `MINIMAX_MODEL`, `MINIMAX_TEMPERATURE`, `MINIMAX_MAX_OUTPUT_TOKENS`, `MINIMAX_TIMEOUT_SECONDS`
- Provider routing: `MARKET_DATA_PROVIDER` (auto|alpaca|yfinance)
- Webui: `KRONOS_WEBUI_DATA_DIR`, `KRONOS_WEBUI_PORT`, `KRONOS_WEBUI_HOST`, `KRONOS_WEBUI_DEBUG`, `KRONOS_WEBUI_CORS_ORIGINS`

**Secrets location:**
- `scanner/.env` (exists; contents never read/committed — `.gitignore:64` covers `.env`). Root `.env` is also checked by `ENV_PATHS` (`scanner/main.py:79`) but does not currently exist. `scanner/README.md:200` instructs rotating the Telegram bot token before enabling live alerts.

## Webhooks & Callbacks

**Incoming:**
- None. No inbound HTTP surface beyond the localhost Flask webui (`/api/*` routes in `webui/app.py`) and Streamlit; neither accepts external callbacks.

**Outgoing:**
- Telegram `sendMessage` push notifications (fire-and-forget POST with retries, `scanner/alerts/telegram.py:80`). No other outbound callbacks; Alpaca/MiniMax/yfinance/HF are request-response only.

---

*Integration audit: 2026-07-02*
