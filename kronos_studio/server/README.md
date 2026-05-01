# Kronos Prediction Server

FastAPI backend for Kronos Studio. Serves OHLCV data and runs Kronos transformer models to predict future prices.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
uv sync
```

## Run (Development)

```bash
uv run fastapi dev main.py
```

API available at `http://localhost:8000`.  
Interactive docs at `http://localhost:8000/docs` or `http://localhost:8000/scalar`.

## Run (Production)

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

## Tests

```bash
uv run pytest
```

## Directory Layout

| Path | Purpose |
|------|---------|
| `main.py` | App factory, middleware, exception handlers |
| `routers/` | API route handlers (`prediction`, `results`) |
| `schemas/` | Pydantic request/response models |
| `services/` | Prediction, raw data, and result business logic |
| `model/` | Kronos model classes (`KronosTokenizer`, `Kronos`, `KronosPredictor`) |
| `db/` | SQLite connection and auto-migrations |
| `errors/` | Custom `PredictionAPIError` exceptions |
| `constants/` | Shared constants |
| `tests/` | Pytest test suite |

## Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/api/prediction/options` | Available models and defaults |
| `POST` | `/api/prediction/price` | Single price prediction |
| `POST` | `/api/prediction/price/batch` | Batch price predictions |
| `POST` | `/api/prediction/local-data/upload` | Upload CSV (max 25 MB) |
| `POST` | `/api/saved-results` | Save a result |
| `GET` | `/api/saved-results` | List saved results |
| `GET` | `/api/saved-results/{id}` | Get result by ID |
| `DELETE` | `/api/saved-results/{id}` | Delete result by ID |
