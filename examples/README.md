# Example Scripts

All scripts in this directory can be run from the repository root and save their outputs to `examples/outputs/`.

## Recommended Order

1. `python examples/prediction_example.py --model small`
2. `python examples/prediction_wo_vol_example.py --model small`
3. `python examples/prediction_batch_example.py --model small`
4. `python examples/prediction_cn_markets_day.py --model small --mode forward --symbol 000001`
5. `python examples/prediction_cn_markets_day.py --model small --mode eval --symbol 000001`

## Model Selection

The examples support:

- `--model small`
- `--model base`

This keeps the input data and evaluation mode fixed while allowing direct `Kronos-small` vs `Kronos-base` comparisons.

## Script Semantics

- `prediction_example.py`
  Runs one OHLCV forecast on a single historical window and saves CSV, PNG, and summary JSON.
- `prediction_wo_vol_example.py`
  Runs one forecast without `volume`/`amount` inputs and saves CSV, PNG, and summary JSON.
- `prediction_batch_example.py`
  Demonstrates batch inference on multiple sequential time windows from the same instrument.
  It is not a multi-asset cross-sectional example.
  By default it saves CSV for every window and one representative PNG.
- `prediction_cn_markets_day.py`
  Supports two explicit modes:
  - `forward`: future prediction only, so there is no ground-truth window yet.
  - `eval`: historical replay mode with known future truth and saved error metrics.

## Outputs

Each run saves structured outputs in `examples/outputs/`:

- Forecast CSV files
- Plot PNG files
- Summary JSON files describing the model, device, time ranges, and output paths

## Apple Silicon Devices

The scripts auto-select a compute device in this order:

1. `cuda:0`
2. `mps`
3. `cpu`

On Apple Silicon Macs, `mps` is used when PyTorch reports that Metal is available.

## Dependencies

- Local CSV examples only need the root `requirements.txt`.
- `prediction_cn_markets_day.py` also requires `akshare` and network access:

```bash
uv pip install --python .venv/bin/python akshare
```
