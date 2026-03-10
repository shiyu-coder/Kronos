# Experiments Overview

This repository currently contains four experiment categories.

## 1. Prediction Demos

Location: `examples/`

Purpose:
- Verify that pretrained Kronos models can run end-to-end inference.
- Compare model sizes under the same inputs.

Typical inputs:
- Local CSV K-line data
- Optional online A-share data from `akshare`

Typical outputs:
- Forecast CSV
- Plot PNG
- Summary JSON

Ground-truth availability:
- `prediction_example.py`: no future truth, visualization only
- `prediction_wo_vol_example.py`: no future truth, visualization only
- `prediction_batch_example.py`: no future truth, visualization only
- `prediction_cn_markets_day.py --mode eval`: includes future truth and error metrics

## 2. Regression Tests

Location: `tests/test_kronos_regression.py`

Purpose:
- Ensure code changes do not silently drift model outputs.
- Check exact regression fixtures and MSE thresholds against fixed model revisions.

Typical outputs:
- `pytest` pass/fail result
- Printed absolute/relative differences and MSE summary in the terminal

## 3. Fine-Tuning Pipelines

Locations:
- `finetune/`
- `finetune_csv/`

Purpose:
- Adapt Kronos to domain-specific datasets.
- Support either Qlib-based A-share workflows or direct CSV-based training.

Notes:
- The `finetune/` training scripts are currently written for CUDA + NCCL + DDP.
- The training scripts now treat `comet_ml` as optional.

## 4. Qlib Backtesting

Location: `finetune/qlib_test.py`

Purpose:
- Convert model forecasts into cross-sectional stock-selection signals.
- Evaluate those signals with Qlib's `TopkDropoutStrategy`.

Typical inputs:
- Processed `train/val/test` pickle datasets from `qlib_data_preprocess.py`
- A tokenizer/model pair, which can be local checkpoints or explicit Hugging Face IDs

Typical outputs:
- `predictions.pkl`
- Per-signal curve CSV
- Per-signal metrics JSON
- Backtest summary CSV/JSON
- Backtest plot PNG

Ground-truth availability:
- Yes, via historical replay using the test split

## Recommended Validation Sequence

1. Run `examples/` to verify inference and output formats.
2. Run `tests/test_kronos_regression.py` to detect output drift.
3. Start `webui` to verify the interactive inference path.
4. Run `finetune/qlib_test.py` with an explicit config file for smoke backtesting.
5. Compare `Kronos-small` vs `Kronos-base` on the same prediction and backtest setups.
