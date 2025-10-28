import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from tqdm import tqdm

from model import Kronos, KronosPredictor, KronosTokenizer
from model.kronos import calc_time_stamps


TEST_DATA_ROOT = Path(__file__).parent / "data"
INPUT_DATA_PATH = TEST_DATA_ROOT / "regression_input.csv"
OUTPUT_DATA_DIR = TEST_DATA_ROOT
MAX_CTX_LEN = 512
TEST_CTX_LEN = [512, 256]
PRED_LEN = 8
NORM_EPS = 1e-5

MSE_SAMPLE_SIZE = 100
MSE_SAMPLE_CTX_LEN = 256
MSE_BATCH_SIZE = 8
# Average MSE (cpu): 0.0017483025
# Average MSE (mps): 0.0017483025
MSE_THRESHOLD = 1.75e-3

MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
SEED = 123

REL_TOLERANCE = 1e-5

DEVICE = "cpu"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.mark.parametrize("context_len", TEST_CTX_LEN)
def test_kronos_predictor_regression(context_len):
    set_seed(SEED)

    expected_output_path = OUTPUT_DATA_DIR / f"regression_output_{context_len}.csv"
    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])
    expected_df = pd.read_csv(expected_output_path, parse_dates=["timestamps"])

    if df.shape[0] < context_len + len(expected_df):
        raise ValueError("Example data does not contain enough rows for the regression test.")

    context_df = df.iloc[:context_len].copy()
    x_timestamp = context_df["timestamps"]
    future_timestamp = df["timestamps"].iloc[context_len:context_len + len(expected_df)]

    x = context_df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
    x_stamp = calc_time_stamps(x_timestamp).values.astype(np.float32)
    y_stamp = calc_time_stamps(future_timestamp).values.astype(np.float32)
    expected = expected_df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small", revision=MODEL_REVISION)
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CTX_LEN)

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    normalized_x = (x - x_mean) / (x_std + NORM_EPS)
    normalized_x = np.clip(normalized_x, -predictor.clip, predictor.clip)

    with torch.no_grad():
        preds = predictor.generate(
            x=normalized_x[np.newaxis, ...],
            x_stamp=x_stamp[np.newaxis, ...],
            y_stamp=y_stamp[np.newaxis, ...],
            pred_len=expected.shape[0],
            T=1.0,
            top_k=1,
            top_p=1.0,
            sample_count=1,
            verbose=False,
        )

    obtained = preds.squeeze(0)
    obtained = obtained * (x_std + NORM_EPS) + x_mean

    abs_diff = np.abs(obtained - expected)
    rel_diff = abs_diff / (np.abs(expected) + 1e-9)
    print(f"Abs diff: {np.max(abs_diff)}, Rel diff: {np.max(rel_diff)}")

    np.testing.assert_allclose(obtained, expected, rtol=REL_TOLERANCE)


def test_kronos_predictor_mse():
    set_seed(SEED)

    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])
    if df.shape[0] <= MSE_SAMPLE_CTX_LEN + PRED_LEN:
        raise ValueError("Example data does not contain enough rows for the random sample regression test.")

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small", revision=MODEL_REVISION)
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CTX_LEN)

    feature_names = ["open", "high", "low", "close", "volume", "amount"]
    mse_feature_names = ["open", "high", "low", "close"]
    mse_feature_idx = [feature_names.index(name) for name in mse_feature_names]

    valid_region = df.iloc[MSE_SAMPLE_CTX_LEN : df.shape[0] - PRED_LEN]
    if valid_region.shape[0] < MSE_SAMPLE_SIZE:
        raise ValueError("Not enough data points to draw the requested random samples.")

    sampled_rows = valid_region.sample(n=MSE_SAMPLE_SIZE, random_state=SEED).sort_index()

    mse_values = []
    sample_indices = sampled_rows.index.to_list()
    with torch.no_grad():
        for start in tqdm(range(0, len(sample_indices), MSE_BATCH_SIZE)):
            batch_indices = sample_indices[start : start + MSE_BATCH_SIZE]

            normalized_batch = []
            expected_batch = []
            x_stamp_batch = []
            y_stamp_batch = []
            mean_batch = []
            std_batch = []

            for row_idx in batch_indices:
                context_df = df.iloc[row_idx - MSE_SAMPLE_CTX_LEN : row_idx]
                future_df = df.iloc[row_idx : row_idx + PRED_LEN]

                x = context_df[feature_names].values.astype(np.float32)
                expected = future_df[feature_names].values.astype(np.float32)

                x_stamp = calc_time_stamps(context_df["timestamps"]).values.astype(np.float32)
                y_stamp = calc_time_stamps(future_df["timestamps"]).values.astype(np.float32)

                x_mean = np.mean(x, axis=0).astype(np.float32)
                x_std = np.std(x, axis=0).astype(np.float32)
                normalized_x = (x - x_mean) / (x_std + NORM_EPS)
                normalized_x = np.clip(normalized_x, -predictor.clip, predictor.clip)

                normalized_batch.append(normalized_x.astype(np.float32))
                expected_batch.append(expected.astype(np.float32))
                x_stamp_batch.append(x_stamp.astype(np.float32))
                y_stamp_batch.append(y_stamp.astype(np.float32))
                mean_batch.append(x_mean)
                std_batch.append(x_std)

            normalized_batch_arr = np.stack(normalized_batch, axis=0)
            x_stamp_batch_arr = np.stack(x_stamp_batch, axis=0)
            y_stamp_batch_arr = np.stack(y_stamp_batch, axis=0)
            mean_batch_arr = np.stack(mean_batch, axis=0)
            std_batch_arr = np.stack(std_batch, axis=0)
            expected_batch_arr = np.stack(expected_batch, axis=0)

            preds = predictor.generate(
                x=normalized_batch_arr,
                x_stamp=x_stamp_batch_arr,
                y_stamp=y_stamp_batch_arr,
                pred_len=PRED_LEN,
                T=1.0,
                top_k=1,
                top_p=1.0,
                sample_count=1,
                verbose=False,
            )

            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy()

            preds = np.asarray(preds, dtype=np.float32)
            obtained = preds * (std_batch_arr[:, np.newaxis, :] + NORM_EPS) + mean_batch_arr[:, np.newaxis, :]

            diff = obtained[:, :, mse_feature_idx] - expected_batch_arr[:, :, mse_feature_idx]
            batch_mse = diff ** 2
            mse_values.extend(batch_mse.mean(axis=(1, 2)).tolist())

    if not mse_values:
        raise AssertionError("Failed to compute any MSE values for the regression test.")

    if len(mse_values) != MSE_SAMPLE_SIZE:
        raise AssertionError(f"Expected {MSE_SAMPLE_SIZE} MSE values, got {len(mse_values)}.")

    mse = np.mean(mse_values).item()
    print(f"Average MSE: {mse}")

    assert mse < MSE_THRESHOLD, f"Average MSE {mse} exceeds threshold {MSE_THRESHOLD}"
