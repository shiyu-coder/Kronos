import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from tqdm import tqdm
import itertools

from model import Kronos, KronosPredictor, KronosTokenizer

TEST_DATA_ROOT = Path(__file__).parent / "data"
INPUT_DATA_PATH = TEST_DATA_ROOT / "regression_input.csv"

# Regression test configuration
OUTPUT_DATA_DIR = TEST_DATA_ROOT
TEST_CTX_LEN = [512, 256]
PRED_LEN = 8
REL_TOLERANCE = 1e-5
FEATURE_NAMES = ["open", "high", "low", "close", "volume", "amount"]

# MSE regression test configuration
MSE_SAMPLE_SIZE = 4
MSE_CTX_LEN = [512, 256]
MSE_EXPECTED = [0.008979, 0.003741]
MSE_PRED_LEN = 30
MSE_TOLERANCE = 0.000001
MSE_FEATURE_NAMES = ["open", "high", "low", "close"]

MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
MAX_CTX_LEN = 512
SEED = 123
DEVICE = "cpu"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.mark.parametrize("context_len, use_kv_cache", itertools.product(TEST_CTX_LEN, [False, True]))
def test_kronos_predictor_regression(context_len, use_kv_cache):
    set_seed(SEED)

    expected_output_path = OUTPUT_DATA_DIR / f"regression_output_{context_len}.csv"
    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])
    expected_df = pd.read_csv(expected_output_path, parse_dates=["timestamps"])

    if df.shape[0] < context_len + len(expected_df):
        raise ValueError("Example data does not contain enough rows for the regression test.")

    context_df = df.iloc[:context_len].copy()
    context_features = context_df[FEATURE_NAMES].reset_index(drop=True)
    x_timestamp = context_df["timestamps"].reset_index(drop=True)
    future_timestamp = df["timestamps"].iloc[context_len:context_len + len(expected_df)].reset_index(drop=True)
    expected = expected_df[FEATURE_NAMES].values.astype(np.float32)

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small", revision=MODEL_REVISION)
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CTX_LEN)

    with torch.no_grad():
        pred_df = predictor.predict(
            df=context_features,
            x_timestamp=x_timestamp,
            y_timestamp=future_timestamp,
            pred_len=expected.shape[0],
            T=1.0,
            top_k=1,
            top_p=1.0,
            verbose=False,
            sample_count=1,
            use_kv_cache=use_kv_cache,
        )

    obtained = pred_df[FEATURE_NAMES].to_numpy(dtype=np.float32)

    abs_diff = np.abs(obtained - expected)
    rel_diff = abs_diff / (np.abs(expected) + 1e-9)
    print(f"Abs diff: {np.max(abs_diff)}, Rel diff: {np.max(rel_diff):.6e}")

    np.testing.assert_allclose(obtained, expected, rtol=REL_TOLERANCE)

@pytest.mark.parametrize("context_len, expected_mse", zip(MSE_CTX_LEN, MSE_EXPECTED))
@pytest.mark.parametrize("use_kv_cache", [False, True])
def test_kronos_predictor_mse(context_len, expected_mse, use_kv_cache):
    set_seed(SEED)

    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])
    if df.shape[0] <= context_len + MSE_PRED_LEN:
        raise ValueError("Example data does not contain enough rows for the random sample regression test.")

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small", revision=MODEL_REVISION)
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CTX_LEN)

    valid_region = df.iloc[context_len : df.shape[0] - MSE_PRED_LEN]
    if valid_region.shape[0] < MSE_SAMPLE_SIZE:
        raise ValueError("Not enough data points to draw the requested random samples.")

    sampled_rows = valid_region.sample(n=MSE_SAMPLE_SIZE, random_state=SEED).sort_index()

    mse_values = []
    sample_indices = sampled_rows.index.to_list()
    with torch.no_grad():
        for row_idx in tqdm(sample_indices, disable=True):
            context_slice = df.iloc[row_idx - context_len : row_idx].copy()
            future_slice = df.iloc[row_idx : row_idx + MSE_PRED_LEN].copy()

            pred_df = predictor.predict(
                df=context_slice[FEATURE_NAMES].reset_index(drop=True),
                x_timestamp=context_slice["timestamps"].reset_index(drop=True),
                y_timestamp=future_slice["timestamps"].reset_index(drop=True),
                pred_len=MSE_PRED_LEN,
                T=1.0,
                top_k=1,
                top_p=1.0,
                verbose=False,
                sample_count=1,
                use_kv_cache=use_kv_cache,
            )

            obtained = pred_df[MSE_FEATURE_NAMES].to_numpy(dtype=np.float32)
            expected = future_slice[MSE_FEATURE_NAMES].to_numpy(dtype=np.float32)
            mse_values.append(float(np.mean((obtained - expected) ** 2)))

    assert len(mse_values) == MSE_SAMPLE_SIZE, f"Expected {MSE_SAMPLE_SIZE} MSE values, got {len(mse_values)}."

    mse = np.mean(mse_values).item()
    mse_diff = mse - expected_mse
    print(f"Average MSE: {mse} (Diff vs expected: {mse_diff:+.6e})")

    assert abs(mse_diff) <= MSE_TOLERANCE, f"MSE {mse} differs from expected {expected_mse}"
