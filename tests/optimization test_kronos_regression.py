import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from model import Kronos, KronosPredictor, KronosTokenizer

# ---------------- CONFIG ---------------- #

TEST_DATA_ROOT = Path(__file__).parent / "data"
INPUT_DATA_PATH = TEST_DATA_ROOT / "regression_input.csv"

FEATURE_NAMES = ["open", "high", "low", "close", "volume", "amount"]
MSE_FEATURE_NAMES = ["open", "high", "low", "close"]

TEST_CTX_LEN = [512, 256]
PRED_LEN = 8
REL_TOLERANCE = 1e-5

MSE_SAMPLE_SIZE = 4
MSE_CTX_LEN = [512, 256]
MSE_EXPECTED = [0.008979, 0.003741]
MSE_PRED_LEN = 30
MSE_TOLERANCE = 1e-6

MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"

MAX_CTX_LEN = 512
SEED = 123
DEVICE = torch.device("cpu")

# ---------------- UTILITIES ---------------- #

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.fixture(scope="session")
def data_df():
    """Load dataset once for all tests."""
    return pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])


@pytest.fixture(scope="session")
def predictor():
    """Load model & tokenizer only once (huge speedup)."""
    tokenizer = KronosTokenizer.from_pretrained(
        "NeoQuasar/Kronos-Tokenizer-base",
        revision=TOKENIZER_REVISION
    )
    model = Kronos.from_pretrained(
        "NeoQuasar/Kronos-small",
        revision=MODEL_REVISION
    ).to(DEVICE)

    model.eval()
    tokenizer.eval()

    return KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CTX_LEN)


def run_prediction(predictor, context_df, future_timestamps, pred_len):
    """Reusable prediction helper."""
    with torch.no_grad():
        return predictor.predict(
            df=context_df[FEATURE_NAMES].reset_index(drop=True),
            x_timestamp=context_df["timestamps"].reset_index(drop=True),
            y_timestamp=future_timestamps.reset_index(drop=True),
            pred_len=pred_len,
            T=1.0,
            top_k=1,
            top_p=1.0,
            verbose=False,
            sample_count=1,
        )

# ---------------- REGRESSION TEST ---------------- #

@pytest.mark.parametrize("context_len", TEST_CTX_LEN)
def test_kronos_predictor_regression(context_len, data_df, predictor):
    set_seed()

    expected_output_path = TEST_DATA_ROOT / f"regression_output_{context_len}.csv"
    expected_df = pd.read_csv(expected_output_path, parse_dates=["timestamps"])

    assert data_df.shape[0] >= context_len + len(expected_df), \
        "Not enough rows for regression test"

    context_df = data_df.iloc[:context_len]
    future_timestamps = data_df["timestamps"].iloc[
        context_len:context_len + len(expected_df)
    ]

    pred_df = run_prediction(predictor, context_df, future_timestamps, len(expected_df))

    obtained = pred_df[FEATURE_NAMES].to_numpy(np.float32)
    expected = expected_df[FEATURE_NAMES].to_numpy(np.float32)

    np.testing.assert_allclose(obtained, expected, rtol=REL_TOLERANCE)

# ---------------- MSE TEST ---------------- #

@pytest.mark.parametrize("context_len, expected_mse", zip(MSE_CTX_LEN, MSE_EXPECTED))
def test_kronos_predictor_mse(context_len, expected_mse, data_df, predictor):
    set_seed()

    assert data_df.shape[0] > context_len + MSE_PRED_LEN, \
        "Not enough rows for MSE test"

    valid_region = data_df.iloc[context_len:-MSE_PRED_LEN]
    sampled_rows = valid_region.sample(n=MSE_SAMPLE_SIZE, random_state=SEED).index

    mse_values = []

    for row_idx in sampled_rows:
        context_slice = data_df.iloc[row_idx - context_len: row_idx]
        future_slice = data_df.iloc[row_idx: row_idx + MSE_PRED_LEN]

        pred_df = run_prediction(
            predictor,
            context_slice,
            future_slice["timestamps"],
            MSE_PRED_LEN
        )

        obtained = pred_df[MSE_FEATURE_NAMES].to_numpy(np.float32)
        expected = future_slice[MSE_FEATURE_NAMES].to_numpy(np.float32)

        mse_values.append(np.mean((obtained - expected) ** 2))

    mse = float(np.mean(mse_values))
    assert abs(mse - expected_mse) <= MSE_TOLERANCE, \
        f"MSE {mse} differs from expected {expected_mse}"
