import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from model import Kronos, KronosPredictor, KronosTokenizer
from model.kronos import calc_time_stamps


TEST_DATA_ROOT = Path(__file__).parent / "data"
INPUT_DATA_PATH = TEST_DATA_ROOT / "regression_input.csv"
OUTPUT_DATA_DIR = TEST_DATA_ROOT
MAX_CTX_LEN = 512
TEST_CTX_LEN = [512, 256]
PRED_LEN = 8
NORM_EPS = 1e-5

MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
SEED = 123

REL_TOLERANCE = 1e-7

DEVICE = "cpu"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _run_regression_scenario(
    context_len: int, device: str
) -> None:
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

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=MAX_CTX_LEN)

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

    np.testing.assert_allclose(obtained, expected, rtol=REL_TOLERANCE)

    abs_diff = np.abs(obtained - expected)
    rel_diff = abs_diff / (np.abs(expected) + 1e-9)
    print(f"Abs diff: {np.max(abs_diff)}, Rel diff: {np.max(rel_diff)}")


@pytest.mark.parametrize("ctx_len", TEST_CTX_LEN)
def test_kronos_predictor_regression(ctx_len):
    _run_regression_scenario(ctx_len, DEVICE)
