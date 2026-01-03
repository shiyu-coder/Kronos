import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from model import Kronos, KronosPredictor, KronosTokenizer


TEST_DATA_ROOT = Path(__file__).parent / "data"
INPUT_DATA_PATH = TEST_DATA_ROOT / "regression_input.csv"

# Reuse same revisions as regression tests
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
DEVICE = "cpu"
SEED = 123
MAX_CTX_LEN = 512


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.mark.parametrize("context_len,pred_len", [(256, 8), (512, 8)])
def test_kv_cache_equivalence(context_len, pred_len):
    set_seed(SEED)

    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])

    if df.shape[0] < context_len + pred_len:
        raise ValueError("Example data does not contain enough rows for the equivalence test.")

    context_df = df.iloc[:context_len].copy()
    x_timestamp = context_df["timestamps"].reset_index(drop=True)
    future_timestamp = df["timestamps"].iloc[context_len:context_len + pred_len].reset_index(drop=True)

    features = ["open", "high", "low", "close", "volume", "amount"]
    context_features = context_df[features].reset_index(drop=True)

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small", revision=MODEL_REVISION)
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CTX_LEN)

    with torch.no_grad():
        pred_df_no_cache = predictor.predict(
            df=context_features,
            x_timestamp=x_timestamp,
            y_timestamp=future_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_k=1,
            top_p=1.0,
            verbose=False,
            sample_count=1,
            use_kv_cache=False,
        )

        pred_df_cache = predictor.predict(
            df=context_features,
            x_timestamp=x_timestamp,
            y_timestamp=future_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_k=1,
            top_p=1.0,
            verbose=False,
            sample_count=1,
            use_kv_cache=True,
        )

    np.testing.assert_allclose(
        pred_df_no_cache[features].to_numpy(dtype=np.float32),
        pred_df_cache[features].to_numpy(dtype=np.float32),
        rtol=0,
        atol=0,
    )

