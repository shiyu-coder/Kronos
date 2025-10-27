import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import Kronos, KronosPredictor, KronosTokenizer
from model.kronos import calc_time_stamps


TEST_DATA_ROOT = Path(__file__).parent
INPUT_DATA_PATH = TEST_DATA_ROOT / "regression_input.csv"
OUTPUT_DATA_DIR = TEST_DATA_ROOT
MAX_CTX_LEN = 512
TEST_CTX_LEN = [512, 256]
PRED_LEN = 8
NORM_EPS = 1e-5

MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
SEED = 123

DEVICE = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_output(ctx_len: int) -> None:
    if ctx_len > MAX_CTX_LEN:
        raise ValueError(
            f"Context length for output generation ({ctx_len}) "
            f"cannot exceed maximum context length ({MAX_CTX_LEN})."
        )

    context_df = df.iloc[:ctx_len].copy()
    future_timestamps = df["timestamps"].iloc[
        ctx_len : ctx_len + PRED_LEN
    ].reset_index(drop=True)

    x = (
        context_df[["open", "high", "low", "close", "volume", "amount"]]
        .values.astype(np.float32)
    )
    x_stamp = calc_time_stamps(context_df["timestamps"]).values.astype(np.float32)
    y_stamp = calc_time_stamps(future_timestamps).values.astype(np.float32)

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    normalized_x = (x - x_mean) / (x_std + NORM_EPS)

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small", revision=MODEL_REVISION)
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(
        model, tokenizer, device=DEVICE, max_context=MAX_CTX_LEN
    )

    normalized_x = np.clip(normalized_x, -predictor.clip, predictor.clip)

    with torch.no_grad():
        preds = predictor.generate(
            x=normalized_x[np.newaxis, ...],
            x_stamp=x_stamp[np.newaxis, ...],
            y_stamp=y_stamp[np.newaxis, ...],
            pred_len=PRED_LEN,
            T=1.0,
            top_k=1,
            top_p=1.0,
            sample_count=1,
            verbose=False,
        )
    
    assert preds.shape == (1, PRED_LEN, 6), f"Unexpected prediction shape: {preds.shape}"

    obtained = preds.squeeze(0)
    obtained = obtained * (x_std + NORM_EPS) + x_mean

    output_df = pd.DataFrame(
        obtained,
        columns=["open", "high", "low", "close", "volume", "amount"],
    )
    output_df["timestamps"] = future_timestamps
    output_df.to_csv(OUTPUT_DATA_DIR / f"regression_output_{ctx_len}.csv", index=False)
    print(f"Saved {ctx_len} fixture to {OUTPUT_DATA_DIR / f'regression_output_{ctx_len}.csv'}")


if __name__ == "__main__":
    set_seed(SEED)


    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])
    if df.shape[0] < MAX_CTX_LEN + PRED_LEN:
        raise ValueError(
            f"Input data must have at least {MAX_CTX_LEN + PRED_LEN} rows, "
            f"found {df.shape[0]} instead."
        )

    for ctx_len in TEST_CTX_LEN:
        generate_output(ctx_len)
