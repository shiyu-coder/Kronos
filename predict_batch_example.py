import sys

import pandas as pd

from model import Kronos, KronosTokenizer, KronosPredictor

csv_path = sys.argv[1] if len(sys.argv) > 1 else "./finetune_csv/data/HK_ali_09988_kline_5min_all.csv"

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

df = pd.read_csv(csv_path)
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120
cols = ['open', 'high', 'low', 'close', 'volume', 'amount']

# Three equal-length windows from the same file stand in for three separate series
df_list, x_timestamp_list, y_timestamp_list = [], [], []
for start in (0, lookback + pred_len, 2 * (lookback + pred_len)):
    hist = df.iloc[start:start + lookback]
    fut = df.iloc[start + lookback:start + lookback + pred_len]
    df_list.append(hist[cols].reset_index(drop=True))
    x_timestamp_list.append(hist['timestamps'].reset_index(drop=True))
    y_timestamp_list.append(fut['timestamps'].reset_index(drop=True))

pred_df_list = predictor.predict_batch(
    df_list=df_list,
    x_timestamp_list=x_timestamp_list,
    y_timestamp_list=y_timestamp_list,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True,
)

for i, pred_df in enumerate(pred_df_list):
    print(f"Predictions for series {i}:")
    print(pred_df.head())
