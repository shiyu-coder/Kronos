import sys

import pandas as pd

from model import Kronos, KronosTokenizer, KronosPredictor

# XSHG_5min_600977.csv is not in the repo; pass your own CSV as argv[1] to override.
csv_path = sys.argv[1] if len(sys.argv) > 1 else "./finetune_csv/data/HK_ali_09988_kline_5min_all.csv"

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

df = pd.read_csv(csv_path)
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
)
print("Forecasted Data Head:")
print(pred_df.head())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pred_df.to_csv("prediction.csv")

actual = df.loc[lookback:lookback+pred_len-1].set_index('timestamps')['close']
hist = df.loc[:lookback-1].set_index('timestamps')['close']
fig, ax = plt.subplots(figsize=(12, 5))
hist.iloc[-150:].plot(ax=ax, label="History")
actual.plot(ax=ax, label="Actual")
pred_df['close'].plot(ax=ax, label="Predicted")
ax.legend()
ax.set_title("Kronos forecast (close)")
fig.savefig("prediction.png", dpi=120, bbox_inches="tight")
