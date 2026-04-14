import pandas as pd
import numpy as np

# Generate 600 data points of 5-min intervals
timestamps = pd.date_range("2025-01-01", periods=600, freq="5min")
open_prices = np.random.uniform(100, 110, 600)
high_prices = open_prices + np.random.uniform(0, 2, 600)
low_prices = open_prices - np.random.uniform(0, 2, 600)
close_prices = open_prices + np.random.uniform(-1, 1, 600)
volumes = np.random.uniform(1000, 5000, 600)
amounts = volumes * close_prices

df = pd.DataFrame({
    'timestamps': timestamps,
    'open': open_prices,
    'high': high_prices,
    'low': low_prices,
    'close': close_prices,
    'volume': volumes,
    'amount': amounts
})

import os
os.makedirs('./data', exist_ok=True)
df.to_csv('./data/XSHG_5min_600977.csv', index=False)
print("Dummy data generated at ./data/XSHG_5min_600977.csv")
