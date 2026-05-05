import pandas as pd
import pytz
from datetime import datetime, time as dt_time, timedelta

IST = pytz.timezone('Asia/Kolkata')

NSE_SYMBOLS = {
    'NIFTY 50': '^NSEI',
    'BANK NIFTY': '^NSEBANK',
    'NIFTY IT': '^CNXIT',
    'NIFTY MIDCAP 50': '^NSEMDCP50',
    'RELIANCE': 'RELIANCE.NS',
    'TCS': 'TCS.NS',
    'HDFC BANK': 'HDFCBANK.NS',
    'INFOSYS': 'INFY.NS',
    'ICICI BANK': 'ICICIBANK.NS',
    'KOTAK BANK': 'KOTAKBANK.NS',
    'AXIS BANK': 'AXISBANK.NS',
    'SBI': 'SBIN.NS',
}

INTERVAL_PERIOD = {
    '1m':  '7d',
    '5m':  '60d',
    '15m': '60d',
    '30m': '60d',
    '60m': '730d',
    '1d':  'max',
}


def fetch_ohlcv(symbol: str, interval: str = '5m') -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance for an NSE symbol."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    period = INTERVAL_PERIOD.get(interval, '60d')
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {symbol} at {interval} interval")

    df = df.rename(columns={
        'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    df.index.name = 'timestamps'
    df = df.reset_index()

    # Normalise timezone to IST
    if df['timestamps'].dt.tz is not None:
        df['timestamps'] = df['timestamps'].dt.tz_convert(IST).dt.tz_localize(None)
    else:
        df['timestamps'] = pd.to_datetime(df['timestamps'])

    df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume']].dropna()
    df = df.sort_values('timestamps').reset_index(drop=True)
    return df


def calculate_orb(df: pd.DataFrame, orb_minutes: int = 15) -> dict:
    """
    Calculate Opening Range Breakout levels per trading day.
    NSE opens at 09:15 IST; ORB window = first orb_minutes after open.
    Returns {date_str: {'high': float, 'low': float}}.
    """
    df = df.copy()
    df['_date'] = df['timestamps'].dt.date
    df['_time'] = df['timestamps'].dt.time

    open_time = dt_time(9, 15)
    end_time = (datetime.combine(datetime.today(), open_time) + timedelta(minutes=orb_minutes)).time()

    orb_levels = {}
    for date, day_df in df.groupby('_date'):
        window = day_df[(day_df['_time'] >= open_time) & (day_df['_time'] < end_time)]
        if not window.empty:
            orb_levels[str(date)] = {
                'high': float(window['high'].max()),
                'low':  float(window['low'].min()),
            }
    return orb_levels


def df_to_records(df: pd.DataFrame) -> list:
    """Convert DataFrame rows to JSON-serialisable dicts."""
    records = []
    for _, row in df.iterrows():
        records.append({
            'timestamp': row['timestamps'].isoformat(),
            'open':   float(row['open']),
            'high':   float(row['high']),
            'low':    float(row['low']),
            'close':  float(row['close']),
            'volume': float(row.get('volume', 0)),
        })
    return records
