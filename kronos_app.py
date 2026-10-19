"""
Kronos Financial Predictor - Streamlit App
Powered by NeoQuasar/Kronos-base (102.3M params)
Jake's one-click financial prediction tool
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import warnings
import torch
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# ── Path setup so model/ is importable ──────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kronos Financial Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 800; color: #1a1a2e;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #666; text-align: center; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 1rem 1.5rem;
        color: white; text-align: center; margin: 0.3rem;
    }
    .metric-label { font-size: 0.75rem; opacity: 0.7; text-transform: uppercase; }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .up { color: #00ff88; } .down { color: #ff4444; } .neutral { color: #ffcc00; }
    .status-box {
        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.5rem 0;
        font-size: 0.9rem; font-weight: 500;
    }
    .status-ok { background: #d4edda; color: #155724; border-left: 4px solid #28a745; }
    .status-warn { background: #fff3cd; color: #856404; border-left: 4px solid #ffc107; }
    .status-err { background: #f8d7da; color: #721c24; border-left: 4px solid #dc3545; }
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached so it only loads once per session) ──────────────────
@st.cache_resource(show_spinner=False)
def load_kronos(model_key: str, device: str):
    """Load Kronos tokenizer + model. Cached across reruns."""
    from model import Kronos, KronosTokenizer, KronosPredictor

    MODEL_MAP = {
        "kronos-mini":  ("NeoQuasar/Kronos-Tokenizer-2k",  "NeoQuasar/Kronos-mini",  2048),
        "kronos-small": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-small", 512),
        "kronos-base":  ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-base",  512),
    }
    tok_id, mdl_id, ctx = MODEL_MAP[model_key]
    tokenizer = KronosTokenizer.from_pretrained(tok_id)
    model     = Kronos.from_pretrained(mdl_id)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=ctx)
    return predictor, mdl_id, ctx


# ── Data fetcher ─────────────────────────────────────────────────────────────
def fetch_ohlcv(ticker: str, interval: str, lookback_bars: int) -> pd.DataFrame:
    """Fetch OHLCV data via yfinance and return clean DataFrame."""
    import yfinance as yf

    # Map interval → yfinance period/interval params
    INTERVAL_MAP = {
        "1D": ("1d",  "2y"),    # daily, 2 years
        "4H": ("1h",  "60d"),   # fetch 1h, resample to 4h
        "1H": ("1h",  "60d"),   # hourly, 60 days
    }
    yf_interval, period = INTERVAL_MAP[interval]

    raw = yf.download(ticker, interval=yf_interval, period=period,
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

    # Flatten multi-index if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]

    # Resample to 4H if needed
    if interval == "4H":
        raw = raw.resample("4h").agg({
            "open": "first", "high": "max",
            "low": "min",   "close": "last", "volume": "sum"
        }).dropna()

    raw = raw[["open", "high", "low", "close", "volume"]].dropna()
    raw.index = pd.to_datetime(raw.index)
    # Strip timezone info — Kronos predictor expects naive timestamps
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    return raw.tail(lookback_bars + 200)  # extra buffer


def make_future_timestamps(last_ts: pd.Timestamp, interval: str,
                           pred_len: int, is_crypto: bool) -> pd.DatetimeIndex:
    """Generate future timestamps, skipping weekends for non-crypto assets."""
    freq_map = {"1H": "1h", "4H": "4h", "1D": "1D"}
    freq = freq_map[interval]
    delta_map = {"1H": timedelta(hours=1), "4H": timedelta(hours=4), "1D": timedelta(days=1)}
    delta = delta_map[interval]

    timestamps = []
    current = last_ts + delta
    while len(timestamps) < pred_len:
        # Skip weekends for stocks (not crypto)
        if not is_crypto and current.weekday() >= 5:
            current += delta
            continue
        timestamps.append(current)
        current += delta
    return pd.DatetimeIndex(timestamps)


def is_crypto_ticker(ticker: str) -> bool:
    crypto_suffixes = ["-USD", "-USDT", "-BTC", "BTC", "ETH", "BNB", "SOL"]
    t = ticker.upper()
    return any(s in t for s in crypto_suffixes)


# ── Chart builder ────────────────────────────────────────────────────────────
def build_chart(hist_df: pd.DataFrame, pred_df: pd.DataFrame,
                ticker: str, interval: str) -> go.Figure:
    """Candlestick chart: last 100 historical bars + predicted bars."""
    # Use last 100 candles for chart clarity
    plot_hist = hist_df.tail(100)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{ticker} — {interval} | Kronos-base Forecast", "Volume"),
        vertical_spacing=0.05
    )

    # Historical candles
    fig.add_trace(go.Candlestick(
        x=plot_hist.index, open=plot_hist["open"], high=plot_hist["high"],
        low=plot_hist["low"],  close=plot_hist["close"],
        name="Historical", increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350"
    ), row=1, col=1)

    # Predicted candles
    fig.add_trace(go.Candlestick(
        x=pred_df.index, open=pred_df["open"], high=pred_df["high"],
        low=pred_df["low"],  close=pred_df["close"],
        name="Predicted", increasing_line_color="#4fc3f7",
        decreasing_line_color="#f48fb1", opacity=0.85
    ), row=1, col=1)

    # Vertical divider line
    fig.add_vline(
        x=hist_df.index[-1], line_dash="dash",
        line_color="rgba(255,200,0,0.8)", line_width=2
    )

    # Historical volume bars (if available)
    if "volume" in plot_hist.columns and plot_hist["volume"].notna().any():
        vol_colors = ["#26a69a" if c >= o else "#ef5350"
                      for c, o in zip(plot_hist["close"], plot_hist["open"])]
        fig.add_trace(go.Bar(
            x=plot_hist.index, y=plot_hist["volume"],
            name="Volume", marker_color=vol_colors, showlegend=False
        ), row=2, col=1)

    fig.update_layout(
        height=620, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", y=1.02, x=0),
        font=dict(family="Inter, sans-serif")
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    ticker = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        help="Examples: AAPL, TSLA, SPY, BTC-USD, ETH-USD, EUR=X, NVDA, MSFT"
    ).strip().upper()

    interval = st.selectbox(
        "Candle Timeframe",
        ["1D", "4H", "1H"],
        index=0,
        help="1D = daily, 4H = 4-hour, 1H = hourly"
    )

    model_key = st.selectbox(
        "Kronos Model",
        ["kronos-base", "kronos-small", "kronos-mini"],
        index=0,
        help="kronos-base = 102M params, best accuracy. kronos-mini = 4M, longest lookback."
    )

    # Auto-detect device
    auto_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_label = st.selectbox(
        "Compute Device",
        ["Auto (GPU)" if torch.cuda.is_available() else "Auto (CPU)", "Force CPU"],
        index=0
    )
    device = auto_device if "Auto" in device_label else "cpu"

    st.markdown("---")
    st.markdown("**Advanced**")
    lookback = st.slider("Lookback bars", 100, 400, 400, step=50,
                          help="Historical candles fed to model (max 400 for base)")
    pred_len  = st.slider("Prediction bars", 20, 120, 60, step=10,
                           help="How many future candles to forecast")
    temperature = st.slider("Temperature", 0.1, 2.0, 1.0, step=0.1,
                             help="Higher = more creative/diverse forecasts")
    top_p = st.slider("Top-p (nucleus)", 0.1, 1.0, 0.9, step=0.05)
    sample_count = st.number_input("Sample paths (averaged)", 1, 5, 1, step=1)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict", type="primary", width="stretch")


# ── Main content area ────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 Kronos Financial Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by NeoQuasar/Kronos-base</div>',
            unsafe_allow_html=True)

# GPU status badge
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
if gpu_name:
    st.markdown(
        f'<div class="status-box status-ok">✅ GPU: {gpu_name} | '
        f'{torch.cuda.get_device_properties(0).total_memory // 1024**3}GB VRAM | CUDA {torch.version.cuda}</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="status-box status-warn">⚠️ No GPU detected — running on CPU (slower)</div>',
        unsafe_allow_html=True
    )

if not predict_btn:
    st.markdown("""
    ### How to use
    1. Enter a **ticker symbol** in the sidebar (stocks, ETFs, crypto, forex all work)
    2. Select a **timeframe** — daily is most reliable
    3. Hit **Predict** to fetch live data and run the Kronos-base model

    **Popular tickers to try:** `AAPL` · `TSLA` · `SPY` · `NVDA` · `BTC-USD` · `ETH-USD` · `EUR=X`

    > **Note:** The first prediction will download the Kronos model weights (~400MB) from HuggingFace.
    > Subsequent runs load instantly from cache.
    """)
    st.stop()


# ── Prediction flow ──────────────────────────────────────────────────────────
with st.spinner(f"Fetching live {interval} data for **{ticker}**..."):
    try:
        raw_df = fetch_ohlcv(ticker, interval, lookback)
        st.markdown(
            f'<div class="status-box status-ok">✅ Fetched {len(raw_df)} candles for {ticker} '
            f'({raw_df.index[0].date()} → {raw_df.index[-1].date()})</div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.markdown(
            f'<div class="status-box status-err">❌ Data fetch failed: {e}</div>',
            unsafe_allow_html=True
        )
        st.stop()

with st.spinner(f"Loading {model_key} on {device}... (first run downloads ~400MB)"):
    try:
        predictor, mdl_id, ctx = load_kronos(model_key, device)
        st.markdown(
            f'<div class="status-box status-ok">✅ Model loaded: {mdl_id} | ctx={ctx} | device={device}</div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.markdown(
            f'<div class="status-box status-err">❌ Model load failed: {e}<br>'
            f'Try selecting "Force CPU" if GPU ran out of VRAM.</div>',
            unsafe_allow_html=True
        )
        st.stop()

with st.spinner("Running Kronos inference..."):
    try:
        # Trim to lookback window
        hist_df = raw_df.tail(lookback).copy()

        # Build input: reset index so KronosPredictor gets clean 0-based rows
        x_df = hist_df[["open", "high", "low", "close", "volume"]].copy().reset_index(drop=True)
        x_timestamp = pd.Series(hist_df.index)  # DatetimeSeries of historical bars

        crypto = is_crypto_ticker(ticker)
        future_ts = make_future_timestamps(hist_df.index[-1], interval, pred_len, crypto)
        y_timestamp = pd.Series(future_ts)

        predict_kwargs = dict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=float(temperature),
            top_p=float(top_p),
            sample_count=int(sample_count),
        )
        try:
            pred_df = predictor.predict(**predict_kwargs, verbose=False)
        except TypeError:
            pred_df = predictor.predict(**predict_kwargs)
        # Keep only OHLCV columns, ensure index is future timestamps
        keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in pred_df.columns]
        pred_df = pred_df[keep_cols].copy()
        pred_df.index = future_ts[:len(pred_df)]

    except Exception as e:
        st.markdown(
            f'<div class="status-box status-err">❌ Inference failed: {e}</div>',
            unsafe_allow_html=True
        )
        import traceback
        st.code(traceback.format_exc())
        st.stop()


# ── Results ──────────────────────────────────────────────────────────────────
current_price  = float(hist_df["close"].iloc[-1])
pred_end_close = float(pred_df["close"].iloc[-1])
pred_high      = float(pred_df["high"].max())
pred_low       = float(pred_df["low"].min())
direction      = "UP 📈" if pred_end_close > current_price else "DOWN 📉"
pct_change     = (pred_end_close - current_price) / current_price * 100
dir_class      = "up" if pred_end_close > current_price else "down"

# Summary metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Predicted Direction</div>
        <div class="metric-value {dir_class}">{direction}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Current Price</div>
        <div class="metric-value">${current_price:,.4f}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Predicted End Price</div>
        <div class="metric-value {dir_class}">${pred_end_close:,.4f} ({pct_change:+.2f}%)</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Predicted Range</div>
        <div class="metric-value neutral">${pred_low:,.2f} – ${pred_high:,.2f}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Chart
fig = build_chart(hist_df, pred_df, ticker, interval)
st.plotly_chart(fig, width="stretch")

# Predicted data table (collapsible)
with st.expander("📊 View predicted OHLCV data"):
    display_pred = pred_df.copy()
    display_pred.index = display_pred.index.strftime("%Y-%m-%d %H:%M")
    st.dataframe(display_pred.round(4), width="stretch")

# Footer
st.markdown("---")
st.markdown(
    f"**Model:** `{mdl_id}` · **Lookback:** {lookback} bars · "
    f"**Pred length:** {pred_len} bars · **Device:** `{device}` · "
    f"**Temp:** {temperature} · **Top-p:** {top_p}",
    help="Parameters used for this prediction run"
)
