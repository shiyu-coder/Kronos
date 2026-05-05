import os
import sys
import json
import uuid
import warnings
import datetime
import threading
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

from data_fetcher import fetch_ohlcv, calculate_orb, NSE_SYMBOLS, INTERVAL_PERIOD

app = Flask(__name__)
CORS(app)

# --------------------------------------------------------------------------- #
# Global state
# --------------------------------------------------------------------------- #
tokenizer = None
model     = None
predictor = None
_cached_df: pd.DataFrame | None = None
_cached_symbol  = ''
_cached_interval = ''

AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini', 'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048, 'params': '4.1M',
        'description': 'Lightweight – fast prediction',
    },
    'kronos-small': {
        'name': 'Kronos-small', 'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,  'params': '24.7M',
        'description': 'Balanced performance & speed',
    },
    'kronos-base': {
        'name': 'Kronos-base', 'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,  'params': '102.3M',
        'description': 'Best open-source quality',
    },
}

# Job store for async prediction {job_id: {'status': ..., 'result': ...}}
_jobs: dict = {}
_jobs_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _bar_freq(df: pd.DataFrame) -> pd.Timedelta:
    """Median gap between consecutive bars (robust to market gaps)."""
    if len(df) < 2:
        return pd.Timedelta(minutes=5)
    diffs = df['timestamps'].diff().dropna()
    # Use mode to ignore overnight/weekend gaps
    mode = diffs[diffs < pd.Timedelta(hours=4)].mode()
    if len(mode):
        return mode.iloc[0]
    return diffs.median()


def _to_str(ts_series) -> list:
    """Convert a timestamp Series / DatetimeIndex to ISO strings for Plotly."""
    if isinstance(ts_series, pd.DatetimeIndex):
        return ts_series.strftime('%Y-%m-%d %H:%M').tolist()
    if isinstance(ts_series, pd.Series):
        return ts_series.dt.strftime('%Y-%m-%d %H:%M').tolist()
    return [str(t) for t in ts_series]


def build_chart(df: pd.DataFrame,
                pred_df: pd.DataFrame | None,
                orb_levels: dict,
                lookback: int) -> str:

    hist = df.iloc[-lookback:].copy().reset_index(drop=True)
    hist_ts = _to_str(hist['timestamps'])

    fig = go.Figure()

    # ── Historical candlesticks ──────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=hist_ts,
        open=hist['open'].tolist(),
        high=hist['high'].tolist(),
        low=hist['low'].tolist(),
        close=hist['close'].tolist(),
        name='Historical',
        increasing=dict(line=dict(color='#26de81', width=1), fillcolor='#26de81'),
        decreasing=dict(line=dict(color='#ff4757', width=1), fillcolor='#ff4757'),
        whiskerwidth=0,
    ))

    # ── Volume bars (secondary y-axis) ──────────────────────────────────────
    if 'volume' in hist.columns and hist['volume'].sum() > 0:
        vol_colors = [
            'rgba(34,222,129,0.35)' if c >= o else 'rgba(255,71,87,0.35)'
            for o, c in zip(hist['open'], hist['close'])
        ]
        fig.add_trace(go.Bar(
            x=hist_ts,
            y=hist['volume'].tolist(),
            name='Volume',
            marker_color=vol_colors,
            yaxis='y2',
            showlegend=False,
        ))

    # ── Prediction candlesticks ──────────────────────────────────────────────
    pred_ts_strs = []
    if pred_df is not None and not pred_df.empty:
        freq = _bar_freq(df)
        last_ts = hist['timestamps'].iloc[-1]
        pred_dates = pd.date_range(start=last_ts + freq, periods=len(pred_df), freq=freq)
        pred_ts_strs = _to_str(pred_dates)

        # Use .values to avoid Series-index misalignment
        fig.add_trace(go.Candlestick(
            x=pred_ts_strs,
            open=pred_df['open'].values.tolist(),
            high=pred_df['high'].values.tolist(),
            low=pred_df['low'].values.tolist(),
            close=pred_df['close'].values.tolist(),
            name='Prediction',
            increasing=dict(line=dict(color='#ffd32a', width=1), fillcolor='rgba(255,211,42,0.25)'),
            decreasing=dict(line=dict(color='#ffa502', width=1), fillcolor='rgba(255,165,2,0.25)'),
            whiskerwidth=0,
        ))

    # ── ORB levels (shapes + annotations) ───────────────────────────────────
    shapes, annotations = [], []
    if orb_levels:
        latest = orb_levels[max(orb_levels.keys())]
        for label, val, color in [
            ('ORB High', latest['high'], '#00d4ff'),
            ('ORB Low',  latest['low'],  '#ff6b81'),
        ]:
            shapes.append(dict(
                type='line', xref='paper', yref='y',
                x0=0, x1=1, y0=val, y1=val,
                line=dict(color=color, width=1.2, dash='dot'),
            ))
            annotations.append(dict(
                xref='paper', yref='y', x=1.002, y=val,
                text=f'<b>{label}</b><br>{val:,.2f}',
                showarrow=False,
                font=dict(color=color, size=10),
                xanchor='left', align='left',
            ))

    # ── Layout ───────────────────────────────────────────────────────────────
    all_x = hist_ts + pred_ts_strs

    fig.update_layout(
        paper_bgcolor='#0d0f1a',
        plot_bgcolor='#0d0f1a',
        font=dict(color='#c9d1d9', family='JetBrains Mono, Consolas, monospace'),
        margin=dict(l=60, r=110, t=36, b=36),
        height=520,
        dragmode='pan',

        # Main price axis
        yaxis=dict(
            gridcolor='#161b22',
            gridwidth=1,
            zerolinecolor='#161b22',
            tickformat=',.0f',
            side='right',
            showgrid=True,
        ),
        # Volume axis
        yaxis2=dict(
            overlaying='y', side='left',
            showgrid=False, showticklabels=False,
            domain=[0, 0.2],
        ),

        # X axis – use category type to skip gaps (no weekends/nights shown)
        xaxis=dict(
            type='category',
            rangeslider=dict(visible=False),
            gridcolor='#161b22',
            gridwidth=1,
            tickangle=-30,
            tickfont=dict(size=10),
            nticks=12,
            # Show only the last N labels to keep it clean
            range=[max(0, len(all_x) - len(hist_ts)), len(all_x) - 1],
        ),

        legend=dict(
            bgcolor='rgba(13,15,26,0.85)',
            bordercolor='#30363d',
            borderwidth=1,
            x=0.01, y=0.99,
            xanchor='left', yanchor='top',
            font=dict(size=11),
        ),
        shapes=shapes,
        annotations=annotations,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#161b22',
            bordercolor='#30363d',
            font=dict(color='#c9d1d9', size=11),
        ),
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.route('/')
def index():
    return render_template('nse_dashboard.html')


@app.route('/api/symbols')
def get_symbols():
    return jsonify({
        'symbols': dict(NSE_SYMBOLS),
        'intervals': list(INTERVAL_PERIOD.keys()),
    })


@app.route('/api/fetch-data', methods=['POST'])
def fetch_data():
    global _cached_df, _cached_symbol, _cached_interval
    data = request.get_json()
    symbol      = data.get('symbol', '^NSEI')
    interval    = data.get('interval', '5m')
    orb_minutes = int(data.get('orb_minutes', 15))

    try:
        df = fetch_ohlcv(symbol, interval)
        orb_levels = calculate_orb(df, orb_minutes)

        _cached_df       = df
        _cached_symbol   = symbol
        _cached_interval = interval

        lookback   = min(200, len(df))
        chart_json = build_chart(df, None, orb_levels, lookback)

        latest_orb = None
        if orb_levels:
            d = max(orb_levels.keys())
            latest_orb = {**orb_levels[d], 'date': d}

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        chg  = float(last['close'] - prev['close'])
        pct  = chg / float(prev['close']) * 100 if float(prev['close']) else 0

        return jsonify({
            'success': True,
            'rows': len(df),
            'start_date': df['timestamps'].min().isoformat(),
            'end_date':   df['timestamps'].max().isoformat(),
            'last_price': round(float(last['close']), 2),
            'change':     round(chg, 2),
            'change_pct': round(pct, 2),
            'orb': latest_orb,
            'chart': chart_json,
            'message': f'Loaded {len(df):,} candles for {symbol} @ {interval}',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/available-models')
def available_models():
    return jsonify({'models': AVAILABLE_MODELS, 'model_available': MODEL_AVAILABLE})


@app.route('/api/load-model', methods=['POST'])
def load_model():
    global tokenizer, model, predictor
    if not MODEL_AVAILABLE:
        return jsonify({'error': 'Kronos library not installed (pip install torch einops)'}), 400

    data      = request.get_json()
    model_key = data.get('model_key', 'kronos-small')
    device    = data.get('device', 'cpu')

    if model_key not in AVAILABLE_MODELS:
        return jsonify({'error': f'Unknown model: {model_key}'}), 400

    cfg = AVAILABLE_MODELS[model_key]
    try:
        tokenizer = KronosTokenizer.from_pretrained(cfg['tokenizer_id'])
        model     = Kronos.from_pretrained(cfg['model_id'])
        predictor = KronosPredictor(model, tokenizer, device=device,
                                    max_context=cfg['context_length'])
        return jsonify({
            'success': True,
            'message': f'{cfg["name"]} ({cfg["params"]}) ready on {device}',
            'model_info': cfg,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-status')
def model_status():
    if MODEL_AVAILABLE and predictor is not None:
        dev = str(next(predictor.model.parameters()).device)
        return jsonify({'available': True, 'loaded': True,
                        'device': dev, 'message': f'Ready on {dev}'})
    elif MODEL_AVAILABLE:
        return jsonify({'available': True, 'loaded': False,
                        'message': 'Library ready – model not loaded'})
    else:
        return jsonify({'available': False, 'loaded': False,
                        'message': 'torch not installed'})


def _safe(v) -> float:
    """Return a JSON-safe float: replace NaN/Inf with 0.0."""
    import math
    try:
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else round(f, 2)
    except (TypeError, ValueError):
        return 0.0


def _run_prediction(job_id: str, df: pd.DataFrame, x_df, x_ts, y_ts,
                    freq, pred_len, temperature, top_p, sample_count,
                    orb_minutes, lookback):
    """Runs in a background thread. Writes result into _jobs[job_id]."""
    try:
        with _jobs_lock:
            _jobs[job_id]['status'] = 'running'

        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=temperature, top_p=top_p,
            sample_count=sample_count, verbose=False,
        )

        # Kronos can output NaN during denormalization; replace before JSON serialization
        # because Python's json.dumps encodes float('nan') as the literal NaN token
        # which is not valid JSON and causes JSON.parse to throw in the browser.
        pred_df = pred_df.fillna(0.0)
        pred_df = pred_df.replace([np.inf, -np.inf], 0.0)

        orb_levels   = calculate_orb(df, orb_minutes)
        chart_json   = build_chart(df, pred_df, orb_levels, lookback)

        pred_ts_list = pd.date_range(start=x_ts.iloc[-1] + freq, periods=pred_len, freq=freq)
        pred_opens   = pred_df['open'].values.tolist()
        pred_highs   = pred_df['high'].values.tolist()
        pred_lows    = pred_df['low'].values.tolist()
        pred_closes  = pred_df['close'].values.tolist()

        prediction_records = [
            {'timestamp': pred_ts_list[i].isoformat(),
             'open':  _safe(pred_opens[i]),
             'high':  _safe(pred_highs[i]),
             'low':   _safe(pred_lows[i]),
             'close': _safe(pred_closes[i])}
            for i in range(pred_len)
        ]

        last_close = float(df['close'].iloc[-1])
        pred_close = _safe(pred_closes[-1])
        trend      = 'BULLISH' if pred_close > last_close else 'BEARISH'
        pct_change = (pred_close - last_close) / last_close * 100 if last_close else 0.0

        valid_highs = [h for h in pred_highs if not (h != h)]  # filter NaN
        valid_lows  = [l for l in pred_lows  if not (l != l)]

        with _jobs_lock:
            _jobs[job_id] = {
                'status': 'done',
                'result': {
                    'success': True,
                    'chart':   chart_json,
                    'prediction_results': prediction_records,
                    'stats': {
                        'last_close': round(last_close, 2),
                        'pred_close': pred_close,
                        'pct_change': round(pct_change, 2),
                        'trend':      trend,
                        'pred_high':  _safe(max(valid_highs)) if valid_highs else 0.0,
                        'pred_low':   _safe(min(valid_lows))  if valid_lows  else 0.0,
                    },
                    'message': f'Predicted {pred_len} candles · Trend: {trend} ({pct_change:+.2f}%)',
                }
            }
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id] = {'status': 'error', 'error': str(e)}


@app.route('/api/predict', methods=['POST'])
def predict():
    global _cached_df
    if _cached_df is None:
        return jsonify({'error': 'No data loaded – fetch data first.'}), 400
    if not MODEL_AVAILABLE or predictor is None:
        return jsonify({'error': 'Model not loaded.'}), 400

    data         = request.get_json()
    lookback     = int(data.get('lookback', 200))
    pred_len     = int(data.get('pred_len', 60))
    temperature  = float(data.get('temperature', 1.0))
    top_p        = float(data.get('top_p', 0.9))
    sample_count = int(data.get('sample_count', 1))
    orb_minutes  = int(data.get('orb_minutes', 15))

    df = _cached_df
    if len(df) < lookback:
        return jsonify({'error': f'Need {lookback} candles, only have {len(df)}.'}), 400

    cols = ['open', 'high', 'low', 'close']
    if 'volume' in df.columns:
        cols.append('volume')

    x_df = df.iloc[-lookback:][cols].copy()
    x_ts = df.iloc[-lookback:]['timestamps'].reset_index(drop=True)
    freq = _bar_freq(df)
    y_ts = pd.Series(
        pd.date_range(start=x_ts.iloc[-1] + freq, periods=pred_len, freq=freq),
        name='timestamps',
    )

    # Launch background thread and return job_id immediately
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {'status': 'pending'}

    t = threading.Thread(
        target=_run_prediction,
        args=(job_id, df.copy(), x_df, x_ts, y_ts,
              freq, pred_len, temperature, top_p, sample_count,
              orb_minutes, lookback),
        daemon=True,
    )
    t.start()

    return jsonify({'job_id': job_id, 'status': 'pending'})


@app.route('/api/predict-status/<job_id>')
def predict_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({'error': 'Unknown job ID'}), 404
    if job['status'] == 'done':
        return jsonify({'status': 'done', **job['result']})
    if job['status'] == 'error':
        return jsonify({'status': 'error', 'error': job['error']}), 500
    return jsonify({'status': job['status']})


if __name__ == '__main__':
    print('-' * 50)
    print(' Kronos NSE Dashboard')
    print(f' Model available : {MODEL_AVAILABLE}')
    print(' URL             : http://localhost:7072')
    print('-' * 50)
    app.run(debug=True, host='0.0.0.0', port=7072, use_reloader=False)
