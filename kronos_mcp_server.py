"""
Kronos MCP Server - exposes Kronos K-line prediction as MCP tools
for OpenClaw Gateway / Claude Code to invoke.

Usage:
  python kronos_mcp_server.py          # stdio mode (for Claude Code / OpenClaw)
  python kronos_mcp_server.py --sse    # SSE mode on port 8101
"""

import os
import sys
import json
import argparse
import traceback
import base64
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Kronos, KronosTokenizer, KronosPredictor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")
# OpenClaw media dir - images saved here are accessible via /api/media
MEDIA_DIR = os.path.join(os.path.expanduser("~"), ".openclaw", "media")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_cache = {}


def get_predictor(size="base"):
    if size in _cache:
        return _cache[size]
    tok_name = "Kronos-Tokenizer-2k" if size == "mini" else "Kronos-Tokenizer-base"
    tok_path = os.path.join(MODEL_DIR, tok_name)
    model_path = os.path.join(MODEL_DIR, f"Kronos-{size}")
    tokenizer = KronosTokenizer.from_pretrained(tok_path)
    model = Kronos.from_pretrained(model_path)
    ctx = 2048 if size == "mini" else 512
    pred = KronosPredictor(model, tokenizer, device=DEVICE, max_context=ctx)
    _cache[size] = pred
    return pred


def future_trading_dates(last_date, n):
    dates, cur = [], last_date + timedelta(days=1)
    while len(dates) < n:
        if cur.weekday() < 5:
            dates.append(cur)
        cur += timedelta(days=1)
    return dates


def generate_kline_chart(hist_df, pred_df, future_dates, stock_code, current_price, end_price, change_pct):
    """Generate K-line chart and save to OpenClaw media dir. Returns relative path."""
    try:
        os.makedirs(MEDIA_DIR, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                        gridspec_kw={'height_ratios': [3, 1]},
                                        facecolor='#1a1a2e')
        fig.subplots_adjust(hspace=0.08)

        for ax in (ax1, ax2):
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='#a0a0b0', labelsize=9)
            ax.spines['bottom'].set_color('#2a2a4a')
            ax.spines['top'].set_color('#2a2a4a')
            ax.spines['left'].set_color('#2a2a4a')
            ax.spines['right'].set_color('#2a2a4a')
            ax.grid(color='#2a2a4a', linewidth=0.5, alpha=0.7)

        # ---- Historical candlesticks ----
        hist = hist_df.tail(60).reset_index(drop=True)
        h_dates = hist['timestamps'].tolist()
        h_x = list(range(len(h_dates)))

        for i, row in hist.iterrows():
            o, c, h, l = row['open'], row['close'], row['high'], row['low']
            color = '#ef5350' if c >= o else '#26a69a'
            ax1.plot([i, i], [l, h], color=color, linewidth=0.8)
            ax1.add_patch(FancyBboxPatch((i - 0.3, min(o, c)), 0.6, abs(c - o) or 0.01,
                                          boxstyle="square,pad=0", color=color, alpha=0.9))
            # Volume
            ax2.bar(i, row['volume'], color=color, alpha=0.7, width=0.8)

        # ---- Predicted line ----
        p_start = len(h_dates)
        pred_x = list(range(p_start - 1, p_start + len(pred_df)))
        pred_close = [float(hist.iloc[-1]['close'])] + [float(pred_df['close'].iloc[i])
                                                         for i in range(len(pred_df))]
        ax1.plot(pred_x, pred_close, color='#ffd700', linewidth=2,
                 linestyle='--', label='Kronos 预测', zorder=5)

        # Mark divider
        ax1.axvline(x=p_start - 0.5, color='#888', linewidth=1, linestyle=':')
        ax1.text(p_start - 0.5, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 0 else 1,
                 '预测起点', color='#888', fontsize=8, ha='center', va='top')

        # ---- Labels ----
        direction = '↑ 看涨' if change_pct > 1 else ('↓ 看跌' if change_pct < -1 else '→ 震荡')
        color_dir = '#ef5350' if change_pct > 1 else ('#26a69a' if change_pct < -1 else '#ffd700')

        ax1.set_title(
            f'{stock_code}  当前: ¥{current_price:.2f}  →  预测: ¥{end_price:.2f}  {direction} {change_pct:+.2f}%',
            color='#e0e0e0', fontsize=13, pad=10, fontweight='bold'
        )
        ax1.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='#2a2a4a',
                   labelcolor='#ffd700', fontsize=9)
        ax1.set_ylabel('价格 (¥)', color='#a0a0b0', fontsize=9)
        ax2.set_ylabel('成交量', color='#a0a0b0', fontsize=9)

        # x-axis ticks
        tick_step = max(1, len(h_dates) // 6)
        ticks = list(range(0, len(h_dates), tick_step))
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(
            [h_dates[i].strftime('%m-%d') for i in ticks],
            rotation=30, ha='right', color='#a0a0b0', fontsize=8
        )
        ax1.set_xticks([])

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fname = f"kronos_{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fpath = os.path.join(MEDIA_DIR, fname)
        fig.savefig(fpath, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return fname  # relative path for /api/media?path=
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def tool_predict_stock(arguments: dict) -> str:
    """Predict future K-line for a Chinese A-share stock."""
    stock_code = arguments.get("stock_code", "")
    pred_days = int(arguments.get("pred_days", 30))
    model_size = arguments.get("model_size", "base")

    if not stock_code:
        return json.dumps({"error": "stock_code is required"}, ensure_ascii=False)

    try:
        import akshare as ak
    except ImportError:
        return json.dumps({"error": "akshare not installed, run: pip install akshare"}, ensure_ascii=False)

    # Fetch data
    try:
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")
        if df is None or df.empty:
            return json.dumps({"error": f"No data for stock {stock_code}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch stock data: {e}"}, ensure_ascii=False)

    col_map = {'日期': 'timestamps', '开盘': 'open', '收盘': 'close',
               '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount'}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)

    if 'amount' not in df.columns:
        df['amount'] = df['close'] * df['volume']

    pred_len = min(pred_days, 120)
    lookback = min(max(100, pred_len * 3), 400, len(df) - 1)

    # Load model & predict
    predictor = get_predictor(model_size)
    tail = df.tail(lookback).reset_index(drop=True)
    x_df = tail[['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_ts = tail['timestamps']
    future = future_trading_dates(df['timestamps'].iloc[-1], pred_len)

    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=pd.Series(future),
        pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False,
    )

    current_price = float(df['close'].iloc[-1])
    end_price = float(pred_df['close'].iloc[-1])
    change_pct = (end_price / current_price - 1) * 100

    # Build daily predictions
    rows = []
    for i, d in enumerate(future[:len(pred_df)]):
        r = {"date": d.strftime("%Y-%m-%d")}
        for c in ['open', 'high', 'low', 'close', 'volume']:
            if c in pred_df.columns:
                r[c] = round(float(pred_df[c].iloc[i]), 2)
        rows.append(r)

    direction = "上涨" if change_pct > 1 else ("下跌" if change_pct < -1 else "震荡")

    # Generate K-line chart
    chart_path = generate_kline_chart(df, pred_df, future, stock_code,
                                       current_price, end_price, change_pct)

    result = {
        "stock_code": stock_code,
        "model": f"Kronos-{model_size}",
        "current_price": round(current_price, 2),
        "predicted_end_price": round(end_price, 2),
        "change_pct": round(change_pct, 2),
        "pred_days": pred_len,
        "direction": direction,
        "chart": chart_path,  # OpenClaw will render this as image
        "summary": f"Kronos模型预测：{stock_code} 当前价{current_price:.2f}元，未来{pred_len}个交易日预计{direction}至{end_price:.2f}元（{change_pct:+.2f}%）\n\n![K线预测图]({chart_path})",
        "predictions": rows,
    }
    return json.dumps(result, ensure_ascii=False)


def tool_predict_kline(arguments: dict) -> str:
    """Predict from raw OHLCV data (any asset, any timeframe)."""
    kline_data = arguments.get("kline_data", [])
    pred_len = int(arguments.get("pred_len", 30))
    model_size = arguments.get("model_size", "base")

    if not kline_data or len(kline_data) < 50:
        return json.dumps({"error": "kline_data must have at least 50 rows"}, ensure_ascii=False)

    df = pd.DataFrame(kline_data)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    if 'amount' not in df.columns:
        df['amount'] = df['close'] * df['volume']

    pred_len = min(pred_len, 120)
    lookback = min(max(100, pred_len * 3), 400, len(df) - 1)

    predictor = get_predictor(model_size)
    tail = df.tail(lookback).reset_index(drop=True)
    x_df = tail[['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_ts = tail['timestamps']
    future = future_trading_dates(df['timestamps'].iloc[-1], pred_len)

    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=pd.Series(future),
        pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False,
    )

    current = float(df['close'].iloc[-1])
    end = float(pred_df['close'].iloc[-1])
    pct = (end / current - 1) * 100

    rows = []
    for i, d in enumerate(future[:len(pred_df)]):
        r = {"date": d.strftime("%Y-%m-%d")}
        for c in ['open', 'high', 'low', 'close', 'volume']:
            if c in pred_df.columns:
                r[c] = round(float(pred_df[c].iloc[i]), 2)
        rows.append(r)

    return json.dumps({
        "current_price": round(current, 2),
        "predicted_end_price": round(end, 2),
        "change_pct": round(pct, 2),
        "pred_len": pred_len,
        "predictions": rows,
    }, ensure_ascii=False)


def tool_list_models(arguments: dict) -> str:
    """List available Kronos model variants."""
    variants = []
    for size in ["mini", "small", "base"]:
        tok = "Kronos-Tokenizer-2k" if size == "mini" else "Kronos-Tokenizer-base"
        avail = os.path.isdir(os.path.join(MODEL_DIR, f"Kronos-{size}")) and os.path.isdir(os.path.join(MODEL_DIR, tok))
        variants.append({
            "name": f"Kronos-{size}",
            "available": avail,
            "loaded": size in _cache,
            "context_length": 2048 if size == "mini" else 512,
        })
    return json.dumps({"models": variants, "device": DEVICE}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# MCP protocol (JSON-RPC over stdio)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "kronos_predict_stock",
        "description": "使用Kronos金融K线模型预测A股股票未来走势。输入股票代码，返回未来N个交易日的OHLCV预测数据和涨跌幅分析。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string", "description": "A股股票代码，如 600036（招商银行）、000001（平安银行）"},
                "pred_days": {"type": "integer", "description": "预测天数（交易日），默认30，最大120", "default": 30},
                "model_size": {"type": "string", "enum": ["mini", "small", "base"], "description": "模型大小，base最准确", "default": "base"},
            },
            "required": ["stock_code"],
        },
    },
    {
        "name": "kronos_predict_kline",
        "description": "使用Kronos模型对自定义K线数据进行预测。适用于任意资产（股票/期货/加密货币等）。需提供至少50条OHLCV历史数据。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kline_data": {
                    "type": "array",
                    "description": "K线历史数据数组，每条包含 open/high/low/close/volume/timestamps",
                    "items": {"type": "object"},
                },
                "pred_len": {"type": "integer", "description": "预测条数，默认30", "default": 30},
                "model_size": {"type": "string", "enum": ["mini", "small", "base"], "default": "base"},
            },
            "required": ["kline_data"],
        },
    },
    {
        "name": "kronos_list_models",
        "description": "列出可用的Kronos模型变体及其状态。",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

TOOL_DISPATCH = {
    "kronos_predict_stock": tool_predict_stock,
    "kronos_predict_kline": tool_predict_kline,
    "kronos_list_models": tool_list_models,
}


def handle_request(req: dict) -> dict:
    method = req.get("method", "")
    rid = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": rid,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "kronos-prediction", "version": "1.0.0"},
            },
        }

    if method == "notifications/initialized":
        return None  # no response for notifications

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        fn = TOOL_DISPATCH.get(tool_name)
        if not fn:
            return {
                "jsonrpc": "2.0", "id": rid,
                "result": {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True},
            }

        try:
            result_text = fn(arguments)
            return {
                "jsonrpc": "2.0", "id": rid,
                "result": {"content": [{"type": "text", "text": result_text}]},
            }
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return {
                "jsonrpc": "2.0", "id": rid,
                "result": {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True},
            }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}

    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def run_stdio():
    """Run MCP server over stdio (standard for Claude Code / OpenClaw)."""
    print("[Kronos MCP] Starting stdio server...", file=sys.stderr)
    print(f"[Kronos MCP] Device: {DEVICE}", file=sys.stderr)

    # Preload model
    try:
        get_predictor("base")
        print("[Kronos MCP] Kronos-base model loaded", file=sys.stderr)
    except Exception as e:
        print(f"[Kronos MCP] Warning: {e}", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sse", action="store_true", help="Run in SSE mode on port 8101")
    args = parser.parse_args()

    if args.sse:
        print("SSE mode not implemented yet, use stdio mode.")
        sys.exit(1)
    else:
        run_stdio()
