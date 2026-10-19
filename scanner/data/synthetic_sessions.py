from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class SyntheticSessionDiagnostics:
    source_interval: str
    prepost_enabled: bool
    input_bars: int
    output_sessions: int
    session_anchor: str


def build_synthetic_sessions(
    intraday_df: pd.DataFrame,
    session_anchor_hour: int,
    session_anchor_minute: int,
    source_interval: str,
    prepost_enabled: bool = True,
) -> tuple[pd.DataFrame, dict]:
    if intraday_df is None or intraday_df.empty:
        return pd.DataFrame(), {
            "source_interval": source_interval,
            "prepost_enabled": prepost_enabled,
            "input_bars": 0,
            "output_sessions": 0,
            "session_start_end_rules": f"session starts {session_anchor_hour:02d}:{session_anchor_minute:02d}",
        }

    df = intraday_df.copy().sort_index()
    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not set(expected_cols).issubset(set(df.columns)):
        raise ValueError(f"intraday data missing required columns: {expected_cols}")

    anchor_minutes = session_anchor_hour * 60 + session_anchor_minute
    minute_of_day = df.index.hour * 60 + df.index.minute
    session_date = df.index.tz_convert(df.index.tz).normalize()
    session_date = session_date.where(minute_of_day >= anchor_minutes, session_date - pd.Timedelta(days=1))

    df = df.assign(_session_date=session_date)
    grouped = df.groupby("_session_date", observed=True)

    synthetic = pd.DataFrame(
        {
            "Open": grouped["Open"].first(),
            "High": grouped["High"].max(),
            "Low": grouped["Low"].min(),
            "Close": grouped["Close"].last(),
            "Volume": grouped["Volume"].sum(min_count=1),
        }
    ).dropna(subset=["Open", "High", "Low", "Close"], how="any")

    idx = pd.DatetimeIndex(synthetic.index)
    synthetic.index = idx.tz_localize(df.index.tz) if idx.tz is None else idx.tz_convert(df.index.tz)
    synthetic.attrs.update(intraday_df.attrs)
    synthetic.attrs["latest_source_timestamp"] = pd.Timestamp(df.index[-1]).isoformat()

    diagnostics = {
        "source_interval": source_interval,
        "prepost_enabled": prepost_enabled,
        "input_bars": int(len(df)),
        "output_sessions": int(len(synthetic)),
        "session_start_end_rules": (
            f"session starts daily at {session_anchor_hour:02d}:{session_anchor_minute:02d} ET; "
            "bars earlier than anchor are assigned to prior synthetic session"
        ),
    }
    return synthetic, diagnostics
