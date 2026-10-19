"""Scanner runtime configuration (non-secret values only)."""

import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
RETIREMENT_MARKER_PATH = ROOT_DIR / "RETIRED.json"
LOG_DIR = ROOT_DIR / "logs"
REPORT_DIR = ROOT_DIR / "reports"
EVIDENCE_DIR = REPORT_DIR / "evidence"
TUNING_DIR = ROOT_DIR / "tuning"
OVERRIDES_PATH = TUNING_DIR / "overrides.json"
EDGE_INDEX_PATH = REPORT_DIR / "edge_retrieval_index.json"
EDGE_SCAN_REPORT_PATH = REPORT_DIR / "edge_scan_report.json"
EDGE_VALIDATION_REPORT_PATH = REPORT_DIR / "edge_validation_report.json"
EDGE_DIAGNOSTIC_REPORT_PATH = REPORT_DIR / "edge_diagnostic_report.json"
EDGE_AUDIT_REPORT_PATH = REPORT_DIR / "edge_audit_report.json"
META_MODEL_PATH = REPORT_DIR / "meta_model.json"

MIN_STOCK_PRICE = 5.00
MIN_KRONOS_AGREEMENT = 0.65
MIN_RR = 1.5
MIN_EMPTY_SPACE_SCORE = 2
EARNINGS_BLOCK_DAYS = 10
# Yahoo forward earnings dates are estimates (the raw payload's
# isEarningsDateEstimate flag is hidden by yfinance's calendar), with
# documented off-by-one and estimate-window failure modes. The cushion
# widens the block window to absorb one day of source error in either
# direction; it does not make an unreliable source reliable.
EARNINGS_ESTIMATE_CUSHION_DAYS = 1
BLOCK_ON_UNKNOWN_EARNINGS = True
BLOCK_ON_EX_DIVIDEND = False
MIN_ATM_OPEN_INTEREST = 500
MAX_ATM_BID_ASK_SPREAD_PCT = 0.12
CONSOLIDATION_BARS = 15
ATR_PERIOD = 14
ATR_COMPRESSION = 0.75
RANGE_COMPRESSION = 0.65
NO_TREND_SLOPE_ABS_MAX = 0.0015
MIN_BOX_TOP_TOUCHES = 2
MIN_BOX_BOTTOM_TOUCHES = 2
BOX_TOUCH_TOLERANCE_PCT = 0.0015
USE_CLOSE_BASED_CONTROL = True
RESEARCH_CANDIDATE_MIN_SCORE = 62
DOCTRINE_V2_SCORE_BASELINE = 70
RESEARCH_NEAR_BREAKOUT_PCT = 0.012
RESEARCH_MIN_VOLUME_EXPANSION = 1.15
PRED_DAYS = 5
DRY_RUN_DEFAULT = True

INTRADAY_INTERVAL = "30m"
INTRADAY_LOOKBACK = "60d"
DAILY_PROXY_LOOKBACK = "2y"
MARKET_DATA_PROVIDER_DEFAULT = "auto"  # auto | alpaca | yfinance
ALPACA_FEED = "iex"  # iex (free) or sip (subscription)

SYNTHETIC_SESSION_ANCHOR_HOUR = 20
SYNTHETIC_SESSION_ANCHOR_MINUTE = 0
TIMEZONE = "America/New_York"

KRONOS_MODEL_NAME = "NeoQuasar/Kronos-small"
KRONOS_TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_LOOKBACK_BARS = 60
# The research path builds synthetic sessions from a 60-calendar-day intraday
# window (~42 trading sessions), so demanding a full 60-bar lookback made
# Kronos unrunnable there. Use what's available down to this floor.
KRONOS_MIN_BARS = 30
KRONOS_SAMPLE_COUNT = 10
# Run Kronos on research candidates (a few per day) so the journal
# accumulates agree/disagree outcome evidence; the strict pipeline only
# reaches Kronos after the options gate, which historically never happened,
# so the model's lift was unmeasurable.
KRONOS_RESEARCH_ENABLED = True

# The scanner strategy was retired on 2026-08-20 after mature-negative
# out-of-sample evidence. Manual research remains available for audit, but no
# scheduled status message should imply that the go-live program is active.
BRIEF_TELEGRAM_ENABLED = False

MINIMAX_ENABLED_DEFAULT = False
MINIMAX_MODEL = "MiniMax-M2.7-highspeed"
MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_TIMEOUT_SECONDS = 20
MINIMAX_MAX_OUTPUT_TOKENS = 220
MINIMAX_TEMPERATURE = 0.2

CALIBRATION_PASS_AVG_ABS_MAX = 0.35
CALIBRATION_WARN_AVG_ABS_MAX = 0.90
CALIBRATION_MIN_MATCHED_ROWS = 20

TUNING_ENABLED_DEFAULT = True
OUTCOME_REVIEW_MAX_RECORDS = 500
OUTCOME_MIN_AGE_DAYS = 3
# Pending decisions older than this can never resolve once they fall out of
# the 60d intraday window; expire them instead of re-checking forever.
OUTCOME_EXPIRY_DAYS = 45
AUTOTUNE_MIN_SAMPLES = 20
AUTOTUNE_STEP_SIZE = 0.05
AUTOTUNE_EMPTY_SPACE_STEP = 1
EDGE_ANALOG_K = 7
# Purge window: an analog's outcome must be RESOLVED before the query bar or
# its realized R leaks future market moves into the query's score. Outcomes
# span PRED_DAYS=5 trading bars (7 calendar days, 8-9 across a normal
# holiday, and up to ~10 across holiday-extended long weekends), so both
# embargoes must cover at least 11 calendar days. The old values (5
# same-ticker / 1 cross-ticker, then 9/9) admitted edge cases whose outcome
# windows could overlap the query's own future - inflating apparent
# walk-forward skill with concurrent-week correlation.
EDGE_EMBARGO_DAYS = 11
EDGE_MIN_ANALOGS = 3
# Analogs must share the query's breakout direction; a bullish setup should
# not borrow expectancy from bearish history.
EDGE_ANALOG_DIRECTION_MATCH = True
# During validation, block analogs from any ticker within this many days of
# the query so same-week market-wide moves (and unresolved analog outcomes)
# cannot inflate apparent skill.
EDGE_CROSS_TICKER_EMBARGO_DAYS = 11
EDGE_VALIDATION_MAX_RECORDS = 1500
EDGE_VALIDATION_THRESHOLDS = (45, 55, 65)
EDGE_VALIDATION_TOP_K = 25

# Round-trip transaction cost charged against every outcome BEFORE any gate
# metric is computed. `walk_triple_barrier` stores outcomes gross - it never
# charges a cost - so rank IC, average R, Wilson-LB precision and the HAC
# t-stats were all measuring an edge that no execution could actually capture.
# A gate that reads 0 bps can only ever be too permissive, which is the one
# direction a fail-closed scanner must not fail in.
#
# 25 bps/side is the pre-registered basis from the 2026-07-10 sprint
# (research/experiments/20260710_sprint/e4_decision), which is where the
# bullish cohort's expectancy died. It is the honest floor for this universe:
# sub-$20 small caps (CHPT, LUNR, SOUN, LYFT) traded through option chains
# that frequently fail the liquidity filters outright.
EDGE_COST_BPS_PER_SIDE = float(os.getenv("KRONOS_COST_BPS_PER_SIDE", "25"))

# Exit geometry for the lab's encoded trade plan. The stop side stays the
# empty-space risk (ATR/2% fallback); these choose the TARGET. Env overrides
# let a sweep flip variants per process without code edits; the committed
# defaults are the shipped geometry. These are NOT adaptive-policy tunables -
# the outcome definition must not drift under the feedback loop that is
# judged against it.
#
# Shipped default "none" (no profit target; stop/horizon exits only) per the
# 2026-07-02 six-variant sweep (trial_registry kind=exit_geometry_trial):
# every tested target truncated more bullish upside than it locked in -
# nearest level -0.014 avg R, 1.5R floor +0.135, 2R floor +0.153, 2xATR
# +0.158, no target +0.195 (t=5.16, n=910 walk-forward bullish samples).
# Bearish stayed negative under all six geometries and remains
# direction-blocked by the audit.
EDGE_EXIT_TARGET_MODE = os.getenv("KRONOS_EXIT_TARGET_MODE", "none")
EDGE_EXIT_TARGET_R_FLOOR = float(os.getenv("KRONOS_EXIT_TARGET_R_FLOOR", "0.0"))
EDGE_EXIT_TARGET_ATR_MULT = float(os.getenv("KRONOS_EXIT_TARGET_ATR_MULT", "2.0"))

# Corporate-action basis for the DAILY bars that feed the edge index. "raw"
# (the pre-2026-07-02 behaviour) let splits and dividends read as real price
# moves inside the 5-bar outcome window - a reverse split looks like a
# catastrophic gap and a dividend like a stop-clipping drop (the index's top
# ticker by record count, AGNC, distributes ~1%/month). "split" removes the
# corruption without crediting dividends an options holder never receives.
# Env-overridable for sweeps, NOT an adaptive-policy tunable - the outcome
# definition must not drift under the feedback loop judged against it.
EDGE_BARS_ADJUSTMENT = os.getenv("KRONOS_BARS_ADJUSTMENT", "split")

# Two-sided adaptive policy guards. Loosening only touches the research
# threshold (a data-collection throttle for paper counterfactuals, not a live
# gate) and only when a lower-threshold cohort dominates the current one on
# conservative bounds, after a cooldown and a second-day confirmation.
ADAPTIVE_LOOSEN_MIN_SAMPLES = 12
ADAPTIVE_LOOSEN_MIN_WILSON = 0.30
ADAPTIVE_LOOSEN_LB_MARGIN = 0.05
ADAPTIVE_LOOSEN_RET_MARGIN = 0.25
ADAPTIVE_LOOSEN_MAX_STEP = 10
ADAPTIVE_CHANGE_COOLDOWN_DAYS = 7

MIN_RR_BOUNDS = (1.1, 2.5)
MIN_KRONOS_AGREEMENT_BOUNDS = (0.50, 0.85)
MIN_EMPTY_SPACE_SCORE_BOUNDS = (1, 3)
MAX_ATM_BID_ASK_SPREAD_PCT_BOUNDS = (0.08, 0.25)
MIN_ATM_OPEN_INTEREST_BOUNDS = (200, 3000)
ATR_COMPRESSION_BOUNDS = (0.55, 1.10)
RANGE_COMPRESSION_BOUNDS = (0.45, 1.05)
NO_TREND_SLOPE_ABS_MAX_BOUNDS = (0.0008, 0.0040)
RESEARCH_CANDIDATE_MIN_SCORE_BOUNDS = (45, 80)
DOCTRINE_V2_SCORE_BASELINE_BOUNDS = (60, 90)

DEFAULT_WATCHLIST = [
    "SOFI", "MARA", "RIOT", "HIMS", "TTD",
    "SOUN", "UPST", "SNAP", "LYFT", "RIVN",
    "GME", "AAL", "CCL", "PFE", "T",
    "F", "KEY", "OPEN", "CHPT", "NIO",
    "CLSK", "CIFR", "LUNR", "BBAI", "EVGO",
    "PLTR", "HOOD", "RKLB", "AFRM", "IONQ",
]

# Extra liquid, optionable names used ONLY to build the historical retrieval
# index and validation cohort. They nearly double the honest walk-forward
# sample without touching the live scan watchlist.
EDGE_INDEX_EXTRA_UNIVERSE = [
    "AMD", "INTC", "MU", "MRVL", "PYPL",
    "UBER", "ROKU", "PINS", "DKNG", "PLUG",
    "LCID", "BAC", "WFC", "C", "DAL",
    "UAL", "JBLU", "CLF", "VALE", "GOLD",
    "KGC", "AGNC", "ET", "PBR", "CVS",
]


def _as_int(value) -> int:
    # Accept float-looking strings ("600.0") for int tunables.
    return int(float(value))


# Tunable defaults captured before any override application, so a reload
# resets keys that were removed from the overrides file instead of keeping
# stale values until the next process start.
_TUNABLES = {
    "MIN_RR": (MIN_RR, float),
    "MIN_KRONOS_AGREEMENT": (MIN_KRONOS_AGREEMENT, float),
    "MIN_EMPTY_SPACE_SCORE": (MIN_EMPTY_SPACE_SCORE, _as_int),
    "MAX_ATM_BID_ASK_SPREAD_PCT": (MAX_ATM_BID_ASK_SPREAD_PCT, float),
    "MIN_ATM_OPEN_INTEREST": (MIN_ATM_OPEN_INTEREST, _as_int),
    "ATR_COMPRESSION": (ATR_COMPRESSION, float),
    "RANGE_COMPRESSION": (RANGE_COMPRESSION, float),
    "NO_TREND_SLOPE_ABS_MAX": (NO_TREND_SLOPE_ABS_MAX, float),
    "RESEARCH_CANDIDATE_MIN_SCORE": (RESEARCH_CANDIDATE_MIN_SCORE, _as_int),
    "DOCTRINE_V2_SCORE_BASELINE": (DOCTRINE_V2_SCORE_BASELINE, _as_int),
}


def _apply_overrides():
    payload = {}
    if OVERRIDES_PATH.exists():
        try:
            raw = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload = raw
        except Exception:
            payload = {}

    for name, (default, caster) in _TUNABLES.items():
        if name in payload:
            try:
                globals()[name] = caster(payload[name])
                continue
            except (TypeError, ValueError):
                pass
        globals()[name] = default


def reload_overrides():
    """Refresh in-process tuning values after an override file is updated."""
    _apply_overrides()


_apply_overrides()
