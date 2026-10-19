# Phase 1 — ruling on the consolidation gate set

**Question:** do `ATR_COMPRESSION` / `RANGE_COMPRESSION` / `NO_TREND_SLOPE_ABS_MAX` belong in the
Potter Box detector, and are they why it fires 5 times in 2 years?

**Measured over 24,254 decision bars, 55 tickers, ~2 years** (`compare_gates.py`, `gate_comparison.json`).

## Finding 1 — the gates are not from the method

| source | what it specifies |
|---|---|
| `Potter_Box_Visual_Dataset.md` (charts from @potter.stocks) | box, cost basis midline, empty space, multiple touches of top/bottom. **No compression gate.** |
| `10 videos transcribed.txt` (~82k words, his own words) | levels, cost basis, empty space, contracts. **No compression gate.** |
| `scanner/research/potter_visual_doctrine.md` (repo's own video-derived doctrine) | close-based control, cost-basis midpoint, touch minimums, "break requires close outside control level plus prior-close bias vs cost basis". **No compression gate.** |
| `PotterBox_Scanner_Handoff.docx` — *"Claude Code Handoff Document", April 2026* | **the only source of all three compression rules** |

All three gates trace to a single AI-authored engineering handoff, not to the strategy. They read
like plausible technical analysis; nothing validated them against a chart.

## Finding 2 — the implementation also diverges from that handoff's own formulas

| gate | handoff spec | implemented (`potter_box.py`) | same thing? |
|---|---|---|---|
| ATR compression | ATR(14) < 0.75 × **mean ATR over prior 30 bars** | `atr_value <= 0.75 ×` **mean raw High−Low of prior window** | **no** — ATR ≥ mean raw range (true range includes gaps), so the test is biased to fail |
| range compression | **latest candle's** range < 0.65 × **median** range of consolidation window | **mean** consolidation range / **mean prior-window** range ≤ 0.65 | **no** — different numerator, denominator, and statistic |
| no trend | \|first close − last close\| < **1.5 × ATR** | normalized regression slope ≤ **0.0015** (absolute constant, no ATR, no volatility scaling) | **no** — different quantity, and it does not scale across stocks |

The config *constants* (0.75, 0.65, 15, 14) were copied faithfully; the *quantities they are compared
against* were not. Right numbers, wrong metrics.

## Finding 3 — fixing the formulas does not fix the starvation

| gate set | firings, 2y × 55 tickers |
|---|--:|
| implemented | **10** |
| spec formulas, close-based box | **9** |
| spec formulas, spec high/low box | **13** |
| **doctrine only (no compression)** | **2,212** (1,192 bullish) |

Marginal pass rates loosen a lot under the spec (no-trend 14.9% → 42.0%, range 5.5% → 12.3%) and the
joint rate does not move (0.04% → 0.04%). **The conjunction is the problem, not the formulas.**
Three volatility-compression conditions AND a breakout condition are near-mutually-exclusive: they
require price to be maximally quiet on the same bar it breaks decisively out of range.

So my working hypothesis going into Phase 1 — "the gates are misimplemented, fix them and the
detector works" — is **wrong**. The misimplementation is real (Finding 2) but costs ~1 firing.

## Recommendation

**Adopt the doctrine gate set as the detector's pass condition:** close-based control levels,
cost-basis midpoint, touch minimums (≥2/≥2), close outside control, prior-close bias vs cost basis.
Yield: 2,212 firings, ~20 per ticker per year — a rate consistent with the handoff's own described
workflow (scanner alerts, human visually confirms, human decides).

**Keep the three compression metrics as FEATURES, not gates.** `range_compression_ratio`,
`no_trend_score` and `atr_value` are already extracted and stored per record. Demoting them from
hard gates to ranking inputs lets the evidence layer measure whether compression predicts outcomes —
which is precisely the stated design intent for the doctrine-v2 layer ("so the system can learn which
v2 mechanics actually improve outcomes before any promotion").

### Why this is not "loosening a gate to manufacture signals"

The repo's standing rule forbids weakening thresholds to make a blocked system produce output. This
is a different action, and the distinction should be checked rather than assumed:

1. It is not a threshold change — no constant is being relaxed. It is the removal of conditions that
   no primary source attributes to the strategy.
2. The conditions being removed are demonstrably misimplemented relative to their own only source.
3. The metrics are retained as features, so nothing is discarded, only demoted from gate to evidence.
4. **No promotion, readiness, or cost gate changes.** This governs what enters the evidence corpus,
   not what is allowed to reach a live alert. Readiness stays blocked and must be re-earned on the
   corrected corpus.
5. The honest cost: signals go from ~0 to ~20/ticker/year. If the corrected corpus then shows no edge,
   that is a real answer — this change cannot by itself make anything look profitable.

## Open ambiguities (flagged, not silently resolved)

- **"the most recent candle"** in the handoff's range rule could mean the breakout candle or the last
  consolidation candle. Read as the breakout candle it is self-defeating. Both measured (12.3% / 14.2%);
  neither changes the joint result.
- **Box extremes:** handoff says highest high / lowest low; the video-derived doctrine says closes have
  priority over wicks. Doctrine is later and source-derived, so close-based control is used, with the
  high/low variant measured alongside (13 vs 9 firings — immaterial).
- **`MIN_BOX_*_TOUCHES = 2` and `BOX_TOUCH_TOLERANCE_PCT`** are doctrine-documented, so they stay.
  Touch counting uses closes within `max(10% of box range, 0.15% of price)` — the 10%-of-range term is
  undocumented and is doing most of the work. Not changed in this phase; flagged for Phase 3.
