# Chart-pattern feature discovery — ranking test (2026-07-10 eve)

## Verdict
**NULL — 0 of 13 pre-registered cells pass the production ranking gate. Every IC is negative (features rank backwards). Classic chart "shapes," encoded as point-in-time computable features and tested honestly, carry no usable ranking signal on this index.**

## Motivation
Test whether classic chart patterns predict the 5-bar R outcome — without the overfitting trap of visually eyeballing charts. Each shape became a computable, point-in-time feature function run through the same purged walk-forward harness with production gates and multiplicity control. Visual intuition sources the hypotheses; the harness kills the false ones.

## Method
- **Data availability**: the index stores ~61 scalar per-record fields — no raw OHLCV and no bar cache (bars are fetched live per run). All features are pure functions of stored, already-point-in-time fields. Two originally-planned shapes (wick-rejection ratio, gap state) genuinely require raw per-bar OHLC that isn't stored — **excluded and documented** rather than bolted on with a slow, non-reproducible live-bar fetch.
- **6 candidate features** (formulas + one-line hypotheses in `preregistration.json`): `compression_breakout_thrust`, `box_retest_asymmetry`, `volume_thrust_interaction`, `atr_relative_coil`, `box_stack_coil`, `breakout_snap_ratio` — none duplicating existing `range_compression_ratio` / `volume_expansion` / `breakout_strength_pct`.
- **13 cells**: 6 solo + 1 combined (standard-10 + all 6 patterns) + 6 leave-one-out ablations. Gates identical to production. Purge 11 days (live from config). Harness-sanity control reproduced exactly (IC −0.09, n = 5,849) before cells were trusted.
  - results.json sha256: `044ca5c5782dd6a7f52174e3c1684fd974e38967f4a8a5d69b35c4fa9637791e`

## Results (bullish, n_eval = 5,849 each)
| cell | rank IC | spread CI-low | tail obs/exp | 6-gate |
|---|---|---|---|---|
| solo_compression_breakout_thrust | −0.0808 | −0.4025 | 0.345 / 0.333 | FAIL |
| solo_box_retest_asymmetry | −0.0937 | −0.4142 | 0.308 / 0.333 | FAIL |
| solo_volume_thrust_interaction | −0.0519 | −0.3250 | 0.339 / 0.333 | FAIL |
| solo_atr_relative_coil | −0.1066 | −0.2352 | 0.500 / 0.333 | FAIL |
| solo_box_stack_coil | −0.0415 | −0.2242 | 0.390 / 0.333 | FAIL |
| solo_breakout_snap_ratio | −0.0739 | −0.4038 | 0.325 / 0.333 | FAIL |
| combined_all_patterns | −0.1057 | −0.1902 | 0.500 / 0.333 | FAIL |
| ablate_drop_compression_breakout_thrust | −0.1068 | −0.1751 | 0.506 / 0.333 | FAIL |
| ablate_drop_box_retest_asymmetry | −0.1016 | −0.1768 | 0.506 / 0.333 | FAIL |
| ablate_drop_volume_thrust_interaction | −0.1076 | −0.1958 | 0.514 / 0.333 | FAIL |
| ablate_drop_atr_relative_coil | −0.0838 | −0.1557 | 0.463 / 0.333 | FAIL |
| ablate_drop_box_stack_coil | −0.1145 | −0.2262 | 0.503 / 0.333 | FAIL |
| ablate_drop_breakout_snap_ratio | −0.1024 | −0.1795 | 0.540 / 0.333 | FAIL |

(day-clustered p ranges 0.88–0.9999, one-sided against IC > 0 — the associations run the wrong way, not borderline. No cell beats naive MSE.)

## Interpretation
0 / 13 pass; every IC is negative. Nonlinear ratio/interaction features built from the existing geometry+volume fields land in the same negative-IC basin as prior ranking attempts (loss-function iteration, tail_prob). Patterns are a *re-transform of information already in the index*, not new information — a closed door.

**Methodological note**: several of these shapes would have looked predictive on cherry-picked charts. Run point-in-time with purge and multiplicity control, they're worthless-to-harmful. This is the value of the harness over visual pattern-hunting.

## Integrity note
The preregistration was authored **blind** — features, cells, and gates fixed before any outcome correlation was computed. Committed post-run in this session; the blind-design property (not git timestamp) is the integrity guarantee.
