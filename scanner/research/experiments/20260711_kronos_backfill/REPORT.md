# Kronos forecast backfill — ranking test (2026-07-10 eve)

## Verdict
**NULL — 0 of 6 pre-registered cells pass the production ranking gate. Real Kronos forecast features add no usable ranking signal to the bullish edge. Adversary HOLDS.**

This resolves the highest-value open experiment from the 2026-07-10 sprint: the fork's namesake model, tested for the first time as a ranking feature, does not clear the gate.

## Background
The historical edge index had 0 / 11,026 finite Kronos forecast values — `build_edge_records_from_bars` never passed `kronos=` into `extract_edge_features`. This experiment wired that path (leak-free, opt-in, live-parity), backfilled a snapshot with real forecasts, and ran the pre-registered ranking cells.

## Method
- **Wiring** (`scanner/edge/retrieval.py`): opt-in `kronos` + `kronos_direction_filter` params on `build_edge_records_from_bars`. Default `None` → byte-identical to prior behavior (278/278 tests green). A non-strippable guard raises if Kronos is ever handed a bar later than the decision timestamp. Live-parity via `dataclasses.asdict` on the raw `KronosResult` (the same fields the live scan feeds `extract_edge_features`).
- **Backfill** (`run_backfill.py --universe full --bullish-only-kronos`): 55/55 tickers OK, 10,919 records, 6,194 bullish, **100% finite Kronos on bullish rows**. ~1h57m wall (bullish-only, ~1.2 s/eval, CPU). Written to a SEPARATE snapshot — production `edge_retrieval_index.json` untouched.
  - Snapshot sha256: `b34bcb20dcdf3edd9f669eeaea2c4d001593d274c3f7e2736cf3644ba983a13e`
  - results.json sha256: `ac330a8daee0750299542d7c96decdbfc4668ed3527ac0256996078fd4c07cea`
- **Cells** (`run_cells.py --run-cells`): the mandatory harness-sanity control reproduced the repo's production `walk_forward_calibration` expected_r IC exactly (−0.0996 == −0.0996) before any cell was trusted. Gates identical to the production ranking gate (IC ≥ +0.07, day-clustered p ≤ 0.05, n ≥ 300, tercile-spread CI-low > 0, tail retention ≥ pro-rata, beats-naive MSE). Purge 11 calendar days (read live from `EDGE_EMBARGO_DAYS`).

## Results (bullish, n_eval = 5,792 each)
| cell | rank IC | Δ vs control | 6-gate |
|---|---|---|---|
| control_standard10 | −0.0996 | — | FAIL |
| combined_standard10_plus_kronos3 | −0.1024 | −0.0028 | FAIL |
| solo_kronos3 | −0.0625 | +0.0371 | FAIL |
| ablate_drop_kronos_directional_agreement | −0.0980 | +0.0016 | FAIL |
| ablate_drop_kronos_median_forecast_return_pct | −0.1010 | −0.0014 | FAIL |
| ablate_drop_kronos_worst_sampled_return_pct | −0.1095 | −0.0099 | FAIL |

All ICs negative (gate needs ≥ +0.07). Adding Kronos to the baseline marginally worsens IC; Kronos-only is least-negative but still ranks backwards.

## Why the null is real, not a plumbing artifact
- `solo_kronos3` IC (−0.0625) differs from control (−0.0996) → the features are genuinely read and influential, not silently zeroed (contrast the old 0-populated `kronos_only` cell, which degenerated to intercept-only).
- n = 5,792 → well-powered; IC is firmly negative, not borderline.

## Adversary review — HOLDS (no MEDIUM+ findings)
An adversarial agent tried to break the null as an artifact and could not:
- **Horizon aligned**: Kronos forecasts `pred_len = PRED_DAYS = 5`; the label is the 5-bar triple-barrier outcome. Same horizon.
- **Outputs non-degenerate**: 6,194 unique median forecasts, range −64% to +20%. Real dispersion.
- **Information ceiling below gate**: even a fully-leaky, in-sample, sign-optimal fit tops at +0.0615 for the Kronos features — under +0.07. The signal isn't there to find.
- Alignment, liveness, sign-handling, and reproduction all clean.

## Leftover threads (future pre-registered hypotheses — NOT tested here)
1. **Kronos as a contrarian filter**: raw associations are weakly *negative* (Kronos is systematically bearish on these bullish breakouts, median forecast ≈ −4.4% / 5d). "De-prioritize setups where Kronos is most bullish." Effect ~0.05 — adversary's read: likely won't survive a binary gate.
2. **Richer Kronos-path features** (dispersion / skew / other quantiles of the 10 sampled paths) — not persisted in the snapshot; would need a new backfill.
3. **LOW parity nit**: the −4.4% bearish level bias may be an input-representation issue (backfill fed vendor-daily bars; live feeds synthetic-from-intraday sessions). Offset-invariant → doesn't affect the rank null, but worth checking before any live Kronos use.

## Integrity note
The preregistration was authored **blind** — cells, features, and gates were fixed before any Kronos outcome was computed. It was committed post-run in this session, so the git ordering reflects organization, not a timestamped advance-registration proof; the blind-design property is the integrity guarantee. The snapshot (sha above) plus results.json are the reproducible record; regenerating the snapshot via `run_backfill.py` will not be byte-identical because vendor history revises over time (same known limitation the 2026-07-10 sprint noted).

## Reproduce
```
venv\Scripts\python.exe scanner\research\experiments\20260711_kronos_backfill\run_backfill.py --universe full --bullish-only-kronos
venv\Scripts\python.exe scanner\research\experiments\20260711_kronos_backfill\run_cells.py --run-cells
```
