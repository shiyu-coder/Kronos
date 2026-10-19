# Backtest sprint — 2026-07-10 (audited correction)

> **Audit status (2026-07-10): the original `HOLDS` verdict is withdrawn.** All preregistrations and results were first committed together, so Git does not prove advance registration; local mtimes are mutable, and the E4/W2 runners were created after their result files. The historical input was not snapshotted or hashed and has since changed (11,026 → 11,055 records), so an exact historical rerun is impossible from the repository as committed. The saved files do contain every declared cell, but the p-value labeled “day-clustered” used an IID shortcut rather than clustered covariance. See `audit_sprint.py` for the mechanical audit.

Orchestrated run: 6 experiment agents (5 workers + 1 adversary), 48 serialized walk-forward verdicts over the evidence index (11,026 records, 6,298 bullish, ~434 OOF days, n_eval ≈ 5,800 per cell). The historical machinery used an expanding window, 21-day refit, 9-day purge, and 6-gate acceptance. These are locally time-ordered experiment artifacts, not independently proven preregistrations or exact independent reproductions.

**Corrected headline: 0 of 48 serialized gated verdicts passes acceptance (E1 4 + E2 16 + E3 14 + W2 8 + E4 6); 47 are nondegenerate because E3's `kronos_only` cell reduces to an intercept-only model. These are not 48 independent tests. Nothing ships, but stronger scientific claims below are downgraded as noted.**

## The four findings

### 1. A fixed-return-subtraction sensitivity crosses zero near 25bps/side (E4)
Take-all bullish on the index: WR 48.75%, avg R +0.108, tail-driven (median R −0.034, tail rate 6.05%). Subtracting a flat 0.50 percentage point round trip from every stored outcome and retaining the original risk denominator produces avg R +0.0005; the 50bps/side scenario produces −0.107. This is a useful sensitivity alarm, **not** a bar-level execution model and not evidence that realized slippage is 25bps/side. “The edge dies at 25bps” was too strong.

### 2. The planned sample size is reached around 2026-11-06 only under stated assumptions (E4)
Journal: 29 resolved research candidates, WR 44.8%, Wilson 95% CI [28.4%, 62.5%]. A one-sample, two-sided normal approximation with the observed baseline treated as fixed, true lift exactly +10pp, 80% power, and an assumed 12 independent resolutions/week requires 195 new observations, reached around **2026-11-06**; +5pp requires 780, around **2027-10-08**. These dates are scenario outputs, not universal embargoes on WR claims; baseline uncertainty, dependence, attrition, or a different true effect changes them.

### 3. Exploratory top-K estimates are worse for two objectives (E4, E3, W2)
- p_win: no top-K discrimination at any K (all bootstrap CIs straddle 0).
- expected_r and tail_prob: top-K win-rate deltas have unadjusted 95% day-block bootstrap intervals fully below zero at some scanned K values. Because 24 objective/K/tail comparisons were explored without multiplicity adjustment, describe these as exploratory interval results, not confirmatory “significance.”
- tail_prob is doing its designed job — trading WR for tail capture (top tercile holds ~62% of ≥2R events; K=33 avg-R delta CI [+0.002, +0.139], the only positive selection signal in the sprint, and a fragile one).
- Bearish: nothing ranks OOF (best +0.017, p=0.13); the sprint's own bearish take-all is negative expectancy (n=4,728 index records, avg R −0.0735), consistent with the production validation cohort's block (brief cohort: n=563, avg R −0.23). Bearish stays blocked.

### 4. No Kronos forecast values are present in the saved index (E3)
The three forecast-valued fields have 0 finite values across all 11,026 saved records; `kronos_sample_count` is populated on every row but is constant zero. Therefore “all kronos_* features are 0%-populated” was false, while the substantive conclusion—no historical record contains a usable Kronos forecast—holds. Source tracing shows `build_edge_records_from_bars` does not pass `kronos=` into `extract_edge_features`.

## Secondary findings

- **Feature geometry is objective-conditional noise (E3 + W2)**: dropping the box-geometry/compression cluster (compact5 = volume/momentum features only) cuts tail_prob |IC| by two-thirds (−0.0664 → −0.0233); the improvement is stable across λ ∈ {0.3,1,3,10} and purge ∈ {9,14}, decays second-half (−0.016 → −0.041), and **reverses under p_win** (geometry helps there: −0.0115 vs −0.0488 without it). Interpretation: curse-of-dimensionality under class imbalance for the rare-event objective, not worthless features. No compact5 variant approaches gates.
- **Quarterly regime swings dominate (E4)**: take-all WR ranges 36.9%–55.2% by quarter; three quarters have negative take-all avg R. Any future WR comparison must be regime-controlled.

## E1 — robust magnitude objectives (Huber / quantile-0.75 / winsorized-R / rank-R): hypothesis rejected

Audit correction: “exact reproduction” in the historical paragraph below means same-input agreement by a copied harness, not an independent rerun from an immutable snapshot.

Harness sanity: exact reproduction of the registry expected_r control (−0.0876, n 5849). **0/4 cells pass.** huber_r and rank_ridge lift IC to small insignificant positives (+0.013/+0.016, p_day ≈ 0.37–0.39) but with **significantly negative tercile spread** (CI entirely below zero) and below-pro-rata tail retention (~23% vs 33%) — the top-ranked bucket underperforms the bottom and steers away from the tail trades the edge lives on. quantile_r_75 and winsor_ridge reproduce expected_r's negative IC almost exactly. The informative part: four loss functions built specifically to defeat tail-value domination all land in the same place, so the original expected_r failure was never about tail domination of the squared loss. Three objective families (win-probability, raw-R ridge, tail-robust/rank variants) have now failed on the same 10-feature linear model — the binding question is the feature set / model class, not the loss function.

## E2 — tail_prob grid (tail_r × λ, 16 cells): family is dead in the whole neighborhood

Audit correction: the control match below has the same shared-input/shared-machinery limitation, and the saved p-values use the superseded IID day-count shortcut.

Harness sanity: exact reproduction of the registry control (−0.0664, n 5809). 16 pre-registered cells: tail_r ∈ {1.5, 2.0, 2.5, 3.0} × λ ∈ {0.3, 1, 3, 10}. **0/16 pass.** IC range −0.0635…−0.0771 (never crosses zero), day-clustered p range 0.90–0.95 (flat null, not a near-miss — with 16 cells you'd expect ~0.8 false positives, and none appeared because the null held everywhere). tail_r/λ were not the missing ingredient: the L2-logistic-on-tail-threshold approach itself carries no ranking signal on this feature set. Persistent non-ranking wrinkle across all 16 cells: top-tercile tail capture stays at 61–65% vs 33% pro-rata and Brier beats naive — a model can over-represent the tail in its top bucket while mis-ranking the bulk. Consistent with E4's finding that tail_prob buys tail exposure at the cost of win rate.

## Adversary review (post-sprint): original verdict withdrawn

The next paragraph is retained as the historical review record, not as the current conclusion. It matched saved cells and calculations but treated mutable mtimes as preregistration proof, shared harnesses as independent reproduction, and the old IID day-count shortcut as clustered inference. Those conclusions are superseded by the audit status at the top of this report.

An adversarial agent independently reproduced every anchor number with the repo's own machinery (both harness controls to the digit, the slippage collapse, the power-analysis dates, the top-K bootstrap CIs, the Kronos 0%-populate claim at data and source level), verified preregistration mtimes precede results in all five experiments with exact cell-set matches, and audited the shared walk-forward machinery for leakage (none: train-window-only standardization, purged refits, no self-prediction). Verdict: HOLDS, no MEDIUM+ defect. Its LOW findings, all addressed or recorded:

1. This report originally cited the stale brief-cohort bearish figure (n=563, avg R −0.23) where the sprint's own bearish universe (n=4,728, avg R −0.0735) belonged — corrected above; conclusion unchanged.
2. E1's preregistration.json lists a stale expected control (−0.086/n=5764, from the pre-CLSK-fix index) — the harness validated against the true current control (−0.0876/5849) and the tolerance absorbed the gap. The prereg file is deliberately left untouched (post-hoc edits to preregistrations are worse than stale targets).
3. **Production finding worth keeping**: the 9-calendar-day purge can be ~1 day short of a 5-trading-day outcome window across a holiday-extended long weekend (worst case ~10 calendar days). Any residual leakage only inflates apparent skill, so it cannot manufacture this sprint's null results — and W2's purge-14 robustness cell empirically bounds the effect (ΔIC 0.0046). For the nightly pre-registered suite, bumping purge (9 → 11) would be a deliberate protocol-version change, not a hot patch; flagged as a candidate for the next protocol revision.
4. Cell count is 48 serialized verdicts, or 47 nondegenerate verdicts when the intercept-only Kronos cell is excluded. Neither number is a count of independent experiments.

## Multiplicity ledger

48 serialized verdicts were evaluated at a nominal one-sided α=0.05 gate, with extensive dependence and reuse across cells. Observed full-gate passes: 0. Every declared cell (including failures) is recorded; nothing was dropped. The original “~2–3 expected false positives” arithmetic assumes independent, valid uniform-null p-values and should not be used for this dependent sweep.

## What this changes about the plan

1. **Cost model before ranking research.** The 25bps sensitivity says the binding constraint is expectancy-after-friction, not selection. Next lab investment: per-ticker spread/slippage estimates from the options and quote data already collected, and an after-cost acceptance gate.
2. **Kronos backfill** (task chip filed) — the only untested feature family, and the point of the fork.
3. **Stop iterating loss functions on the current linear model + 10 features.** E1 closed that door: four purpose-built robust losses land where expected_r did. The remaining ranking threads, in value order: (a) Kronos-forecast backfill (new information, not a new loss), (b) tail_prob/compact5 with explicit class-imbalance handling, (c) nonlinear interactions — only after (a).
4. Directions unchanged: take-all-bullish, bearish blocked, live alerting stays gated by the readiness audit.

## Post-sprint production follow-up

Evening 2026-07-10: the purge-window production finding above was converted into a protocol hardening patch. Production edge analog embargo, cross-ticker embargo, and the nightly calibration purge now all use 11 calendar days, with regression coverage for the 10-day holiday-long-weekend boundary. This does not change the pre-registered sprint results above; it tightens future validation runs.
