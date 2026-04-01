# Thesis Current Model Pipeline

Canonical reference for the model that should be described in the thesis as of 2026-04-01.

This document is the project-level answer to four questions:

1. What is the current final model?
2. Which files and directories were actually run from start to finish?
3. What is the architecture of the upstream model and the downstream overlay?
4. What caveats must be stated so that the thesis description is technically accurate?

---

## 1. Canonical Final Model

The current final model is the Round 3 promoted champion that survived Round 4 unchanged.

- Canonical result file:
  - `results_runpod/nvidia_tournament_round3_group_constraints/promoted_best_result.json`
- Round 3 promotion record:
  - `results_runpod/nvidia_tournament_round3_group_constraints/promotion_decision.json`
- Round 4 survival record:
  - `results_runpod/nvidia_tournament_round4_mdd_reopt/search/promotion_decision.json`
- Round 4 evaluated seed snapshot:
  - `results_runpod/nvidia_tournament_round4_mdd_reopt/search/best_result.json`

Current final metrics:

| Metric | Value |
| --- | ---: |
| Sharpe | 1.0724 |
| Annual return | 10.1035% |
| Max drawdown | -9.4414% |
| Bootstrap P(MDD >= -10%) | 57.98% in Round 4 baseline replay |
| Track name | `grp_eq60` |
| Candidate name | `grp_eq60_baseline` in Round 3, then `baseline_seed` in Round 4 replay |

What this means in plain language:

- The current model is not the older precision-search champion with Sharpe 1.0838 and MDD -9.996%.
- The current model is the more robust NVIDIA-inspired winner that improved drawdown margin and bootstrap robustness.
- The winning change that survived was the Round 3 upstream `equity_cap_60` group-constraint track.
- Round 4 tested MDD-triggered overlay re-optimization ideas, but none beat the Round 3 champion under the promotion rule.

---

## 2. Final Result Lineage

The final model was reached in two layers of search.

### 2.1 Pre-NVIDIA Triple Achievement Layer

This was the first layer that achieved point-estimate triple:

1. Base Phase 18 upstream snapshot:
   - `results_runpod/phase18_e2e_fresh_runpod_20260330_022828UTC/upstream_snapshot`
2. Structural tournament seed:
   - `results_runpod/phase18_nonleveraged_v2_structural_tournament_fresh_20260330_local_v1/best_near_triple.json`
3. Precision searches:
   - `scripts/run_triple_precision_search.py`
   - `scripts/run_triple_precision_round2.py`
   - `scripts/run_triple_precision_round3.py`
4. Verification:
   - `scripts/verify_triple_champion.py`
5. Result:
   - `results_runpod/phase18_triple_precision_round3/best_result.json`
   - `results_runpod/phase18_triple_verification/verification_report.json`

That older champion achieved:

- Sharpe 1.0838
- Annual return 10.0619%
- MDD -9.9964%
- Bootstrap P(MDD >= -10%) 36.82%

### 2.2 NVIDIA-Inspired Robustness Layer

This was the second layer that replaced the old champion with a more robust one.

Round sequence:

1. Round 0 baseline freeze:
   - `scripts/run_nvidia_round0_freeze.py`
2. Round 1 KDE scenarios:
   - `scripts/run_nvidia_round1_kde_full_runpod.sh`
   - Tested but did not become the lasting champion
3. Round 2 turnover-constrained upstream:
   - `scripts/run_nvidia_round2_turnover_runpod.sh`
   - Promoted `tov25 / upstream_expert` as the entering champion for Round 3
4. Round 3 group constraints:
   - `scripts/run_nvidia_round3_group_constraints_runpod.sh`
   - Promoted `grp_eq60_baseline`
5. Round 4 overlay-only MDD re-optimization:
   - `scripts/run_nvidia_round4_mdd_reopt_runpod.sh`
   - `scripts/run_nvidia_round4_mdd_reopt_search.py`
   - No further promotion

Promotion chain summary:

| Stage | Champion | Key artifact |
| --- | --- | --- |
| Pre-NVIDIA triple | `ref_q_hc10_s30.80_rf0.60_sc0.005_s1-0.01_sb+0.02` | `results_runpod/phase18_triple_precision_round3/best_result.json` |
| Round 2 winner | `upstream_expert` on `tov25` | preserved in Round 3 freeze and promotion records |
| Round 3 winner | `grp_eq60_baseline` | `results_runpod/nvidia_tournament_round3_group_constraints/promoted_best_result.json` |
| Round 4 winner | no change | `results_runpod/nvidia_tournament_round4_mdd_reopt/search/promotion_decision.json` |

---

## 3. Exact Files Run From Start To Finish

This section lists the important files in chronological order, grouped by function.

### 3.1 Data And Regime Preparation

Files:

- `src/data_loader.py`
- `src/gen_regime_4state_pit.py`
- `data/processed/regime_4state_pit.csv`
- `data/cache/`

What happened:

1. Asset and macro data were loaded and aligned in `src/data_loader.py`.
2. Point-in-time regime probabilities were rebuilt with:
   - `python -m src.gen_regime_4state_pit --end-date 2026-01-01`
3. The output PIT regime file was written to:
   - `data/processed/regime_4state_pit.csv`

Why this matters:

- The thesis-safe pipeline relies on PIT regime probabilities, not full-sample hindsight regime labels.
- This is the main mechanism that prevents regime look-ahead.

### 3.2 Upstream Walk-Forward Model Training And Inference

Primary files:

- `run_walkforward.py`
- `src/models.py`
- `src/optimization.py`
- `src/trainer.py`
- `src/loss.py`

Important run labels for the final champion track:

- `R6_13_e2e_pit_shifted_safe_grp_eq60`
- `R7_13_offline_ea_pit_shifted_safe_grp_eq60`

Current final champion was produced by Round 3 group-constraint runner:

- `scripts/run_nvidia_round3_group_constraints_runpod.sh`

That runner internally executed:

1. R6:
   - `python -u run_walkforward.py --model_type transformer --dist_type t --t_df 5.0 --group-constraint-preset equity_cap_60 --e2e_regime --assets_13 --scaler-type expanding --pit_hmm --no_nfci --start-date 2007-07-01 --end-date 2026-01-01 --oos-start 2016-07-31 --test-window-months 24 --batch_size 32 --label R6_13_e2e_pit_shifted_safe_grp_eq60`
2. R7:
   - `python -u run_walkforward.py --model_type transformer --dist_type t --t_df 5.0 --group-constraint-preset equity_cap_60 --ea_midas --assets_13 --scaler-type expanding --pit_hmm --no_nfci --start-date 2007-07-01 --end-date 2026-01-01 --oos-start 2016-07-31 --test-window-months 24 --batch_size 32 --label R7_13_offline_ea_pit_shifted_safe_grp_eq60`

Immediate outputs written by `run_walkforward.py`:

- `results/walkforward/R6_13_e2e_pit_shifted_safe_grp_eq60_port_returns.csv`
- `results/walkforward/R6_13_e2e_pit_shifted_safe_grp_eq60_port_weights.csv`
- `results/walkforward/R7_13_offline_ea_pit_shifted_safe_grp_eq60_port_returns.csv`
- `results/walkforward/R7_13_offline_ea_pit_shifted_safe_grp_eq60_port_weights.csv`
- per-label metrics and fold summaries in the same directory

Packaged upstream snapshot for the final winner:

- `results_runpod/nvidia_tournament_round3_group_constraints/upstream_tracks/grp_eq60/ExpA_R6_pit_shifted_returns.csv`
- `results_runpod/nvidia_tournament_round3_group_constraints/upstream_tracks/grp_eq60/ExpA_R6_pit_shifted_weights.csv`
- `results_runpod/nvidia_tournament_round3_group_constraints/upstream_tracks/grp_eq60/ExpA_R7_pit_shifted_returns.csv`
- `results_runpod/nvidia_tournament_round3_group_constraints/upstream_tracks/grp_eq60/ExpA_R7_pit_shifted_weights.csv`
- `results_runpod/nvidia_tournament_round3_group_constraints/upstream_tracks/grp_eq60/pit_shift_manifest.json`

Validation step:

- `scripts/validate_thesis_safe_coverage.py --run-end-date 2026-01-01 --upstream-dir results_runpod/nvidia_tournament_round3_group_constraints/upstream_tracks/grp_eq60`

### 3.3 Phase 18 Overlay Evaluation

Primary files:

- `scripts/run_phase18_nonleveraged_v2_benchmark.py`
- `scripts/run_phase18_nonleveraged_v2_structural_tournament.py`

What these files do:

- `run_phase18_nonleveraged_v2_benchmark.py` builds the evaluation context and runs the overlay engine.
- `run_phase18_nonleveraged_v2_structural_tournament.py` compares baseline and challenger tracks under a common overlay candidate set, computes point metrics, computes bootstrap statistics, and records tournament outputs.

Round 3 structural evaluation command:

- `python scripts/run_phase18_nonleveraged_v2_structural_tournament.py --baseline-upstream-dir results_runpod/phase18_e2e_fresh_runpod_20260330_022828UTC/upstream_snapshot --upstream-track grp_eq60=results_runpod/nvidia_tournament_round3_group_constraints/upstream_tracks/grp_eq60 --cache-seed-dir results_runpod/phase18_nonleveraged_v2_thesis_safe_2025_repro_onepass_v1 --seed-path results_runpod/phase18_nonleveraged_v2_structural_tournament_fresh_20260330_local_v1/best_near_triple.json --out-dir results_runpod/nvidia_tournament_round3_group_constraints/structural_tournament --bootstrap-samples 1000 --bootstrap-block-size 6`

Important outputs:

- `results_runpod/nvidia_tournament_round3_group_constraints/structural_tournament/`
- `results_runpod/nvidia_tournament_round3_group_constraints/promoted_best_result.json`
- `results_runpod/nvidia_tournament_round3_group_constraints/promotion_decision.json`

### 3.4 Round 4 Overlay-Only Search

Primary files:

- `scripts/run_nvidia_round4_mdd_reopt_runpod.sh`
- `scripts/run_nvidia_round4_mdd_reopt_search.py`

What happened:

1. The Round 3 promoted winner was frozen again.
2. The same upstream `grp_eq60` track was reused.
3. Only overlay-side hierarchical and policy stress settings were searched.
4. Promotion rule was applied.
5. No challenger replaced the incumbent.

Important outputs:

- `results_runpod/nvidia_tournament_round4_mdd_reopt/search/leaderboard.csv`
- `results_runpod/nvidia_tournament_round4_mdd_reopt/search/best_result.json`
- `results_runpod/nvidia_tournament_round4_mdd_reopt/search/promotion_decision.json`

---

## 4. Current Upstream Model Architecture

The upstream model is a decision-aware Black-Litterman plus differentiable-CVaR network.

### 4.1 Data Domain

Universe:

- `SPY`
- `QQQ`
- `XLV`
- `XLP`
- `XLE`
- `TLT`
- `IEF`
- `GLD`
- `VNQ`
- `TIP`
- `DBC`
- `ACWX`
- `BIL`

The 13-asset universe is defined in:

- `src/data_loader.py`

Sequence setup:

- 12-month lookback windows
- OOS starts at `2016-07-31`
- test window size is `24` months per fold
- expanding walk-forward folds are defined in `run_walkforward.py`

### 4.2 Encoder

The current final upstream track uses `TransformerModel`, not GRU.

Definition:

- `src/models.py`

Structure:

1. Linear input projection:
   - `input_dim -> hidden_dim`
2. Transformer encoder stack:
   - `num_layers=2`
   - `nhead=4`
   - feedforward size `hidden_dim * 4`
3. Last time-step representation:
   - `out[:, -1, :]`

Project defaults inherited from `run_walkforward.py` unless overridden:

- `hidden_dim = 32`
- `seq_length = 12`
- `dist_type = t`
- `t_df = 5.0`
- `omega_mode = learnable`
- `sigma_mode = prior`
- `lambda_risk = 2.0`

### 4.3 Black-Litterman Heads

Implemented in `BaseBLModel` in `src/models.py`.

The encoder state feeds:

- `P` head:
  - diagonal asset selection weights
- `Q` head:
  - expected-return views from hidden state plus regime and drawdown state
- `Omega` head:
  - learnable uncertainty of the views

Then Black-Litterman posterior moments are constructed:

- posterior mean `mu_BL`
- posterior covariance `Sigma_BL` or prior `Sigma` depending on sigma mode

### 4.4 Differentiable CVaR Layer

Implemented in:

- `src/optimization.py`

What it does:

- samples return scenarios
- solves a convex long-only CVaR portfolio optimization
- returns differentiable weights through `cvxpylayers`

In the current final champion track:

- scenario family: Student-t
- `df = 5.0`
- confidence level: 95%
- long-only budget-constrained portfolio
- group constraint preset: `equity_cap_60`

Important nuance:

- the `equity_cap_60` constraint is enforced on the upstream solver weights
- it is not a hard cap on the final executed overlay weights

### 4.5 Loss And Training

Main loss file:

- `src/loss.py`

Main trainer file:

- `src/trainer.py`

The project uses a decision-aware loss combining:

- negative return term
- risk penalty
- turnover penalty
- optional drawdown penalty

In code, the loss family is broader than the exact final winner story, but the training pipeline still follows the same idea:

- the network is not trained to minimize forecast error alone
- it is trained to improve downstream portfolio decisions

---

## 5. Current Overlay Architecture

The overlay is where the upstream sleeve outputs are turned into the final thesis portfolio.

Primary file:

- `scripts/run_phase18_nonleveraged_v2_benchmark.py`

Main function:

- `simulate_overlay_v2_strategy(...)`

### 5.1 Input Sleeves

The overlay consumes two upstream sleeves:

- R6 sleeve
- R7 sleeve

These are PIT-shifted walk-forward outputs saved as:

- `ExpA_R6_pit_shifted_returns.csv`
- `ExpA_R6_pit_shifted_weights.csv`
- `ExpA_R7_pit_shifted_returns.csv`
- `ExpA_R7_pit_shifted_weights.csv`

### 5.2 Overlay Pipeline

The execution order is:

1. Build context
   - load sleeve returns and weights
   - load PIT regime
   - align dates and cache inputs
2. Sleeve mixing
   - `mix.base_mix`, `mix.high_mix`, `mix.high_threshold`
3. Expert scaling
   - stress, inflation, recovery, trend, and volatility-aware budget adjustments
4. S3 conviction filter
   - scale down low-conviction exposures
5. SRB regime budget
   - regime-specific risk multiplier
6. S1 volatility targeting
   - target monthly risk profile
7. Sleeve allocation and tilt
   - growth, defensive, cash behavior
8. Policy controller
   - caution and defense transition rules
9. Daily stateful stop-loss logic
   - soft stop
   - hard stop
   - reentry
10. Trading cost application
11. Final monthly return aggregation

### 5.3 Final Champion Overlay Parameters

The current final model inherited the baseline overlay from the Round 2 incumbent and changed the upstream track.

That means the final winner is mostly:

- same overlay family as the incumbent
- different upstream sleeve regularization via `grp_eq60`

High-level parameter blocks inside `promoted_best_result.json`:

- `expert`
- `mix`
- `policy`
- `s1`
- `s3`
- `sleeve`
- `srb`
- `stop`
- `tilt`

Important operational values in the final winner:

- `trade_cost_bps = 10`
- `weights_semantics = realized_average_executed_weights`
- `pit_warmup_policy = pre_start_neutral_no_backfill`

---

## 6. Winner-Survival Tournament Logic

The final champion is not just the top row of one leaderboard.

### 6.1 Round 3 Promotion Rule

Stored in:

- `results_runpod/nvidia_tournament_round3_group_constraints/promotion_decision.json`

Applied rule:

- if both incumbent and challenger satisfy point triple, higher bootstrap `P(MDD >= -10%)` wins

Why Round 3 promoted `grp_eq60_baseline`:

- incumbent on `tov25` had higher Sharpe and return
- challenger on `grp_eq60` had materially better MDD robustness
- bootstrap success probability improved from `47.18%` to `57.82%`

### 6.2 Round 4 Promotion Rule

Stored in:

- `results_runpod/nvidia_tournament_round4_mdd_reopt/search/promotion_decision.json`

Applied rule order:

1. point triple first
2. bootstrap probability second
3. if bootstrap within 2 percentage points, better point MDD
4. then annual return
5. then Sharpe

Result:

- no Round 4 candidate replaced the incumbent

---

## 7. Reproduction Map For The Current Final Model

If the goal is to reproduce the current final model rather than re-run every historical experiment, the minimum canonical sequence is:

1. Rebuild PIT regime:
   - `python -m src.gen_regime_4state_pit --end-date 2026-01-01`
2. Run Round 3 group-constraint pipeline:
   - `bash scripts/run_nvidia_round3_group_constraints_runpod.sh`
3. Confirm promoted champion:
   - `results_runpod/nvidia_tournament_round3_group_constraints/promoted_best_result.json`
   - `results_runpod/nvidia_tournament_round3_group_constraints/promotion_decision.json`
4. Run Round 4 overlay-only search:
   - `bash scripts/run_nvidia_round4_mdd_reopt_runpod.sh`
5. Confirm no further promotion:
   - `results_runpod/nvidia_tournament_round4_mdd_reopt/search/promotion_decision.json`

If the goal is to reproduce the full historical path to the current model, then the sequence is:

1. Build original Phase 18 upstream and seed artifacts
2. Run triple precision searches
3. Verify the older triple champion
4. Freeze Round 0 baseline
5. Run NVIDIA rounds 1 to 4
6. Keep the Round 3 promoted winner because Round 4 does not replace it

---

## 8. Thesis-Safe Interpretation

The following wording is accurate and safe.

### 8.1 Safe Description Of The Final Model

"The final thesis portfolio is a decision-aware Black-Litterman plus differentiable-CVaR upstream model, evaluated through a Phase 18 risk-management overlay. The final promoted model uses a PIT-conditioned transformer upstream track with a solver-level broad-equity cap of 60 percent, and it is selected through a winner-survival rule that prioritizes triple feasibility and bootstrap drawdown robustness."

### 8.2 Phrases To Avoid

Do not write the following:

- "The final portfolio has a hard 60% equity cap."
- "The final transformer uses an active end-to-end regime head."
- "The final model is the official structural-tournament winner satisfying all COVID guard filters."
- "The subperiod COVID MDD is a true pre-peak-to-trough shock drawdown."

### 8.3 What Must Be Qualified

These caveats should be stated in the thesis:

1. `grp_eq60` is an upstream solver regularizer, not a final executed-weight cap.
2. `--e2e_regime` is passed in the Round 3 R6 command, but transformer does not activate the GRU-only end-to-end regime head.
3. `covid_shock_mdd` in tournament outputs is a local window-reset MDD, not a true pre-shock peak-to-trough guard.
4. The final model is a promoted robust winner, not simply the official point-metric tournament winner.
5. Bootstrap improves robustness assessment, but it does not remove post-selection bias because selection still occurs on the same historical OOS span.

---

## 9. Key Directories And What They Mean

### Core code

- `run_walkforward.py`
- `src/models.py`
- `src/optimization.py`
- `src/trainer.py`
- `src/loss.py`
- `src/data_loader.py`
- `src/gen_regime_4state_pit.py`
- `scripts/run_phase18_nonleveraged_v2_benchmark.py`
- `scripts/run_phase18_nonleveraged_v2_structural_tournament.py`

### Historical triple milestone

- `results_runpod/phase18_triple_precision_round3/`
- `results_runpod/phase18_triple_verification/`

### Current canonical winner

- `results_runpod/nvidia_tournament_round3_group_constraints/`
- `results_runpod/nvidia_tournament_round4_mdd_reopt/search/`

### Supporting docs

- `docs/COMPLETE_MODEL_DOCUMENTATION.md`
- `docs/NVIDIA_INSPIRED_TOURNAMENT.md`
- `docs/ROBUST_TRIPLE_STRATEGY.md`

---

## 10. Final One-Paragraph Thesis Summary

This project starts with PIT-safe data preparation and regime estimation, trains a decision-aware Black-Litterman plus differentiable-CVaR upstream allocator with walk-forward validation, passes the resulting R6 and R7 sleeves through a multi-layer Phase 18 overlay, and then selects the final thesis model through a winner-survival robustness tournament. The current canonical thesis model is the Round 3 promoted `grp_eq60` winner, preserved in `results_runpod/nvidia_tournament_round3_group_constraints/promoted_best_result.json` and confirmed to survive Round 4 unchanged. It improves drawdown robustness relative to the pre-NVIDIA triple champion, while requiring explicit thesis caveats about guard semantics, upstream-only group constraints, transformer regime-head wiring, and post-selection interpretation.
