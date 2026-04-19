# Remediation Plan — Decision-Aware Tail Risk Optimization

**Status as of 2026-04-19.** Based on a full audit of the repository at commit `ffe6b59`. This document defines a concrete, sequenced action plan to bring the thesis pipeline to a defensible, reproducible, publication-ready state.

The plan is organized by **priority tier** (P0 → P2) and each item specifies: *what to do*, *why*, *files to touch*, *acceptance criteria*, and *rough effort*.

---

## Summary of Issues to Fix

| # | Issue | Severity | Tier |
|---|-------|----------|------|
| 1 | `regime_4state_pit.csv` and `gen_regime_4state_pit.py` do not exist — pipeline silently falls back to full-sample HMM | **Critical** | P0 |
| 2 | `TransformerModel.__init__` drops `regime_dim / macro_dim / dist_type / t_df / e2e_regime / max_bil_floor` | High | P0 |
| 3 | `DecisionAwareLoss` with `risk_type='cvar'` computes batch-CVaR, not scenario CVaR — misleading naming | High | P0 |
| 4 | Docs reference non-existent scripts and result directories (`nvidia_tournament_round*`, `run_phase18_*`) | High | P0 |
| 5 | `PathAwareMDDLoss` uses `torch.no_grad()` — no gradient path to MDD; adaptive loop is heuristic only | Medium | P1 |
| 6 | Post-selection bias on tournament winner — no Deflated Sharpe / SPA / Reality Check | **Critical for thesis defense** | P1 |
| 7 | Block-bootstrap P(MDD ≥ -10%) misinterpreted as unconditional probability | Medium | P1 |
| 8 | Overlay hyperparameters tuned on the same OOS folds used for reporting — leakage | Medium | P1 |
| 9 | Group constraints enforced upstream but overridden by overlay → non-homomorphic constraint | Low | P2 |
| 10 | Universe (13 assets, monthly) too small for statistically stable Sharpe/MDD | Low | P2 |
| 11 | Student-T `df=5` hardcoded; not regime-adaptive | Low | P2 |
| 12 | `cvxpylayers` GPU↔CPU round-trip slows training | Low | P2 |

---

# Tier P0 — Must Fix Before Any Further Experiments

These items block the core thesis claims. Do not run additional experiments or refresh results until P0 is complete.

## P0.1 — Implement true PIT-HMM (leakage-free regime labels)

**Why.** The thesis calls the 4-state HMM "PIT-safe", but the HMM is fit on the full 2007–2025 sample. Baum–Welch smoothing uses the entire observation sequence, so every regime label for past dates is contaminated by future observations. This directly biases every downstream metric.

**What to do.**

1. Create `src/gen_regime_4state_pit.py`:
   - For each walk-forward fold cutoff `t`, fit a fresh HMM on `[2007-01, t]` only.
   - Emit `data/processed/regime_4state_pit.csv` with columns `[date, fold_id, regime, p_bull, p_sideways, p_correction, p_crisis]` — **one row per (date, fold)**, so downstream code always reads the regime that was knowable at that point given that fold's training window.
   - Apply label alignment across folds via Hungarian assignment on posterior means (to prevent regime-index permutation across refits).
2. Modify `src/data_loader.py::get_regime_4state()`:
   - Remove the silent fallback. If `--pit_hmm` is set and `regime_4state_pit.csv` is missing, **raise** `FileNotFoundError` with instructions.
   - Add a `fold_id` argument; slice the PIT CSV accordingly.
3. Modify `run_walkforward.py`:
   - `--pit_hmm` must pass `fold_id` into the data loader.
   - Add a CI check: if `--pit_hmm` is absent, log a prominent warning and tag results with `leakage=full_sample_hmm`.
4. Produce a **head-to-head table** (full-sample HMM vs PIT-HMM) in `docs/THESIS_CURRENT_MODEL_PIPELINE.md`:
   - Sharpe, annual return, MDD, bootstrap P, with 95% CIs for both regimes.
   - The delta quantifies leakage magnitude — this is a valuable thesis contribution on its own.

**Files.** `src/gen_regime_4state_pit.py` (new), `src/data_loader.py`, `run_walkforward.py`, `data/processed/regime_4state_pit.csv` (new), `docs/THESIS_CURRENT_MODEL_PIPELINE.md`.

**Acceptance.** `python run_walkforward.py --pit_hmm ...` refuses to run without the PIT CSV. PIT vs non-PIT results reported side-by-side.

**Effort.** 2–3 days (HMM refit loop + label alignment + rerun of headline experiments).

---

## P0.2 — Fix `TransformerModel` argument plumbing

**Why.** `TransformerModel.__init__` in `src/models.py:400-411` does not accept or forward the regime/macro/distribution/e2e flags that every other encoder uses. Any benchmark comparison involving the Transformer is invalid because it runs with regime-conditioning, Student-T scenarios, and crisis overlay silently disabled.

**What to do.**

1. Edit `src/models.py`:
   ```python
   class TransformerModel(BaseBLModel):
       def __init__(self, input_dim, num_assets, hidden_dim=64, num_layers=2, nhead=4, dropout=0.2,
                    omega_mode='learnable', sigma_mode='prior', lambda_risk=0.0,
                    regime_dim=0, macro_dim=0, dist_type='gaussian', t_df=5,
                    e2e_regime=False, max_bil_floor=0.0):
           super().__init__(input_dim, num_assets, hidden_dim, dropout,
                            omega_mode, sigma_mode, lambda_risk,
                            regime_dim=regime_dim, macro_dim=macro_dim,
                            dist_type=dist_type, t_df=t_df,
                            e2e_regime=e2e_regime, max_bil_floor=max_bil_floor)
           ...
   ```
   Mirror whatever signature `LSTMModel` / `GRUModel` use — copy it verbatim.
2. Add a unit test `tests/test_model_args.py` that asserts all five encoder classes accept and store the same set of regime-related attributes.
3. Rerun any Transformer results previously reported; tag prior Transformer numbers as "regime-off ablation" in docs.

**Files.** `src/models.py`, `tests/test_model_args.py` (new).

**Acceptance.** `pytest tests/test_model_args.py` passes; Transformer training actually uses `e2e_regime=True` when the flag is set (verify by logging `model.regime_head is not None`).

**Effort.** 2–4 hours.

---

## P0.3 — Rename or replace batch-CVaR in `DecisionAwareLoss`

**Why.** `risk_type='cvar'` in `src/loss.py` computes the mean of the worst α-fraction of portfolio returns **within the current mini-batch**. This is not the Rockafellar–Uryasev CVaR that `optimization.py` uses. The two are close only for large, time-homogeneous batches. Reporting "CVaR loss" invites reviewer pushback.

**What to do (choose one).**

- **Option A (preferred):** Replace the batch-CVaR with scenario CVaR. In the loss, sample Student-T scenarios from the predicted `(μ, Σ)` (same generator as the optimization layer) and compute CVaR over the scenarios. This keeps the loss definition consistent with the optimizer.
- **Option B (minimal):** Rename `risk_type='cvar'` to `risk_type='batch_tail_mean'` and document in the loss docstring that this is a proxy, not a distributional CVaR. Keep existing semantics; remove the ambiguity.

**Files.** `src/loss.py`, `src/trainer.py` (if it references the old name).

**Acceptance.** Docstring and thesis section unambiguously state which CVaR definition is used in the loss vs the optimizer.

**Effort.** Option A: 1 day. Option B: 1 hour.

---

## P0.4 — Repair docs ↔ code drift

**Why.** The docs (`README.md`, `docs/THESIS_CURRENT_MODEL_PIPELINE.md`) treat `nvidia_tournament_round3/4`, `phase18_triple`, `run_phase18_paper_safe_ablation.py`, and `run_phase18_nonleveraged_v2_benchmark.py` as canonical artifacts, but none exist in this worktree or on `origin/main`. The headline numbers (Sharpe 1.0724, MDD -9.44%, bootstrap 57.98%) cannot be reproduced from the pushed code.

**What to do.**

1. Inventory: list every file/directory referenced in `README.md` and `docs/**/*.md`. Cross-check existence.
2. For each missing artifact, **either**:
   - Recover it from a local-only directory and push it, **or**
   - Remove the reference from the docs and replace it with a reproducible alternative.
3. Add `scripts/reproduce_canonical_result.sh` that re-runs the exact pipeline producing the headline numbers from scratch. Include a fixed random seed set (3 seeds as per current protocol).
4. Add a CI job (GitHub Actions) that runs a **smoke version** of the pipeline on a 2-fold subset to catch future drift.

**Files.** `README.md`, `docs/**/*.md`, `scripts/reproduce_canonical_result.sh` (new), `.github/workflows/smoke.yml` (new, optional).

**Acceptance.** A fresh clone → `bash scripts/reproduce_canonical_result.sh` reproduces the reported metrics within Monte-Carlo tolerance.

**Effort.** 1–2 days.

---

# Tier P1 — Must Fix Before Thesis Submission

These items do not block further experimentation but are required for a defensible thesis.

## P1.1 — Post-selection inference (Deflated Sharpe Ratio, SPA test)

**Why.** The tournament evaluated hundreds of candidates on the same OOS folds and promoted the best one. Without a multiple-testing correction, the winner's Sharpe is inflated. This is the single largest statistical vulnerability.

**What to do.**

1. Add `src/stats/deflated_sharpe.py` implementing Bailey & López de Prado (2014) DSR:
   ```
   DSR = Φ( (SR - E[max SR_N]) / σ(SR) )
   ```
   where `N` is the number of trials in the tournament (count from `scripts/run_triple_precision_search.py` logs — typically 80–120 per axis × multiple axes).
2. Add White's Reality Check and Hansen's SPA test in `src/stats/spa_test.py` using the Politis–Romano stationary bootstrap.
3. Report **DSR, SPA p-value, Reality Check p-value** alongside every headline metric in the thesis.
4. If DSR < 0.95 confidence, the "Sharpe > 1.0" claim must be softened to "Sharpe > 1.0 on the selected path; post-selection-adjusted confidence XX%".

**Files.** `src/stats/deflated_sharpe.py` (new), `src/stats/spa_test.py` (new), `scripts/report_selection_adjusted.py` (new).

**Acceptance.** Every table reporting a Sharpe includes DSR and SPA-adjusted p-value.

**Effort.** 2–3 days.

---

## P1.2 — Reinterpret the block-bootstrap probability

**Why.** `P(MDD ≥ -10%) = 57.98%` is currently presented as "the probability the strategy stays above the -10% floor". It is actually the empirical frequency under block resampling of a **single, already-selected** path. That conflates path variability with strategy uncertainty.

**What to do.**

1. Rename the metric in docs to `P_bootstrap(MDD ≥ -10% | selected path)`.
2. Add a second bootstrap on a **family of candidate strategies** (not only the winner): resample both block-of-returns *and* the selected strategy index (via Bayesian-bootstrap weights on the tournament). Report this as `P_bootstrap(MDD ≥ -10% | tournament distribution)` — this is the statistically honest number.
3. Report both; discuss their difference in the thesis Appendix.

**Files.** `scripts/run_triple_precision_search.py`, `docs/THESIS_CURRENT_MODEL_PIPELINE.md`.

**Effort.** 1–2 days.

---

## P1.3 — Nested cross-validation for overlay hyperparameters

**Why.** Phase 18 overlay parameters (stop/s1/s3/SRB/policy) are tuned using the same OOS folds that the thesis reports on. That is a classic leakage pattern.

**What to do.**

1. Restructure walk-forward:
   - **Outer fold**: reports the final OOS metric.
   - **Inner fold**: tunes overlay hyperparameters only on training-window data; test window is untouched by the tuner.
2. Implement in `run_walkforward.py` via a `--nested` flag that wraps the existing expanding window.
3. Rerun the tournament with nested CV; compare headline metrics pre/post nesting.

**Files.** `run_walkforward.py`, `scripts/run_triple_precision_search.py`.

**Effort.** 2–4 days (includes rerun time).

---

## P1.4 — Path-aware MDD with actual gradient flow

**Why.** `PathAwareMDDLoss` uses `torch.no_grad()` internally and the adaptive-λ loop only adjusts coefficients between epochs. There is no gradient signal from MDD to the model weights — the current setup is a heuristic scheduler, not end-to-end MDD optimization.

**What to do (choose one).**

- **Option A — soft drawdown.** Replace `max` with a smooth surrogate:
  ```
  soft_peak_t = (1/β) logsumexp(β · cum_ret_{≤t})
  soft_dd_t   = cum_ret_t - soft_peak_t
  soft_mdd    = -(1/β) logsumexp(-β · soft_dd_t)
  ```
  Then MDD is differentiable. Tune β ∈ [10, 100].
- **Option B — sequential rollout.** Turn the trainer into a recurrent simulator: at each step, feed the prev-step weights into next-step features (transaction cost, regime), keeping gradients attached. MDD over the rollout is then differentiable directly.

**Files.** `src/loss.py`, `src/trainer.py`.

**Acceptance.** Ablation: MDD improves when the differentiable term is on vs. off, at matched Sharpe.

**Effort.** Option A: 1 day. Option B: 3–5 days.

---

# Tier P2 — Nice to Have (post-submission or revision round)

## P2.1 — Enforce group constraints inside the optimizer, not via overlay

Move the equity/bond/alternatives caps from the overlay into `cvxpy` constraints in `src/optimization.py`. This makes the upstream policy constraint-consistent with the executed portfolio; removes the need for post-hoc sleeve/policy re-weighting.

**Effort.** 2–3 days.

## P2.2 — Regime-adaptive Student-T degrees of freedom

Make `t_df` a learnable per-regime parameter (e.g., df=2.5 for Crisis, df=10 for Bull). Train via gradient through the scenario generator (reparameterization available for Student-T).

**Effort.** 1–2 days.

## P2.3 — Expand universe and frequency

Extend from 13 assets monthly to ≥30 assets weekly. This roughly 4× the effective sample size and narrows Sharpe CIs. Use the extra assets to add out-of-universe stress ablations (EM, commodities, crypto).

**Effort.** 1–2 weeks including data plumbing and re-training.

## P2.4 — GPU-native CVaR layer

`cvxpylayers` drops to CPU for the LP solve. For 3 seeds × many folds × long training, this dominates wall-clock. Replace with a GPU-friendly proximal / ADMM solver, or a differentiable projection onto the CVaR-constrained simplex.

**Effort.** 1 week, research-grade.

## P2.5 — Ablation ladder

Add a progressive ablation table to the thesis:
1. Equal-weight.
2. Risk-parity.
3. Min-CVaR (no regime).
4. BL-only.
5. BL + CVaR (no regime).
6. BL + CVaR + regime (no overlay).
7. Full model (with overlay).

Each step isolates the marginal contribution of one component. This is strong evidence for any reviewer.

**Effort.** 2–3 days (uses existing modules, turns flags on/off).

---

# Proposed Schedule

| Week | Focus |
|------|-------|
| **Week 1** | P0.2 (Transformer fix), P0.3 (CVaR naming), P0.4 (docs-code alignment). Low-risk, high-trust-recovery work. |
| **Week 2** | P0.1 (PIT-HMM) — implement + rerun headline experiments. |
| **Week 3** | P1.1 (Deflated Sharpe / SPA) + P1.2 (bootstrap reinterpretation). |
| **Week 4** | P1.3 (nested CV) rerun. |
| **Week 5** | P1.4 (path-MDD gradient). Ablation ladder P2.5 in parallel. |
| **Week 6** | Thesis write-up: incorporate all new tables, caveats, and comparisons. |
| **Buffer** | P2.x items only if time permits. |

---

# Exit Criteria (Thesis-Ready)

The pipeline is ready for thesis defense when **all** of the following hold:

- [ ] Running `python run_walkforward.py --pit_hmm ...` without the PIT file fails loudly.
- [ ] Transformer results pass the regime-attribute unit test.
- [ ] Every reported Sharpe has a companion Deflated Sharpe and SPA p-value.
- [ ] A full pre-registered reproduction script runs end-to-end in CI.
- [ ] The thesis explicitly reports both full-sample HMM and PIT-HMM headline numbers.
- [ ] Overlay hyperparameters were selected via nested CV (outer folds untouched).
- [ ] Path-MDD either has a documented differentiable form **or** is explicitly demoted from "optimized" to "monitored/scheduled".
- [ ] Docs reference only files that exist in the repository at the release commit.

---

# Out of Scope for This Plan

- Moving to a different RL / control framework (diffusion-based planning, MPC).
- Multi-period CVaR (the current single-period formulation is sufficient for the thesis).
- Alternative data (news sentiment, options-implied distributions). These are follow-up-paper material.

---

*Prepared by Claude (Opus 4.7) for YoungJun0814, 2026-04-19. Commit reference: `ffe6b59`.*
