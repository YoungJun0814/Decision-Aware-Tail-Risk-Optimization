# Robust Triple Target Strategy

**Date:** 2026-03-30
**Goal:** Achieve the triple target (Sharpe >= 1.0, Return >= 10%, MDD >= -10%) with statistical robustness, not just point-estimate satisfaction.

---

## Current State

| Metric | Current Value | Target | Margin |
|--------|--------------|--------|--------|
| Sharpe | 1.0838 | >= 1.0 | +0.084 (comfortable) |
| Return | 10.06% | >= 10% | +0.06%p (tight) |
| MDD | -9.996% | >= -10% | +0.004%p (extremely tight) |
| Bootstrap P(MDD >= -10%) | 36.8% | >= 50% | -13.2%p (insufficient) |
| Bootstrap MDD 95% CI | [-16.4%, -5.2%] | - | Wide range |

**Core problem:** The triple target is achieved on the point estimate but MDD has near-zero margin. In a bootstrap resampling test, only 36.8% of paths satisfy MDD >= -10%. This means the result is not statistically robust.

---

## Strategy 1: Upstream Model Improvement (Highest Impact)

The overlay system can only work with what the upstream GRU + CVaR models produce. Improving the base model returns gives the overlay more room to tighten risk without sacrificing return/Sharpe.

### 1A. Ensemble Upstream Models

**Idea:** Instead of blending just R6 (E2E) and R7 (offline) at 50/50, train multiple seeds and ensemble.

```
Current:  0.5 * R6_seed0 + 0.5 * R7_seed0
Proposed: 0.5 * mean(R6_seed0..2) + 0.5 * mean(R7_seed0..2)
```

**Why it helps:** Ensembling reduces variance of upstream weights, which directly reduces MDD. Each seed encounters different local optima; averaging smooths out the worst single-seed drawdowns.

**Expected impact:**
- MDD improvement: 0.5-2.0%p (from variance reduction alone)
- Sharpe: roughly preserved or slightly improved
- Return: roughly preserved

**Implementation:**
1. Re-run `run_walkforward.py` with 3 different seeds for both R6 and R7
2. Average the weight outputs per fold before feeding into Phase 18 overlay
3. No overlay re-tuning needed if upstream is strictly better

### 1B. GRU Architecture Variants

**Idea:** Try LSTM, Transformer, or TCN as the encoder and ensemble the best 2-3.

The Phase 12 encoder benchmark already tested 11 architectures. GRU won for single-model performance, but the top 3 (GRU, TCN, LSTM) have low correlation in their errors, making them ideal ensemble candidates.

**Expected impact:**
- Cross-architecture ensemble could reduce MDD by 1-3%p
- Sharpe improvement from decorrelated alpha signals

### 1C. Longer Training Window

**Idea:** Extend training data to include pre-2007 if available (SPY/TLT exist back to 2002-2003).

More training data generally improves generalization. The current 2007-07 start was chosen to align all 13 assets, but a 5-asset core (SPY, TLT, IEF, GLD, VNQ) could start from 2005.

**Caveat:** Requires handling missing assets in early periods (zero-fill or reduced universe).

---

## Strategy 2: Overlay System Enhancements (Medium Impact)

### 2A. Bootstrap-Aware Overlay Optimization

**Idea:** Instead of optimizing overlay parameters for point-estimate MDD, optimize for bootstrap-robust MDD.

```python
def robust_objective(cfg):
    returns = simulate(cfg)
    point_mdd = compute_mdd(returns)
    boot_mdds = block_bootstrap_mdd(returns, n=1000, block=6)
    # Optimize for 75th percentile MDD (worst 25% of bootstraps)
    robust_mdd = np.percentile(boot_mdds, 25)  # 25th pctile = worse MDDs
    return score(sharpe, return, robust_mdd)
```

**Why it helps:** Current optimization targets point MDD = -9.996%, which is a single draw. Bootstrap-aware optimization would target the distribution tail, ensuring the overlay parameters are robust to path reordering.

**Expected impact:**
- Point MDD may worsen slightly (e.g., -9.5% instead of -9.996%)
- But P(MDD >= -10%) jumps from 36.8% to 60-80%
- Trades 0.5%p of point margin for much higher confidence

**Implementation cost:** Each candidate evaluation goes from O(1) simulation to O(1000) bootstraps. For 200 candidates, that's 200K evaluations. Feasible on RunPod in ~2-4 hours.

### 2B. Drawdown Control Layer (S7) Re-calibration

**Idea:** The current champion doesn't use S7 (drawdown scaling). Re-introduce it with conservative parameters designed to clip only the worst 2-3 drawdown months.

S7 scales exposure based on trailing drawdown depth:
```
dd0=1.0 (no DD: full exposure)
dd3=0.98 (3% DD: slight reduction)
dd6=0.90 (6% DD: meaningful reduction)
dd9=0.75 (9% DD: aggressive protection)
```

This acts as a nonlinear brake that activates only near the -10% MDD boundary.

**Expected impact:**
- MDD improvement: 0.5-1.5%p
- Return cost: 0.2-0.5%p (from reduced exposure during recoveries)
- Net: wider MDD margin with manageable return cost

### 2C. Regime-Conditional Stop-Loss Tightening

**Idea:** The current stop-loss thresholds are static (`soft=-2%, hard=-3.5%`). Make them regime-dependent:

```
Bull:       soft=-2.5%, hard=-4.0% (loose - let profits run)
Sideways:   soft=-2.0%, hard=-3.5% (current defaults)
Correction: soft=-1.5%, hard=-2.5% (tighter)
Crisis:     soft=-1.0%, hard=-2.0% (very tight)
```

**Why it helps:** Most MDD occurs during crisis/correction regimes. Tighter stops in these regimes catch drawdowns earlier without affecting bull-market returns.

### 2D. Adaptive Rebalance Frequency

**Idea:** Current system rebalances monthly. During high-vol regimes, switch to more frequent intra-month risk checks.

Not full daily rebalancing (that would require daily data in the overlay), but a simulated "mid-month check" that can de-risk if a month is on track for large losses:

```
If month-to-date return < -3% at mid-month:
    Reduce equity exposure by 30%
    Re-enter at month-end rebalance
```

**Caveat:** Requires daily return data, which is available in the cache but not currently used in the overlay pipeline.

---

## Strategy 3: Statistical Robustness Testing (No Model Change)

### 3A. Walk-Forward Fold Jackknife

**Idea:** Run 5 leave-one-fold-out evaluations. If the triple holds in 4/5 jackknife samples, the result is more defensible.

```
Jackknife 1: Folds 2,3,4,5 (drop fold 1)
Jackknife 2: Folds 1,3,4,5 (drop fold 2)
...
Jackknife 5: Folds 1,2,3,4 (drop fold 5)
```

**Why it matters:** If triple holds only with all 5 folds, it might be driven by one lucky fold. If it holds in 4/5 jackknife samples, the result is robust to individual fold outcomes.

### 3B. Transaction Cost Sensitivity

**Idea:** Verify triple at 5, 10, 15, 20 bps transaction costs. Currently verified at 0 bps.

The champion config has moderate turnover (monthly rebalance with regime-dependent adjustments). Expected that triple survives 10 bps, may fail at 15-20 bps.

### 3C. Perturbed Parameter Stability

**Idea:** For each overlay parameter, perturb by +/-5% and check if triple still holds.

The Round 3 leaderboard already shows 18 triple winners with varying parameters, suggesting some stability. But a formal sensitivity analysis would map the "triple-feasible region" in parameter space.

```
Volume of triple-feasible region / Volume of search space = "robustness score"
```

---

## Strategy 4: Alternative Approaches (Research-Grade)

### 4A. RMT-Based Covariance Cleaning

**Idea (from quant-traderr-lab):** Use Random Matrix Theory to clean the sample covariance matrix before feeding it to the CVaR optimizer.

The Marchenko-Pastur distribution identifies eigenvalues that are pure noise vs. signal. Clipping noise eigenvalues reduces estimation error in the covariance matrix, which directly improves portfolio weight stability.

```python
def clean_covariance_rmt(cov_matrix, T, N):
    """Marchenko-Pastur covariance cleaning."""
    q = N / T  # ratio of assets to observations
    lambda_plus = (1 + np.sqrt(q))**2  # upper MP bound

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Clip eigenvalues below MP upper bound to their average
    noise_mask = eigenvalues < lambda_plus
    eigenvalues[noise_mask] = eigenvalues[noise_mask].mean()

    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
```

**Expected impact:** Reduces weight instability -> lower turnover -> lower MDD from whipsaw trades.

### 4B. MST-Based Diversification Constraint

**Idea:** Use Minimum Spanning Tree of the correlation matrix to enforce diversification constraints. If the portfolio over-concentrates in one cluster of the MST, the overlay penalizes or re-distributes.

This goes beyond simple correlation guards by understanding the hierarchical structure of asset co-movements.

### 4C. Wasserstein Distributional Robustness

**Idea:** Replace point-estimate CVaR with distributionally robust CVaR (DRO-CVaR). Instead of optimizing over a single scenario distribution, optimize over the worst-case distribution within a Wasserstein ball around the estimated distribution.

```
min_w max_{P: W(P, P_hat) <= epsilon} CVaR_alpha(w'r)
```

This naturally produces more conservative portfolios that are robust to distribution shift.

**Implementation complexity:** HIGH. Requires reformulating the cvxpylayers problem with a DRO constraint, which may or may not be tractable with the current differentiable solver.

### 4D. Hierarchical Risk Parity (HRP) Fallback

**Idea:** When the CVaR optimizer fails or produces extreme weights, fall back to HRP allocation instead of the current diagonal-covariance fallback.

HRP uses hierarchical clustering on the correlation matrix to build a diversified portfolio without matrix inversion. It's more robust to estimation error than mean-variance or CVaR approaches.

**Expected impact:** Reduces worst-case weights in fallback scenarios, which are exactly the scenarios contributing to MDD.

---

## Strategy 5: Data Enhancements

### 5A. Extended Asset Universe

**Idea:** Add 2-3 defensive assets that provide unique diversification:

| Candidate | Ticker | Role |
|-----------|--------|------|
| Managed Futures | DBMF | Crisis alpha, low correlation to equities |
| Long Vol | VIXM | Direct tail hedge |
| Utilities | XLU | Defensive equity with yield |
| Int'l Bonds | BNDX | Currency diversification |

**Why it helps:** More defensive options give the optimizer more room to hedge without going to cash.

**Caveat:** Adding assets requires re-training the GRU on the expanded universe, and the 13-asset setup is already well-tested.

### 5B. Alternative Risk-Free Rate

**Idea:** Current cash position earns BIL returns (~5% in 2023-24). Verify this is being applied correctly and consider using actual T-bill rates for historical accuracy.

### 5C. Daily Data Integration

**Idea:** Use daily returns for intra-month risk monitoring, even if rebalancing remains monthly. The daily cache already exists (`daily_returns.csv`).

Daily monitoring enables:
- Mid-month stop-loss triggers
- Real-time drawdown tracking
- More responsive regime detection

---

## Recommended Priority Order

| Priority | Strategy | Expected MDD Gain | Effort | Risk |
|----------|----------|--------------------|--------|------|
| 1 | 2A: Bootstrap-aware optimization | +3-5%p confidence | Low | Low |
| 2 | 1A: Multi-seed ensemble | +0.5-2.0%p | Medium | Low |
| 3 | 2B: S7 drawdown layer | +0.5-1.5%p | Low | Low |
| 4 | 3A-C: Robustness testing | Documentation only | Low | None |
| 5 | 2C: Regime-conditional stops | +0.3-1.0%p | Low | Low |
| 6 | 4A: RMT covariance cleaning | +0.5-1.0%p | Medium | Medium |
| 7 | 1B: Architecture ensemble | +1-3%p | High | Medium |
| 8 | 4C: DRO-CVaR | +1-2%p | Very High | High |

**Quick wins (1-2 days):** Strategies 1, 3, 4 - bootstrap-aware optimization and S7 layer can be implemented by modifying the existing precision search scripts.

**Medium effort (1 week):** Strategies 2, 5 - multi-seed ensemble and regime-conditional stops require re-training upstream models.

**Research projects (2+ weeks):** Strategies 6, 7, 8 - RMT, architecture ensemble, DRO-CVaR require significant new code.

---

## Concrete Next Step: Bootstrap-Aware Round 4

The single highest-ROI action is modifying the precision search to optimize for bootstrap-robust MDD rather than point MDD.

```python
# Modification to evaluate() in precision search scripts
def evaluate_robust(mod, ctx, cfg, tc=0.0, n_boot=500, block=6):
    returns, weights, diag = mod.simulate_overlay_v2_strategy(...)
    m = mod.psa.evaluate_returns(returns, "R4")
    sh, rt, md = float(m["sharpe"]), float(m["return"]), float(m["mdd"])

    # Bootstrap MDD distribution
    ret_arr = returns.values
    n = len(ret_arr)
    boot_mdds = []
    for _ in range(n_boot):
        indices = []
        while len(indices) < n:
            start = np.random.randint(0, n)
            for j in range(block):
                indices.append((start + j) % n)
        boot_ret = ret_arr[np.array(indices[:n])]
        cum = np.cumprod(1 + boot_ret)
        peak = np.maximum.accumulate(cum)
        boot_mdds.append(float((cum / peak - 1).min()))

    p75_mdd = float(np.percentile(boot_mdds, 25))  # 25th pctile = worse MDDs
    prob_above = float((np.array(boot_mdds) >= -0.10).mean())

    return {
        "sharpe": sh, "return": rt, "mdd": md,
        "robust_mdd": p75_mdd,
        "prob_mdd_above_10": prob_above,
        "triple": sh >= 1.0 and rt >= 0.10 and p75_mdd >= -0.10,
    }
```

This changes the optimization target from "MDD >= -10% on the single realized path" to "MDD >= -10% on 75% of bootstrap paths," which is a much stronger condition.

**Expected outcome:** A config with point MDD around -8% to -9%, bootstrap P(MDD >= -10%) around 70-85%, and the triple target comfortably met on both point and robust metrics.
