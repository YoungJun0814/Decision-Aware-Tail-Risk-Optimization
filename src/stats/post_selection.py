"""
Post-Selection Statistics (P1.1)
================================

When the tournament in ``scripts/run_triple_precision_search.py`` evaluates
N strategy candidates on the same out-of-sample fold and then promotes the
best one, the winning Sharpe ratio is positively biased even under the null
of no skill. This module provides the two standard corrections:

  * **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) — closed-form
    probability that the observed SR exceeds what the best of N iid
    no-skill strategies would have produced, adjusted for the sample's
    third and fourth moments (skew / kurtosis).

  * **Reality Check** (White, 2000; Politis–Romano stationary bootstrap) —
    resampling test of H₀: ``max_k E[excess_k] ≤ 0`` across the whole
    candidate family. Reports a bootstrapped p-value for the observation
    ``max_k mean(excess_k)``.

Both are designed to be called from reporting scripts, not training loops.

References
----------
* Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio.
  *Journal of Portfolio Management*, 40(5), 94-107.
* White, H. (2000). A reality check for data snooping. *Econometrica*,
  68(5), 1097-1126.
* Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
  *Journal of the American Statistical Association*, 89(428), 1303-1313.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm, skew, kurtosis


# ---------------------------------------------------------------------------
# Sharpe ratio utilities
# ---------------------------------------------------------------------------


def sharpe_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    periods_per_year: Optional[int] = None,
) -> float:
    """Sample Sharpe ratio.

    Parameters
    ----------
    returns : array_like, shape (T,)
        Per-period returns (e.g. monthly).
    risk_free : float
        Per-period risk-free return (same units as ``returns``).
    periods_per_year : int or None
        If given, return annualised Sharpe (× sqrt(periods_per_year)).
        If None, return the per-period Sharpe.
    """
    r = np.asarray(returns, dtype=float) - risk_free
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = mu / sd
    if periods_per_year is not None:
        sr *= math.sqrt(periods_per_year)
    return float(sr)


# ---------------------------------------------------------------------------
# Expected maximum Sharpe under the null
# ---------------------------------------------------------------------------


def expected_max_sharpe(n_trials: int) -> float:
    """Expected maximum per-period Sharpe of N iid no-skill strategies.

    Uses the Bailey-López de Prado (2014) approximation

        SR* ≈ (1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N·e))

    where γ ≈ 0.5772 is the Euler-Mascheroni constant. This is the mean of
    the distribution of the maximum of N iid standard normals.

    For N == 1 returns 0 (no selection bias).
    """
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    e = math.e
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float((1.0 - gamma) * z1 + gamma * z2)


# ---------------------------------------------------------------------------
# Probabilistic / Deflated Sharpe
# ---------------------------------------------------------------------------


@dataclass
class DSRResult:
    psr: float          # P(true_SR > threshold) given the sample moments
    dsr: float          # P(true_SR > SR_star) — adjusted for selection
    sr: float           # observed per-period Sharpe
    sr_star: float      # expected max Sharpe under null
    n_trials: int
    n_obs: int
    skew: float
    excess_kurt: float  # kurtosis - 3 (Fisher definition used by scipy's default)

    def reject_null(self, alpha: float = 0.05) -> bool:
        return self.dsr > 1.0 - alpha


def _sr_standard_error(sr: float, n: int, skew_: float, excess_kurt: float) -> float:
    """Mertens (2002) standard error of the Sharpe ratio estimator."""
    # var(SR) = (1 - γ3·SR + (γ4 - 1)/4 · SR²) / (T - 1)
    # where γ4 here is the (non-excess) kurtosis = excess_kurt + 3.
    gamma4 = excess_kurt + 3.0
    var = (1.0 - skew_ * sr + (gamma4 - 1.0) / 4.0 * sr * sr) / max(n - 1, 1)
    return math.sqrt(max(var, 1e-12))


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio — P(true_SR > benchmark_sr).

    Non-Gaussian-adjusted per Bailey & LdP (2014 eq. 1).
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    sr = sharpe_ratio(r)
    sk = float(skew(r, bias=False))
    ek = float(kurtosis(r, fisher=True, bias=False))
    se = _sr_standard_error(sr, n, sk, ek)
    z = (sr - benchmark_sr) / se
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
) -> DSRResult:
    """Deflated Sharpe Ratio — PSR against the expected max-Sharpe null.

    Parameters
    ----------
    returns : (T,) per-period portfolio returns (e.g. monthly)
    n_trials : number of candidate strategies evaluated during selection

    Returns
    -------
    DSRResult with .dsr in [0, 1]. High values (close to 1) indicate the
    observed Sharpe is unlikely to be explained by selection from ``n_trials``
    no-skill alternatives.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    sr = sharpe_ratio(r)
    sk = float(skew(r, bias=False))
    ek = float(kurtosis(r, fisher=True, bias=False))

    sr_star = expected_max_sharpe(n_trials)

    se = _sr_standard_error(sr, n, sk, ek)
    z_dsr = (sr - sr_star) / se
    dsr = float(norm.cdf(z_dsr))

    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)

    return DSRResult(
        psr=psr,
        dsr=dsr,
        sr=sr,
        sr_star=sr_star,
        n_trials=n_trials,
        n_obs=n,
        skew=sk,
        excess_kurt=ek,
    )


# ---------------------------------------------------------------------------
# White's Reality Check via stationary bootstrap
# ---------------------------------------------------------------------------


def _stationary_bootstrap_indices(n: int, block_p: float, rng: np.random.Generator) -> np.ndarray:
    """Politis–Romano stationary bootstrap index vector of length n.

    At each step stay at the current index with prob 1-block_p, otherwise
    jump to a uniform random index. Average block length = 1/block_p.
    """
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < block_p:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


@dataclass
class RealityCheckResult:
    best_statistic: float
    p_value: float
    n_candidates: int
    n_bootstrap: int
    block_length: float


def reality_check(
    excess_returns: np.ndarray,
    n_bootstrap: int = 2000,
    block_length: float = 6.0,
    seed: Optional[int] = None,
) -> RealityCheckResult:
    """White's Reality Check with stationary bootstrap.

    Parameters
    ----------
    excess_returns : (T, K) array
        K candidate strategies' per-period excess returns over a common
        benchmark (e.g. equal-weight). Column k is strategy k.
    n_bootstrap : int
        Number of bootstrap repetitions.
    block_length : float
        Expected stationary-bootstrap block length (months). Inverse of the
        geometric switching probability.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    RealityCheckResult with ``p_value = P(max_k mean(r*_k) >= max_k mean(r_k))``
    under the resampled null (each candidate's mean is centred to zero inside
    the bootstrap loop).
    """
    X = np.asarray(excess_returns, dtype=float)
    if X.ndim != 2:
        raise ValueError("excess_returns must be 2-D (T, K)")
    T, K = X.shape
    if T < 2 or K < 1:
        raise ValueError("Need T>=2 and K>=1")
    block_p = 1.0 / max(float(block_length), 1.0)
    rng = np.random.default_rng(seed)

    # Observed statistic V = max_k mean_t X[t, k]
    means = X.mean(axis=0)
    V_hat = float(means.max())

    # Centre columns to impose H0: mean <= 0 (White 2000, eq. 2.8).
    X_centred = X - means

    V_star = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(T, block_p, rng)
        V_star[b] = X_centred[idx].mean(axis=0).max()

    p = float((V_star >= V_hat).mean())

    return RealityCheckResult(
        best_statistic=V_hat,
        p_value=p,
        n_candidates=K,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
    )
