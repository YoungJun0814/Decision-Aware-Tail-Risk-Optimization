"""
Bootstrap Reinterpretation (P1.2)
=================================

The headline number ``P_bootstrap(MDD >= -10%) = 57.98%`` is, as reported
in ``TRIPLE_TARGET_RESULTS.md`` and ``README.md``, the empirical frequency
under a block bootstrap of a **single, already-selected** return path.
That is a statement about *path variability of the winner*, not a
statement about the probability the tournament would produce a path that
stays above the -10% floor.

This module separates the two quantities cleanly:

  * ``path_bootstrap_mdd``  — block bootstrap of the selected path only.
                              Reports ``P(MDD >= threshold | selected path)``.
                              Matches the legacy 57.98% number.

  * ``tournament_bootstrap_mdd`` — joint bootstrap over
      (i) block resamples of each candidate's return series,
      (ii) Bayesian-bootstrap weights on the tournament of K candidates.
      Reports ``P(MDD >= threshold | tournament distribution)`` — the
      statistically honest number that includes strategy-selection
      uncertainty.

Both use the Politis–Romano stationary bootstrap (same machinery as
``reality_check``) so the dependence structure in returns is preserved.

References
----------
* Politis & Romano (1994). The stationary bootstrap. *JASA*.
* Rubin (1981). The Bayesian bootstrap. *Annals of Statistics*.
* White (2000). A reality check for data snooping. *Econometrica*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.stats.post_selection import _stationary_bootstrap_indices


# ---------------------------------------------------------------------------
# MDD utility
# ---------------------------------------------------------------------------


def max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown of a per-period return series, in path units.

    Returns a non-positive number. ``returns`` are simple per-period
    returns; wealth is ``cumprod(1 + r)``.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    return float(dd.min())


# ---------------------------------------------------------------------------
# Path-only bootstrap (legacy semantics)
# ---------------------------------------------------------------------------


@dataclass
class PathBootstrapResult:
    threshold: float
    prob_above: float            # P(MDD >= threshold | selected path)
    mdd_mean: float
    mdd_quantiles: dict          # 5/50/95 percentiles
    n_bootstrap: int
    block_length: float


def path_bootstrap_mdd(
    returns: np.ndarray,
    threshold: float = -0.10,
    n_bootstrap: int = 5000,
    block_length: float = 6.0,
    seed: Optional[int] = None,
) -> PathBootstrapResult:
    """Block bootstrap of a single selected path.

    ``prob_above`` is the fraction of bootstrap replicates whose MDD is
    **greater-or-equal** to ``threshold`` (i.e. less severe). For the
    canonical -10% floor this reproduces the legacy 57.98%-style number.
    """
    r = np.asarray(returns, dtype=float)
    T = len(r)
    if T < 2:
        raise ValueError("Need at least 2 return observations")
    block_p = 1.0 / max(float(block_length), 1.0)
    rng = np.random.default_rng(seed)

    mdds = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(T, block_p, rng)
        mdds[b] = max_drawdown(r[idx])

    return PathBootstrapResult(
        threshold=threshold,
        prob_above=float((mdds >= threshold).mean()),
        mdd_mean=float(mdds.mean()),
        mdd_quantiles={
            "p05": float(np.quantile(mdds, 0.05)),
            "p50": float(np.quantile(mdds, 0.50)),
            "p95": float(np.quantile(mdds, 0.95)),
        },
        n_bootstrap=n_bootstrap,
        block_length=block_length,
    )


# ---------------------------------------------------------------------------
# Tournament bootstrap (P1.2 — the honest number)
# ---------------------------------------------------------------------------


@dataclass
class TournamentBootstrapResult:
    threshold: float
    prob_above: float            # P(MDD >= threshold | tournament dist)
    mdd_mean: float
    mdd_quantiles: dict
    n_bootstrap: int
    n_candidates: int
    block_length: float
    selection: str               # "bayesian" or "uniform" or "weighted"


def _dirichlet_weights(k: int, rng: np.random.Generator) -> np.ndarray:
    """Bayesian bootstrap weights: Dirichlet(1, ..., 1)."""
    w = rng.standard_exponential(k)
    return w / w.sum()


def tournament_bootstrap_mdd(
    candidates: np.ndarray,
    threshold: float = -0.10,
    n_bootstrap: int = 5000,
    block_length: float = 6.0,
    selection: str = "bayesian",
    prior_weights: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> TournamentBootstrapResult:
    """Joint bootstrap over block-of-returns AND strategy choice.

    Parameters
    ----------
    candidates : (T, K) array
        Column k is per-period return series of candidate strategy k
        (already selected into the tournament).
    threshold : float
        MDD floor (non-positive). ``prob_above`` counts replicates with
        MDD >= threshold, i.e. less severe than the floor.
    selection : {"bayesian", "uniform", "weighted"}
        How a winner is drawn on each bootstrap replicate:
          * "bayesian"  — Dirichlet(1,...,1) weights over candidates;
            winner = argmax_k w_k * mean_return_k (Rubin's Bayesian
            bootstrap of the selection procedure).
          * "uniform"   — pick a uniform-random candidate.
          * "weighted"  — fixed ``prior_weights`` multinomial pick
            (e.g. tournament posterior from a meta-model).
    prior_weights : (K,) array, optional
        Required when ``selection == "weighted"``.
    """
    X = np.asarray(candidates, dtype=float)
    if X.ndim != 2:
        raise ValueError("candidates must be 2-D (T, K)")
    T, K = X.shape
    if T < 2 or K < 1:
        raise ValueError("Need T>=2 and K>=1")
    if selection == "weighted":
        if prior_weights is None:
            raise ValueError("prior_weights required when selection='weighted'")
        pw = np.asarray(prior_weights, dtype=float)
        if pw.shape != (K,) or pw.min() < 0 or pw.sum() <= 0:
            raise ValueError("prior_weights must be non-negative, length K, sum>0")
        pw = pw / pw.sum()

    block_p = 1.0 / max(float(block_length), 1.0)
    rng = np.random.default_rng(seed)

    mdds = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(T, block_p, rng)
        X_star = X[idx]  # (T, K) bootstrapped panel — shared resample across k
        if selection == "bayesian":
            w = _dirichlet_weights(K, rng)
            k_star = int(np.argmax(w * X_star.mean(axis=0)))
        elif selection == "uniform":
            k_star = int(rng.integers(0, K))
        elif selection == "weighted":
            k_star = int(rng.choice(K, p=pw))
        else:
            raise ValueError(f"unknown selection mode: {selection!r}")
        mdds[b] = max_drawdown(X_star[:, k_star])

    return TournamentBootstrapResult(
        threshold=threshold,
        prob_above=float((mdds >= threshold).mean()),
        mdd_mean=float(mdds.mean()),
        mdd_quantiles={
            "p05": float(np.quantile(mdds, 0.05)),
            "p50": float(np.quantile(mdds, 0.50)),
            "p95": float(np.quantile(mdds, 0.95)),
        },
        n_bootstrap=n_bootstrap,
        n_candidates=K,
        block_length=block_length,
        selection=selection,
    )
