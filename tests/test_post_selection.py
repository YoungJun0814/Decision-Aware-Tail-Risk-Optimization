"""Regression test for P1.1 — Deflated Sharpe + Reality Check."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.stats import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    reality_check,
    sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Expected-max-Sharpe sanity checks
# ---------------------------------------------------------------------------


def test_expected_max_sharpe_monotone_and_zero_at_n1():
    assert expected_max_sharpe(1) == 0.0
    # Expected max of the best of N standard normals strictly increases with N.
    prev = 0.0
    for n in [2, 10, 100, 1000]:
        v = expected_max_sharpe(n)
        assert v > prev, f"expected_max_sharpe({n}) = {v} not > {prev}"
        prev = v


def test_expected_max_sharpe_monte_carlo():
    """Compare closed-form against Monte-Carlo max of iid normals."""
    rng = np.random.default_rng(0)
    for n in [10, 100]:
        mc = rng.standard_normal((200_000, n)).max(axis=1).mean()
        cf = expected_max_sharpe(n)
        # Bailey-LdP formula has ~0.05 absolute error across this range.
        assert abs(mc - cf) < 0.05, f"N={n}: MC={mc:.4f} vs CF={cf:.4f}"


# ---------------------------------------------------------------------------
# DSR behaviour
# ---------------------------------------------------------------------------


def test_dsr_reduces_to_psr_when_n_trials_is_one():
    rng = np.random.default_rng(42)
    r = rng.normal(0.01, 0.05, 120)  # ~monthly, skill present
    res = deflated_sharpe_ratio(r, n_trials=1)
    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    # N=1 → sr_star=0 → DSR ≡ PSR(0).
    assert abs(res.dsr - psr) < 1e-10


def test_dsr_decreases_as_n_trials_grows():
    rng = np.random.default_rng(1)
    r = rng.normal(0.006, 0.04, 180)
    dsr_values = [deflated_sharpe_ratio(r, n_trials=n).dsr for n in [1, 10, 100, 1000]]
    for a, b in zip(dsr_values, dsr_values[1:]):
        assert a >= b - 1e-12, f"DSR not monotone non-increasing: {dsr_values}"
    # Usually strictly decreasing when SR > 0.
    assert dsr_values[0] > dsr_values[-1]


def test_dsr_below_half_when_sr_equals_sr_star():
    """If observed SR exactly equals the selection-inflated null, DSR ≈ 0.5."""
    rng = np.random.default_rng(7)
    # Build a return series whose Sharpe is (very close to) sr_star for N=500.
    target = expected_max_sharpe(500)
    r = rng.normal(0.0, 1.0, 400)
    r = r - r.mean()
    r = r / r.std(ddof=1)
    r = r + target  # shift so mean/std ≈ target (per-period)
    res = deflated_sharpe_ratio(r, n_trials=500)
    assert abs(res.sr - target) < 1e-6
    assert abs(res.dsr - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Reality Check
# ---------------------------------------------------------------------------


def test_reality_check_null_has_large_pvalue():
    """With all candidates centred at 0, p-value should be uniform → large."""
    rng = np.random.default_rng(11)
    T, K = 150, 20
    X = rng.normal(0.0, 0.04, (T, K))
    res = reality_check(X, n_bootstrap=1000, block_length=6.0, seed=3)
    # Under H0 the p-value is uniform; not-too-small is enough for a smoke test.
    assert res.p_value > 0.05, f"p-value too small under null: {res.p_value}"


def test_reality_check_rejects_strong_signal():
    """Inject one candidate with real skill; p-value should be tiny."""
    rng = np.random.default_rng(13)
    T, K = 150, 20
    X = rng.normal(0.0, 0.04, (T, K))
    X[:, 0] += 0.03  # large per-period edge on strategy 0
    res = reality_check(X, n_bootstrap=1000, block_length=6.0, seed=5)
    assert res.p_value < 0.05, f"p-value too large with true signal: {res.p_value}"
    assert res.best_statistic > 0.02


def test_reality_check_shape_validation():
    with pytest.raises(ValueError):
        reality_check(np.zeros(10), n_bootstrap=100)  # 1-D
    with pytest.raises(ValueError):
        reality_check(np.zeros((1, 5)), n_bootstrap=100)  # T<2


# ---------------------------------------------------------------------------
# Basic Sharpe utility
# ---------------------------------------------------------------------------


def test_sharpe_annualisation():
    rng = np.random.default_rng(2)
    r = rng.normal(0.01, 0.04, 240)
    sr_m = sharpe_ratio(r)
    sr_a = sharpe_ratio(r, periods_per_year=12)
    assert abs(sr_a - sr_m * math.sqrt(12)) < 1e-10
