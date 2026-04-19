"""Regression tests for P1.2 — bootstrap reinterpretation."""

from __future__ import annotations

import numpy as np
import pytest

from src.stats import (
    max_drawdown,
    path_bootstrap_mdd,
    tournament_bootstrap_mdd,
)


# ---------------------------------------------------------------------------
# max_drawdown basic properties
# ---------------------------------------------------------------------------


def test_max_drawdown_monotone_up_is_zero():
    r = np.full(50, 0.01)
    assert max_drawdown(r) == 0.0


def test_max_drawdown_known_path():
    # +10%, -20%, +10% → wealth 1.10, 0.88, 0.968; peak 1.10 → dd = -0.20.
    r = np.array([0.10, -0.20, 0.10])
    assert abs(max_drawdown(r) - (0.88 / 1.10 - 1.0)) < 1e-12


# ---------------------------------------------------------------------------
# Path bootstrap — legacy semantics
# ---------------------------------------------------------------------------


def test_path_bootstrap_reproducible():
    rng = np.random.default_rng(0)
    r = rng.normal(0.008, 0.04, 240)
    a = path_bootstrap_mdd(r, n_bootstrap=500, seed=42)
    b = path_bootstrap_mdd(r, n_bootstrap=500, seed=42)
    assert a.prob_above == b.prob_above


def test_path_bootstrap_better_series_has_higher_prob_above():
    """Higher-Sharpe series should clear the -10% floor more often."""
    rng = np.random.default_rng(1)
    good = rng.normal(0.015, 0.03, 240)   # SR ~ 0.5/month
    poor = rng.normal(-0.002, 0.05, 240)  # negative drift
    pg = path_bootstrap_mdd(good, n_bootstrap=1000, seed=7).prob_above
    pp = path_bootstrap_mdd(poor, n_bootstrap=1000, seed=7).prob_above
    assert pg > pp


# ---------------------------------------------------------------------------
# Tournament bootstrap — the new (honest) number
# ---------------------------------------------------------------------------


def test_tournament_bootstrap_shape_validation():
    with pytest.raises(ValueError):
        tournament_bootstrap_mdd(np.zeros(10), n_bootstrap=100)  # 1-D
    with pytest.raises(ValueError):
        tournament_bootstrap_mdd(
            np.zeros((50, 3)),
            selection="weighted",
            n_bootstrap=100,
        )  # missing prior_weights


def test_tournament_bootstrap_selection_modes_run():
    rng = np.random.default_rng(2)
    X = rng.normal(0.005, 0.04, (200, 5))
    for sel in ("bayesian", "uniform"):
        res = tournament_bootstrap_mdd(
            X, n_bootstrap=300, selection=sel, seed=3
        )
        assert 0.0 <= res.prob_above <= 1.0
        assert res.n_candidates == 5

    res_w = tournament_bootstrap_mdd(
        X,
        n_bootstrap=300,
        selection="weighted",
        prior_weights=np.array([0.6, 0.1, 0.1, 0.1, 0.1]),
        seed=3,
    )
    assert 0.0 <= res_w.prob_above <= 1.0


def test_tournament_prob_bounded_by_best_and_worst_paths():
    """The tournament prob must lie between the best and worst candidate's
    path-only prob. Winner-take-all inflates beyond the tournament mixture,
    so the mixture can't exceed the best."""
    rng = np.random.default_rng(4)
    T, K = 240, 6
    means = np.linspace(-0.003, 0.012, K)  # ranked candidates
    X = rng.normal(0.0, 0.04, (T, K)) + means
    per_cand = [
        path_bootstrap_mdd(X[:, k], n_bootstrap=500, seed=11).prob_above
        for k in range(K)
    ]
    tourn = tournament_bootstrap_mdd(
        X, n_bootstrap=2000, selection="uniform", seed=11
    ).prob_above
    # Under uniform selection the bootstrap is essentially a mixture, so
    # tournament prob is in [min, max] of per-candidate probs, up to MC noise.
    assert min(per_cand) - 0.05 <= tourn <= max(per_cand) + 0.05


def test_tournament_less_optimistic_than_winner_path():
    """Honest claim: the tournament number should be no larger than the
    winner's path-only number once strategy uncertainty is added."""
    rng = np.random.default_rng(5)
    T, K = 200, 4
    X = rng.normal(0.004, 0.04, (T, K))
    # Pick the observed winner by mean return.
    winner = int(X.mean(axis=0).argmax())
    winner_prob = path_bootstrap_mdd(
        X[:, winner], n_bootstrap=1500, seed=9
    ).prob_above
    tourn_prob = tournament_bootstrap_mdd(
        X, n_bootstrap=1500, selection="bayesian", seed=9
    ).prob_above
    # Bayesian tournament averages over plausible winners; with candidates
    # near each other this must be <= winner's number + small MC tolerance.
    assert tourn_prob <= winner_prob + 0.05
