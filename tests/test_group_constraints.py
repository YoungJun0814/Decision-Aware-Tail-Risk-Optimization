"""Regression tests for P2.1 — group constraints inside the CVaR optimizer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

cvxpy = pytest.importorskip("cvxpy")
pytest.importorskip("cvxpylayers")

from src.optimization import CVaROptimizationLayer


def _mu_sigma(n_assets=5, scale=0.02):
    mu = torch.tensor([[0.02, 0.015, 0.01, 0.005, 0.0]], dtype=torch.float32)
    assert mu.shape[1] == n_assets
    sigma = torch.eye(n_assets).unsqueeze(0) * (scale ** 2)
    return mu, sigma


def test_group_cap_respected():
    # Two groups: equities = {0,1,2}, safe = {3,4}. Cap equities at 50%.
    masks = np.array([
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
    ])
    layer = CVaROptimizationLayer(
        num_assets=5, num_scenarios=100,
        bil_index=4, safety_threshold=0.0,
        group_masks=masks, group_caps=np.array([0.5, 1.0]),
        dist_type='normal',
    )
    mu, sigma = _mu_sigma()
    w = layer(mu, sigma, is_crisis=0.0)
    equities_share = float(w[0, :3].sum())
    assert equities_share <= 0.5 + 1e-4, f"equity cap violated: {equities_share:.4f}"
    assert abs(float(w.sum()) - 1.0) < 1e-4


def test_group_floor_respected():
    masks = np.array([[0, 0, 0, 1, 1]])  # bonds+safe as one group
    layer = CVaROptimizationLayer(
        num_assets=5, num_scenarios=100,
        bil_index=4, safety_threshold=0.0,
        group_masks=masks, group_floors=np.array([0.3]),
        dist_type='normal',
    )
    mu, sigma = _mu_sigma()
    w = layer(mu, sigma, is_crisis=0.0)
    safe_share = float(w[0, 3:].sum())
    assert safe_share >= 0.3 - 1e-4, f"floor violated: {safe_share:.4f}"


def test_validation_errors():
    with pytest.raises(ValueError):
        CVaROptimizationLayer(num_assets=5, group_masks=np.zeros((2, 4)))
    with pytest.raises(ValueError):
        CVaROptimizationLayer(
            num_assets=5, group_masks=np.array([[0.5, 0.5, 0, 0, 0]])
        )
    with pytest.raises(ValueError):
        CVaROptimizationLayer(
            num_assets=5,
            group_masks=np.array([[1, 1, 0, 0, 0]]),
            group_caps=np.array([0.5, 0.5]),  # wrong K
        )


def test_no_groups_matches_legacy_behavior():
    """Omitting groups must not change the solution shape / budget."""
    layer = CVaROptimizationLayer(
        num_assets=5, num_scenarios=100, bil_index=4, safety_threshold=0.0,
        dist_type='normal',
    )
    mu, sigma = _mu_sigma()
    w = layer(mu, sigma, is_crisis=0.0)
    assert w.shape == (1, 5)
    assert abs(float(w.sum()) - 1.0) < 1e-4
    assert float(w.min()) >= -1e-6
