"""Regression tests for P1.4 — differentiable soft path-MDD."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.loss import SoftPathMDDLoss


def _hard_mdd(r: torch.Tensor) -> torch.Tensor:
    w = torch.cumprod(1.0 + r, dim=0)
    peak = torch.cummax(w, dim=0).values
    return (1.0 - w / peak).max()


def test_soft_mdd_converges_to_hard_as_beta_grows():
    torch.manual_seed(0)
    r = torch.randn(120) * 0.04 + 0.005
    hard = float(_hard_mdd(r))
    errs = []
    for beta in [10.0, 40.0, 160.0]:
        soft = float(SoftPathMDDLoss(beta=beta).soft_mdd(r))
        errs.append(abs(soft - hard))
    assert errs[0] > errs[2]
    assert errs[-1] < 0.015, f"soft MDD not close enough to hard: {errs}"


def test_soft_mdd_is_upper_bound_of_hard():
    """Logsumexp max-surrogate is always >= true max — for any beta."""
    torch.manual_seed(1)
    for _ in range(5):
        r = torch.randn(80) * 0.05
        hard = float(_hard_mdd(r))
        soft = float(SoftPathMDDLoss(beta=30.0).soft_mdd(r))
        # The soft running peak is >= true peak; so soft drawdown is
        # >= true drawdown; hence soft_mdd >= hard_mdd up to numerical eps.
        assert soft >= hard - 1e-6


def test_soft_mdd_every_timestep_has_gradient():
    """Key property vs the hard MDD: every observation feeds a gradient."""
    torch.manual_seed(2)
    r = (torch.randn(60) * 0.04).clone().requires_grad_(True)
    mdd = SoftPathMDDLoss(beta=40.0).soft_mdd(r)
    mdd.backward()
    g = r.grad
    assert g is not None
    # Under hard MDD only 1-2 entries would be non-zero; here we expect
    # essentially all of them to receive non-trivial gradient.
    nonzero_frac = (g.abs() > 1e-10).float().mean().item()
    assert nonzero_frac > 0.9, f"gradient too sparse: {nonzero_frac:.2f}"


def test_hinge_zero_when_below_threshold():
    """With a flat up-trending path, soft MDD ~ 0 and hinge must be 0."""
    r = torch.full((50,), 0.01)
    loss = SoftPathMDDLoss(mdd_target=-0.10, soft_margin=0.02)(r)
    assert float(loss) == 0.0


def test_short_sequence_returns_zero():
    r = torch.tensor([0.01])
    loss = SoftPathMDDLoss()(r)
    assert float(loss) == 0.0


def test_gradient_pushes_toward_lower_mdd():
    """A single SGD step on the soft loss should reduce MDD."""
    torch.manual_seed(3)
    r = (torch.randn(80) * 0.05 - 0.005).clone().requires_grad_(True)
    loss_fn = SoftPathMDDLoss(mdd_target=-0.05, soft_margin=0.0, beta=40.0,
                              mdd_lambda=1.0)
    loss0 = loss_fn(r)
    loss0.backward()
    with torch.no_grad():
        r_new = (r - 0.05 * r.grad).detach()
    mdd_before = float(_hard_mdd(r.detach()))
    mdd_after = float(_hard_mdd(r_new))
    assert mdd_after <= mdd_before + 1e-4, (
        f"gradient step did not reduce MDD: before={mdd_before:.4f} "
        f"after={mdd_after:.4f}"
    )
