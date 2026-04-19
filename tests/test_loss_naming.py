"""
Regression test for P0.3 — CVaR / batch_tail_mean naming.

DecisionAwareLoss.risk_type='cvar' computes the mean of the worst α% of
portfolio returns within the mini-batch. This is NOT the distributional
Rockafellar-Uryasev CVaR used by the optimization layer. The two only
coincide in the large-batch stationary limit.

This test guards two properties:
  1. The new canonical name ``'batch_tail_mean'`` produces the same
     numerical value as the legacy ``'cvar'`` name (behavior preserved).
  2. The legacy ``'cvar'`` alias emits a DeprecationWarning exactly once
     per loss instance, so downstream users migrate.
"""

import warnings

import torch
import pytest

from src.loss import DecisionAwareLoss


@pytest.fixture
def dummy_batch():
    torch.manual_seed(0)
    B, N = 64, 13
    weights = torch.softmax(torch.randn(B, N), dim=1)
    future_returns = torch.randn(B, N) * 0.02
    vix = torch.rand(B) * 30 + 10
    return weights, future_returns, vix


def test_batch_tail_mean_equals_legacy_cvar(dummy_batch):
    weights, returns, vix = dummy_batch
    loss_new = DecisionAwareLoss(risk_type='batch_tail_mean')
    loss_old = DecisionAwareLoss(risk_type='cvar')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v_new = loss_new(weights, returns, vix).item()
        v_old = loss_old(weights, returns, vix).item()
    assert abs(v_new - v_old) < 1e-6, (
        "'batch_tail_mean' and 'cvar' must yield identical values "
        f"(got new={v_new}, old={v_old})"
    )


def test_cvar_alias_emits_deprecation_warning(dummy_batch):
    weights, returns, vix = dummy_batch
    loss_old = DecisionAwareLoss(risk_type='cvar')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss_old(weights, returns, vix)
    msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(msgs) == 1, f"Expected one DeprecationWarning, got {len(msgs)}"
    assert "batch_tail_mean" in msgs[0]
    assert "distributional CVaR" in msgs[0]


def test_cvar_alias_warns_only_once_per_instance(dummy_batch):
    weights, returns, vix = dummy_batch
    loss_old = DecisionAwareLoss(risk_type='cvar')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            loss_old(weights, returns, vix)
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep) == 1, "DeprecationWarning must be emitted only on first call"


def test_canonical_name_does_not_warn(dummy_batch):
    weights, returns, vix = dummy_batch
    loss_new = DecisionAwareLoss(risk_type='batch_tail_mean')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss_new(weights, returns, vix)
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep) == 0, "'batch_tail_mean' must not emit DeprecationWarning"
