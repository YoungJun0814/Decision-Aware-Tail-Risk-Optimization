"""Regression tests for P2.2 — regime-adaptive learnable Student-T df."""

from __future__ import annotations

import torch

from src.regime_t_df import RegimeAdaptiveTDf, sample_unit_t


def test_df_at_init_matches_prior():
    prior = [3.0, 5.0, 7.0, 10.0]
    layer = RegimeAdaptiveTDf(prior_df=prior)
    df = layer.df.detach().numpy()
    assert all(abs(df[k] - prior[k]) < 1e-4 for k in range(4))


def test_df_strictly_above_floor():
    layer = RegimeAdaptiveTDf(prior_df=[3.0, 5.0], df_floor=2.001)
    # Drive theta way negative — df should still be > floor.
    with torch.no_grad():
        layer.theta.fill_(-50.0)
    assert float(layer.df.min()) >= 2.001 - 1e-6


def test_effective_df_mixes_by_probs():
    layer = RegimeAdaptiveTDf(prior_df=[3.0, 10.0])
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    df_eff = layer(probs)
    assert abs(float(df_eff[0]) - 3.0) < 1e-4
    assert abs(float(df_eff[1]) - 10.0) < 1e-4
    assert abs(float(df_eff[2]) - 6.5) < 1e-4


def test_gradient_flows_to_theta():
    layer = RegimeAdaptiveTDf(prior_df=[3.0, 5.0, 7.0, 10.0])
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    df_eff = layer(probs)
    loss = (df_eff - 8.0).pow(2).mean()
    loss.backward()
    assert layer.theta.grad is not None
    assert layer.theta.grad.abs().sum() > 0


def test_sample_unit_t_has_unit_variance_approx():
    torch.manual_seed(0)
    df = torch.tensor(6.0)
    x = sample_unit_t((200_000,), df)
    v = float(x.var(unbiased=True))
    assert abs(v - 1.0) < 0.05, f"variance={v}"


def test_sample_unit_t_per_batch_df_runs():
    torch.manual_seed(1)
    df = torch.tensor([3.1, 5.0, 10.0])
    x = sample_unit_t((3, 1000), df)
    assert x.shape == (3, 1000)
    # Heaviest tail (df=3.1) should have highest kurtosis among the three.
    kurts = [float(((x[k] - x[k].mean()) ** 4).mean() / x[k].var() ** 2)
             for k in range(3)]
    assert kurts[0] >= kurts[1] >= kurts[2] - 1.0  # MC noise tolerance
