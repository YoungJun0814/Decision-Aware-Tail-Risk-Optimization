"""Integration smoke tests for P2.1 / P2.2 wiring through BaseBLModel.

Guarantees that:
  * ``get_model`` and ``BaseBLModel`` accept and forward the new
    ``group_masks`` / ``group_caps`` / ``group_floors`` / ``learnable_t_df``
    / ``t_df_prior`` kwargs.
  * ``regime_t_df`` gets attached to the model and produces a tensor df
    that the CVaR layer can consume.
  * Gradient flows from the learnable df into the df parameters under
    the cvxpy-less Softmax fallback path (integration without the LP).
"""

from __future__ import annotations

import numpy as np
import torch

from src.models import get_model


def test_wiring_attaches_regime_t_df_and_group_masks():
    masks = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
    m = get_model(
        model_type='gru', input_dim=6, num_assets=4,
        regime_dim=4, macro_dim=0,
        learnable_t_df=True, t_df_prior=[3.0, 5.0, 7.0, 10.0],
        group_masks=masks, group_caps=[0.6, 1.0], group_floors=[0.0, 0.2],
    )
    assert m.regime_t_df is not None
    assert m.opt_layer.group_masks is not None
    assert m.opt_layer.group_masks.shape == (2, 4)
    assert m.opt_layer.group_caps.tolist() == [0.6, 1.0]
    assert m.opt_layer.group_floors.tolist() == [0.0, 0.2]


def test_no_wiring_is_backward_compatible():
    m = get_model(model_type='gru', input_dim=6, num_assets=4,
                  regime_dim=4, macro_dim=0)
    assert m.regime_t_df is None
    assert m.opt_layer.group_masks is None
    assert isinstance(m.opt_layer.t_df, (int, float))


def test_learnable_t_df_gradient_flows_via_sampler():
    """Gradient from a downstream tensor must reach regime_t_df.theta.

    We bypass the cvxpy LP (not installed in this env) and exercise the
    scenario sampler directly: the CVaR layer's _sample_noise is what
    actually consumes the learnable df.
    """
    m = get_model(
        model_type='gru', input_dim=6, num_assets=4,
        regime_dim=4, macro_dim=0,
        learnable_t_df=True, t_df_prior=[3.0, 5.0, 7.0, 10.0],
    )
    # Mimic the forward-time handoff that BaseBLModel.forward does.
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    m.opt_layer.t_df = m.regime_t_df(probs)

    eps = m.opt_layer._sample_noise((1, 200, 4), torch.device('cpu'))
    loss = eps.pow(2).mean()
    loss.backward()
    assert m.regime_t_df.theta.grad is not None
    assert m.regime_t_df.theta.grad.abs().sum() > 0
