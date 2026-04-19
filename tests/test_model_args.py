"""
Regression test for encoder argument plumbing (P0.2).

Ensures every encoder class (LSTM / GRU / TCN / Transformer / TFT) accepts
and forwards the full set of regime-related kwargs so that `--e2e_regime`,
macro features, crisis overlay, and Student-T scenarios actually take
effect. Previously only GRU honoured these — the other encoders silently
dropped them, which invalidated any multi-encoder comparison.
"""

import torch
import pytest

from src.models import (
    LSTMModel,
    GRUModel,
    TCNModel,
    TransformerModel,
    TFTModel,
)

ENCODERS = [LSTMModel, GRUModel, TCNModel, TransformerModel, TFTModel]

COMMON_KWARGS = dict(
    input_dim=13,
    num_assets=13,
    hidden_dim=16,
    dropout=0.1,
    omega_mode="learnable",
    sigma_mode="prior",
    lambda_risk=0.0,
    regime_dim=4,
    macro_dim=0,
    max_bil_floor=0.5,
    dist_type="t",
    t_df=5.0,
    e2e_regime=True,
)


@pytest.mark.parametrize("cls", ENCODERS)
def test_encoder_accepts_regime_kwargs(cls):
    model = cls(**COMMON_KWARGS)
    # Every encoder must expose the regime-related attributes.
    assert model.regime_dim == 4, f"{cls.__name__}: regime_dim not stored"
    assert model.macro_dim == 0, f"{cls.__name__}: macro_dim not stored"
    assert model.e2e_regime is True, f"{cls.__name__}: e2e_regime flag lost"
    # With regime_dim>0 and e2e_regime=True the submodules must exist.
    assert hasattr(model, "crisis_overlay"), f"{cls.__name__}: crisis_overlay missing"
    assert hasattr(model, "e2e_regime_head"), f"{cls.__name__}: e2e_regime_head missing"
    # Student-T scenario generator must be wired into the opt layer.
    assert model.opt_layer.dist_type == "t"
    assert float(model.opt_layer.t_df) == 5.0


@pytest.mark.parametrize("cls", ENCODERS)
def test_encoder_forward_returns_e2e_tuple(cls):
    """When e2e_regime=True, forward must return (weights, regime_probs)."""
    model = cls(**COMMON_KWARGS).eval()
    B, T = 2, 12
    x = torch.randn(B, T, COMMON_KWARGS["input_dim"])
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, tuple), f"{cls.__name__}: did not return tuple"
    weights, regime_probs = out
    assert weights.shape == (B, COMMON_KWARGS["num_assets"])
    assert regime_probs.shape == (B, COMMON_KWARGS["regime_dim"])
    assert torch.allclose(weights.sum(dim=1), torch.ones(B), atol=1e-3)
    assert torch.allclose(regime_probs.sum(dim=1), torch.ones(B), atol=1e-3)


@pytest.mark.parametrize("cls", ENCODERS)
def test_encoder_non_e2e_backcompat(cls):
    """Default kwargs (regime_dim=0) must still produce a plain weight tensor."""
    kwargs = dict(
        input_dim=13,
        num_assets=13,
        hidden_dim=16,
        dropout=0.1,
    )
    model = cls(**kwargs).eval()
    B, T = 2, 12
    x = torch.randn(B, T, kwargs["input_dim"])
    with torch.no_grad():
        out = model(x)
    assert torch.is_tensor(out), f"{cls.__name__}: expected tensor when e2e off"
    assert out.shape == (B, kwargs["num_assets"])
