"""
Regime-Adaptive Student-T Degrees of Freedom (P2.2)
===================================================

``CVaROptimizationLayer`` currently hardcodes ``t_df=5`` across every
regime. Tail behaviour in crisis vs bull regimes is obviously not
identical: credit / VIX panics are well-fit by df ~ 3, while calm
regimes look almost Gaussian (df >> 10). A single global df is a
conservative compromise that over-fattens tails in calm periods and
under-fattens them in crises.

This module provides ``RegimeAdaptiveTDf``: K learnable df values (one
per regime) with a strict lower bound df > 2 so that the Student-T has
finite variance, which the scenario generator relies on for its unit-
variance normalisation.

Parameterisation
----------------

For each regime k:

    df_k = 2 + eps + softplus(theta_k)

with ``eps`` a small positive slack (default 1e-3). ``theta_k`` is
initialised so that df_k starts at a user-supplied prior (e.g.
``[3, 5, 7, 10]`` for Crisis / Correction / Sideways / Bull).

Usage
-----

    t_df_layer = RegimeAdaptiveTDf(prior_df=[3, 5, 7, 10])
    df_eff = t_df_layer(regime_probs)   # (B,) or scalar — E[df] per sample
    eps    = sample_unit_t(shape, df_eff)

``regime_probs`` must be a (B, K) non-negative vector; rows do not need
to sum to 1 (the layer normalises).

Notes on sampling
-----------------

Because df itself is learnable, we cannot use
``torch.distributions.StudentT(df=df).sample()`` with a gradient path to
df. Instead we build the Student-T from its canonical stochastic
representation:

    X = Z / sqrt(G / df)   with   Z ~ N(0,1),  G ~ Chi2(df)

We treat Z and G as *draws* (no reparameterisation for df), which is
sufficient for the common usage: df flows into the CVaR layer's
scenario-generation variance scaling, and gradients come through the
normalisation factor ``sqrt(df / (df - 2))``. If full pathwise
gradients w.r.t. df are required later, one can switch to an implicit
reparameterisation (Figurnov et al. 2018) — out of scope for P2.2.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimeAdaptiveTDf(nn.Module):
    """Learnable per-regime Student-T degrees of freedom."""

    def __init__(
        self,
        prior_df: Sequence[float] = (3.0, 5.0, 7.0, 10.0),
        df_floor: float = 2.001,
        df_ceiling: float = 50.0,
    ):
        super().__init__()
        prior = torch.tensor(list(prior_df), dtype=torch.float32)
        if (prior <= df_floor).any():
            raise ValueError(
                f"prior_df entries must be > df_floor={df_floor}; got {prior_df}"
            )
        # Invert parameterisation: theta such that
        #   df = df_floor + softplus(theta)
        # => softplus(theta) = prior - df_floor
        # => theta = log(exp(prior - df_floor) - 1)
        with torch.no_grad():
            softplus_inv = torch.log(torch.expm1(prior - df_floor))
        self.theta = nn.Parameter(softplus_inv)
        self.df_floor = float(df_floor)
        self.df_ceiling = float(df_ceiling)

    @property
    def df(self) -> torch.Tensor:
        """Current df vector of shape (K,). Bounded in (df_floor, df_ceiling]."""
        raw = self.df_floor + F.softplus(self.theta)
        return torch.clamp(raw, max=self.df_ceiling)

    def forward(self, regime_probs: torch.Tensor) -> torch.Tensor:
        """Effective df per sample.

        Parameters
        ----------
        regime_probs : (B, K) or (K,) non-negative

        Returns
        -------
        df_eff : same leading shape, each entry in (df_floor, df_ceiling].
        """
        probs = regime_probs
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
        probs = probs.clamp(min=0.0)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        if probs.shape[-1] != self.theta.shape[0]:
            raise ValueError(
                f"regime_probs last dim {probs.shape[-1]} != K={self.theta.shape[0]}"
            )
        df_eff = probs @ self.df  # (B,)
        # Ensure the expectation is also strictly > floor (rare edge case
        # with ceilings clamped and probs all on ceiling bucket).
        return df_eff.clamp(min=self.df_floor + 1e-6)


def sample_unit_t(
    shape: Sequence[int],
    df: torch.Tensor,
    device: torch.device = None,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Sample unit-variance Student-T noise with given df.

    ``df`` may be a scalar or a (B,) tensor broadcast across the leading
    dim of ``shape`` — the returned noise has exactly the shape passed
    in, with variance 1 (not df/(df-2)).
    """
    if device is None:
        device = df.device if torch.is_tensor(df) else torch.device("cpu")
    z = torch.randn(*shape, device=device, generator=generator)
    df_t = df if torch.is_tensor(df) else torch.tensor(float(df), device=device)
    df_t = df_t.clamp(min=2.001)

    if df_t.ndim == 0:
        g = torch.distributions.Gamma(df_t / 2.0, 0.5).rsample(shape)
        df_b = df_t
    else:
        # Per-batch df: df_t has shape (B,) and shape[0] must equal B.
        if df_t.shape[0] != shape[0]:
            raise ValueError(
                f"df batch dim {df_t.shape[0]} != leading shape {shape[0]}"
            )
        g = torch.distributions.Gamma(df_t / 2.0, 0.5).rsample(shape[1:])
        # Gamma.rsample with sample_shape places it *before* the batch dim.
        # We want (B, ...) — permute if needed.
        g = g.movedim(-1, 0) if g.ndim > 1 else g
        # Broadcast df across trailing dims.
        tail = (1,) * (z.ndim - 1)
        df_b = df_t.view(-1, *tail)
        while g.ndim < z.ndim:
            g = g.unsqueeze(-1)

    t_raw = z / torch.sqrt(g / df_b)
    scale = torch.sqrt(df_b / (df_b - 2.0))
    return t_raw / scale
