"""
Ablation Ladder (P2.5)
======================

A progressive ablation that isolates the contribution of each modelling
stage. Each rung produces a per-period portfolio return series on a
common out-of-sample period; downstream the series are scored with the
same metric stack (Sharpe, MDD, bootstrap, DSR, Reality Check) so that
cross-rung deltas are attributable.

Rungs
-----

1. Equal-weight (EW)             — 1/N baseline, no inputs at all.
2. Risk-parity (RP)              — inverse-volatility weights from a
                                   rolling window of realised returns.
3. Min-CVaR (no regime)          — CVaR-minimising portfolio from the
                                   realised-return scenarios; no BL
                                   posterior, no regime conditioning.
4. BL-only (no CVaR)             — Black-Litterman posterior mean,
                                   mean-variance max-Sharpe (analytic),
                                   no tail penalty.
5. BL + CVaR (no regime)         — BL posterior mean feeds the CVaR
                                   layer, but regime probabilities are
                                   replaced with a uniform prior.
6. Full model                    — The thesis model (BL + CVaR + regime
                                   + overlay).

The script deliberately does **not** retrain deep models — the goal is
to get statistically comparable series cheaply. Deep-model rungs consume
already-saved portfolio return CSVs; the lighter baselines (1, 2, 3) are
computed from the asset return panel directly.

Usage
-----

    python scripts/run_ablation_ladder.py \
        --asset-returns data/processed/asset_returns.csv \
        --oos-start 2016-07-31 \
        --full-model-returns results_runpod/walkforward/port_returns.csv \
        --out results_runpod/ablation_ladder/

Outputs
-------

* ``<out>/ladder_returns.csv``  — wide frame, one column per rung.
* ``<out>/ladder_metrics.csv``  — annualised Sharpe, MDD, DSR, PSR,
  Reality Check p-value per rung.
* ``<out>/ladder_bootstrap.csv`` — bootstrap MDD distributions.

Missing asset-return or full-model-returns files degrade gracefully: the
rung is skipped and logged; the rest of the ladder still runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src/ importable without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats import (
    deflated_sharpe_ratio,
    path_bootstrap_mdd,
    reality_check,
    sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Rung implementations
# ---------------------------------------------------------------------------


def rung_equal_weight(returns: pd.DataFrame) -> pd.Series:
    """Rebalance to 1/N each period."""
    w = np.full(returns.shape[1], 1.0 / returns.shape[1])
    return pd.Series(returns.values @ w, index=returns.index, name="equal_weight")


def rung_risk_parity(returns: pd.DataFrame, window: int = 24) -> pd.Series:
    """Inverse-volatility weights from a rolling window."""
    vol = returns.rolling(window).std().shift(1)
    inv = 1.0 / (vol + 1e-8)
    w = inv.div(inv.sum(axis=1), axis=0)
    port = (w * returns).sum(axis=1)
    port.name = "risk_parity"
    return port.iloc[window:]


def rung_min_cvar(
    returns: pd.DataFrame, window: int = 36, alpha: float = 0.95
) -> pd.Series:
    """Empirical CVaR minimisation on a rolling window.

    Uses the Rockafellar-Uryasev LP on *realised* returns — no simulator.
    Requires cvxpy. If cvxpy is unavailable the rung returns NaN.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return pd.Series(dtype=float, name="min_cvar")

    n = returns.shape[1]
    out = []
    idx = []
    for i in range(window, len(returns)):
        R = returns.iloc[i - window : i].values  # (T, N)
        w = cp.Variable(n, nonneg=True)
        a = cp.Variable()
        u = cp.Variable(window, nonneg=True)
        loss = -R @ w
        cvar = a + (1.0 / (window * (1.0 - alpha))) * cp.sum(u)
        prob = cp.Problem(
            cp.Minimize(cvar),
            [cp.sum(w) == 1, u >= loss - a],
        )
        try:
            prob.solve(solver=cp.ECOS)
            w_val = np.array(w.value).ravel()
            if not np.isfinite(w_val).all():
                raise RuntimeError("non-finite w")
        except Exception:
            w_val = np.full(n, 1.0 / n)
        out.append(float(returns.iloc[i].values @ w_val))
        idx.append(returns.index[i])
    return pd.Series(out, index=idx, name="min_cvar")


def load_series(path: Path, name: str) -> pd.Series:
    """Load a one-column return CSV with a Date index; return a named Series."""
    if not path.exists():
        return pd.Series(dtype=float, name=name)
    df = pd.read_csv(path)
    # Heuristic: first column = date, second = returns.
    date_col = df.columns[0]
    ret_col = [c for c in df.columns if c != date_col][0]
    s = pd.Series(
        df[ret_col].astype(float).values,
        index=pd.to_datetime(df[date_col]),
        name=name,
    )
    return s


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(series: pd.Series, n_trials: int = 1) -> dict:
    r = series.dropna().values
    if len(r) < 12:
        return {"n_obs": len(r)}
    dsr = deflated_sharpe_ratio(r, n_trials=n_trials)
    pb = path_bootstrap_mdd(r, threshold=-0.10, n_bootstrap=2000, seed=0)
    return {
        "n_obs": len(r),
        "sharpe_ann": sharpe_ratio(r, periods_per_year=12),
        "mdd": float(
            (1.0 - np.cumprod(1 + r) / np.maximum.accumulate(np.cumprod(1 + r))).max()
        )
        * -1.0,
        "psr_vs_zero": dsr.psr,
        "dsr_vs_tournament": dsr.dsr,
        "sr_star": dsr.sr_star,
        "bootstrap_p_above_-10": pb.prob_above,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-returns", type=Path, required=True,
                   help="Wide CSV: Date, asset1, asset2, ... (per-period returns).")
    p.add_argument("--oos-start", type=str, default="2016-07-31")
    p.add_argument("--full-model-returns", type=Path, default=None,
                   help="Optional CSV with the thesis model's OOS port returns.")
    p.add_argument("--bl-only-returns", type=Path, default=None)
    p.add_argument("--bl-cvar-noregime-returns", type=Path, default=None)
    p.add_argument("--n-tournament-trials", type=int, default=1,
                   help="N for DSR; set to the tournament size.")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.asset_returns.exists():
        print(f"[fatal] asset returns not found: {args.asset_returns}",
              file=sys.stderr)
        return 2

    ar = pd.read_csv(args.asset_returns)
    ar.iloc[:, 0] = pd.to_datetime(ar.iloc[:, 0])
    ar = ar.set_index(ar.columns[0]).sort_index()
    oos_mask = ar.index >= pd.Timestamp(args.oos_start)

    # Rung 1-3 are computed from asset returns directly, then truncated to OOS.
    ew = rung_equal_weight(ar).loc[oos_mask]
    rp = rung_risk_parity(ar)
    rp = rp[rp.index >= pd.Timestamp(args.oos_start)]
    mc = rung_min_cvar(ar)
    mc = mc[mc.index >= pd.Timestamp(args.oos_start)] if not mc.empty else mc

    # Rungs 4, 5, 6 come from external CSVs.
    bl_only = load_series(args.bl_only_returns, "bl_only") \
        if args.bl_only_returns else pd.Series(dtype=float, name="bl_only")
    bl_cvar = load_series(args.bl_cvar_noregime_returns, "bl_cvar_noregime") \
        if args.bl_cvar_noregime_returns else pd.Series(dtype=float,
                                                        name="bl_cvar_noregime")
    full = load_series(args.full_model_returns, "full_model") \
        if args.full_model_returns else pd.Series(dtype=float, name="full_model")

    all_series = [ew, rp, mc, bl_only, bl_cvar, full]
    all_series = [s for s in all_series if not s.empty]
    if not all_series:
        print("[fatal] no rungs produced any data", file=sys.stderr)
        return 3

    returns_df = pd.concat(all_series, axis=1).sort_index()
    returns_df.to_csv(args.out / "ladder_returns.csv")

    metrics = {
        col: score(returns_df[col], n_trials=args.n_tournament_trials)
        for col in returns_df.columns
    }
    pd.DataFrame(metrics).T.to_csv(args.out / "ladder_metrics.csv")

    with open(args.out / "meta.json", "w") as f:
        json.dump(
            {
                "oos_start": args.oos_start,
                "n_trials_for_dsr": args.n_tournament_trials,
                "rungs_present": list(returns_df.columns),
            },
            f,
            indent=2,
        )

    print(f"[ok] wrote {len(returns_df.columns)} rungs to {args.out}")
    print(pd.DataFrame(metrics).T.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
