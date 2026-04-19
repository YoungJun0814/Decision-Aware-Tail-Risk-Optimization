"""
Point-in-Time 4-State Regime Generator (P0.1)
=============================================

Produces `data/processed/regime_4state_pit.csv`, a regime-probability time
series where **each row is conditioned only on information available up to
that row's date**.

Why this file exists
--------------------
The non-PIT generator ``src.gen_regime_4state`` fits a Gaussian HMM on the
*entire* 2007–2025 sample. Because Baum–Welch is a smoothing algorithm, a
regime label assigned to e.g. 2008-10 is influenced by observations from
2020. Using that label in walk-forward training leaks future information
into every upstream decision. Any headline metric built on top of it is
therefore biased.

Strategy
--------
1.  **Pre-OOS period** (`features.index < start_oos`): a single HMM fit on
    those features. This is the first walk-forward training window; nothing
    in the OOS period is visible. The entire pre-OOS row block is written
    from this single fit, which is equivalent to the walk-forward harness
    seeing the whole train window at once — no leakage relative to OOS.
2.  **OOS period** (`features.index >= start_oos`): for every month `t` we
    re-fit the hierarchical HMM on features sliced to ``features.index <= t``
    and keep only the last row (time `t`) of the resulting posterior. This
    yields a posterior for time `t` that uses exactly the information
    knowable at time `t`.
3.  Causal exponential smoothing (``adjust=False``) is applied after each
    fit. Because smoothing is a running EWMA it never pulls future
    information backwards.

Label-switching is handled inside ``fit_level1_hmm`` / ``fit_level2_hmm``
by the existing ``argmax(state_means)`` rule — Bull / Sideways /
Correction / Crisis are identified by the sign and magnitude of returns
within each refit, which is stable across refits when the underlying
regimes are well-separated. A small residual risk of label drift is
acceptable because downstream loss functions use regime probabilities as
soft mixtures (not hard argmax).

Output schema
-------------
Index:       Date (month-end)
Columns:     Prob_Bull, Prob_Sideways, Prob_Correction, Prob_Crisis
             fit_end_date, n_train_months, source

``fit_end_date`` should equal the Date for OOS rows — easy audit.
``source`` ∈ {"pre_oos_single_fit", "oos_per_month_refit"}.

Usage
-----
    python -m src.gen_regime_4state_pit \
        --start-date 2007-01-01 \
        --end-date 2025-06-01 \
        --start-oos 2016-07-01
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.gen_regime_4state import (
    HMM_AVAILABLE,
    generate_4state_probs,
    get_regime_features,
    smooth_probs,
)

warnings.filterwarnings("ignore")


DEFAULT_OUTPUT = (
    Path(__file__).parent.parent / "data" / "processed" / "regime_4state_pit.csv"
)


def generate_pit_probs(
    features: pd.DataFrame,
    start_oos: str = "2016-07-01",
    min_train_months: int = 60,
    smoothing_alpha: float = 0.3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate point-in-time 4-state regime probabilities.

    Parameters
    ----------
    features : DataFrame
        Output of ``get_regime_features`` — expects columns
        ``Returns``, ``RollingVol``, ``VIX`` indexed by month-end.
    start_oos : str
        First OOS month. All dates strictly before this use a single
        train-only fit; all dates from here onward are per-month refits.
    min_train_months : int
        Guard against degenerate short windows in OOS per-month refits.
    smoothing_alpha : float
        Causal EWMA alpha applied to each refit's posterior.

    Returns
    -------
    DataFrame indexed by Date with four probability columns plus audit
    metadata columns.
    """
    if not HMM_AVAILABLE:
        raise ImportError(
            "hmmlearn is required for PIT regime generation. "
            "pip install hmmlearn"
        )

    start_oos_ts = pd.Timestamp(start_oos)
    rows = []

    # ------------------------------------------------------------------
    # 1) Pre-OOS: single fit on features strictly before start_oos.
    # ------------------------------------------------------------------
    pre_mask = features.index < start_oos_ts
    pre_features = features.loc[pre_mask]
    if len(pre_features) < min_train_months:
        raise ValueError(
            f"Pre-OOS window has only {len(pre_features)} months; need "
            f">= {min_train_months}. Extend start_date or lower start_oos."
        )
    if verbose:
        print(
            f"[PIT] Pre-OOS single fit on {len(pre_features)} months "
            f"({pre_features.index[0].date()} -> {pre_features.index[-1].date()})"
        )
    pre_probs = generate_4state_probs(pre_features)
    pre_smoothed = smooth_probs(pre_probs, alpha=smoothing_alpha)
    pre_fit_end = pre_features.index[-1]
    for date, row in pre_smoothed.iterrows():
        rows.append(
            {
                "Date": date,
                "Prob_Bull": row["Prob_Bull"],
                "Prob_Sideways": row["Prob_Sideways"],
                "Prob_Correction": row["Prob_Correction"],
                "Prob_Crisis": row["Prob_Crisis"],
                "fit_end_date": pre_fit_end,
                "n_train_months": len(pre_features),
                "source": "pre_oos_single_fit",
            }
        )

    # ------------------------------------------------------------------
    # 2) OOS: refit per month on features[:t] and keep last row only.
    # ------------------------------------------------------------------
    oos_dates = features.index[features.index >= start_oos_ts]
    if verbose:
        print(f"[PIT] OOS per-month refit for {len(oos_dates)} months")

    for i, t in enumerate(oos_dates, 1):
        sub = features.loc[features.index <= t]
        if len(sub) < min_train_months:
            if verbose:
                print(
                    f"  [skip] {t.date()}: only {len(sub)} months "
                    f"< min_train_months={min_train_months}"
                )
            continue

        # Silence the hierarchical generator during per-month refits to
        # keep the log readable.
        with _suppress_hmm_prints():
            probs = generate_4state_probs(sub)
            smoothed = smooth_probs(probs, alpha=smoothing_alpha)

        last = smoothed.iloc[-1]
        assert smoothed.index[-1] == t, (
            f"Last row date {smoothed.index[-1]} != refit date {t}"
        )
        rows.append(
            {
                "Date": t,
                "Prob_Bull": float(last["Prob_Bull"]),
                "Prob_Sideways": float(last["Prob_Sideways"]),
                "Prob_Correction": float(last["Prob_Correction"]),
                "Prob_Crisis": float(last["Prob_Crisis"]),
                "fit_end_date": t,
                "n_train_months": len(sub),
                "source": "oos_per_month_refit",
            }
        )
        if verbose and (i % 12 == 0 or i == len(oos_dates)):
            print(
                f"  [{i:3d}/{len(oos_dates)}] {t.date()}  "
                f"Bull={last['Prob_Bull']:.2f} "
                f"Side={last['Prob_Sideways']:.2f} "
                f"Corr={last['Prob_Correction']:.2f} "
                f"Crisis={last['Prob_Crisis']:.2f}"
            )

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    df.index.name = "Date"
    return df


class _suppress_hmm_prints:
    """Context manager that silences stdout from noisy HMM refits."""

    def __enter__(self):
        import io
        import sys

        self._stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, exc_type, exc, tb):
        import sys

        sys.stdout = self._stdout
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2007-01-01")
    parser.add_argument("--end-date", default="2025-06-01")
    parser.add_argument("--start-oos", default="2016-07-01")
    parser.add_argument("--min-train-months", type=int, default=60)
    parser.add_argument("--smoothing-alpha", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--features-cache", type=Path, default=None,
        help="Optional pre-downloaded feature CSV (Date, Returns, RollingVol, VIX).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Point-in-Time 4-State Regime Generator")
    print("=" * 70)
    print(f"  start_date:       {args.start_date}")
    print(f"  end_date:         {args.end_date}")
    print(f"  start_oos:        {args.start_oos}")
    print(f"  min_train_months: {args.min_train_months}")
    print(f"  smoothing_alpha:  {args.smoothing_alpha}")
    print(f"  output:           {args.output}")

    if args.features_cache and args.features_cache.exists():
        features = pd.read_csv(
            args.features_cache, parse_dates=["Date"], index_col="Date"
        )
        print(f"[INFO] Loaded features from cache: {features.shape}")
    else:
        features = get_regime_features(args.start_date, args.end_date)

    result = generate_pit_probs(
        features,
        start_oos=args.start_oos,
        min_train_months=args.min_train_months,
        smoothing_alpha=args.smoothing_alpha,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output)
    print(f"\n[SAVED] {args.output}")
    print(f"  Shape: {result.shape}")
    print(f"  Pre-OOS rows: {(result['source'] == 'pre_oos_single_fit').sum()}")
    print(f"  OOS  rows:    {(result['source'] == 'oos_per_month_refit').sum()}")


if __name__ == "__main__":
    main()
