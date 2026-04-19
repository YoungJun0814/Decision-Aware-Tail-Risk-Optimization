"""
Post-Selection Report
=====================

Takes an already-run portfolio's OOS return series and (optionally) the
tournament-candidate return panel, and emits the post-selection
statistics that the thesis needs to defend:

  * Sharpe / annualised Sharpe
  * Probabilistic Sharpe Ratio (PSR vs 0)
  * Deflated Sharpe Ratio (PSR vs the selection-inflated null for a
    tournament of N trials)
  * White's Reality Check p-value (if the candidate panel is given)
  * Path-conditional bootstrap P(MDD >= -10%)
  * Tournament bootstrap P(MDD >= -10%) (if the candidate panel is given)

Inputs
------
--port-returns        CSV with [Date, return] — the selected strategy.
--candidates          Optional CSV with Date as first column and one
                      column per tournament candidate's OOS return.
--n-trials            Tournament size N used to compute the DSR null
                      when --candidates is absent. When --candidates is
                      provided, K = candidates.shape[1] is used instead.
--threshold           MDD floor (default -0.10).
--out                 Output directory; writes report.json + report.md.

Usage
-----
    python scripts/report_post_selection.py \
        --port-returns results/walkforward/port_returns.csv \
        --candidates   results/tournament/candidates.csv \
        --threshold    -0.10 \
        --out          results/walkforward/post_selection/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats import (
    deflated_sharpe_ratio,
    path_bootstrap_mdd,
    probabilistic_sharpe_ratio,
    reality_check,
    sharpe_ratio,
    tournament_bootstrap_mdd,
)


def _load_port(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    date_col = df.columns[0]
    ret_col = [c for c in df.columns if c != date_col][0]
    return pd.Series(
        df[ret_col].astype(float).values,
        index=pd.to_datetime(df[date_col]),
        name="port",
    ).sort_index()


def _load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df.astype(float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port-returns", type=Path, required=True)
    p.add_argument("--candidates", type=Path, default=None)
    p.add_argument("--n-trials", type=int, default=1,
                   help="Tournament size for DSR when --candidates is omitted.")
    p.add_argument("--threshold", type=float, default=-0.10)
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--block-length", type=float, default=6.0)
    p.add_argument("--periods-per-year", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    port = _load_port(args.port_returns)
    r = port.dropna().values
    if len(r) < 24:
        print(f"[fatal] port series too short: {len(r)} < 24", file=sys.stderr)
        return 2

    cand_df = _load_candidates(args.candidates) if args.candidates else None
    n_trials = cand_df.shape[1] if cand_df is not None else args.n_trials

    report = {
        "n_obs": len(r),
        "period_start": str(port.index.min().date()),
        "period_end": str(port.index.max().date()),
        "n_trials_for_dsr": int(n_trials),
    }

    # Sharpe & PSR & DSR.
    report["sharpe_per_period"] = sharpe_ratio(r)
    report["sharpe_annualised"] = sharpe_ratio(r, periods_per_year=args.periods_per_year)
    report["psr_vs_zero"] = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    dsr = deflated_sharpe_ratio(r, n_trials=int(n_trials))
    report["dsr"] = {
        "value": dsr.dsr,
        "sr_star_per_period": dsr.sr_star,
        "sr_observed_per_period": dsr.sr,
        "skew": dsr.skew,
        "excess_kurt": dsr.excess_kurt,
        "reject_null_at_alpha=0.05": dsr.reject_null(0.05),
    }

    # Path bootstrap.
    pb = path_bootstrap_mdd(
        r, threshold=args.threshold, n_bootstrap=args.n_bootstrap,
        block_length=args.block_length, seed=args.seed,
    )
    report["path_bootstrap"] = {
        "prob_mdd_above_threshold": pb.prob_above,
        "mdd_mean": pb.mdd_mean,
        "mdd_quantiles": pb.mdd_quantiles,
        "threshold": args.threshold,
    }

    # Candidate-panel-only analyses.
    if cand_df is not None:
        # Align candidates to port index — use their common intersection.
        common = cand_df.index.intersection(port.index)
        if len(common) < 24:
            print("[warn] candidate panel has <24 overlapping dates; skipping.")
        else:
            X = cand_df.loc[common].values  # (T, K)
            rc = reality_check(
                X, n_bootstrap=args.n_bootstrap,
                block_length=args.block_length, seed=args.seed,
            )
            report["reality_check"] = {
                "p_value": rc.p_value,
                "best_candidate_mean": rc.best_statistic,
                "n_candidates": rc.n_candidates,
            }
            tb = tournament_bootstrap_mdd(
                X, threshold=args.threshold, n_bootstrap=args.n_bootstrap,
                block_length=args.block_length, selection="bayesian",
                seed=args.seed,
            )
            report["tournament_bootstrap"] = {
                "prob_mdd_above_threshold": tb.prob_above,
                "mdd_mean": tb.mdd_mean,
                "mdd_quantiles": tb.mdd_quantiles,
                "selection": tb.selection,
            }

    # --- Write outputs ------------------------------------------------------
    (args.out / "report.json").write_text(json.dumps(report, indent=2))

    md = []
    md.append(f"# Post-Selection Report\n")
    md.append(f"- observations: **{report['n_obs']}**  "
              f"({report['period_start']} to {report['period_end']})")
    md.append(f"- Sharpe (annualised, p/y={args.periods_per_year}): **"
              f"{report['sharpe_annualised']:.4f}**")
    md.append(f"- PSR(SR > 0): **{report['psr_vs_zero']:.4f}**")
    md.append(f"- DSR (N={n_trials}): **{report['dsr']['value']:.4f}**  "
              f"(SR*={report['dsr']['sr_star_per_period']:.4f}/period)")
    md.append(f"- Path bootstrap P(MDD ≥ {args.threshold:.2f}): **"
              f"{report['path_bootstrap']['prob_mdd_above_threshold']:.4f}**")
    if "reality_check" in report:
        md.append(f"- Reality Check p-value: **{report['reality_check']['p_value']:.4f}**")
        md.append(f"- Tournament bootstrap P(MDD ≥ {args.threshold:.2f}): **"
                  f"{report['tournament_bootstrap']['prob_mdd_above_threshold']:.4f}**")
    md.append("")
    md.append("## Interpretation")
    md.append("- **DSR** close to 1.0 means the observed Sharpe is unlikely "
              "under the null that the best of N no-skill strategies was picked. "
              "Report this alongside the raw Sharpe.")
    md.append("- **Tournament bootstrap** vs **path bootstrap** difference "
              "measures how much of the path-bootstrap optimism is driven by "
              "strategy-selection uncertainty.")
    (args.out / "report.md").write_text("\n".join(md))

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
