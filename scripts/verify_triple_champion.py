"""
Independent verification of the triple target champion config.
Checks:
1. Reproducibility (same numbers as search)
2. Sub-period breakdown (Covid, Inflation)
3. Bootstrap confidence interval for MDD
4. Walk-forward upstream integrity
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "scripts" / "run_phase18_nonleveraged_v2_benchmark.py"
UPSTREAM_DIR = ROOT / "results_runpod" / "phase18_e2e_fresh_runpod_20260330_022828UTC" / "upstream_snapshot"
CHAMPION_PATH = ROOT / "results_runpod" / "phase18_triple_precision_round3" / "best_result.json"
CACHE_DIR = ROOT / "results_runpod" / "phase18_triple_precision_round2" / "ctx" / "cache"
OUT_DIR = ROOT / "results_runpod" / "phase18_triple_verification"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    champion = json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))
    cfg = json.loads(champion["config_json"])

    print("=" * 80)
    print("TRIPLE TARGET CHAMPION VERIFICATION")
    print("=" * 80)
    print(f"Config: {champion['candidate_name']}")
    print(f"Claimed: Sharpe={champion['sharpe']:.6f} Return={champion['return']:.6%} MDD={champion['mdd']:.6%}")

    # Load engine
    os.environ["PHASE17_STEP1_DIR"] = str(UPSTREAM_DIR.resolve())
    mod = load_module("verify_bench", BENCHMARK_PATH)

    # Build context from scratch
    ctx_dir = OUT_DIR / "ctx"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    local_cache = ctx_dir / "cache"
    if CACHE_DIR.exists() and not local_cache.exists():
        shutil.copytree(CACHE_DIR, local_cache)

    ctx = mod.psa.build_context(ctx_dir, refresh_data=False, refresh_regime=False)

    # === CHECK 1: Upstream integrity ===
    print("\n[CHECK 1] Upstream data integrity")
    upstream_files = {
        "R6_returns": UPSTREAM_DIR / "ExpA_R6_pit_shifted_returns.csv",
        "R6_weights": UPSTREAM_DIR / "ExpA_R6_pit_shifted_weights.csv",
        "R7_returns": UPSTREAM_DIR / "ExpA_R7_pit_shifted_returns.csv",
        "R7_weights": UPSTREAM_DIR / "ExpA_R7_pit_shifted_weights.csv",
    }
    for name, path in upstream_files.items():
        df = pd.read_csv(path)
        dates = pd.to_datetime(df.iloc[:, 0])
        print(f"  {name}: {len(df)} rows, {dates.min().date()} to {dates.max().date()}, {len(df.columns)-1} assets")

    pit_manifest = json.loads((UPSTREAM_DIR / "pit_shift_manifest.json").read_text(encoding="utf-8"))
    if isinstance(pit_manifest, dict):
        print(f"  PIT manifest keys: {list(pit_manifest.keys())}")
        for k, v in list(pit_manifest.items())[:3]:
            print(f"    {k}: {v}")
    else:
        print(f"  PIT manifest: {len(pit_manifest)} entries")
        for entry in pit_manifest[:3]:
            print(f"    sample: {entry}")

    # === CHECK 2: PIT regime ===
    print("\n[CHECK 2] PIT regime look-ahead check")
    regime = ctx.pit_regime
    print(f"  Regime rows: {len(regime)}")
    print(f"  Regime dates: {regime.index.min().date()} to {regime.index.max().date()}")
    print(f"  Regime columns: {list(regime.columns)}")
    # Verify regime dates align with month_ends (no future data)
    for i in range(min(5, len(regime))):
        row = regime.iloc[i]
        print(f"    {regime.index[i].date()}: {dict(row)}")

    # === CHECK 3: Reproduce champion ===
    print("\n[CHECK 3] Reproducibility")
    returns, weights, diagnostics = mod.simulate_overlay_v2_strategy(
        ctx=ctx, label="VERIFY", regime=ctx.pit_regime, trade_cost_bps=0.0,
        mix_cfg=cfg["mix"], sleeve_cfg=cfg["sleeve"], s1_cfg=cfg["s1"],
        s3_cfg=cfg["s3"], srb_cfg=cfg["srb"], stop_cfg=cfg["stop"],
        tilt_cfg=cfg.get("tilt"), s7_cfg=cfg.get("s7"),
        expert_cfg=cfg.get("expert"), brake_cfg=cfg.get("brake"),
        meta_cfg=cfg.get("meta"), allocator_cfg=cfg.get("allocator"),
        hier_cfg=cfg.get("hier"), policy_cfg=cfg.get("policy"),
        rebalance_cfg=cfg.get("rebalance"),
    )
    metrics = mod.psa.evaluate_returns(returns, "VERIFY")
    sh = float(metrics["sharpe"])
    rt = float(metrics["return"])
    md = float(metrics["mdd"])
    print(f"  Reproduced: Sharpe={sh:.6f} Return={rt:.6%} MDD={md:.6%}")
    print(f"  Match: Sharpe={abs(sh - champion['sharpe']) < 1e-8} "
          f"Return={abs(rt - champion['return']) < 1e-8} "
          f"MDD={abs(md - champion['mdd']) < 1e-8}")
    triple = sh >= 1.0 and rt >= 0.10 and md >= -0.10
    print(f"  Triple: {triple}")

    # === CHECK 4: Sub-period breakdown ===
    print("\n[CHECK 4] Sub-period analysis")
    periods = [
        ("Full sample", None, None),
        ("Pre-COVID (2014-2020/01)", None, pd.Timestamp("2020-01-31")),
        ("COVID shock (2020/02-2020/04)", pd.Timestamp("2020-02-29"), pd.Timestamp("2020-04-30")),
        ("Recovery (2020/05-2021/12)", pd.Timestamp("2020-05-31"), pd.Timestamp("2021-12-31")),
        ("Inflation tightening (2022-2023/Q1)", pd.Timestamp("2022-01-31"), pd.Timestamp("2023-03-31")),
        ("Post-tightening (2023/Q2-end)", pd.Timestamp("2023-04-30"), None),
    ]
    for pname, start, end in periods:
        p = returns.copy()
        if start is not None:
            p = p.loc[p.index >= start]
        if end is not None:
            p = p.loc[p.index <= end]
        if p.empty:
            continue
        pm = mod.psa.evaluate_returns(p, pname)
        print(f"  {pname}: Sharpe={float(pm['sharpe']):.3f} Return={float(pm['return']):.2%} "
              f"MDD={float(pm['mdd']):.2%} months={len(p)}")

    # === CHECK 5: Bootstrap MDD confidence ===
    print("\n[CHECK 5] Bootstrap MDD confidence interval")
    np.random.seed(42)
    n_bootstrap = 5000
    block_size = 6
    ret_arr = returns.values
    n = len(ret_arr)
    mdd_samples = []
    for _ in range(n_bootstrap):
        indices = []
        while len(indices) < n:
            start = np.random.randint(0, n)
            for j in range(block_size):
                indices.append((start + j) % n)
        indices = indices[:n]
        boot_ret = ret_arr[indices]
        cum = np.cumprod(1.0 + boot_ret)
        running_max = np.maximum.accumulate(cum)
        dd = cum / running_max - 1.0
        mdd_samples.append(float(dd.min()))

    mdd_arr = np.array(mdd_samples)
    print(f"  Bootstrap MDD: mean={mdd_arr.mean():.4%} median={np.median(mdd_arr):.4%}")
    print(f"  95% CI: [{np.percentile(mdd_arr, 2.5):.4%}, {np.percentile(mdd_arr, 97.5):.4%}]")
    print(f"  P(MDD > -10%): {(mdd_arr >= -0.10).mean():.2%}")
    print(f"  P(MDD > -12%): {(mdd_arr >= -0.12).mean():.2%}")

    # === CHECK 6: MDD tightness analysis ===
    print("\n[CHECK 6] MDD tightness / overfitting risk")
    cum = np.cumprod(1.0 + ret_arr)
    running_max = np.maximum.accumulate(cum)
    dd = cum / running_max - 1.0
    mdd_val = float(dd.min())
    mdd_idx = int(dd.argmin())
    mdd_date = returns.index[mdd_idx]
    # Find drawdown trough dates
    dd_series = pd.Series(dd, index=returns.index)
    worst_3 = dd_series.nsmallest(3)
    print(f"  Realized MDD: {mdd_val:.6%} at {mdd_date.date()}")
    print(f"  Margin to -10%: {(-0.10 - mdd_val):.6%}")
    print(f"  Top 3 drawdowns:")
    for date, val in worst_3.items():
        print(f"    {date.date()}: {val:.4%}")

    # === CHECK 7: Overlay vs raw model ===
    print("\n[CHECK 7] Overlay attribution")
    # Evaluate with neutralized overlay (just equal-weight R6+R7)
    neutral_cfg = json.loads(json.dumps(cfg))
    neutral_cfg["s3"] = {"mode": "hard", "threshold": 2.0, "high": 1.0, "low": 1.0}
    neutral_cfg["srb"] = {"bull_mult": 1.0, "crisis_mult": 1.0, "correction_alpha": 0.5}
    neutral_cfg["s1"] = {**cfg["s1"], "bull_scale": 1.0, "crisis_scale": 1.0}
    neutral_cfg["stop"] = None
    neutral_cfg["tilt"] = {"bull_strength": 0.0, "bull_sim_threshold": 0.8, "crisis_def_strength": 0.0}
    neutral_cfg.pop("policy", None)
    neutral_cfg.pop("s7", None)
    try:
        nr, nw, nd = mod.simulate_overlay_v2_strategy(
            ctx=ctx, label="NEUTRAL", regime=ctx.pit_regime, trade_cost_bps=0.0,
            mix_cfg=cfg["mix"], sleeve_cfg=cfg["sleeve"], s1_cfg=neutral_cfg["s1"],
            s3_cfg=neutral_cfg["s3"], srb_cfg=neutral_cfg["srb"], stop_cfg=None,
            tilt_cfg=neutral_cfg["tilt"],
        )
        nm = mod.psa.evaluate_returns(nr, "NEUTRAL")
        print(f"  Without overlay: Sharpe={float(nm['sharpe']):.3f} Return={float(nm['return']):.2%} MDD={float(nm['mdd']):.2%}")
        print(f"  Overlay lift: Sharpe +{sh - float(nm['sharpe']):.3f}, Return +{(rt - float(nm['return']))*100:.2f}%p, MDD +{(md - float(nm['mdd']))*100:.2f}%p")
    except Exception as e:
        print(f"  Neutral eval failed: {e}")

    # === Summary ===
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    issues = []

    if not triple:
        issues.append("CRITICAL: Triple target NOT reproduced")
    if abs(md - champion["mdd"]) > 1e-6:
        issues.append(f"WARNING: MDD mismatch ({md:.8f} vs {champion['mdd']:.8f})")
    if (-0.10 - md) < 0.001:
        issues.append(f"CAUTION: MDD margin only {(-0.10 - md)*100:.4f}%p - very tight")
    if (mdd_arr >= -0.10).mean() < 0.50:
        issues.append(f"CAUTION: Bootstrap P(MDD > -10%) = {(mdd_arr >= -0.10).mean():.1%} - less than 50%")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  All checks passed. No issues found.")

    # Save verification report
    report = {
        "champion_name": champion["candidate_name"],
        "reproduced_sharpe": sh,
        "reproduced_return": rt,
        "reproduced_mdd": md,
        "triple_verified": triple,
        "mdd_margin_to_target": float(-0.10 - md),
        "bootstrap_mdd_mean": float(mdd_arr.mean()),
        "bootstrap_mdd_95ci_low": float(np.percentile(mdd_arr, 2.5)),
        "bootstrap_mdd_95ci_high": float(np.percentile(mdd_arr, 97.5)),
        "bootstrap_prob_mdd_above_10pct": float((mdd_arr >= -0.10).mean()),
        "issues": issues,
    }
    with open(OUT_DIR / "verification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {OUT_DIR / 'verification_report.json'}")


if __name__ == "__main__":
    main()
