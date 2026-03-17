"""
Final analysis for thesis: sub-period validation, drawdown profile, guard precision.
"""
import sys, importlib.util, os, math
from pathlib import Path
import numpy as np, pandas as pd

os.environ["PHASE17_STEP1_DIR"] = "results_runpod/phase17/step1_2025_repro_onepass_regimefix_v1"
ROOT = Path("/workspace")

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

psa = load_module("psa", ROOT / "scripts" / "run_phase18_paper_safe_ablation.py")
bench = load_module("bench", ROOT / "scripts" / "run_phase18_nonleveraged_v2_benchmark.py")
v7 = load_module("v7", ROOT / "scripts" / "run_correlation_guard_v7_exclusion.py")

ctx = psa.build_context(out_dir=ROOT / "results_runpod" / "verify_triple", refresh_data=False)
guard_signals = v7.compute_guard_signals(ctx.daily_returns, ctx.month_ends)

AUDITED = {
    "mix": {"base_mix": 0.5, "high_mix": 0.75, "high_threshold": 0.82, "lookback": 3, "mode": "hard", "tie_tol": 0.002},
    "s1": {"base_target_vol": 0.1, "bull_scale": 1.2, "correction_alpha": 0.5, "crisis_scale": 0.7, "lookback": 3, "max_leverage": 1.6},
    "s3": {"high": 1.45, "low": 0.92, "mode": "smooth", "sim_high": 0.97, "sim_low": 0.78},
    "s7": {"allow_leverage": False, "dd0": 1.1, "dd3": 1.0, "dd6": 0.8, "dd9": 0.65, "gross_cap": 1.0},
    "sleeve": {"cash_bull": 0.0, "cash_correction": 0.0, "cash_crisis": 0.02, "cash_sideways": 0.0, "confidence_cash_relief": 0.02, "sim_high": 0.9},
    "srb": {"bull_mult": 1.15, "correction_alpha": 0.5, "crisis_mult": 0.5},
    "stop": {"bull_hard_scale": 1.1, "bull_soft_scale": 1.2, "crisis_hard_scale": 0.8, "crisis_soft_scale": 0.85, "hard_cash": 0.05, "hard_growth_frac": 0.2, "hard_sl_thresh": -0.035, "reentry_fraction": 0.7, "reentry_mode": "portfolio", "reentry_thresh": 0.008, "sideways_hard_scale": 1.0, "sideways_soft_scale": 1.0, "soft_cash": 0.0, "soft_growth_frac": 0.55, "soft_sl_thresh": -0.02},
    "tilt": {"bull_sim_threshold": 0.8, "bull_strength": 0.2, "crisis_def_strength": 0.2},
}

# Best: corr<=-0.25, BIL=0%, s7+stop+vol (target_vol=0.09)
best_mods = {"s7": {"dd0": 1.5, "dd3": 1.1}, "stop": {"soft_sl_thresh": -0.02, "hard_sl_thresh": -0.03}, "s1": {"base_target_vol": 0.09}}
best_rets, best_wt, best_diag, best_guard = v7.simulate_with_exclusion_guard(
    ctx, cfg=AUDITED, guard_signals=guard_signals,
    corr_thresh=-0.25, vol_thresh=99.0,
    base_bil=0.0, exclude_assets=["TLT", "IEF", "TIP"],
    mode="corr_only", trade_cost_bps=0.0,
    label="best_config", overlay_mods=best_mods,
)

# Baseline: no guard, no mods
base_rets, _, base_diag, _ = v7.simulate_with_exclusion_guard(
    ctx, cfg=AUDITED, guard_signals=guard_signals,
    corr_thresh=99.0, vol_thresh=99.0,
    base_bil=0.0, exclude_assets=["TLT", "IEF", "TIP"],
    mode="corr_only", trade_cost_bps=0.0,
    label="baseline",
)

# Conservative config
mid_mods = {"s7": {"dd0": 1.5, "dd3": 1.1}, "stop": {"soft_sl_thresh": -0.02, "hard_sl_thresh": -0.03}, "s1": {"base_target_vol": 0.09}}
mid_rets, _, mid_diag, _ = v7.simulate_with_exclusion_guard(
    ctx, cfg=AUDITED, guard_signals=guard_signals,
    corr_thresh=-0.10, vol_thresh=99.0,
    base_bil=0.05, exclude_assets=["TLT", "IEF", "TIP"],
    mode="corr_only", trade_cost_bps=0.0,
    label="mid_config", overlay_mods=mid_mods,
)

# 1. Full-period metrics
print("=" * 70)
print("1. FULL-PERIOD METRICS (2016-07 to 2025-12)")
print("=" * 70)
configs = {
    "Baseline (no guard)": base_rets,
    "Best (corr<=-0.25, s7+stop+vol)": best_rets,
    "Conservative (corr<=-0.10, bil=5%)": mid_rets,
}
print("%-40s %7s %8s %8s %7s %6s" % ("Config", "Sharpe", "Return", "MDD", "Calmar", "Triple"))
for name, rets in configs.items():
    m = psa.evaluate_returns(rets, name)
    t = "YES" if m["triple"] else "no"
    print("%-40s %7.4f %7.2f%% %7.2f%% %7.4f %6s" % (name, m["sharpe"], m["return"]*100, m["mdd"]*100, m["calmar"], t))

# 2. Sub-period analysis
print("\n" + "=" * 70)
print("2. SUB-PERIOD ANALYSIS")
print("=" * 70)

periods = {
    "Pre-COVID (2016-2019)": (None, "2019-12-31"),
    "COVID (2020)": ("2020-01-31", "2020-12-31"),
    "Post-COVID (2021-2022)": ("2021-01-31", "2022-12-31"),
    "Recent (2023-2025)": ("2023-01-31", None),
    "2025 extension": ("2025-01-31", None),
}

for pname, (start, end) in periods.items():
    print("\n--- %s ---" % pname)
    print("%-40s %7s %8s %8s" % ("Config", "Sharpe", "Return", "MDD"))
    for cname, rets in configs.items():
        sub = rets.copy()
        if start:
            sub = sub[sub.index >= start]
        if end:
            sub = sub[sub.index <= end]
        if len(sub) < 2:
            continue
        m = psa.evaluate_returns(sub, cname)
        print("%-40s %7.4f %7.2f%% %7.2f%%" % (cname, m["sharpe"], m["return"]*100, m["mdd"]*100))

# 3. Guard precision
print("\n" + "=" * 70)
print("3. GUARD PRECISION ANALYSIS (corr <= -0.25)")
print("=" * 70)

guard_on = best_diag[best_diag["guard_active"]]
guard_off = best_diag[~best_diag["guard_active"]]
diff = best_rets - base_rets
good_catch = diff[guard_on.index]
good_catch_positive = good_catch[good_catch > 0.005]
good_catch_neutral = good_catch[(good_catch >= -0.005) & (good_catch <= 0.005)]
good_catch_negative = good_catch[good_catch < -0.005]

print("  Guard active months: %d/%d (%.1f%%)" % (len(guard_on), len(best_diag), len(guard_on)/len(best_diag)*100))
print("  Guard OFF months: %d" % len(guard_off))
print()
print("  When guard is ON:")
print("    Helped (>0.5%% better):  %d months" % len(good_catch_positive))
print("    Neutral (+/-0.5%%):      %d months" % len(good_catch_neutral))
print("    Hurt (<-0.5%% worse):   %d months" % len(good_catch_negative))
print()
print("  Avg delta (guard ON):  %+.3f%%" % (good_catch.mean()*100))
print("  Avg return (guard ON): %+.3f%%" % (guard_on["monthly_return"].mean()*100))
print("  Avg return (guard OFF):%+.3f%%" % (guard_off["monthly_return"].mean()*100))

# 4. Drawdown profile
print("\n" + "=" * 70)
print("4. DRAWDOWN PROFILE")
print("=" * 70)

for cname, rets in [("Baseline", base_rets), ("Best", best_rets)]:
    wealth = (1 + rets).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    mdd_end = dd.idxmin()
    mdd_val = dd.min()
    mdd_peak_idx = wealth[:mdd_end].idxmax()
    recovery = wealth[mdd_end:]
    recovery_date = recovery[recovery >= wealth[mdd_peak_idx]].index
    rec_str = recovery_date[0].strftime("%Y-%m") if len(recovery_date) > 0 else "Not recovered"

    print("\n  %s:" % cname)
    print("    MDD: %.2f%%" % (mdd_val * 100))
    print("    Peak: %s, Trough: %s, Recovery: %s" % (
        mdd_peak_idx.strftime("%Y-%m"), mdd_end.strftime("%Y-%m"), rec_str))

    print("    Top 5 worst months:")
    worst = rets.nsmallest(5)
    for dt, r in worst.items():
        ga = "GUARD" if dt in guard_on.index else "     "
        print("      %s %s: %+.2f%%" % (dt.strftime("%Y-%m"), ga, r*100))

# 5. Annual returns
print("\n" + "=" * 70)
print("5. ANNUAL RETURNS")
print("=" * 70)
print("%-6s %12s %12s %8s" % ("Year", "Baseline", "Best", "Delta"))
for year in range(2016, 2026):
    base_yr_data = base_rets[base_rets.index.year == year]
    best_yr_data = best_rets[best_rets.index.year == year]
    if len(base_yr_data) > 0 and len(best_yr_data) > 0:
        base_yr = (1 + base_yr_data).prod() - 1
        best_yr = (1 + best_yr_data).prod() - 1
        print("%-6d %11.2f%% %11.2f%% %+7.2f%%" % (year, base_yr*100, best_yr*100, (best_yr-base_yr)*100))

# 6. Weight allocation
print("\n" + "=" * 70)
print("6. AVERAGE WEIGHT ALLOCATION (guard ON vs OFF)")
print("=" * 70)

guard_on_months = best_diag[best_diag["guard_active"]].index
guard_off_months = best_diag[~best_diag["guard_active"]].index

wt_on = best_wt[best_wt.index.isin(guard_on_months)]
wt_off = best_wt[best_wt.index.isin(guard_off_months)]

if len(wt_on) > 0 and len(wt_off) > 0:
    print("\n  %-8s %8s %8s %8s" % ("Asset", "Guard ON", "Guard OFF", "Delta"))
    for col in sorted(best_wt.columns):
        on_avg = wt_on[col].mean() * 100
        off_avg = wt_off[col].mean() * 100
        print("  %-8s %7.2f%% %7.2f%% %+7.2f%%" % (col, on_avg, off_avg, on_avg - off_avg))

print("\nDone.")
