"""
Correlation Guard V8: Tiered Guard (BIL Floor + Conditional Exclusion)

Two-tier defense:
  Tier 1 (corr > thresh_1): BIL floor only (keeps bonds, mild defense)
  Tier 2 (corr > thresh_2): BIL floor + bond exclusion (strong defense)

This combines V1-V6's BIL floor (which preserved bond upside in mild months)
with V7's asset exclusion (which provides maximum protection in extreme months).

Also tests partial exclusion (reduce bonds by X% instead of 100%).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "scripts" / "run_phase18_paper_safe_ablation.py"
BENCHMARK_PATH = ROOT / "scripts" / "run_phase18_nonleveraged_v2_benchmark.py"
DEFAULT_OUT_DIR = ROOT / "results_runpod" / "correlation_guard_v8"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod

psa = _load_module("phase18_paper_safe_engine", ENGINE_PATH)
bench = _load_module("phase18_nonlev_v2_benchmark", BENCHMARK_PATH)

AUDITED_FINAL_CONFIG = {
    "mix": {"base_mix": 0.5, "high_mix": 0.75, "high_threshold": 0.82, "lookback": 3, "mode": "hard", "tie_tol": 0.002},
    "s1": {"base_target_vol": 0.1, "bull_scale": 1.2, "correction_alpha": 0.5, "crisis_scale": 0.7, "lookback": 3, "max_leverage": 1.6},
    "s3": {"high": 1.45, "low": 0.92, "mode": "smooth", "sim_high": 0.97, "sim_low": 0.78},
    "s7": {"allow_leverage": False, "dd0": 1.1, "dd3": 1.0, "dd6": 0.8, "dd9": 0.65, "gross_cap": 1.0},
    "sleeve": {"cash_bull": 0.0, "cash_correction": 0.0, "cash_crisis": 0.02, "cash_sideways": 0.0, "confidence_cash_relief": 0.02, "sim_high": 0.9},
    "srb": {"bull_mult": 1.15, "correction_alpha": 0.5, "crisis_mult": 0.5},
    "stop": {"bull_hard_scale": 1.1, "bull_soft_scale": 1.2, "crisis_hard_scale": 0.8, "crisis_soft_scale": 0.85, "hard_cash": 0.05, "hard_growth_frac": 0.2, "hard_sl_thresh": -0.035, "reentry_fraction": 0.7, "reentry_mode": "portfolio", "reentry_thresh": 0.008, "sideways_hard_scale": 1.0, "sideways_soft_scale": 1.0, "soft_cash": 0.0, "soft_growth_frac": 0.55, "soft_sl_thresh": -0.02},
    "tilt": {"bull_sim_threshold": 0.8, "bull_strength": 0.2, "crisis_def_strength": 0.2},
}


def compute_guard_signals(daily_returns, month_ends, corr_lookback=60):
    spy = daily_returns["SPY"]
    tlt = daily_returns["TLT"]
    rolling_corr = spy.rolling(corr_lookback).corr(tlt)
    rolling_vol = tlt.rolling(corr_lookback).std() * np.sqrt(252)
    monthly_corr = rolling_corr.resample("ME").last()
    monthly_vol = rolling_vol.resample("ME").last()
    rows = []
    for date in month_ends:
        prev_corr = monthly_corr[monthly_corr.index < date]
        prev_vol = monthly_vol[monthly_vol.index < date]
        if len(prev_corr) == 0 or len(prev_vol) == 0:
            rows.append({"date": date, "spy_tlt_corr": np.nan, "tlt_vol": np.nan})
            continue
        rows.append({"date": date, "spy_tlt_corr": float(prev_corr.iloc[-1]), "tlt_vol": float(prev_vol.iloc[-1])})
    return pd.DataFrame(rows).set_index("date")


def partial_exclude_bonds(weights, exclude_list, bil_col, reduction_pct, min_bil=0.0):
    """Reduce bond weights by reduction_pct (0-1) and add to BIL."""
    w = weights.copy()
    transferred = 0.0
    for asset in exclude_list:
        if asset in w.index and w[asset] > 0:
            amount = w[asset] * reduction_pct
            w[asset] -= amount
            transferred += amount
    w[bil_col] = w.get(bil_col, 0.0) + transferred
    if w[bil_col] < min_bil:
        deficit = min_bil - w[bil_col]
        non_bil = w.drop(bil_col)
        pos_sum = non_bil[non_bil > 0].sum()
        if pos_sum > 1e-9:
            scale = max(0.0, 1.0 - deficit / pos_sum)
            for a in non_bil.index:
                if w[a] > 0:
                    w[a] *= scale
        w[bil_col] = min_bil
    return w


def simulate_tiered_guard(
    ctx, cfg, guard_signals,
    tier1_corr_thresh, tier1_bil_floor,
    tier2_corr_thresh, tier2_exclude_list, tier2_reduction_pct,
    base_bil, vol_thresh=0.18, mode="corr_only",
    trade_cost_bps=10.0, label="tiered",
    overlay_mods=None,
):
    working_cfg = copy.deepcopy(cfg)
    if overlay_mods:
        for section, params in overlay_mods.items():
            if section in working_cfg:
                working_cfg[section].update(params)
    cfg = working_cfg

    growth_assets, defensive_assets = bench.classify_assets(ctx)

    # Pre-compute guard tier for each month
    guard_tier_map = {}  # 0=off, 1=bil_floor, 2=bil_floor+exclusion
    for date in ctx.month_ends:
        if date not in guard_signals.index:
            guard_tier_map[date] = 0
            continue
        row = guard_signals.loc[date]
        corr = row.get("spy_tlt_corr", np.nan)
        vol = row.get("tlt_vol", np.nan)
        if np.isnan(corr):
            guard_tier_map[date] = 0
            continue

        corr_t2 = (not np.isnan(corr)) and corr > tier2_corr_thresh
        vol_t2 = (not np.isnan(vol)) and vol > vol_thresh
        if mode == "either":
            is_tier2 = corr_t2 or vol_t2
        elif mode == "corr_only":
            is_tier2 = corr_t2
        else:
            is_tier2 = corr_t2

        if is_tier2:
            guard_tier_map[date] = 2
        else:
            corr_t1 = corr > tier1_corr_thresh
            vol_t1 = (not np.isnan(vol)) and vol > vol_thresh
            if mode == "either":
                is_tier1 = corr_t1 or vol_t1
            elif mode == "corr_only":
                is_tier1 = corr_t1
            else:
                is_tier1 = corr_t1
            guard_tier_map[date] = 1 if is_tier1 else 0

    guard_log = guard_signals.copy()
    guard_log["guard_tier"] = [guard_tier_map.get(d, 0) for d in guard_log.index]
    guard_log["guard_active"] = guard_log["guard_tier"] > 0

    month_returns = []
    ret_history = []
    weights_rows = []
    diagnostics_rows = []
    prev_end_weights = None
    prev_preferred_sleeve = None

    for date in ctx.month_ends:
        w6 = ctx.w6.loc[date] if date in ctx.w6.index else pd.Series(0.0, index=ctx.assets)
        w7 = ctx.w7.loc[date] if date in ctx.w7.index else pd.Series(0.0, index=ctx.assets)
        similarity = psa.cosine_similarity(w6, w7)

        base_weights, mix_info = bench.blend_sleeves_v2(
            ctx=ctx, date=date, w6=w6, w7=w7, similarity=similarity,
            mix_cfg=cfg.get("mix", {}), prev_preferred=prev_preferred_sleeve,
            prev_end_weights=prev_end_weights, trade_cost_bps=trade_cost_bps,
        )
        prev_preferred_sleeve = mix_info.get("preferred_sleeve")

        regime_row = ctx.pit_regime.loc[date] if date in ctx.pit_regime.index else pd.Series(psa.NEUTRAL_REGIME)
        drawdown_prev = psa.current_drawdown(ret_history)

        s3_factor = bench.s3_factor_from_cfg(similarity, cfg.get("s3"))
        s7_cfg = cfg.get("s7")
        s7_factor = psa.budget_scale_from_drawdown(drawdown_prev, s7_cfg) if s7_cfg else 1.0
        srb_cfg = cfg.get("srb")
        srb_factor = psa.regime_multiplier(regime_row, bull_mult=float(srb_cfg["bull_mult"]), crisis_mult=float(srb_cfg["crisis_mult"]), correction_alpha=float(srb_cfg["correction_alpha"])) if srb_cfg else 1.0

        s1_cfg = cfg.get("s1")
        realized_prev = psa.realized_vol(ret_history, int(s1_cfg["lookback"])) if s1_cfg else None
        if s1_cfg and realized_prev is not None:
            vol_scale = psa.regime_target_vol_scale(regime_row, bull_scale=float(s1_cfg["bull_scale"]), crisis_scale=float(s1_cfg["crisis_scale"]), correction_alpha=float(s1_cfg["correction_alpha"]))
            target_vol = float(s1_cfg["base_target_vol"]) * vol_scale
            vt_scalar = float(np.clip(target_vol / realized_prev, 1.0 / float(s1_cfg["max_leverage"]), float(s1_cfg["max_leverage"])))
        else:
            target_vol = float(s1_cfg["base_target_vol"]) if s1_cfg else np.nan
            vt_scalar = 1.0

        expert_base_weights, expert_info = bench.specialist_router_from_cfg(ctx=ctx, date=date, base_weights=base_weights, regime_row=regime_row, similarity=similarity, drawdown_prev=drawdown_prev, realized_prev=realized_prev, target_vol=float(target_vol), expert_cfg=None)
        brake_info = bench.absolute_risk_brake_from_cfg(ctx=ctx, date=date, regime_row=regime_row, similarity=similarity, realized_prev=realized_prev, target_vol=float(target_vol), brake_cfg=None)
        meta_info = bench.exposure_meta_from_cfg(ctx=ctx, date=date, regime_row=regime_row, similarity=similarity, realized_prev=realized_prev, target_vol=float(target_vol), meta_cfg=None)
        allocator_info = bench.allocator_budget_from_cfg(ctx=ctx, date=date, regime_row=regime_row, similarity=similarity, realized_prev=realized_prev, target_vol=float(target_vol), allocator_cfg=None)
        active_hier_cfg, _ = bench.resolve_hier_cfg_for_date(regime_row=regime_row, similarity=similarity, drawdown_prev=drawdown_prev, realized_prev=realized_prev, target_vol=float(target_vol), expert_info=expert_info, hier_cfg=None)
        hier_info = bench.hierarchical_allocator_from_cfg(regime_row=regime_row, similarity=similarity, drawdown_prev=drawdown_prev, realized_prev=realized_prev, target_vol=float(target_vol), expert_info=expert_info, hier_cfg=active_hier_cfg)
        policy_info = bench.policy_state_from_cfg(ctx=ctx, date=date, regime_row=regime_row, similarity=similarity, realized_prev=realized_prev, target_vol=float(target_vol), policy_cfg=None)

        pre_brake = psa.risky_budget_of(expert_base_weights, ctx.bil_col) * s3_factor * s7_factor * srb_factor * vt_scalar
        post_brake = pre_brake * brake_info["factor"]
        pre_alloc = post_brake * meta_info["factor"]
        pre_hier = min(pre_alloc, float(allocator_info["budget_cap"]))
        pre_policy = min(pre_hier, float(hier_info["risk_cap"]))
        raw_growth_target = min(pre_policy, float(policy_info["risk_cap"]))

        adjusted_weights, _ = bench.allocate_overlay_v2_weights(ctx=ctx, base_weights=expert_base_weights, regime_row=regime_row, similarity=similarity, growth_budget_target=raw_growth_target, sleeve_cfg=cfg["sleeve"], tilt_cfg=cfg.get("tilt"))
        adjusted_weights, _ = bench.enforce_hierarchical_budget(adjusted_weights, growth_assets=growth_assets, defensive_assets=defensive_assets, bil_col=ctx.bil_col, hier_info=hier_info)
        adjusted_weights = bench.enforce_cash_floor(adjusted_weights, ctx.bil_col, float(policy_info["cash_floor"]))

        # ================================================================
        # >>> TIERED GUARD <<<
        # ================================================================
        tier = guard_tier_map.get(date, 0)

        if tier == 2:
            # Tier 2: BIL floor + partial/full bond exclusion
            adjusted_weights = bench.enforce_cash_floor(adjusted_weights, ctx.bil_col, tier1_bil_floor)
            adjusted_weights = partial_exclude_bonds(
                adjusted_weights, tier2_exclude_list, ctx.bil_col,
                tier2_reduction_pct, min_bil=tier1_bil_floor,
            )
        elif tier == 1:
            # Tier 1: BIL floor only (keep bonds)
            adjusted_weights = bench.enforce_cash_floor(adjusted_weights, ctx.bil_col, tier1_bil_floor)
        else:
            # No guard: just base BIL floor
            adjusted_weights = bench.enforce_cash_floor(adjusted_weights, ctx.bil_col, base_bil)

        effective_rebalance_cfg = bench.merge_policy_rebalance_cfg(None, policy_info)
        adjusted_weights, _ = bench.apply_rebalance_guard(prev_end_weights=prev_end_weights, target_weights=adjusted_weights, bil_col=ctx.bil_col, rebalance_cfg=effective_rebalance_cfg)
        entry_turnover = bench.turnover_cost(prev_end_weights, adjusted_weights, trade_cost_bps) if prev_end_weights is not None else 0.0
        effective_stop_cfg = bench.merge_policy_stop_cfg(cfg.get("stop"), policy_info)
        month_ret, stop_info, realized_avg_weights, end_weights = bench.simulate_month_with_stateful_stop_v2(ctx=ctx, month_returns=ctx.daily_by_month.get(date, pd.DataFrame()), base_weights=adjusted_weights, regime_row=regime_row, stop_cfg=effective_stop_cfg, trade_cost_bps=trade_cost_bps, initial_turnover_cost=entry_turnover)

        month_returns.append(month_ret)
        ret_history.append(month_ret)
        prev_end_weights = end_weights
        avg_budget = bench.budget_breakdown(realized_avg_weights, growth_assets, defensive_assets, ctx.bil_col)
        weights_rows.append(realized_avg_weights.rename(date))

        diagnostics_rows.append({
            "date": date, "label": label, "sim": similarity, "dd_prev": drawdown_prev,
            "guard_tier": tier,
            "spy_tlt_corr": guard_signals.loc[date, "spy_tlt_corr"] if date in guard_signals.index else np.nan,
            "tlt_vol": guard_signals.loc[date, "tlt_vol"] if date in guard_signals.index else np.nan,
            "realized_avg_growth_budget": float(avg_budget["growth_budget"]),
            "realized_avg_defensive_budget": float(avg_budget["defensive_budget"]),
            "realized_avg_cash_budget": float(avg_budget["cash_budget"]),
            "bil_weight": float(realized_avg_weights.get(ctx.bil_col, 0.0)),
            "monthly_return": month_ret,
        })

    returns = pd.Series(month_returns, index=pd.DatetimeIndex(ctx.month_ends), name="return")
    weights_df = pd.DataFrame(weights_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows).set_index("date")
    return returns, weights_df, diagnostics_df, guard_log


def _compute_subperiod_metrics(returns, model_label):
    returns = returns.astype(float).dropna().sort_index()
    specs = [("full_sample", None, None), ("pre_covid", pd.Timestamp("2016-07-31"), pd.Timestamp("2019-12-31")), ("covid", pd.Timestamp("2020-01-31"), pd.Timestamp("2021-12-31")), ("inflation", pd.Timestamp("2022-01-31"), pd.Timestamp("2023-12-31"))]
    latest = pd.Timestamp(returns.index.max())
    if latest >= pd.Timestamp("2024-01-31"):
        specs.append(("post_2023", pd.Timestamp("2024-01-31"), latest))
    rows = []
    for name, s, e in specs:
        p = returns.copy()
        if s: p = p.loc[p.index >= s]
        if e: p = p.loc[p.index <= e]
        if len(p) < 6: continue
        ar = float(p.mean() * 12)
        av = float(p.std(ddof=1) * math.sqrt(12)) if len(p) > 1 else 0
        sh = ar / av if av > 1e-12 else 0
        w = (1 + p).cumprod(); pk = w.cummax()
        mdd = float(((w - pk) / pk).min()) if len(w) else 0
        rows.append({"label": model_label, "period": name, "months": len(p), "sharpe": sh, "return": ar, "vol": av, "mdd": mdd, "triple": sh >= 1.0 and ar >= 0.10 and mdd >= -0.10})
    return rows


def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, pd.Timestamp): return str(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def build_search_grid():
    configs = []
    bond_list = ["TLT", "IEF", "TIP"]

    # Tiered configs:
    # tier1_corr is the "mild defense" threshold (BIL floor)
    # tier2_corr is the "strong defense" threshold (BIL floor + exclusion)
    # tier2_corr must be > tier1_corr

    for tier1_corr in [-0.20, -0.15, -0.10]:
        for tier1_bil in [0.30, 0.35, 0.40]:
            for tier2_corr in [-0.05, 0.00, 0.05, 0.10, 0.15]:
                if tier2_corr <= tier1_corr:
                    continue
                for reduction in [0.5, 0.75, 1.0]:
                    for base_bil in [0.05, 0.08]:
                        for mod_name, mods in [("none", {}), ("combined", {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}})]:
                            configs.append({
                                "tier1_corr": tier1_corr,
                                "tier1_bil": tier1_bil,
                                "tier2_corr": tier2_corr,
                                "tier2_reduction": reduction,
                                "base_bil": base_bil,
                                "exclude_assets": bond_list,
                                "mod_name": mod_name,
                                "overlay_mods": mods,
                            })
    return configs


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("TIERED GUARD V8 (BIL Floor + Conditional Exclusion)")
    print(f"  Output: {args.out_dir}")
    print("=" * 72)

    ctx = psa.build_context(args.out_dir, args.refresh_data, args.refresh_regime)
    print(f"  Months: {len(ctx.month_ends)} ({ctx.month_ends[0].date()} -> {ctx.month_ends[-1].date()})")

    guard_signals = compute_guard_signals(ctx.daily_returns, ctx.month_ends)
    guard_signals.to_csv(args.out_dir / "guard_signals.csv")
    print(f"  SPY-TLT corr: [{guard_signals['spy_tlt_corr'].min():.3f}, {guard_signals['spy_tlt_corr'].max():.3f}]")

    if args.quick:
        bond_list = ["TLT", "IEF", "TIP"]
        grid = [
            # Best V4/V5 config (BIL floor only, no exclusion) as reference
            {"tier1_corr": -0.20, "tier1_bil": 0.40, "tier2_corr": 99.0, "tier2_reduction": 0.0, "base_bil": 0.08, "exclude_assets": bond_list, "mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            # Tiered: BIL floor at -0.20, exclusion at various higher thresholds
            {"tier1_corr": -0.20, "tier1_bil": 0.40, "tier2_corr": -0.05, "tier2_reduction": 0.75, "base_bil": 0.08, "exclude_assets": bond_list, "mod_name": "none", "overlay_mods": {}},
            {"tier1_corr": -0.20, "tier1_bil": 0.40, "tier2_corr": 0.00, "tier2_reduction": 0.75, "base_bil": 0.08, "exclude_assets": bond_list, "mod_name": "none", "overlay_mods": {}},
            {"tier1_corr": -0.20, "tier1_bil": 0.40, "tier2_corr": 0.05, "tier2_reduction": 0.75, "base_bil": 0.08, "exclude_assets": bond_list, "mod_name": "none", "overlay_mods": {}},
            {"tier1_corr": -0.20, "tier1_bil": 0.40, "tier2_corr": 0.10, "tier2_reduction": 0.75, "base_bil": 0.08, "exclude_assets": bond_list, "mod_name": "none", "overlay_mods": {}},
            # With combined overlay mods
            {"tier1_corr": -0.20, "tier1_bil": 0.40, "tier2_corr": -0.05, "tier2_reduction": 0.75, "base_bil": 0.08, "exclude_assets": bond_list, "mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            {"tier1_corr": -0.20, "tier1_bil": 0.40, "tier2_corr": 0.00, "tier2_reduction": 0.75, "base_bil": 0.08, "exclude_assets": bond_list, "mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            {"tier1_corr": -0.20, "tier1_bil": 0.40, "tier2_corr": 0.05, "tier2_reduction": 0.75, "base_bil": 0.08, "exclude_assets": bond_list, "mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            {"tier1_corr": -0.20, "tier1_bil": 0.40, "tier2_corr": 0.10, "tier2_reduction": 0.75, "base_bil": 0.08, "exclude_assets": bond_list, "mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            # Full exclusion variants
            {"tier1_corr": -0.20, "tier1_bil": 0.40, "tier2_corr": 0.00, "tier2_reduction": 1.0, "base_bil": 0.08, "exclude_assets": bond_list, "mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            {"tier1_corr": -0.20, "tier1_bil": 0.40, "tier2_corr": 0.05, "tier2_reduction": 1.0, "base_bil": 0.08, "exclude_assets": bond_list, "mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            # 50% reduction
            {"tier1_corr": -0.20, "tier1_bil": 0.40, "tier2_corr": 0.00, "tier2_reduction": 0.50, "base_bil": 0.08, "exclude_assets": bond_list, "mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            # Different tier1 threshold
            {"tier1_corr": -0.15, "tier1_bil": 0.35, "tier2_corr": 0.05, "tier2_reduction": 0.75, "base_bil": 0.08, "exclude_assets": bond_list, "mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            {"tier1_corr": -0.15, "tier1_bil": 0.40, "tier2_corr": 0.05, "tier2_reduction": 0.75, "base_bil": 0.08, "exclude_assets": bond_list, "mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            # Baseline
            {"tier1_corr": 99.0, "tier1_bil": 0.0, "tier2_corr": 99.0, "tier2_reduction": 0.0, "base_bil": 0.0, "exclude_assets": [], "mod_name": "none", "overlay_mods": {}},
        ]
    else:
        grid = build_search_grid()
        grid.append({"tier1_corr": 99.0, "tier1_bil": 0.0, "tier2_corr": 99.0, "tier2_reduction": 0.0, "base_bil": 0.0, "exclude_assets": [], "mod_name": "none", "overlay_mods": {}})

    print(f"\n  Grid: {len(grid)} configs")

    all_results = []
    triple_count = 0

    for i, g in enumerate(grid):
        is_baseline = g["tier1_corr"] > 10.0
        if is_baseline:
            lbl = "BASELINE"
        else:
            lbl = (f"T1c{g['tier1_corr']:+.2f}_b{g['tier1_bil']:.2f}"
                   f"_T2c{g['tier2_corr']:+.2f}_r{g['tier2_reduction']:.0%}"
                   f"_bb{g['base_bil']:.2f}_{g['mod_name']}")

        returns, weights, diagnostics, guard_log = simulate_tiered_guard(
            ctx=ctx, cfg=AUDITED_FINAL_CONFIG, guard_signals=guard_signals,
            tier1_corr_thresh=g["tier1_corr"], tier1_bil_floor=g["tier1_bil"],
            tier2_corr_thresh=g["tier2_corr"], tier2_exclude_list=g["exclude_assets"],
            tier2_reduction_pct=g["tier2_reduction"], base_bil=g["base_bil"],
            mode="corr_only", trade_cost_bps=args.trade_cost_bps, label=lbl,
            overlay_mods=g["overlay_mods"],
        )

        m = psa.evaluate_returns(returns, lbl)
        t1_count = int((guard_log["guard_tier"] == 1).sum()) if "guard_tier" in guard_log.columns else 0
        t2_count = int((guard_log["guard_tier"] == 2).sum()) if "guard_tier" in guard_log.columns else 0

        result = {
            "label": lbl,
            "tier1_corr": g["tier1_corr"], "tier1_bil": g["tier1_bil"],
            "tier2_corr": g["tier2_corr"], "tier2_reduction": g["tier2_reduction"],
            "base_bil": g["base_bil"], "mod": g["mod_name"],
            "sharpe": m["sharpe"], "ann_return": m["return"], "vol": m["vol"],
            "mdd": m["mdd"], "calmar": m["calmar"], "triple": m["triple"],
            "tier1_months": t1_count, "tier2_months": t2_count,
            "total_guard_months": t1_count + t2_count,
        }
        all_results.append(result)
        if m["triple"]:
            triple_count += 1

        flag = " *** TRIPLE ***" if m["triple"] else ""
        print(f"  [{i+1}/{len(grid)}] {lbl}: Sh={m['sharpe']:.3f} Ret={m['return']:.2%} MDD={m['mdd']:.2%} T1={t1_count} T2={t2_count}{flag}")

        if m["triple"] or is_baseline:
            dd = args.out_dir / "details" / lbl
            dd.mkdir(parents=True, exist_ok=True)
            returns.to_csv(dd / "port_returns.csv", header=["return"])
            diagnostics.to_csv(dd / "diagnostics.csv")
            guard_log.to_csv(dd / "guard_log.csv")
            pd.DataFrame(_compute_subperiod_metrics(returns, lbl)).to_csv(dd / "subperiod.csv", index=False)
            with open(dd / "metrics.json", "w") as f:
                json.dump(result, f, indent=2, default=_json_default)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.out_dir / "all_results.csv", index=False)

    if triple_count > 0:
        triple_df = results_df[results_df["triple"]].sort_values("sharpe", ascending=False)
        print(f"\n{'='*72}\nTRIPLE ACHIEVED! ({triple_count} configs)\n{'='*72}")
        best = triple_df.iloc[0]
        print(f"  Best: {best['label']}")
        print(f"  Sh={best['sharpe']:.4f} Ret={best['ann_return']:.2%} MDD={best['mdd']:.2%}")
        triple_df.to_csv(args.out_dir / "triple_frontier.csv", index=False)
        with open(args.out_dir / "best_triple.json", "w") as f:
            json.dump(best.to_dict(), f, indent=2, default=_json_default)
    else:
        print(f"\n{'='*72}\nNO TRIPLE\n{'='*72}")

    print("\n  === Pareto ===")
    for lb in [-13, -12, -11, -10.5, -10.0, -9.5, -9.0]:
        bk = results_df[results_df["mdd"] >= lb / 100]
        if len(bk) > 0:
            b = bk.loc[bk["ann_return"].idxmax()]
            print(f"  MDD>={lb:.1f}%: Ret={b['ann_return']:.2%} Sh={b['sharpe']:.3f} T1={b['tier1_months']:.0f} T2={b['tier2_months']:.0f} [{b['label']}]")

    near = results_df[(results_df["sharpe"] >= 0.95) & (results_df["ann_return"] >= 0.09) & (results_df["mdd"] >= -0.11)].sort_values("mdd", ascending=False)
    if not near.empty:
        near.to_csv(args.out_dir / "near_triple.csv", index=False)
        print(f"\n  Near-triple: {len(near)}")
        for _, r in near.head(10).iterrows():
            print(f"    Sh={r['sharpe']:.3f} Ret={r['ann_return']:.2%} MDD={r['mdd']:.2%} [{r['label']}]")

    with open(args.out_dir / "run_metadata.json", "w") as f:
        json.dump({"total": len(grid), "triple_count": triple_count}, f, indent=2)
    print(f"\n  Done. Results: {args.out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--trade-cost-bps", type=float, default=10.0)
    p.add_argument("--refresh-data", action="store_true")
    p.add_argument("--refresh-regime", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
