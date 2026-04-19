"""
Correlation Guard V7: Asset Exclusion Guard

Instead of just enforcing a BIL floor (which leaves TLT/IEF/TIP untouched),
this version REPLACES bond assets with BIL when the guard is active.

Theory:
    In 2022, the overlay correctly went 97% defensive, but TLT(-8.2%), IEF(-4.7%),
    TIP(-6.7%) all crashed. The existing guard adds BIL floor but defensive bonds
    remain in the portfolio, still causing damage.

    Asset Exclusion solves this by zeroing out specified bond assets and redirecting
    their weight to BIL. Combined with a HIGHER correlation threshold (fewer
    activations), this achieves:
    - Stronger MDD protection per activation (BIL replaces crashing bonds)
    - Less return drag (fewer months affected)

Usage:
    PHASE17_STEP1_DIR=results_runpod/phase17/step1_2025_repro_onepass_regimefix_v1 \
        python scripts/run_correlation_guard_v7_exclusion.py \
        --out-dir results_runpod/correlation_guard_v7
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

# ---------------------------------------------------------------------------
# Bootstrap engine
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "scripts" / "run_phase18_paper_safe_ablation.py"
BENCHMARK_PATH = ROOT / "scripts" / "run_phase18_nonleveraged_v2_benchmark.py"

DEFAULT_OUT_DIR = ROOT / "results_runpod" / "correlation_guard_v7"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


# NOTE (2026-04-19): ENGINE_PATH and BENCHMARK_PATH point to Phase 18 overlay
# scripts that are NOT present in this repository. They existed in the author's
# local workspace but were never pushed. See docs/REPO_STATUS.md. Fail fast with
# an actionable message rather than a cryptic spec-loader error.
for _p in (ENGINE_PATH, BENCHMARK_PATH):
    if not _p.exists():
        raise FileNotFoundError(
            f"Required Phase 18 overlay engine not found: {_p.name}\n"
            f"  Expected at: {_p}\n"
            "This script depends on run_phase18_paper_safe_ablation.py and "
            "run_phase18_nonleveraged_v2_benchmark.py which are not pushed to "
            "this repository. See docs/REPO_STATUS.md and "
            "docs/REMEDIATION_PLAN.md for the fix-forward plan."
        )

psa = _load_module("phase18_paper_safe_engine", ENGINE_PATH)
bench = _load_module("phase18_nonlev_v2_benchmark", BENCHMARK_PATH)


# ---------------------------------------------------------------------------
# Audited final config (frozen from 2023 winner)
# ---------------------------------------------------------------------------
AUDITED_FINAL_CONFIG = {
    "mix": {
        "base_mix": 0.5, "high_mix": 0.75, "high_threshold": 0.82,
        "lookback": 3, "mode": "hard", "tie_tol": 0.002,
    },
    "s1": {
        "base_target_vol": 0.1, "bull_scale": 1.2, "correction_alpha": 0.5,
        "crisis_scale": 0.7, "lookback": 3, "max_leverage": 1.6,
    },
    "s3": {
        "high": 1.45, "low": 0.92, "mode": "smooth",
        "sim_high": 0.97, "sim_low": 0.78,
    },
    "s7": {
        "allow_leverage": False, "dd0": 1.1, "dd3": 1.0,
        "dd6": 0.8, "dd9": 0.65, "gross_cap": 1.0,
    },
    "sleeve": {
        "cash_bull": 0.0, "cash_correction": 0.0, "cash_crisis": 0.02,
        "cash_sideways": 0.0, "confidence_cash_relief": 0.02, "sim_high": 0.9,
    },
    "srb": {
        "bull_mult": 1.15, "correction_alpha": 0.5, "crisis_mult": 0.5,
    },
    "stop": {
        "bull_hard_scale": 1.1, "bull_soft_scale": 1.2,
        "crisis_hard_scale": 0.8, "crisis_soft_scale": 0.85,
        "hard_cash": 0.05, "hard_growth_frac": 0.2,
        "hard_sl_thresh": -0.035, "reentry_fraction": 0.7,
        "reentry_mode": "portfolio", "reentry_thresh": 0.008,
        "sideways_hard_scale": 1.0, "sideways_soft_scale": 1.0,
        "soft_cash": 0.0, "soft_growth_frac": 0.55,
        "soft_sl_thresh": -0.02,
    },
    "tilt": {
        "bull_sim_threshold": 0.8, "bull_strength": 0.2,
        "crisis_def_strength": 0.2,
    },
}


# ---------------------------------------------------------------------------
# Correlation Guard: compute monthly signals (same as V1-V6)
# ---------------------------------------------------------------------------

def compute_guard_signals(
    daily_returns: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
    corr_lookback: int = 60,
) -> pd.DataFrame:
    spy = daily_returns["SPY"]
    tlt = daily_returns["TLT"]
    rolling_corr = spy.rolling(corr_lookback).corr(tlt)
    rolling_vol = tlt.rolling(corr_lookback).std() * np.sqrt(252)
    monthly_corr = rolling_corr.resample("ME").last()
    monthly_vol = rolling_vol.resample("ME").last()

    rows = []
    for date in month_ends:
        prev_corr_vals = monthly_corr[monthly_corr.index < date]
        prev_vol_vals = monthly_vol[monthly_vol.index < date]
        if len(prev_corr_vals) == 0 or len(prev_vol_vals) == 0:
            rows.append({"date": date, "spy_tlt_corr": np.nan, "tlt_vol": np.nan})
            continue
        rows.append({
            "date": date,
            "spy_tlt_corr": float(prev_corr_vals.iloc[-1]),
            "tlt_vol": float(prev_vol_vals.iloc[-1]),
        })
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Asset Exclusion: replace specified assets with BIL
# ---------------------------------------------------------------------------

def exclude_assets_to_bil(
    weights: pd.Series,
    exclude_list: List[str],
    bil_col: str,
    base_bil_floor: float = 0.0,
) -> pd.Series:
    """Zero out exclude_list assets and redirect their weight to BIL.

    Args:
        weights: Current portfolio weights
        exclude_list: Assets to zero out (e.g. ["TLT", "IEF", "TIP"])
        bil_col: BIL column name
        base_bil_floor: Minimum BIL after exclusion

    Returns:
        Modified weights with excluded assets at 0 and BIL increased.
    """
    w = weights.copy()
    excluded_total = 0.0
    for asset in exclude_list:
        if asset in w.index:
            excluded_total += max(0.0, w[asset])
            w[asset] = 0.0
    w[bil_col] = w.get(bil_col, 0.0) + excluded_total
    # Also enforce base floor
    if w[bil_col] < base_bil_floor:
        deficit = base_bil_floor - w[bil_col]
        # Pro-rata reduce remaining non-BIL assets
        non_bil = w.drop(bil_col)
        pos_sum = non_bil[non_bil > 0].sum()
        if pos_sum > 1e-9:
            scale = max(0.0, 1.0 - deficit / pos_sum)
            for a in non_bil.index:
                if w[a] > 0:
                    w[a] *= scale
        w[bil_col] = base_bil_floor
    return w


def determine_guard_action(
    signal_row: pd.Series,
    corr_thresh: float,
    vol_thresh: float,
    mode: str = "corr_only",
) -> bool:
    """Determine if guard should activate for this month."""
    corr_val = signal_row.get("spy_tlt_corr", np.nan)
    vol_val = signal_row.get("tlt_vol", np.nan)

    corr_triggered = (not np.isnan(corr_val)) and corr_val > corr_thresh
    vol_triggered = (not np.isnan(vol_val)) and vol_val > vol_thresh

    if mode == "either":
        return corr_triggered or vol_triggered
    elif mode == "both":
        return corr_triggered and vol_triggered
    elif mode == "corr_only":
        return corr_triggered
    elif mode == "vol_only":
        return vol_triggered
    return False


# ---------------------------------------------------------------------------
# Full simulation with Asset Exclusion Guard
# ---------------------------------------------------------------------------

def simulate_with_exclusion_guard(
    ctx,
    cfg: Dict[str, Any],
    guard_signals: pd.DataFrame,
    corr_thresh: float,
    vol_thresh: float,
    base_bil: float,
    exclude_assets: List[str],
    mode: str = "corr_only",
    trade_cost_bps: float = 10.0,
    label: str = "exclusion_guard",
    overlay_mods: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full Phase 18 overlay + Asset Exclusion Guard."""

    # Apply overlay modifications if any
    working_cfg = copy.deepcopy(cfg)
    if overlay_mods:
        for section, params in overlay_mods.items():
            if section in working_cfg:
                working_cfg[section].update(params)
    cfg = working_cfg

    growth_assets, defensive_assets = bench.classify_assets(ctx)

    # Pre-compute guard activation for each month
    guard_active_map = {}
    for date in ctx.month_ends:
        if date in guard_signals.index:
            guard_active_map[date] = determine_guard_action(
                guard_signals.loc[date], corr_thresh, vol_thresh, mode
            )
        else:
            guard_active_map[date] = False

    # Guard log
    guard_log = guard_signals.copy()
    guard_log["guard_active"] = [guard_active_map.get(d, False) for d in guard_log.index]

    # Simulation loop
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

        mix_cfg = cfg.get("mix", {})
        base_weights, mix_info = bench.blend_sleeves_v2(
            ctx=ctx, date=date, w6=w6, w7=w7,
            similarity=similarity, mix_cfg=mix_cfg,
            prev_preferred=prev_preferred_sleeve,
            prev_end_weights=prev_end_weights,
            trade_cost_bps=trade_cost_bps,
        )
        prev_preferred_sleeve = mix_info.get("preferred_sleeve")

        regime_row = ctx.pit_regime.loc[date] if date in ctx.pit_regime.index else pd.Series(psa.NEUTRAL_REGIME)
        drawdown_prev = psa.current_drawdown(ret_history)

        s3_factor = bench.s3_factor_from_cfg(similarity, cfg.get("s3"))
        s7_cfg = cfg.get("s7")
        s7_factor = psa.budget_scale_from_drawdown(drawdown_prev, s7_cfg) if s7_cfg is not None else 1.0
        srb_cfg = cfg.get("srb")
        srb_factor = (
            psa.regime_multiplier(
                regime_row,
                bull_mult=float(srb_cfg["bull_mult"]),
                crisis_mult=float(srb_cfg["crisis_mult"]),
                correction_alpha=float(srb_cfg["correction_alpha"]),
            ) if srb_cfg is not None else 1.0
        )

        s1_cfg = cfg.get("s1")
        realized_prev = psa.realized_vol(ret_history, int(s1_cfg["lookback"])) if s1_cfg is not None else None
        if s1_cfg is not None and realized_prev is not None:
            vol_scale = psa.regime_target_vol_scale(
                regime_row,
                bull_scale=float(s1_cfg["bull_scale"]),
                crisis_scale=float(s1_cfg["crisis_scale"]),
                correction_alpha=float(s1_cfg["correction_alpha"]),
            )
            target_vol = float(s1_cfg["base_target_vol"]) * vol_scale
            vt_scalar = float(np.clip(
                target_vol / realized_prev,
                1.0 / float(s1_cfg["max_leverage"]),
                float(s1_cfg["max_leverage"]),
            ))
        else:
            target_vol = float(s1_cfg["base_target_vol"]) if s1_cfg is not None else np.nan
            vt_scalar = 1.0

        expert_base_weights, expert_info = bench.specialist_router_from_cfg(
            ctx=ctx, date=date, base_weights=base_weights,
            regime_row=regime_row, similarity=similarity,
            drawdown_prev=drawdown_prev, realized_prev=realized_prev,
            target_vol=float(target_vol), expert_cfg=None,
        )

        brake_info = bench.absolute_risk_brake_from_cfg(
            ctx=ctx, date=date, regime_row=regime_row,
            similarity=similarity, realized_prev=realized_prev,
            target_vol=float(target_vol), brake_cfg=None,
        )
        meta_info = bench.exposure_meta_from_cfg(
            ctx=ctx, date=date, regime_row=regime_row,
            similarity=similarity, realized_prev=realized_prev,
            target_vol=float(target_vol), meta_cfg=None,
        )
        allocator_info = bench.allocator_budget_from_cfg(
            ctx=ctx, date=date, regime_row=regime_row,
            similarity=similarity, realized_prev=realized_prev,
            target_vol=float(target_vol), allocator_cfg=None,
        )
        active_hier_cfg, hier_dynamic_info = bench.resolve_hier_cfg_for_date(
            regime_row=regime_row, similarity=similarity,
            drawdown_prev=drawdown_prev, realized_prev=realized_prev,
            target_vol=float(target_vol), expert_info=expert_info,
            hier_cfg=None,
        )
        hier_info = bench.hierarchical_allocator_from_cfg(
            regime_row=regime_row, similarity=similarity,
            drawdown_prev=drawdown_prev, realized_prev=realized_prev,
            target_vol=float(target_vol), expert_info=expert_info,
            hier_cfg=active_hier_cfg,
        )
        policy_info = bench.policy_state_from_cfg(
            ctx=ctx, date=date, regime_row=regime_row,
            similarity=similarity, realized_prev=realized_prev,
            target_vol=float(target_vol), policy_cfg=None,
        )

        pre_brake_growth_target = psa.risky_budget_of(expert_base_weights, ctx.bil_col) * s3_factor * s7_factor * srb_factor * vt_scalar
        post_brake_growth_target = pre_brake_growth_target * brake_info["factor"]
        pre_allocator_growth_target = post_brake_growth_target * meta_info["factor"]
        pre_hier_growth_target = min(pre_allocator_growth_target, float(allocator_info["budget_cap"]))
        pre_policy_growth_target = min(pre_hier_growth_target, float(hier_info["risk_cap"]))
        raw_growth_target = min(pre_policy_growth_target, float(policy_info["risk_cap"]))

        adjusted_weights, sleeve_info = bench.allocate_overlay_v2_weights(
            ctx=ctx, base_weights=expert_base_weights,
            regime_row=regime_row, similarity=similarity,
            growth_budget_target=raw_growth_target,
            sleeve_cfg=cfg["sleeve"], tilt_cfg=cfg.get("tilt"),
        )
        adjusted_weights, hier_budget_info = bench.enforce_hierarchical_budget(
            adjusted_weights, growth_assets=growth_assets,
            defensive_assets=defensive_assets, bil_col=ctx.bil_col,
            hier_info=hier_info,
        )
        adjusted_weights = bench.enforce_cash_floor(
            adjusted_weights, ctx.bil_col, float(policy_info["cash_floor"])
        )

        # ================================================================
        # >>> ASSET EXCLUSION GUARD <<<
        # When guard active: zero out specified bond assets → BIL
        # When guard inactive: just enforce base_bil floor
        # ================================================================
        is_guard_active = guard_active_map.get(date, False)

        if is_guard_active:
            adjusted_weights = exclude_assets_to_bil(
                adjusted_weights, exclude_assets, ctx.bil_col, base_bil
            )
        else:
            # Even when guard is off, enforce base BIL floor
            adjusted_weights = bench.enforce_cash_floor(
                adjusted_weights, ctx.bil_col, base_bil
            )

        effective_rebalance_cfg = bench.merge_policy_rebalance_cfg(None, policy_info)
        adjusted_weights, rebalance_info = bench.apply_rebalance_guard(
            prev_end_weights=prev_end_weights,
            target_weights=adjusted_weights,
            bil_col=ctx.bil_col,
            rebalance_cfg=effective_rebalance_cfg,
        )
        entry_turnover = bench.turnover_cost(
            prev_end_weights, adjusted_weights, trade_cost_bps
        ) if prev_end_weights is not None else 0.0

        effective_stop_cfg = bench.merge_policy_stop_cfg(cfg.get("stop"), policy_info)
        month_ret, stop_info, realized_avg_weights, end_weights = bench.simulate_month_with_stateful_stop_v2(
            ctx=ctx,
            month_returns=ctx.daily_by_month.get(date, pd.DataFrame()),
            base_weights=adjusted_weights,
            regime_row=regime_row,
            stop_cfg=effective_stop_cfg,
            trade_cost_bps=trade_cost_bps,
            initial_turnover_cost=entry_turnover,
        )

        month_returns.append(month_ret)
        ret_history.append(month_ret)
        prev_end_weights = end_weights

        avg_budget = bench.budget_breakdown(realized_avg_weights, growth_assets, defensive_assets, ctx.bil_col)
        weights_rows.append(realized_avg_weights.rename(date))

        diagnostics_rows.append({
            "date": date,
            "label": label,
            "sim": similarity,
            "dd_prev": drawdown_prev,
            "s3_factor": float(s3_factor),
            "s7_factor": float(s7_factor),
            "srb_factor": float(srb_factor),
            "vt_scalar": float(vt_scalar),
            "guard_active": is_guard_active,
            "spy_tlt_corr": guard_signals.loc[date, "spy_tlt_corr"] if date in guard_signals.index else np.nan,
            "tlt_vol": guard_signals.loc[date, "tlt_vol"] if date in guard_signals.index else np.nan,
            "realized_avg_growth_budget": float(avg_budget["growth_budget"]),
            "realized_avg_defensive_budget": float(avg_budget["defensive_budget"]),
            "realized_avg_cash_budget": float(avg_budget["cash_budget"]),
            "bil_weight": float(realized_avg_weights.get(ctx.bil_col, 0.0)),
            "gross_exposure": float(realized_avg_weights.sum()),
            "monthly_return": month_ret,
            "soft_triggered": stop_info.get("soft_triggered", False),
            "hard_triggered": stop_info.get("hard_triggered", False),
        })

    returns = pd.Series(month_returns, index=pd.DatetimeIndex(ctx.month_ends), name="return")
    weights_df = pd.DataFrame(weights_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows).set_index("date")
    return returns, weights_df, diagnostics_df, guard_log


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_subperiod_metrics(returns: pd.Series, model_label: str) -> List[Dict[str, Any]]:
    returns = returns.astype(float).dropna().sort_index()
    period_specs = [
        ("full_sample", None, None),
        ("pre_covid_expansion", pd.Timestamp("2016-07-31"), pd.Timestamp("2019-12-31")),
        ("covid_recovery", pd.Timestamp("2020-01-31"), pd.Timestamp("2021-12-31")),
        ("inflation_tightening", pd.Timestamp("2022-01-31"), pd.Timestamp("2023-12-31")),
    ]
    latest_end = pd.Timestamp(returns.index.max())
    if latest_end >= pd.Timestamp("2024-01-31"):
        period_specs.append(("post_2023_extension", pd.Timestamp("2024-01-31"), latest_end))

    rows = []
    for period_name, start, end in period_specs:
        period = returns.copy()
        if start is not None:
            period = period.loc[period.index >= start]
        if end is not None:
            period = period.loc[period.index <= end]
        if len(period) < 6:
            continue
        ann_ret = float(period.mean() * 12.0)
        ann_vol = float(period.std(ddof=1) * math.sqrt(12.0)) if len(period) > 1 else 0.0
        sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
        wealth = (1.0 + period).cumprod()
        peak = wealth.cummax()
        mdd = float(((wealth - peak) / peak).min()) if len(wealth) else 0.0
        triple = sharpe >= 1.0 and ann_ret >= 0.10 and mdd >= -0.10
        rows.append({
            "model_label": model_label, "period": period_name,
            "months": len(period), "sharpe": sharpe, "return": ann_ret,
            "vol": ann_vol, "mdd": mdd, "triple": triple,
        })
    return rows


def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, pd.Timestamp): return str(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def build_search_grid() -> List[Dict[str, Any]]:
    """Build Asset Exclusion Guard grid.

    Key insight: Because asset exclusion is STRONGER than BIL floor,
    we can use HIGHER correlation thresholds (fewer activations).
    This reduces return drag while maintaining MDD protection.
    """
    configs = []

    # Exclusion sets
    exclusion_sets = {
        "bonds": ["TLT", "IEF", "TIP"],
        "bonds_xlp": ["TLT", "IEF", "TIP", "XLP"],
    }

    # Core grid: higher thresholds than V1-V6
    corr_thresholds = [-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20]
    base_bil_options = [0.05, 0.08, 0.10]
    modes = ["corr_only", "either"]
    vol_thresh = 0.18  # fixed (secondary signal)

    # Overlay modifications (from V3-V5 best performers)
    overlay_mod_sets = {
        "none": {},
        "aggressive_s3": {"s3": {"high": 1.50}},
        "aggressive_s7": {"s7": {"dd0": 1.2}},
        "combined": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}},
    }

    for excl_name, excl_list in exclusion_sets.items():
        for corr_thresh in corr_thresholds:
            for base_bil in base_bil_options:
                for mode in modes:
                    for mod_name, mod_dict in overlay_mod_sets.items():
                        configs.append({
                            "corr_thresh": corr_thresh,
                            "vol_thresh": vol_thresh,
                            "base_bil": base_bil,
                            "exclude_name": excl_name,
                            "exclude_assets": excl_list,
                            "mode": mode,
                            "overlay_mod_name": mod_name,
                            "overlay_mods": mod_dict,
                        })

    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Asset Exclusion Guard V7")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trade-cost-bps", type=float, default=10.0)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--refresh-regime", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ASSET EXCLUSION GUARD V7")
    print(f"  Output: {args.out_dir}")
    print(f"  Trade cost: {args.trade_cost_bps:.0f} bps")
    print("=" * 72)

    ctx = psa.build_context(args.out_dir, args.refresh_data, args.refresh_regime)
    print(f"  Months: {len(ctx.month_ends)} ({ctx.month_ends[0].date()} -> {ctx.month_ends[-1].date()})")
    print(f"  Assets: {ctx.assets}")

    print("\n  Computing guard signals...")
    guard_signals = compute_guard_signals(ctx.daily_returns, ctx.month_ends)
    guard_signals.to_csv(args.out_dir / "guard_signals.csv")
    print(f"  SPY-TLT corr range: [{guard_signals['spy_tlt_corr'].min():.3f}, {guard_signals['spy_tlt_corr'].max():.3f}]")
    print(f"  TLT vol range: [{guard_signals['tlt_vol'].min():.3f}, {guard_signals['tlt_vol'].max():.3f}]")

    if args.quick:
        # Quick test: most promising configs only
        grid = [
            # Pure exclusion at various thresholds
            {"corr_thresh": -0.15, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds", "exclude_assets": ["TLT","IEF","TIP"], "mode": "corr_only", "overlay_mod_name": "none", "overlay_mods": {}},
            {"corr_thresh": -0.10, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds", "exclude_assets": ["TLT","IEF","TIP"], "mode": "corr_only", "overlay_mod_name": "none", "overlay_mods": {}},
            {"corr_thresh": -0.05, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds", "exclude_assets": ["TLT","IEF","TIP"], "mode": "corr_only", "overlay_mod_name": "none", "overlay_mods": {}},
            {"corr_thresh": 0.00, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds", "exclude_assets": ["TLT","IEF","TIP"], "mode": "corr_only", "overlay_mod_name": "none", "overlay_mods": {}},
            {"corr_thresh": 0.05, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds", "exclude_assets": ["TLT","IEF","TIP"], "mode": "corr_only", "overlay_mod_name": "none", "overlay_mods": {}},
            {"corr_thresh": 0.10, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds", "exclude_assets": ["TLT","IEF","TIP"], "mode": "corr_only", "overlay_mod_name": "none", "overlay_mods": {}},
            {"corr_thresh": 0.15, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds", "exclude_assets": ["TLT","IEF","TIP"], "mode": "corr_only", "overlay_mod_name": "none", "overlay_mods": {}},
            {"corr_thresh": 0.20, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds", "exclude_assets": ["TLT","IEF","TIP"], "mode": "corr_only", "overlay_mod_name": "none", "overlay_mods": {}},
            # With combined overlay mods
            {"corr_thresh": -0.10, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds", "exclude_assets": ["TLT","IEF","TIP"], "mode": "corr_only", "overlay_mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            {"corr_thresh": -0.05, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds", "exclude_assets": ["TLT","IEF","TIP"], "mode": "corr_only", "overlay_mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            {"corr_thresh": 0.00, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds", "exclude_assets": ["TLT","IEF","TIP"], "mode": "corr_only", "overlay_mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            {"corr_thresh": 0.05, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds", "exclude_assets": ["TLT","IEF","TIP"], "mode": "corr_only", "overlay_mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            {"corr_thresh": 0.10, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds", "exclude_assets": ["TLT","IEF","TIP"], "mode": "corr_only", "overlay_mod_name": "combined", "overlay_mods": {"s3": {"high": 1.50}, "s7": {"dd0": 1.2}}},
            # Bonds + XLP exclusion
            {"corr_thresh": 0.00, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds_xlp", "exclude_assets": ["TLT","IEF","TIP","XLP"], "mode": "corr_only", "overlay_mod_name": "none", "overlay_mods": {}},
            {"corr_thresh": 0.05, "vol_thresh": 0.18, "base_bil": 0.08, "exclude_name": "bonds_xlp", "exclude_assets": ["TLT","IEF","TIP","XLP"], "mode": "corr_only", "overlay_mod_name": "none", "overlay_mods": {}},
            # Baseline
            {"corr_thresh": 99.0, "vol_thresh": 99.0, "base_bil": 0.00, "exclude_name": "none", "exclude_assets": [], "mode": "corr_only", "overlay_mod_name": "none", "overlay_mods": {}},
        ]
    else:
        grid = build_search_grid()
        # Add baseline
        grid.append({
            "corr_thresh": 99.0, "vol_thresh": 99.0, "base_bil": 0.00,
            "exclude_name": "none", "exclude_assets": [],
            "mode": "corr_only", "overlay_mod_name": "none", "overlay_mods": {},
        })

    print(f"\n  Search grid size: {len(grid)} configs")

    all_results = []
    triple_count = 0

    for i, gcfg in enumerate(grid):
        is_baseline = gcfg["corr_thresh"] > 10.0
        if is_baseline:
            lbl = "BASELINE_NO_GUARD"
        else:
            lbl = (f"EX_{gcfg['exclude_name']}_corr{gcfg['corr_thresh']:+.2f}"
                   f"_base{gcfg['base_bil']:.2f}_{gcfg['mode']}"
                   f"_mod{gcfg['overlay_mod_name']}")

        returns, weights, diagnostics, guard_log = simulate_with_exclusion_guard(
            ctx=ctx, cfg=AUDITED_FINAL_CONFIG,
            guard_signals=guard_signals,
            corr_thresh=gcfg["corr_thresh"],
            vol_thresh=gcfg["vol_thresh"],
            base_bil=gcfg["base_bil"],
            exclude_assets=gcfg["exclude_assets"],
            mode=gcfg["mode"],
            trade_cost_bps=args.trade_cost_bps,
            label=lbl,
            overlay_mods=gcfg["overlay_mods"],
        )

        m = psa.evaluate_returns(returns, lbl)
        n_active = int(guard_log["guard_active"].sum()) if "guard_active" in guard_log.columns else 0

        result = {
            "label": lbl,
            "corr_thresh": gcfg["corr_thresh"],
            "vol_thresh": gcfg["vol_thresh"],
            "base_bil": gcfg["base_bil"],
            "exclude_name": gcfg["exclude_name"],
            "mode": gcfg["mode"],
            "overlay_mod": gcfg["overlay_mod_name"],
            "sharpe": m["sharpe"],
            "ann_return": m["return"],
            "vol": m["vol"],
            "mdd": m["mdd"],
            "calmar": m["calmar"],
            "triple": m["triple"],
            "months_guard_active": n_active,
            "pct_guard_active": n_active / len(ctx.month_ends),
        }
        all_results.append(result)

        if m["triple"]:
            triple_count += 1

        flag = " *** TRIPLE ***" if m["triple"] else ""
        print(f"  [{i+1}/{len(grid)}] {lbl}: Sh={m['sharpe']:.3f} Ret={m['return']:.2%} MDD={m['mdd']:.2%} guard={n_active}/{len(ctx.month_ends)}{flag}")

        # Save details for triple or baseline
        if m["triple"] or is_baseline:
            detail_dir = args.out_dir / "details" / lbl
            detail_dir.mkdir(parents=True, exist_ok=True)
            returns.to_csv(detail_dir / "port_returns.csv", header=["return"])
            weights.to_csv(detail_dir / "port_weights.csv")
            diagnostics.to_csv(detail_dir / "diagnostics.csv")
            guard_log.to_csv(detail_dir / "guard_log.csv")
            subperiod_rows = _compute_subperiod_metrics(returns, lbl)
            pd.DataFrame(subperiod_rows).to_csv(detail_dir / "subperiod_analysis.csv", index=False)
            with open(detail_dir / "metrics.json", "w") as f:
                json.dump(result, f, indent=2, default=_json_default)

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.out_dir / "all_results.csv", index=False)

    # Summary
    triple_df = results_df[results_df["triple"]].sort_values("sharpe", ascending=False)
    if not triple_df.empty:
        print(f"\n{'='*72}")
        print(f"TRIPLE TARGET ACHIEVED! ({triple_count} configs)")
        print(f"{'='*72}")
        best = triple_df.iloc[0]
        print(f"  Best: {best['label']}")
        print(f"  Sharpe: {best['sharpe']:.4f}  Return: {best['ann_return']:.2%}  MDD: {best['mdd']:.2%}")
        print(f"  Guard: {best['months_guard_active']}/{len(ctx.month_ends)} months ({best['pct_guard_active']:.0%})")
        with open(args.out_dir / "best_triple.json", "w") as f:
            json.dump(best.to_dict(), f, indent=2, default=_json_default)
        triple_df.to_csv(args.out_dir / "triple_frontier.csv", index=False)
    else:
        print(f"\n{'='*72}")
        print("NO TRIPLE ACHIEVED")
        print(f"{'='*72}")

    # Pareto frontier analysis
    print("\n  === Pareto Frontier ===")
    for mdd_lb in [-13, -12, -11, -10.5, -10.0, -9.5, -9.0]:
        bucket = results_df[results_df["mdd"] >= mdd_lb / 100.0]
        if len(bucket) > 0:
            idx = bucket["ann_return"].idxmax()
            b = bucket.loc[idx]
            print(f"  MDD>={mdd_lb:.1f}%: Ret={b['ann_return']:.2%} Sh={b['sharpe']:.3f} guard={b['months_guard_active']:.0f}mo [{b['label']}]")

    # Near-triple
    near = results_df[
        (results_df["sharpe"] >= 0.95) &
        (results_df["ann_return"] >= 0.09) &
        (results_df["mdd"] >= -0.11)
    ].sort_values("mdd", ascending=False)
    if not near.empty:
        near.to_csv(args.out_dir / "near_triple_frontier.csv", index=False)
        print(f"\n  Near-triple configs: {len(near)}")
        for _, r in near.head(10).iterrows():
            print(f"    Sh={r['sharpe']:.3f} Ret={r['ann_return']:.2%} MDD={r['mdd']:.2%} guard={r['months_guard_active']:.0f}mo [{r['label']}]")

    # Save run metadata
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_configs": len(grid),
        "triple_count": triple_count,
        "trade_cost_bps": args.trade_cost_bps,
    }
    with open(args.out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
