"""
Triple target precision micro-search.

Starts from the best near-triple config (policy_refine_recovery) and makes
very small perturbations to stop / srb / s1 / policy parameters, targeting
the 0.25%p MDD gap that separates us from the triple target.

Usage:
    python scripts/run_triple_precision_search.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "scripts" / "run_phase18_nonleveraged_v2_benchmark.py"
DEFAULT_OUT_DIR = ROOT / "results_runpod" / "phase18_triple_precision_search"

# Fresh RunPod upstream snapshot
DEFAULT_UPSTREAM_DIR = (
    ROOT / "results_runpod" / "phase18_e2e_fresh_runpod_20260330_022828UTC" / "upstream_snapshot"
)
# Seed: policy_refine_recovery (best near-triple from structural tournament)
DEFAULT_SEED_PATH = (
    ROOT / "results_runpod" / "phase18_nonleveraged_v2_structural_tournament_fresh_20260330_local_v1"
    / "best_near_triple.json"
)
# Cache from structural tournament
DEFAULT_CACHE_DIR = (
    ROOT / "results_runpod" / "phase18_nonleveraged_v2_structural_tournament_fresh_20260330_local_v1"
)

TARGET_SHARPE = 1.0
TARGET_RETURN = 0.10
TARGET_MDD = -0.10


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def deep_copy(d: Dict) -> Dict:
    return json.loads(json.dumps(d))


def merge_nested(base: Dict, updates: Dict) -> Dict:
    merged = deep_copy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = merge_nested(merged[k], v)
        else:
            merged[k] = v
    return merged


def is_triple(sharpe: float, ret: float, mdd: float) -> bool:
    return sharpe >= TARGET_SHARPE and ret >= TARGET_RETURN and mdd >= TARGET_MDD


def shortfall(sharpe: float, ret: float, mdd: float) -> float:
    gaps = []
    if mdd < TARGET_MDD:
        gaps.append(TARGET_MDD - mdd)
    if ret < TARGET_RETURN:
        gaps.append(TARGET_RETURN - ret)
    if sharpe < TARGET_SHARPE:
        gaps.append(TARGET_SHARPE - sharpe)
    if not gaps:
        return 0.0
    return float(np.sqrt(sum(g ** 2 for g in gaps)))


def evaluate(mod, ctx, cfg: Dict, trade_cost_bps: float = 0.0) -> Dict[str, Any]:
    returns, weights, diagnostics = mod.simulate_overlay_v2_strategy(
        ctx=ctx,
        label="TRIPLE_SEARCH",
        regime=ctx.pit_regime,
        trade_cost_bps=trade_cost_bps,
        mix_cfg=cfg["mix"],
        sleeve_cfg=cfg["sleeve"],
        s1_cfg=cfg["s1"],
        s3_cfg=cfg["s3"],
        srb_cfg=cfg["srb"],
        stop_cfg=cfg["stop"],
        tilt_cfg=cfg.get("tilt"),
        s7_cfg=cfg.get("s7"),
        expert_cfg=cfg.get("expert"),
        brake_cfg=cfg.get("brake"),
        meta_cfg=cfg.get("meta"),
        allocator_cfg=cfg.get("allocator"),
        hier_cfg=cfg.get("hier"),
        policy_cfg=cfg.get("policy"),
        rebalance_cfg=cfg.get("rebalance"),
    )
    metrics = mod.psa.evaluate_returns(returns, "TRIPLE_SEARCH")
    sh = float(metrics["sharpe"])
    rt = float(metrics["return"])
    md = float(metrics["mdd"])

    # Sub-period MDD
    inflation_period = returns.loc[
        (returns.index >= pd.Timestamp("2022-01-31")) & (returns.index <= pd.Timestamp("2023-03-31"))
    ]
    inflation_mdd = float(mod.psa.evaluate_returns(inflation_period, "infl")["mdd"]) if len(inflation_period) > 0 else 0.0

    return {
        "sharpe": sh,
        "return": rt,
        "mdd": md,
        "triple": is_triple(sh, rt, md),
        "shortfall": shortfall(sh, rt, md),
        "inflation_mdd": inflation_mdd,
        "returns_obj": returns,
        "weights_obj": weights,
    }


def generate_candidates(seed_cfg: Dict) -> List[Tuple[str, Dict]]:
    """Generate ~80-120 micro-perturbation candidates around the seed."""
    candidates: List[Tuple[str, Dict]] = []

    # === AXIS 1: Stop-loss thresholds (tighter soft) ===
    for soft_thresh in [-0.018, -0.019, -0.0195]:
        name = f"stop_soft_{abs(soft_thresh):.4f}"
        candidates.append((name, {"stop": {"soft_sl_thresh": soft_thresh}}))

    # === AXIS 2: Hard stop parameters ===
    for hard_gf in [0.15, 0.18, 0.12]:
        for hard_cash in [0.06, 0.07, 0.08]:
            name = f"hard_gf{hard_gf:.2f}_cash{hard_cash:.2f}"
            candidates.append((name, {"stop": {"hard_growth_frac": hard_gf, "hard_cash": hard_cash}}))

    # === AXIS 3: Soft stop cash (currently 0) ===
    for soft_cash in [0.01, 0.02, 0.03]:
        name = f"soft_cash_{soft_cash:.2f}"
        candidates.append((name, {"stop": {"soft_cash": soft_cash}}))

    # === AXIS 4: Crisis regime scales for stop ===
    for crisis_soft in [0.80, 0.82, 0.78]:
        for crisis_hard in [0.75, 0.78, 0.72]:
            name = f"crisis_ss{crisis_soft:.2f}_hs{crisis_hard:.2f}"
            candidates.append((name, {"stop": {"crisis_soft_scale": crisis_soft, "crisis_hard_scale": crisis_hard}}))

    # === AXIS 5: Reentry fraction (less aggressive reentry) ===
    for reentry_frac in [0.60, 0.65, 0.55]:
        name = f"reentry_frac_{reentry_frac:.2f}"
        candidates.append((name, {"stop": {"reentry_fraction": reentry_frac}}))

    # === AXIS 6: Reentry threshold (higher = more cautious) ===
    for reentry_thresh in [0.010, 0.012, 0.006]:
        name = f"reentry_thresh_{reentry_thresh:.3f}"
        candidates.append((name, {"stop": {"reentry_thresh": reentry_thresh}}))

    # === AXIS 7: S1 crisis_scale (lower = more conservative in crisis) ===
    for crisis_scale in [0.60, 0.62, 0.58]:
        name = f"s1_crisis_{crisis_scale:.2f}"
        candidates.append((name, {"s1": {"crisis_scale": crisis_scale}}))

    # === AXIS 8: SRB crisis_mult (lower = more conservative) ===
    for crisis_mult in [0.40, 0.38, 0.42]:
        name = f"srb_crisis_{crisis_mult:.2f}"
        candidates.append((name, {"srb": {"crisis_mult": crisis_mult}}))

    # === AXIS 9: Policy defense tightening ===
    for defense_risk_cap in [0.60, 0.62, 0.58]:
        name = f"pol_def_cap_{defense_risk_cap:.2f}"
        candidates.append((name, {"policy": {"defense_risk_cap": defense_risk_cap}}))

    # === AXIS 10: Policy caution tightening ===
    for caution_risk_cap in [0.82, 0.84, 0.80]:
        name = f"pol_caut_cap_{caution_risk_cap:.2f}"
        candidates.append((name, {"policy": {"caution_risk_cap": caution_risk_cap}}))

    # === AXIS 11: S7 drawdown controller (currently off) ===
    for dd6, dd9 in [(0.85, 0.70), (0.90, 0.75), (0.80, 0.65)]:
        name = f"s7_dd6{dd6:.2f}_dd9{dd9:.2f}"
        candidates.append((name, {"s7": {"dd0": 1.0, "dd3": 1.0, "dd6": dd6, "dd9": dd9, "allow_leverage": False, "gross_cap": 1.0}}))

    # === AXIS 12: Combined stop tightening ===
    for soft_thresh in [-0.018, -0.019]:
        for soft_cash in [0.01, 0.02]:
            name = f"combo_st{abs(soft_thresh):.3f}_sc{soft_cash:.2f}"
            candidates.append((name, {"stop": {"soft_sl_thresh": soft_thresh, "soft_cash": soft_cash}}))

    # === AXIS 13: Combined stop + srb ===
    for soft_thresh in [-0.018, -0.019]:
        for crisis_mult in [0.40, 0.38]:
            name = f"combo_st{abs(soft_thresh):.3f}_srb{crisis_mult:.2f}"
            candidates.append((name, {"stop": {"soft_sl_thresh": soft_thresh}, "srb": {"crisis_mult": crisis_mult}}))

    # === AXIS 14: Combined stop + s1 ===
    for soft_thresh in [-0.018, -0.019]:
        for crisis_scale in [0.60, 0.62]:
            name = f"combo_st{abs(soft_thresh):.3f}_s1{crisis_scale:.2f}"
            candidates.append((name, {"stop": {"soft_sl_thresh": soft_thresh}, "s1": {"crisis_scale": crisis_scale}}))

    # === AXIS 15: Combined stop + policy ===
    for soft_thresh in [-0.018, -0.019]:
        for defense_risk_cap in [0.60, 0.62]:
            name = f"combo_st{abs(soft_thresh):.3f}_pol{defense_risk_cap:.2f}"
            candidates.append((name, {"stop": {"soft_sl_thresh": soft_thresh}, "policy": {"defense_risk_cap": defense_risk_cap}}))

    # === AXIS 16: Triple combo (stop + s1 + srb) ===
    for soft_thresh in [-0.018, -0.019]:
        for crisis_scale in [0.60, 0.62]:
            for crisis_mult in [0.40, 0.38]:
                name = f"tri_st{abs(soft_thresh):.3f}_s1{crisis_scale:.2f}_srb{crisis_mult:.2f}"
                candidates.append((name, {
                    "stop": {"soft_sl_thresh": soft_thresh},
                    "s1": {"crisis_scale": crisis_scale},
                    "srb": {"crisis_mult": crisis_mult},
                }))

    # === AXIS 17: Quad combo (stop + s1 + srb + policy) ===
    for soft_thresh in [-0.018, -0.019]:
        for crisis_scale in [0.60, 0.62]:
            for crisis_mult in [0.40, 0.38]:
                for defense_cap in [0.60, 0.62]:
                    name = f"quad_st{abs(soft_thresh):.3f}_s1{crisis_scale:.2f}_srb{crisis_mult:.2f}_pol{defense_cap:.2f}"
                    candidates.append((name, {
                        "stop": {"soft_sl_thresh": soft_thresh},
                        "s1": {"crisis_scale": crisis_scale},
                        "srb": {"crisis_mult": crisis_mult},
                        "policy": {"defense_risk_cap": defense_cap},
                    }))

    # === AXIS 18: Hard threshold tighter ===
    for hard_thresh in [-0.030, -0.032, -0.028]:
        name = f"hard_thresh_{abs(hard_thresh):.3f}"
        candidates.append((name, {"stop": {"hard_sl_thresh": hard_thresh}}))

    # === AXIS 19: Combined soft+hard threshold tighter ===
    for soft_thresh in [-0.018, -0.019]:
        for hard_thresh in [-0.030, -0.032]:
            name = f"both_soft{abs(soft_thresh):.3f}_hard{abs(hard_thresh):.3f}"
            candidates.append((name, {"stop": {"soft_sl_thresh": soft_thresh, "hard_sl_thresh": hard_thresh}}))

    # === AXIS 20: Soft growth frac (lower = more defensive when soft-stopped) ===
    for soft_gf in [0.50, 0.45, 0.48]:
        name = f"soft_gf_{soft_gf:.2f}"
        candidates.append((name, {"stop": {"soft_growth_frac": soft_gf}}))

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--seed-path", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--trade-cost-bps", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit candidates (0=all)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load seed config
    seed_payload = json.loads(args.seed_path.read_text(encoding="utf-8"))
    seed_cfg = json.loads(seed_payload["config_json"])
    print(f"[INFO] Seed: {seed_payload.get('candidate_name', seed_payload.get('label'))}")
    print(f"[INFO] Seed metrics: Sharpe={seed_payload['sharpe']:.4f} Return={seed_payload['return']:.4%} MDD={seed_payload['mdd']:.4%}")

    # Load engine
    os.environ["PHASE17_STEP1_DIR"] = str(args.upstream_dir.resolve())
    mod = load_module("precision_search_benchmark", BENCHMARK_PATH)

    # Build context
    ctx_dir = args.out_dir / "cache"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    cache_src = args.cache_dir / "contexts" / "baseline" / "cache"
    if not cache_src.exists():
        cache_src = args.cache_dir / "cache"
    if cache_src.exists() and not (ctx_dir / "cache").exists():
        local_cache = ctx_dir / "cache"
        if not local_cache.exists():
            shutil.copytree(cache_src, local_cache)

    ctx = mod.psa.build_context(ctx_dir, refresh_data=False, refresh_regime=False)

    # Evaluate baseline
    print("\n[STAGE 0] Evaluating baseline...")
    baseline_result = evaluate(mod, ctx, seed_cfg, args.trade_cost_bps)
    print(f"  Baseline: Sharpe={baseline_result['sharpe']:.4f} Return={baseline_result['return']:.4%} "
          f"MDD={baseline_result['mdd']:.4%} Triple={baseline_result['triple']} "
          f"Inflation_MDD={baseline_result['inflation_mdd']:.4%}")

    # Generate and evaluate candidates
    candidates = generate_candidates(seed_cfg)
    if args.limit > 0:
        candidates = candidates[:args.limit]
    print(f"\n[STAGE 1] Evaluating {len(candidates)} micro-perturbations...")

    results: List[Dict[str, Any]] = [{
        "name": "baseline",
        "sharpe": baseline_result["sharpe"],
        "return": baseline_result["return"],
        "mdd": baseline_result["mdd"],
        "triple": baseline_result["triple"],
        "shortfall": baseline_result["shortfall"],
        "inflation_mdd": baseline_result["inflation_mdd"],
        "config": seed_cfg,
    }]
    triple_found = False

    for idx, (name, updates) in enumerate(candidates, 1):
        cfg = merge_nested(seed_cfg, updates)
        try:
            result = evaluate(mod, ctx, cfg, args.trade_cost_bps)
        except Exception as e:
            print(f"  [{idx}/{len(candidates)}] {name}: ERROR {e}")
            continue

        results.append({
            "name": name,
            "sharpe": result["sharpe"],
            "return": result["return"],
            "mdd": result["mdd"],
            "triple": result["triple"],
            "shortfall": result["shortfall"],
            "inflation_mdd": result["inflation_mdd"],
            "config": cfg,
        })

        marker = " *** TRIPLE ***" if result["triple"] else ""
        if idx % 5 == 0 or result["triple"] or idx == len(candidates):
            print(f"  [{idx}/{len(candidates)}] {name}: Sharpe={result['sharpe']:.4f} "
                  f"Return={result['return']:.4%} MDD={result['mdd']:.4%} "
                  f"gap={result['shortfall']:.6f}{marker}")

        if result["triple"] and not triple_found:
            triple_found = True
            print(f"\n  >>> TRIPLE TARGET ACHIEVED by '{name}'! <<<\n")

    # Sort by shortfall (0 = triple achieved)
    results.sort(key=lambda r: (not r["triple"], r["shortfall"], -r["sharpe"]))

    # Stage 2: If we found triple winners, do micro-refinement around them
    stage2_results = []
    triple_winners = [r for r in results if r["triple"]]
    if triple_winners:
        print(f"\n[STAGE 2] Refining {len(triple_winners)} triple winners...")
        for winner in triple_winners[:3]:  # top 3 winners
            winner_cfg = winner["config"]
            # Micro-perturbations around winner to maximize Sharpe while keeping triple
            refinements = []
            # Try slightly loosening constraints to boost return/sharpe
            for s1_bull_delta in [-0.02, 0, 0.02]:
                for srb_bull_delta in [-0.01, 0, 0.01]:
                    if s1_bull_delta == 0 and srb_bull_delta == 0:
                        continue
                    rname = f"refine_{winner['name']}_s1b{s1_bull_delta:+.2f}_srbB{srb_bull_delta:+.2f}"
                    rupdate = {}
                    if s1_bull_delta != 0:
                        rupdate["s1"] = {"bull_scale": float(winner_cfg["s1"]["bull_scale"]) + s1_bull_delta}
                    if srb_bull_delta != 0:
                        rupdate["srb"] = {"bull_mult": float(winner_cfg["srb"]["bull_mult"]) + srb_bull_delta}
                    refinements.append((rname, rupdate))

            for ridx, (rname, rupdate) in enumerate(refinements, 1):
                rcfg = merge_nested(winner_cfg, rupdate)
                try:
                    rresult = evaluate(mod, ctx, rcfg, args.trade_cost_bps)
                except Exception:
                    continue
                entry = {
                    "name": rname,
                    "sharpe": rresult["sharpe"],
                    "return": rresult["return"],
                    "mdd": rresult["mdd"],
                    "triple": rresult["triple"],
                    "shortfall": rresult["shortfall"],
                    "inflation_mdd": rresult["inflation_mdd"],
                    "config": rcfg,
                }
                stage2_results.append(entry)
                results.append(entry)

        results.sort(key=lambda r: (not r["triple"], r["shortfall"], -r["sharpe"]))

    # Stage 3: If no triple yet, try deeper combos
    if not triple_found:
        print(f"\n[STAGE 2] No triple found. Trying deeper combinations...")
        # Take top 5 single-axis winners and combine them
        top5 = results[:5]
        combo_count = 0
        for i in range(len(top5)):
            for j in range(i + 1, len(top5)):
                if top5[i]["name"] == "baseline" or top5[j]["name"] == "baseline":
                    continue
                # Extract the delta from each
                cfg_i = top5[i]["config"]
                cfg_j = top5[j]["config"]
                # Merge: take non-seed values from both
                combo_cfg = deep_copy(seed_cfg)
                for key in cfg_i:
                    if isinstance(cfg_i[key], dict) and isinstance(seed_cfg.get(key), dict):
                        if cfg_i[key] != seed_cfg.get(key):
                            combo_cfg[key] = merge_nested(combo_cfg.get(key, {}), cfg_i[key])
                for key in cfg_j:
                    if isinstance(cfg_j[key], dict) and isinstance(seed_cfg.get(key), dict):
                        if cfg_j[key] != seed_cfg.get(key):
                            combo_cfg[key] = merge_nested(combo_cfg.get(key, {}), cfg_j[key])

                cname = f"deep_combo_{top5[i]['name']}+{top5[j]['name']}"
                try:
                    cresult = evaluate(mod, ctx, combo_cfg, args.trade_cost_bps)
                except Exception:
                    continue
                entry = {
                    "name": cname,
                    "sharpe": cresult["sharpe"],
                    "return": cresult["return"],
                    "mdd": cresult["mdd"],
                    "triple": cresult["triple"],
                    "shortfall": cresult["shortfall"],
                    "inflation_mdd": cresult["inflation_mdd"],
                    "config": combo_cfg,
                }
                results.append(entry)
                combo_count += 1
                if cresult["triple"]:
                    triple_found = True
                    print(f"  >>> TRIPLE by deep combo '{cname}'! <<<")

        print(f"  Evaluated {combo_count} deep combos")
        results.sort(key=lambda r: (not r["triple"], r["shortfall"], -r["sharpe"]))

    # Save results
    print("\n" + "=" * 88)
    print("FINAL RANKING")
    print("=" * 88)

    leaderboard_rows = []
    for rank, r in enumerate(results[:30], 1):
        triple_str = "TRIPLE" if r["triple"] else "---"
        print(f"  #{rank:2d} [{triple_str}] {r['name']}: "
              f"Sharpe={r['sharpe']:.4f} Return={r['return']:.4%} MDD={r['mdd']:.4%} "
              f"gap={r['shortfall']:.6f} infl_mdd={r['inflation_mdd']:.4%}")
        leaderboard_rows.append({
            "rank": rank,
            "name": r["name"],
            "sharpe": r["sharpe"],
            "return": r["return"],
            "mdd": r["mdd"],
            "triple": r["triple"],
            "shortfall": r["shortfall"],
            "inflation_mdd": r["inflation_mdd"],
        })

    # Save leaderboard
    df = pd.DataFrame(leaderboard_rows)
    df.to_csv(args.out_dir / "leaderboard.csv", index=False)

    # Save best config
    best = results[0]
    best_payload = {
        "label": "TRIPLE_PRECISION_SEARCH",
        "candidate_name": best["name"],
        "sharpe": best["sharpe"],
        "return": best["return"],
        "mdd": best["mdd"],
        "triple": best["triple"],
        "shortfall": best["shortfall"],
        "inflation_mdd": best["inflation_mdd"],
        "config_json": json.dumps(best["config"], sort_keys=True),
        "family": "fullstack_v2_nonlev",
        "regime_source": "pit",
        "trade_cost_bps": args.trade_cost_bps,
    }
    with open(args.out_dir / "best_result.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, ensure_ascii=False)

    # Save all triple winners
    triple_list = [r for r in results if r["triple"]]
    if triple_list:
        triple_payload = []
        for r in triple_list:
            triple_payload.append({
                "name": r["name"],
                "sharpe": r["sharpe"],
                "return": r["return"],
                "mdd": r["mdd"],
                "config_json": json.dumps(r["config"], sort_keys=True),
            })
        with open(args.out_dir / "triple_winners.json", "w", encoding="utf-8") as f:
            json.dump(triple_payload, f, indent=2, ensure_ascii=False)
        print(f"\n*** {len(triple_list)} TRIPLE WINNER(S) FOUND! ***")
    else:
        print(f"\nNo triple achieved. Best gap: {results[0]['shortfall']:.6f}")

    print(f"\nResults saved to {args.out_dir}")
    return results[0]["triple"]


if __name__ == "__main__":
    main()
