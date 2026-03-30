"""
Triple target precision search — Round 2.

Starts from the Round 1 winner (soft_gf_0.50, gap=0.16%p) and tries
ultra-fine perturbations to close the remaining MDD gap.
"""
from __future__ import annotations

import argparse
import importlib.util
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
DEFAULT_OUT_DIR = ROOT / "results_runpod" / "phase18_triple_precision_round2"
DEFAULT_UPSTREAM_DIR = (
    ROOT / "results_runpod" / "phase18_e2e_fresh_runpod_20260330_022828UTC" / "upstream_snapshot"
)
DEFAULT_SEED_PATH = ROOT / "results_runpod" / "phase18_triple_precision_search" / "best_result.json"
DEFAULT_CACHE_DIR = ROOT / "results_runpod" / "phase18_triple_precision_search"

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
        label="TRIPLE_R2",
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
    metrics = mod.psa.evaluate_returns(returns, "TRIPLE_R2")
    sh = float(metrics["sharpe"])
    rt = float(metrics["return"])
    md = float(metrics["mdd"])
    return {
        "sharpe": sh, "return": rt, "mdd": md,
        "triple": is_triple(sh, rt, md),
        "shortfall": shortfall(sh, rt, md),
        "returns_obj": returns, "weights_obj": weights,
    }


def generate_round2_candidates(seed_cfg: Dict) -> List[Tuple[str, Dict]]:
    """Ultra-fine perturbations around the Round 1 winner."""
    candidates: List[Tuple[str, Dict]] = []

    # Current winner has soft_growth_frac=0.50, reentry_fraction=0.70
    # MDD=-10.16%, need to get to -10.00%

    # === A: Even lower soft_growth_frac ===
    for sgf in [0.47, 0.48, 0.45, 0.43]:
        candidates.append((f"sgf_{sgf:.2f}", {"stop": {"soft_growth_frac": sgf}}))

    # === B: Combine soft_gf with lower reentry_fraction ===
    for rf in [0.60, 0.55, 0.65, 0.50]:
        candidates.append((f"rf_{rf:.2f}", {"stop": {"reentry_fraction": rf}}))

    # === C: Combine soft_gf with soft_cash ===
    for sc in [0.01, 0.015, 0.02, 0.005]:
        candidates.append((f"sc_{sc:.3f}", {"stop": {"soft_cash": sc}}))

    # === D: Combine soft_gf with higher hard_cash ===
    for hc in [0.06, 0.07, 0.08]:
        candidates.append((f"hc_{hc:.2f}", {"stop": {"hard_cash": hc}}))

    # === E: Lower hard_growth_frac ===
    for hgf in [0.18, 0.15, 0.12, 0.10]:
        candidates.append((f"hgf_{hgf:.2f}", {"stop": {"hard_growth_frac": hgf}}))

    # === F: S7 drawdown controller (adds another layer) ===
    for dd6, dd9 in [(0.92, 0.80), (0.90, 0.75), (0.95, 0.85), (0.88, 0.72)]:
        candidates.append((f"s7_{dd6:.2f}_{dd9:.2f}", {
            "s7": {"dd0": 1.0, "dd3": 1.0, "dd6": dd6, "dd9": dd9, "allow_leverage": False, "gross_cap": 1.0}
        }))

    # === G: Sleeve cash in crisis ===
    for cc in [0.04, 0.05, 0.06]:
        candidates.append((f"sleeve_cc_{cc:.2f}", {"sleeve": {"cash_crisis": cc}}))

    # === H: S3 conviction lower-bound ===
    for s3_low in [0.85, 0.80, 0.88]:
        candidates.append((f"s3_low_{s3_low:.2f}", {"s3": {"low": s3_low}}))

    # === I: Tilt stronger crisis defense ===
    for cds in [0.25, 0.30, 0.35]:
        candidates.append((f"tilt_cds_{cds:.2f}", {"tilt": {"crisis_def_strength": cds}}))

    # === J: Combined best pairs ===
    for sgf in [0.47, 0.48]:
        for rf in [0.60, 0.65]:
            candidates.append((f"sgf{sgf:.2f}_rf{rf:.2f}", {
                "stop": {"soft_growth_frac": sgf, "reentry_fraction": rf}
            }))

    for sgf in [0.47, 0.48]:
        for sc in [0.01, 0.015]:
            candidates.append((f"sgf{sgf:.2f}_sc{sc:.3f}", {
                "stop": {"soft_growth_frac": sgf, "soft_cash": sc}
            }))

    for sgf in [0.47, 0.48]:
        for hgf in [0.18, 0.15]:
            candidates.append((f"sgf{sgf:.2f}_hgf{hgf:.2f}", {
                "stop": {"soft_growth_frac": sgf, "hard_growth_frac": hgf}
            }))

    # === K: Triple combo: sgf + rf + sc ===
    for sgf in [0.47, 0.48]:
        for rf in [0.60, 0.65]:
            for sc in [0.01, 0.015]:
                candidates.append((f"tri_sgf{sgf:.2f}_rf{rf:.2f}_sc{sc:.3f}", {
                    "stop": {"soft_growth_frac": sgf, "reentry_fraction": rf, "soft_cash": sc}
                }))

    # === L: sgf + s7 ===
    for sgf in [0.47, 0.48]:
        for dd6, dd9 in [(0.92, 0.80), (0.90, 0.75)]:
            candidates.append((f"sgf{sgf:.2f}_s7_{dd6:.2f}_{dd9:.2f}", {
                "stop": {"soft_growth_frac": sgf},
                "s7": {"dd0": 1.0, "dd3": 1.0, "dd6": dd6, "dd9": dd9, "allow_leverage": False, "gross_cap": 1.0},
            }))

    # === M: sgf + sleeve crisis ===
    for sgf in [0.47, 0.48]:
        for cc in [0.04, 0.05]:
            candidates.append((f"sgf{sgf:.2f}_slcc{cc:.2f}", {
                "stop": {"soft_growth_frac": sgf},
                "sleeve": {"cash_crisis": cc},
            }))

    # === N: sgf + tilt ===
    for sgf in [0.47, 0.48]:
        for cds in [0.25, 0.30]:
            candidates.append((f"sgf{sgf:.2f}_tilt{cds:.2f}", {
                "stop": {"soft_growth_frac": sgf},
                "tilt": {"crisis_def_strength": cds},
            }))

    # === O: Quad combo: sgf + rf + s7 + sc ===
    for sgf in [0.47, 0.48]:
        for rf in [0.60, 0.65]:
            for dd6 in [0.92, 0.90]:
                candidates.append((f"quad_sgf{sgf:.2f}_rf{rf:.2f}_s7{dd6:.2f}", {
                    "stop": {"soft_growth_frac": sgf, "reentry_fraction": rf},
                    "s7": {"dd0": 1.0, "dd3": 1.0, "dd6": dd6, "dd9": 0.75, "allow_leverage": False, "gross_cap": 1.0},
                }))

    # === P: Policy fine-tuning with sgf ===
    for sgf in [0.47, 0.48]:
        for d_cap in [0.60, 0.62]:
            candidates.append((f"sgf{sgf:.2f}_pol{d_cap:.2f}", {
                "stop": {"soft_growth_frac": sgf},
                "policy": {"defense_risk_cap": d_cap},
            }))

    # === Q: Reentry threshold + sgf ===
    for sgf in [0.47, 0.48]:
        for rt in [0.010, 0.012]:
            candidates.append((f"sgf{sgf:.2f}_rth{rt:.3f}", {
                "stop": {"soft_growth_frac": sgf, "reentry_thresh": rt},
            }))

    # === R: All-in combos (5 axes at once) ===
    for sgf in [0.47, 0.48]:
        for rf in [0.60, 0.65]:
            for sc in [0.01, 0.005]:
                for cc in [0.04, 0.05]:
                    candidates.append((f"allin_sgf{sgf:.2f}_rf{rf:.2f}_sc{sc:.3f}_cc{cc:.2f}", {
                        "stop": {"soft_growth_frac": sgf, "reentry_fraction": rf, "soft_cash": sc},
                        "sleeve": {"cash_crisis": cc},
                    }))

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--seed-path", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--trade-cost-bps", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seed_payload = json.loads(args.seed_path.read_text(encoding="utf-8"))
    seed_cfg = json.loads(seed_payload["config_json"])
    print(f"[INFO] Seed: {seed_payload.get('candidate_name', 'unknown')}")
    print(f"[INFO] Seed metrics: Sharpe={seed_payload['sharpe']:.4f} Return={seed_payload['return']:.4%} MDD={seed_payload['mdd']:.4%}")

    os.environ["PHASE17_STEP1_DIR"] = str(args.upstream_dir.resolve())
    mod = load_module("precision_r2_benchmark", BENCHMARK_PATH)

    ctx_dir = args.out_dir / "ctx"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    # Find the actual cache with daily_returns.csv
    for candidate_cache in [
        args.cache_dir / "cache" / "cache",  # nested from round 1
        args.cache_dir / "cache",
        args.cache_dir / "contexts" / "baseline" / "cache",
    ]:
        if (candidate_cache / "daily_returns.csv").exists():
            local_cache = ctx_dir / "cache"
            if not local_cache.exists():
                shutil.copytree(candidate_cache, local_cache)
            print(f"[INFO] Cache from: {candidate_cache}")
            break

    ctx = mod.psa.build_context(ctx_dir, refresh_data=False, refresh_regime=False)

    # Baseline
    print("\n[BASELINE]")
    bl = evaluate(mod, ctx, seed_cfg, args.trade_cost_bps)
    print(f"  Sharpe={bl['sharpe']:.4f} Return={bl['return']:.4%} MDD={bl['mdd']:.4%} Triple={bl['triple']} gap={bl['shortfall']:.6f}")

    candidates = generate_round2_candidates(seed_cfg)
    if args.limit > 0:
        candidates = candidates[:args.limit]
    print(f"\n[SEARCH] {len(candidates)} candidates")

    results: List[Dict[str, Any]] = [{
        "name": "baseline_r1_winner",
        "sharpe": bl["sharpe"], "return": bl["return"], "mdd": bl["mdd"],
        "triple": bl["triple"], "shortfall": bl["shortfall"],
        "config": seed_cfg,
    }]
    triple_found = False

    for idx, (name, updates) in enumerate(candidates, 1):
        cfg = merge_nested(seed_cfg, updates)
        try:
            r = evaluate(mod, ctx, cfg, args.trade_cost_bps)
        except Exception as e:
            print(f"  [{idx}/{len(candidates)}] {name}: ERROR {e}")
            continue

        results.append({
            "name": name, "sharpe": r["sharpe"], "return": r["return"],
            "mdd": r["mdd"], "triple": r["triple"], "shortfall": r["shortfall"],
            "config": cfg,
        })

        marker = " *** TRIPLE ***" if r["triple"] else ""
        if idx % 10 == 0 or r["triple"] or idx == len(candidates):
            print(f"  [{idx}/{len(candidates)}] {name}: Sh={r['sharpe']:.4f} "
                  f"Ret={r['return']:.4%} MDD={r['mdd']:.4%} gap={r['shortfall']:.6f}{marker}")

        if r["triple"] and not triple_found:
            triple_found = True
            print(f"\n  >>> TRIPLE TARGET ACHIEVED: '{name}'! <<<\n")

    results.sort(key=lambda r: (not r["triple"], r["shortfall"], -r["sharpe"]))

    # If triple found, refine around winners
    if triple_found:
        triple_winners = [r for r in results if r["triple"]]
        print(f"\n[REFINE] {len(triple_winners)} triple winners found. Micro-tuning...")
        for tw in triple_winners[:3]:
            tw_cfg = tw["config"]
            # Try loosening slightly to maximize sharpe/return
            for s1bd in [-0.02, 0, 0.02, 0.04]:
                for srbd in [-0.01, 0, 0.01]:
                    if s1bd == 0 and srbd == 0:
                        continue
                    rcfg = deep_copy(tw_cfg)
                    rcfg["s1"]["bull_scale"] = float(rcfg["s1"]["bull_scale"]) + s1bd
                    rcfg["srb"]["bull_mult"] = float(rcfg["srb"]["bull_mult"]) + srbd
                    try:
                        rr = evaluate(mod, ctx, rcfg, args.trade_cost_bps)
                    except Exception:
                        continue
                    results.append({
                        "name": f"refine_{tw['name']}_s1b{s1bd:+.2f}_sb{srbd:+.2f}",
                        "sharpe": rr["sharpe"], "return": rr["return"], "mdd": rr["mdd"],
                        "triple": rr["triple"], "shortfall": rr["shortfall"], "config": rcfg,
                    })
        results.sort(key=lambda r: (not r["triple"], r["shortfall"], -r["sharpe"]))

    # Print ranking
    print("\n" + "=" * 88)
    print("ROUND 2 FINAL RANKING")
    print("=" * 88)

    lb_rows = []
    for rank, r in enumerate(results[:40], 1):
        ts = "TRIPLE" if r["triple"] else "---"
        print(f"  #{rank:2d} [{ts}] {r['name']}: Sh={r['sharpe']:.4f} Ret={r['return']:.4%} "
              f"MDD={r['mdd']:.4%} gap={r['shortfall']:.6f}")
        lb_rows.append({
            "rank": rank, "name": r["name"], "sharpe": r["sharpe"],
            "return": r["return"], "mdd": r["mdd"], "triple": r["triple"],
            "shortfall": r["shortfall"],
        })

    pd.DataFrame(lb_rows).to_csv(args.out_dir / "leaderboard.csv", index=False)

    best = results[0]
    with open(args.out_dir / "best_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "candidate_name": best["name"],
            "sharpe": best["sharpe"], "return": best["return"], "mdd": best["mdd"],
            "triple": best["triple"], "shortfall": best["shortfall"],
            "config_json": json.dumps(best["config"], sort_keys=True),
            "family": "fullstack_v2_nonlev", "regime_source": "pit",
        }, f, indent=2, ensure_ascii=False)

    triples = [r for r in results if r["triple"]]
    if triples:
        with open(args.out_dir / "triple_winners.json", "w", encoding="utf-8") as f:
            json.dump([{
                "name": r["name"], "sharpe": r["sharpe"], "return": r["return"],
                "mdd": r["mdd"], "config_json": json.dumps(r["config"], sort_keys=True),
            } for r in triples], f, indent=2, ensure_ascii=False)
        print(f"\n*** {len(triples)} TRIPLE WINNER(S)! ***")
    else:
        print(f"\nNo triple. Best gap: {best['shortfall']:.6f}")

    print(f"Results: {args.out_dir}")
    return best["triple"]


if __name__ == "__main__":
    main()
