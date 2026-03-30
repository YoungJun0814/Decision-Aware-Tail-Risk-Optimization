"""
Triple target precision search — Round 3.

Starts from R2 winner (hc_0.08 = soft_gf=0.50 + hard_cash=0.08, gap=0.12%p)
and combines with best orthogonal improvements from R2.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "scripts" / "run_phase18_nonleveraged_v2_benchmark.py"
DEFAULT_OUT_DIR = ROOT / "results_runpod" / "phase18_triple_precision_round3"
DEFAULT_UPSTREAM_DIR = (
    ROOT / "results_runpod" / "phase18_e2e_fresh_runpod_20260330_022828UTC" / "upstream_snapshot"
)
DEFAULT_SEED_PATH = ROOT / "results_runpod" / "phase18_triple_precision_round2" / "best_result.json"
DEFAULT_CACHE_DIR = ROOT / "results_runpod" / "phase18_triple_precision_round2"

TARGET_SHARPE = 1.0
TARGET_RETURN = 0.10
TARGET_MDD = -0.10


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def deep_copy(d): return json.loads(json.dumps(d))

def merge_nested(base, updates):
    merged = deep_copy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = merge_nested(merged[k], v)
        else:
            merged[k] = v
    return merged


def is_triple(sh, rt, md): return sh >= TARGET_SHARPE and rt >= TARGET_RETURN and md >= TARGET_MDD

def shortfall(sh, rt, md):
    gaps = []
    if md < TARGET_MDD: gaps.append(TARGET_MDD - md)
    if rt < TARGET_RETURN: gaps.append(TARGET_RETURN - rt)
    if sh < TARGET_SHARPE: gaps.append(TARGET_SHARPE - sh)
    return float(np.sqrt(sum(g**2 for g in gaps))) if gaps else 0.0


def evaluate(mod, ctx, cfg, tc=0.0):
    returns, weights, diagnostics = mod.simulate_overlay_v2_strategy(
        ctx=ctx, label="R3", regime=ctx.pit_regime, trade_cost_bps=tc,
        mix_cfg=cfg["mix"], sleeve_cfg=cfg["sleeve"], s1_cfg=cfg["s1"],
        s3_cfg=cfg["s3"], srb_cfg=cfg["srb"], stop_cfg=cfg["stop"],
        tilt_cfg=cfg.get("tilt"), s7_cfg=cfg.get("s7"),
        expert_cfg=cfg.get("expert"), brake_cfg=cfg.get("brake"),
        meta_cfg=cfg.get("meta"), allocator_cfg=cfg.get("allocator"),
        hier_cfg=cfg.get("hier"), policy_cfg=cfg.get("policy"),
        rebalance_cfg=cfg.get("rebalance"),
    )
    m = mod.psa.evaluate_returns(returns, "R3")
    sh, rt, md = float(m["sharpe"]), float(m["return"]), float(m["mdd"])
    return {"sharpe": sh, "return": rt, "mdd": md, "triple": is_triple(sh, rt, md),
            "shortfall": shortfall(sh, rt, md), "returns_obj": returns}


def generate_r3_candidates(seed_cfg):
    """
    R2 key findings:
    - hc_0.08: MDD -10.12% (best MDD), but Return 9.9995% (barely misses)
    - s3_low_0.80: Sharpe 1.0807, Return 10.07%, MDD -10.16% (same MDD, big Ret boost)
    - s3_low_0.85: Sharpe 1.0775, Return 10.04%, MDD -10.16%
    - sc_0.005: MDD -10.14%, Return 10.01%
    - rf_0.60: Sharpe 1.0769, Return 10.03%, MDD -10.21%

    Strategy: Combine hc_0.08 (best MDD) with s3_low variations and other MDD-neutral boosters.
    """
    candidates = []

    # === A: hc_0.08 + s3_low (the key combo) ===
    for s3_low in [0.80, 0.82, 0.83, 0.85, 0.87, 0.88]:
        candidates.append((f"hc08_s3low{s3_low:.2f}", {
            "stop": {"hard_cash": 0.08},
            "s3": {"low": s3_low},
        }))

    # === B: hc_0.07 + s3_low ===
    for s3_low in [0.80, 0.82, 0.85, 0.88]:
        candidates.append((f"hc07_s3low{s3_low:.2f}", {
            "stop": {"hard_cash": 0.07},
            "s3": {"low": s3_low},
        }))

    # === C: hc + s3_low + sc (triple combo) ===
    for hc in [0.07, 0.08]:
        for s3_low in [0.80, 0.85]:
            for sc in [0.005, 0.01]:
                candidates.append((f"hc{int(hc*100):02d}_s3{s3_low:.2f}_sc{sc:.3f}", {
                    "stop": {"hard_cash": hc, "soft_cash": sc},
                    "s3": {"low": s3_low},
                }))

    # === D: hc + s3_low + rf ===
    for hc in [0.07, 0.08]:
        for s3_low in [0.80, 0.85]:
            for rf in [0.60, 0.65]:
                candidates.append((f"hc{int(hc*100):02d}_s3{s3_low:.2f}_rf{rf:.2f}", {
                    "stop": {"hard_cash": hc, "reentry_fraction": rf},
                    "s3": {"low": s3_low},
                }))

    # === E: hc + s3_low + s7 ===
    for hc in [0.07, 0.08]:
        for s3_low in [0.80, 0.85]:
            for dd6, dd9 in [(0.95, 0.85), (0.92, 0.80)]:
                candidates.append((f"hc{int(hc*100):02d}_s3{s3_low:.2f}_s7{dd6:.2f}", {
                    "stop": {"hard_cash": hc},
                    "s3": {"low": s3_low},
                    "s7": {"dd0": 1.0, "dd3": 1.0, "dd6": dd6, "dd9": dd9, "allow_leverage": False, "gross_cap": 1.0},
                }))

    # === F: hc + s3_low + hgf ===
    for hc in [0.07, 0.08]:
        for s3_low in [0.80, 0.85]:
            for hgf in [0.18, 0.15]:
                candidates.append((f"hc{int(hc*100):02d}_s3{s3_low:.2f}_hgf{hgf:.2f}", {
                    "stop": {"hard_cash": hc, "hard_growth_frac": hgf},
                    "s3": {"low": s3_low},
                }))

    # === G: Ultra-fine hard_cash sweep ===
    for hc in [0.085, 0.09, 0.095, 0.10, 0.11, 0.12]:
        candidates.append((f"hc{hc:.3f}", {"stop": {"hard_cash": hc}}))

    # === H: Ultra-fine hard_cash + s3_low ===
    for hc in [0.09, 0.10, 0.11, 0.12]:
        for s3_low in [0.80, 0.85]:
            candidates.append((f"hc{hc:.2f}_s3{s3_low:.2f}", {
                "stop": {"hard_cash": hc},
                "s3": {"low": s3_low},
            }))

    # === I: Full quad: hc + s3_low + rf + sc ===
    for hc in [0.08, 0.09, 0.10]:
        for s3_low in [0.80, 0.85]:
            for rf in [0.60, 0.65]:
                for sc in [0.005, 0.01]:
                    candidates.append((f"q_hc{int(hc*100):02d}_s3{s3_low:.2f}_rf{rf:.2f}_sc{sc:.3f}", {
                        "stop": {"hard_cash": hc, "reentry_fraction": rf, "soft_cash": sc},
                        "s3": {"low": s3_low},
                    }))

    # === J: hc + sgf sweep (even lower soft_growth_frac) ===
    for hc in [0.08, 0.09, 0.10]:
        for sgf in [0.47, 0.45, 0.43]:
            candidates.append((f"hc{int(hc*100):02d}_sgf{sgf:.2f}", {
                "stop": {"hard_cash": hc, "soft_growth_frac": sgf},
            }))

    # === K: hc + sgf + s3_low ===
    for hc in [0.08, 0.09, 0.10]:
        for sgf in [0.47, 0.45]:
            for s3_low in [0.80, 0.85]:
                candidates.append((f"hc{int(hc*100):02d}_sgf{sgf:.2f}_s3{s3_low:.2f}", {
                    "stop": {"hard_cash": hc, "soft_growth_frac": sgf},
                    "s3": {"low": s3_low},
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
    print(f"[INFO] Seed: {seed_payload.get('candidate_name')}")
    print(f"[INFO] Metrics: Sh={seed_payload['sharpe']:.4f} Ret={seed_payload['return']:.4%} MDD={seed_payload['mdd']:.4%}")

    os.environ["PHASE17_STEP1_DIR"] = str(args.upstream_dir.resolve())
    mod = load_module("r3_bench", BENCHMARK_PATH)

    ctx_dir = args.out_dir / "ctx"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    for cp in [args.cache_dir / "ctx" / "cache", args.cache_dir / "cache" / "cache", args.cache_dir / "cache"]:
        if (cp / "daily_returns.csv").exists():
            lc = ctx_dir / "cache"
            if not lc.exists():
                shutil.copytree(cp, lc)
            print(f"[INFO] Cache: {cp}")
            break

    ctx = mod.psa.build_context(ctx_dir, refresh_data=False, refresh_regime=False)

    bl = evaluate(mod, ctx, seed_cfg, args.trade_cost_bps)
    print(f"[BL] Sh={bl['sharpe']:.4f} Ret={bl['return']:.4%} MDD={bl['mdd']:.4%} T={bl['triple']} gap={bl['shortfall']:.6f}")

    candidates = generate_r3_candidates(seed_cfg)
    if args.limit > 0:
        candidates = candidates[:args.limit]
    print(f"\n[SEARCH] {len(candidates)} candidates")

    results = [{"name": "R2_winner", "sharpe": bl["sharpe"], "return": bl["return"],
                "mdd": bl["mdd"], "triple": bl["triple"], "shortfall": bl["shortfall"], "config": seed_cfg}]
    triple_found = False

    for idx, (name, updates) in enumerate(candidates, 1):
        cfg = merge_nested(seed_cfg, updates)
        try:
            r = evaluate(mod, ctx, cfg, args.trade_cost_bps)
        except Exception as e:
            print(f"  [{idx}/{len(candidates)}] {name}: ERR {e}")
            continue

        results.append({"name": name, "sharpe": r["sharpe"], "return": r["return"],
                       "mdd": r["mdd"], "triple": r["triple"], "shortfall": r["shortfall"], "config": cfg})

        m = " *** TRIPLE ***" if r["triple"] else ""
        if idx % 10 == 0 or r["triple"] or idx == len(candidates):
            print(f"  [{idx}/{len(candidates)}] {name}: Sh={r['sharpe']:.4f} Ret={r['return']:.4%} MDD={r['mdd']:.4%} gap={r['shortfall']:.6f}{m}")

        if r["triple"] and not triple_found:
            triple_found = True
            print(f"\n  >>> TRIPLE! '{name}' <<<\n")

    results.sort(key=lambda r: (not r["triple"], r["shortfall"], -r["sharpe"]))

    # If triple found, micro-refine
    if triple_found:
        tw = [r for r in results if r["triple"]]
        print(f"\n[REFINE] {len(tw)} triple winners")
        for w in tw[:5]:
            wc = w["config"]
            for s1d in [-0.03, -0.01, 0.01, 0.03]:
                for sbd in [-0.02, 0, 0.02]:
                    if s1d == 0 and sbd == 0: continue
                    rc = deep_copy(wc)
                    rc["s1"]["bull_scale"] = float(rc["s1"]["bull_scale"]) + s1d
                    rc["srb"]["bull_mult"] = float(rc["srb"]["bull_mult"]) + sbd
                    try:
                        rr = evaluate(mod, ctx, rc, args.trade_cost_bps)
                    except: continue
                    results.append({"name": f"ref_{w['name']}_s1{s1d:+.2f}_sb{sbd:+.2f}",
                                   "sharpe": rr["sharpe"], "return": rr["return"], "mdd": rr["mdd"],
                                   "triple": rr["triple"], "shortfall": rr["shortfall"], "config": rc})
        results.sort(key=lambda r: (not r["triple"], r["shortfall"], -r["sharpe"]))

    print("\n" + "=" * 88)
    print("ROUND 3 FINAL RANKING")
    print("=" * 88)
    lb = []
    for rank, r in enumerate(results[:50], 1):
        ts = "TRIPLE" if r["triple"] else "---"
        print(f"  #{rank:2d} [{ts}] {r['name']}: Sh={r['sharpe']:.4f} Ret={r['return']:.4%} MDD={r['mdd']:.4%} gap={r['shortfall']:.6f}")
        lb.append({"rank": rank, "name": r["name"], "sharpe": r["sharpe"],
                   "return": r["return"], "mdd": r["mdd"], "triple": r["triple"], "shortfall": r["shortfall"]})

    pd.DataFrame(lb).to_csv(args.out_dir / "leaderboard.csv", index=False)
    best = results[0]
    with open(args.out_dir / "best_result.json", "w") as f:
        json.dump({"candidate_name": best["name"], "sharpe": best["sharpe"], "return": best["return"],
                   "mdd": best["mdd"], "triple": best["triple"], "shortfall": best["shortfall"],
                   "config_json": json.dumps(best["config"], sort_keys=True)}, f, indent=2)

    triples = [r for r in results if r["triple"]]
    if triples:
        with open(args.out_dir / "triple_winners.json", "w") as f:
            json.dump([{"name": r["name"], "sharpe": r["sharpe"], "return": r["return"],
                       "mdd": r["mdd"], "config_json": json.dumps(r["config"], sort_keys=True)} for r in triples], f, indent=2)
        print(f"\n*** {len(triples)} TRIPLE WINNER(S)! ***")
    else:
        print(f"\nNo triple. Best gap: {best['shortfall']:.6f}")

    print(f"Results: {args.out_dir}")


if __name__ == "__main__":
    main()
