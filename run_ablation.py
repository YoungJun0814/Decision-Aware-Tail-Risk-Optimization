"""
Ablation Study — Component-Level Contribution Analysis
=======================================================
각 컴포넌트를 하나씩 제거하여 마진기여도를 측정합니다.

실험:
  1. full           — 전체 모델 (baseline)
  2. no_crisis      — CrisisOverlay OFF (regime_dim=0)
  3. no_momentum    — Momentum features OFF
  4. no_dd_control  — Drawdown defense OFF (thresholds disabled)
  5. no_regime_lev  — Regime-adaptive leverage OFF (fixed max_leverage)

Usage:
  python run_ablation.py
"""

import os
import sys
import json
import copy
import time
import numpy as np
import torch

# --- 기존 run_walkforward 모듈 재사용 ---
from run_walkforward import (
    CONFIG as BASE_CONFIG,
    ASSET_TICKERS,
    define_folds,
    train_fold,
    apply_vol_targeting,
    evaluate_portfolio,
)
from src.data_loader import (
    prepare_training_data,
    get_monthly_asset_data,
    get_regime_4state,
)
from src.utils import get_device


# =============================================================================
# Ablation Experiments Definition
# =============================================================================

EXPERIMENTS = {
    'full': {
        'desc': 'Full Model (all components)',
        'overrides': {},  # No changes
    },
    'no_crisis': {
        'desc': 'No CrisisOverlay (regime_dim=0)',
        'overrides': {
            'regime_dim': 0,
        },
    },
    'no_momentum': {
        'desc': 'No Momentum Features',
        'overrides': {
            'use_momentum': False,
        },
    },
    'no_dd_control': {
        'desc': 'No Drawdown Defense (thresholds disabled)',
        'overrides': {
            'dd_threshold_1': 1.0,   # never triggers
            'dd_threshold_2': 1.0,   # never triggers
        },
    },
    'no_regime_lev': {
        'desc': 'No Regime-Adaptive Leverage (fixed 1.5x)',
        'overrides': {
            'bull_leverage': 1.5,    # same as max_leverage
            'crisis_leverage': 1.5,  # same as max_leverage
        },
    },
}


# =============================================================================
# Single Experiment Runner
# =============================================================================

def run_experiment(exp_name, overrides, base_config):
    """Run one ablation experiment with CONFIG overrides."""
    config = copy.deepcopy(base_config)
    config.update(overrides)
    
    # --- Data Loading (may differ based on use_momentum) ---
    _, asset_returns_df = get_monthly_asset_data(
        ASSET_TICKERS, config['start_date'], config['end_date'])
    
    X, y, vix, scaler, asset_names, y_dates, _ = prepare_training_data(
        start_date=config['start_date'],
        end_date=config['end_date'],
        seq_length=config['seq_length'],
        normalize=True,
        use_momentum=config.get('use_momentum', False),
    )
    
    y_raw = asset_returns_df.reindex(y_dates).values
    nan_mask = np.isnan(y_raw).any(axis=1)
    if nan_mask.any():
        y_raw = np.nan_to_num(y_raw, nan=0.0)
    
    n_samples = min(len(X), len(y_raw))
    X = X[:n_samples]
    y = y[:n_samples]
    vix = vix[:n_samples]
    y_raw = y_raw[:n_samples]
    dates = y_dates[:n_samples]
    
    # --- 4-State Regime ---
    regime_df = get_regime_4state()
    if regime_df.empty:
        regime_probs_np = np.ones((n_samples, 4)) / 4
    else:
        regime_aligned = regime_df.reindex(dates).ffill().bfill()
        regime_probs_np = regime_aligned.values
        if np.isnan(regime_probs_np).any():
            regime_probs_np = np.ones((n_samples, 4)) / 4
    
    regime_probs_t = torch.FloatTensor(regime_probs_np)
    
    # --- Walk-Forward ---
    folds = define_folds(dates, config['seq_length'])
    seeds = [42, 43, 44]
    
    all_seed_weights = []
    all_seed_results = []
    
    for seed in seeds:
        oos_weights_list = []
        oos_returns_list = []
        oos_regime_list = []
        
        for fold in folds:
            weights, history = train_fold(
                X, y, y_raw, vix, regime_probs_t, fold, config, seed)
            test_returns = y_raw[fold['test_idx']]
            oos_weights_list.append(weights)
            oos_returns_list.append(test_returns)
            oos_regime_list.append(regime_probs_np[fold['test_idx']])
        
        all_weights = np.vstack(oos_weights_list)
        all_returns = np.vstack(oos_returns_list)
        all_seed_weights.append(all_weights)
        
        # Before VT
        before = evaluate_portfolio(all_weights, all_returns, f"")
        
        # After VT
        if config['vol_targeting']:
            adj_w, _ = apply_vol_targeting(
                all_weights, all_returns,
                target_vol=config['target_vol'],
                lookback=config['vol_lookback'],
                max_leverage=config['max_leverage'],
                dd_threshold_1=config['dd_threshold_1'],
                dd_threshold_2=config['dd_threshold_2'],
                dd_recovery_months=config['dd_recovery_months'],
                regime_probs=np.vstack(oos_regime_list),
                bull_leverage=config['bull_leverage'],
                crisis_leverage=config['crisis_leverage'],
            )
            after = evaluate_portfolio(adj_w, all_returns, f"")
        else:
            after = before
        
        all_seed_results.append({'before': before, 'after': after})
    
    # Ensemble
    all_test_returns = np.vstack([y_raw[f['test_idx']] for f in folds])
    all_test_regime = np.vstack([regime_probs_np[f['test_idx']] for f in folds])
    ensemble_w = np.mean(all_seed_weights, axis=0)
    
    if config['vol_targeting']:
        adj_ens, _ = apply_vol_targeting(
            ensemble_w, all_test_returns,
            target_vol=config['target_vol'],
            lookback=config['vol_lookback'],
            max_leverage=config['max_leverage'],
            dd_threshold_1=config['dd_threshold_1'],
            dd_threshold_2=config['dd_threshold_2'],
            dd_recovery_months=config['dd_recovery_months'],
            regime_probs=all_test_regime,
            bull_leverage=config['bull_leverage'],
            crisis_leverage=config['crisis_leverage'],
        )
        ensemble = evaluate_portfolio(adj_ens, all_test_returns, "")
    else:
        ensemble = evaluate_portfolio(ensemble_w, all_test_returns, "")
    
    # Aggregate across seeds (after VT)
    metrics = {}
    for key in ['sharpe', 'annual_return', 'mdd']:
        vals = [r['after'][key] for r in all_seed_results]
        metrics[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    metrics['ensemble'] = {
        'sharpe': float(ensemble['sharpe']),
        'annual_return': float(ensemble['annual_return']),
        'mdd': float(ensemble['mdd']),
    }
    
    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  Ablation Study: Component-Level Contribution Analysis")
    print("=" * 70)
    
    results = {}
    
    for i, (exp_name, exp_info) in enumerate(EXPERIMENTS.items()):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(EXPERIMENTS)}] {exp_name}: {exp_info['desc']}")
        print(f"{'='*60}")
        
        t0 = time.time()
        metrics = run_experiment(exp_name, exp_info['overrides'], BASE_CONFIG)
        elapsed = time.time() - t0
        
        results[exp_name] = {
            'description': exp_info['desc'],
            'overrides': {k: str(v) for k, v in exp_info['overrides'].items()},
            'metrics': metrics,
            'elapsed_sec': round(elapsed, 1),
        }
        
        ens = metrics['ensemble']
        print(f"\n  Result: Sharpe={ens['sharpe']:.4f}  "
              f"Return={ens['annual_return']:.2%}  "
              f"MDD={ens['mdd']:.2%}  "
              f"({elapsed:.0f}s)")
    
    # --- Summary Table ---
    print("\n\n" + "=" * 70)
    print("  ABLATION SUMMARY")
    print("=" * 70)
    
    full = results.get('full', {}).get('metrics', {}).get('ensemble', {})
    
    print(f"\n  {'Experiment':<20s} {'Sharpe':>8s} {'Return':>8s} {'MDD':>8s}  "
          f"{'dSharpe':>8s} {'dReturn':>8s} {'dMDD':>8s}")
    print(f"  {'-'*80}")
    
    for exp_name, data in results.items():
        ens = data['metrics']['ensemble']
        s, r, m = ens['sharpe'], ens['annual_return'], ens['mdd']
        
        if full and exp_name != 'full':
            ds = s - full.get('sharpe', 0)
            dr = r - full.get('annual_return', 0)
            dm = m - full.get('mdd', 0)
            delta = f"  {ds:>+8.4f} {dr:>+8.2%} {dm:>+8.2%}"
        else:
            delta = f"  {'(base)':>8s} {'(base)':>8s} {'(base)':>8s}"
        
        print(f"  {exp_name:<20s} {s:>8.4f} {r:>8.2%} {m:>8.2%}{delta}")
    
    # --- Save ---
    os.makedirs('results/ablation', exist_ok=True)
    with open('results/ablation/summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to results/ablation/summary.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
