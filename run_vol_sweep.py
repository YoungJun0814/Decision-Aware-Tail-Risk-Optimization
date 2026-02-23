"""
Target Volatility Sweep — Efficient Frontier Analysis
=======================================================
다양한 Target Volatility 수준(5%~15%)에서 포트폴리오 성과를 시뮬레이션하고
Efficient Frontier를 시각화합니다.

핵심: 모델의 "raw weights"는 한 번만 학습하고 캐시합니다.
      각 target_vol에 대해 apply_vol_targeting()만 재적용합니다.

Usage:
    python run_vol_sweep.py
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import json
import warnings

warnings.filterwarnings('ignore')

from src.data_loader import (
    prepare_training_data, get_monthly_asset_data,
    get_regime_4state, ASSET_TICKERS,
)
from src.models import get_model
from src.loss import DecisionAwareLoss
from src.trainer import Trainer, TrajectoryBatchSampler
from src.utils import set_seed, get_device, calculate_mdd
from torch.utils.data import DataLoader, TensorDataset

from run_walkforward import (
    CONFIG, define_folds, train_fold,
    apply_vol_targeting, evaluate_portfolio,
)
from run_baselines import (
    strategy_60_40, strategy_risk_parity,
    strategy_min_variance, strategy_hrp, strategy_momentum,
)


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = 'results/vol_sweep'
CACHE_DIR = 'results/vol_sweep/cache'

# Target volatility sweep range
TARGET_VOLS = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
               0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20]

# Baselines to compare (fixed points on the frontier)
BASELINE_STRATEGIES = {
    '1/N Equal': strategy_60_40,  # placeholder, computed separately
    '60/40': strategy_60_40,
    'Risk Parity': strategy_risk_parity,
    'Min Variance': strategy_min_variance,
    'HRP': strategy_hrp,
    'Momentum': strategy_momentum,
}


# =============================================================================
# Walk-Forward Weight Generation (with caching)
# =============================================================================

def get_or_generate_weights():
    """
    Walk-forward OOS weights를 캐시에서 로드하거나 새로 생성합니다.
    
    Returns:
        raw_weights: (N_oos, N_assets) — Vol Targeting 전 raw weights (seed ensemble)
        oos_returns: (N_oos, N_assets) — OOS 구간 실제 수익률
        oos_regime: (N_oos, 4) — OOS 구간 regime 확률
        oos_dates: DatetimeIndex — OOS 구간 날짜
    """
    cache_file = os.path.join(CACHE_DIR, 'oos_data.npz')
    dates_file = os.path.join(CACHE_DIR, 'oos_dates.csv')
    
    if os.path.exists(cache_file) and os.path.exists(dates_file):
        print("[Cache] 기존 OOS 데이터를 로드합니다...")
        data = np.load(cache_file)
        dates_df = pd.read_csv(dates_file, parse_dates=['date'])
        return (
            data['raw_weights'],
            data['oos_returns'],
            data['oos_regime'],
            pd.DatetimeIndex(dates_df['date']),
        )
    
    print("[Walk-Forward] OOS weights 생성 중 (첫 실행 시에만 소요)...")
    
    # --- Data Loading (same as run_walkforward.py) ---
    _, asset_returns_df = get_monthly_asset_data(
        ASSET_TICKERS, CONFIG['start_date'], CONFIG['end_date'])
    
    X, y, vix, scaler, asset_names, y_dates, macro_tensor = prepare_training_data(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        seq_length=CONFIG['seq_length'],
        normalize=True,
        use_momentum=CONFIG.get('use_momentum', False),
    )
    
    y_raw = asset_returns_df.reindex(y_dates)              # row align by date
    y_raw = y_raw.reindex(columns=ASSET_TICKERS)           # col align: ASSET_TICKERS order!
    y_raw = y_raw.values
    y_raw = np.nan_to_num(y_raw, nan=0.0)
    
    n_samples = min(len(X), len(y_raw))
    X = X[:n_samples]
    y = y[:n_samples]
    vix = vix[:n_samples]
    y_raw = y_raw[:n_samples]
    dates = y_dates[:n_samples]
    
    # 4-State Regime
    regime_df = get_regime_4state()
    if regime_df.empty:
        regime_probs_np = np.ones((n_samples, CONFIG['regime_dim'])) / CONFIG['regime_dim']
    else:
        regime_aligned = regime_df.reindex(dates).ffill().bfill()
        regime_probs_np = regime_aligned.values
        if np.isnan(regime_probs_np).any():
            regime_probs_np = np.ones((n_samples, CONFIG['regime_dim'])) / CONFIG['regime_dim']
    
    regime_probs_t = torch.FloatTensor(regime_probs_np)
    
    # Walk-Forward Folds
    folds = define_folds(dates, CONFIG['seq_length'])
    
    # Multi-seed Walk-Forward
    all_seed_weights = []
    
    for seed_i in range(CONFIG['n_seeds']):
        seed = 42 + seed_i
        print(f"\n  Seed {seed_i + 1}/{CONFIG['n_seeds']} (seed={seed})")
        
        oos_weights_list = []
        
        for fold_i, fold in enumerate(folds):
            print(f"    Fold {fold_i + 1}/{len(folds)}: ", end="")
            
            weights, history = train_fold(
                X, y, y_raw, vix, regime_probs_t, fold, CONFIG, seed)
            
            best_epoch = len(history['train_losses'])
            print(f"trained {best_epoch} epochs")
            
            oos_weights_list.append(weights)
        
        all_seed_weights.append(np.vstack(oos_weights_list))
    
    # Seed Ensemble (weight averaging)
    ensemble_weights = np.mean(all_seed_weights, axis=0)
    
    # OOS data
    all_test_idx = np.concatenate([f['test_idx'] for f in folds])
    oos_returns = y_raw[all_test_idx]
    oos_regime = regime_probs_np[all_test_idx]
    oos_dates = dates[all_test_idx]
    
    # Cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_file,
             raw_weights=ensemble_weights,
             oos_returns=oos_returns,
             oos_regime=oos_regime)
    pd.DataFrame({'date': oos_dates}).to_csv(dates_file, index=False)
    print(f"\n[Cache] OOS 데이터 저장 완료: {cache_file}")
    
    return ensemble_weights, oos_returns, oos_regime, oos_dates


# =============================================================================
# Vol Sweep
# =============================================================================

def run_vol_sweep(raw_weights, oos_returns, oos_regime):
    """
    다양한 target_vol에 대해 apply_vol_targeting을 적용하고 성과 기록.
    """
    results = []
    
    for tv in TARGET_VOLS:
        adj_weights, scalars = apply_vol_targeting(
            raw_weights, oos_returns,
            target_vol=tv,
            lookback=CONFIG['vol_lookback'],
            max_leverage=CONFIG['max_leverage'],
            dd_threshold_1=CONFIG['dd_threshold_1'],
            dd_threshold_2=CONFIG['dd_threshold_2'],
            dd_recovery_months=CONFIG['dd_recovery_months'],
            regime_probs=oos_regime,
            bull_leverage=CONFIG['bull_leverage'],
            crisis_leverage=CONFIG['crisis_leverage'],
        )
        
        metrics = evaluate_portfolio(adj_weights, oos_returns)
        metrics['target_vol'] = tv
        metrics['avg_scalar'] = float(scalars.mean())
        results.append(metrics)
        
        print(f"  target_vol={tv:.0%}: Sharpe={metrics['sharpe']:.4f}, "
              f"Return={metrics['annual_return']:.2%}, MDD={metrics['mdd']:.2%}")
    
    return results


def compute_baseline_metrics(oos_returns, full_returns, all_test_idx, asset_names):
    """
    벤치마크 전략들의 성과 (Efficient Frontier 위의 고정점).
    """
    baseline_results = {}
    num_assets = len(ASSET_TICKERS)
    
    # 1/N Equal Weight
    equal_w = np.ones((len(oos_returns), num_assets)) / num_assets
    baseline_results['1/N Equal'] = evaluate_portfolio(equal_w, oos_returns)
    
    # Other strategies
    strategy_funcs = {
        '60/40': strategy_60_40,
        'Risk Parity': strategy_risk_parity,
        'Min Variance': strategy_min_variance,
        'HRP': strategy_hrp,
        'Momentum': strategy_momentum,
    }
    
    for name, func in strategy_funcs.items():
        full_weights = func(full_returns, asset_names=asset_names)
        oos_weights = full_weights[all_test_idx]
        baseline_results[name] = evaluate_portfolio(oos_weights, oos_returns)
    
    return baseline_results


# =============================================================================
# Visualization
# =============================================================================

def plot_efficient_frontier(sweep_results, baseline_results, save_path):
    """
    Efficient Frontier 시각화:
      - x축: MDD (위험)
      - y축: Annual Return (수익)
      - 우리 모델: 선 (target_vol 변화에 따른 궤적)
      - 벤치마크: 개별 점
    """
    # --- Style setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # --- Our model: Efficient Frontier curve ---
    mdds = [r['mdd'] for r in sweep_results]
    returns = [r['annual_return'] for r in sweep_results]
    sharpes = [r['sharpe'] for r in sweep_results]
    tvols = [r['target_vol'] for r in sweep_results]
    
    # Main frontier curve
    scatter = ax.scatter(mdds, returns, c=sharpes, cmap='RdYlGn', 
                         s=120, zorder=5, edgecolors='white', linewidth=1.5)
    ax.plot(mdds, returns, '-', color='#2196F3', linewidth=2.5, alpha=0.7, 
            label='Our Model (GRU+BL+CVaR)', zorder=4)
    
    # Annotate target_vol labels on selected points
    label_indices = [0, len(tvols)//4, len(tvols)//2, 3*len(tvols)//4, -1]
    for idx in label_indices:
        idx = idx if idx >= 0 else len(tvols) + idx
        if idx < len(tvols):
            ax.annotate(f'{tvols[idx]:.0%}',
                       (mdds[idx], returns[idx]),
                       textcoords="offset points", xytext=(10, 8),
                       fontsize=9, fontweight='bold', color='#1565C0',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='#1565C0', alpha=0.8))
    
    # --- Baselines: individual points ---
    baseline_markers = {
        '1/N Equal':    ('o', '#FF5722', 11),
        '60/40':        ('s', '#9C27B0', 11),
        'Risk Parity':  ('^', '#FF9800', 11),
        'Min Variance': ('D', '#4CAF50', 11),
        'HRP':          ('v', '#795548', 11),
        'Momentum':     ('P', '#607D8B', 11),
    }
    
    for name, metrics in baseline_results.items():
        marker, color, size = baseline_markers.get(name, ('o', 'gray', 10))
        ax.scatter(metrics['mdd'], metrics['annual_return'],
                  marker=marker, c=color, s=size**2, zorder=6,
                  edgecolors='white', linewidth=1.5,
                  label=f"{name} (S={metrics['sharpe']:.2f})")
    
    # --- Colorbar for Sharpe ---
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('Sharpe Ratio', fontsize=12, fontweight='bold')
    
    # --- Labels & formatting ---
    ax.set_xlabel('Maximum Drawdown (MDD) →', fontsize=13, fontweight='bold')
    ax.set_ylabel('← Annual Return', fontsize=13, fontweight='bold')
    ax.set_title('Efficient Frontier: Risk-Return Trade-off\n'
                 'GRU + BL + Mean-CVaR with Adaptive Vol Targeting',
                 fontsize=15, fontweight='bold', pad=15)
    
    # Format axes as percentages
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9,
              edgecolor='gray', fancybox=True)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[Saved] Efficient Frontier → {save_path}")
    plt.close()


def plot_sharpe_vs_vol(sweep_results, baseline_results, save_path):
    """
    Sharpe Ratio vs Target Volatility 차트.
    어느 target_vol에서 최적 Sharpe를 달성하는지 보여줍니다.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tvols = [r['target_vol'] for r in sweep_results]
    sharpes = [r['sharpe'] for r in sweep_results]
    
    # Our model curve
    ax.plot(tvols, sharpes, 'o-', color='#2196F3', linewidth=2.5,
            markersize=8, markerfacecolor='white', markeredgewidth=2,
            label='Our Model', zorder=5)
    
    # Highlight optimal
    best_idx = np.argmax(sharpes)
    ax.scatter([tvols[best_idx]], [sharpes[best_idx]], 
               s=300, c='#F44336', marker='*', zorder=6,
               label=f'Optimal (σ*={tvols[best_idx]:.0%}, S={sharpes[best_idx]:.3f})')
    
    # Baselines as horizontal lines
    baseline_colors = {
        '1/N Equal': '#FF5722', '60/40': '#9C27B0',
        'Risk Parity': '#FF9800', 'Min Variance': '#4CAF50',
        'HRP': '#795548', 'Momentum': '#607D8B',
    }
    
    for name, metrics in baseline_results.items():
        color = baseline_colors.get(name, 'gray')
        ax.axhline(y=metrics['sharpe'], color=color, linestyle='--',
                   alpha=0.6, linewidth=1.5,
                   label=f"{name} (S={metrics['sharpe']:.3f})")
    
    ax.set_xlabel('Target Volatility', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=13, fontweight='bold')
    ax.set_title('Sharpe Ratio Sensitivity to Target Volatility',
                 fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] Sharpe vs Vol → {save_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  Target Volatility Sweep — Efficient Frontier Analysis")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Step 1: Get OOS weights (cached or fresh) ---
    print("\n[Step 1] OOS Weights 준비")
    raw_weights, oos_returns, oos_regime, oos_dates = get_or_generate_weights()
    print(f"  OOS shape: weights={raw_weights.shape}, returns={oos_returns.shape}")
    print(f"  OOS period: {oos_dates[0].date()} ~ {oos_dates[-1].date()}")
    
    # --- Step 2: Vol Sweep ---
    print(f"\n[Step 2] Target Vol Sweep ({len(TARGET_VOLS)} levels)")
    print(f"  Range: {min(TARGET_VOLS):.0%} ~ {max(TARGET_VOLS):.0%}")
    sweep_results = run_vol_sweep(raw_weights, oos_returns, oos_regime)
    
    # --- Step 3: Baseline Metrics ---
    print(f"\n[Step 3] Baseline Strategies 계산")
    
    # Need full returns for strategies requiring lookback
    _, asset_returns_df = get_monthly_asset_data(
        ASSET_TICKERS, CONFIG['start_date'], CONFIG['end_date'])
    
    X_tmp, _, _, _, asset_names, y_dates_tmp, _ = prepare_training_data(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        seq_length=CONFIG['seq_length'],
        normalize=True,
        use_momentum=False,
    )
    
    y_raw_full = asset_returns_df.reindex(y_dates_tmp)          # row align
    y_raw_full = y_raw_full.reindex(columns=ASSET_TICKERS)      # col align
    y_raw_full = y_raw_full.values
    y_raw_full = np.nan_to_num(y_raw_full, nan=0.0)
    n_full = min(len(X_tmp), len(y_raw_full))
    y_raw_full = y_raw_full[:n_full]
    dates_full = y_dates_tmp[:n_full]
    
    folds = define_folds(dates_full, CONFIG['seq_length'])
    all_test_idx = np.concatenate([f['test_idx'] for f in folds])
    
    baseline_results = compute_baseline_metrics(
        oos_returns, y_raw_full, all_test_idx, asset_names)
    
    for name, m in baseline_results.items():
        print(f"  {name:<15s}: Sharpe={m['sharpe']:.4f}, "
              f"Return={m['annual_return']:.2%}, MDD={m['mdd']:.2%}")
    
    # --- Step 4: Visualize ---
    print(f"\n[Step 4] 시각화")
    
    plot_efficient_frontier(
        sweep_results, baseline_results,
        os.path.join(OUTPUT_DIR, 'efficient_frontier.png'))
    
    plot_sharpe_vs_vol(
        sweep_results, baseline_results,
        os.path.join(OUTPUT_DIR, 'sharpe_vs_vol.png'))
    
    # --- Step 5: Save summary ---
    summary = {
        'description': 'Target Volatility Sweep Results',
        'target_vols': TARGET_VOLS,
        'sweep': [{
            'target_vol': r['target_vol'],
            'sharpe': float(r['sharpe']),
            'annual_return': float(r['annual_return']),
            'annual_vol': float(r['annual_vol']),
            'mdd': float(r['mdd']),
        } for r in sweep_results],
        'baselines': {
            name: {
                'sharpe': float(m['sharpe']),
                'annual_return': float(m['annual_return']),
                'mdd': float(m['mdd']),
            } for name, m in baseline_results.items()
        },
        'optimal': {
            'target_vol': float(sweep_results[np.argmax([r['sharpe'] for r in sweep_results])]['target_vol']),
            'sharpe': float(max(r['sharpe'] for r in sweep_results)),
        },
    }
    
    summary_path = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # --- Final Report ---
    best = sweep_results[np.argmax([r['sharpe'] for r in sweep_results])]
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Optimal Target Vol:  {best['target_vol']:.0%}")
    print(f"  Best Sharpe:         {best['sharpe']:.4f}")
    print(f"  Annual Return:       {best['annual_return']:.2%}")
    print(f"  MDD:                 {best['mdd']:.2%}")
    print(f"\n  Outputs:")
    print(f"    {os.path.join(OUTPUT_DIR, 'efficient_frontier.png')}")
    print(f"    {os.path.join(OUTPUT_DIR, 'sharpe_vs_vol.png')}")
    print(f"    {summary_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
