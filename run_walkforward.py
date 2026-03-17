"""
Walk-Forward Cross-Validation (v4: Regime-Conditional CVaR)
============================================================
GRU + CVaR + DecisionAwareLoss (v1 base)
+ Regime-Conditional λ_dd + Q Head Regime Concat + Crisis Overlay
+ hidden_dim=32 + Weight Decay + Cosine LR + Seed Ensemble

Usage:
    python run_walkforward.py
"""

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
import os
import json

warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import (
    prepare_training_data, get_monthly_asset_data, 
    get_regime_4state, ASSET_TICKERS,
)
from src.models import get_model
from src.loss import DecisionAwareLoss, PathAwareMDDLoss, SharpeLoss
from src.trainer import Trainer, TrajectoryBatchSampler
from src.utils import set_seed, get_device, calculate_mdd


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation Script")
    parser.add_argument('--model_type', type=str, default='gru',
                        help="Model type to run (e.g. gru, dual_shock_expert)")
    parser.add_argument('--dist_type', type=str, default='t', choices=['normal', 't'],
                        help="Distribution type for CVaR (normal or t)")
    parser.add_argument('--t_df', type=float, default=5.0,
                        help="Degrees of freedom for t-distribution")
    parser.add_argument('--no_correlation', action='store_true',
                        help="Disable rolling correlation features (Ablation study)")
    parser.add_argument('--no_daily_panic', action='store_true',
                        help="Disable daily panic features (Ablation study)")
    parser.add_argument('--no_nfci', action='store_true',
                        help="Disable NFCI feature (Ablation study)")
    parser.add_argument('--e2e_regime', action='store_true',
                        help="Enable End-to-End Regime learning (Gumbel-Softmax)")
    parser.add_argument('--ea_midas', action='store_true',
                        help="Enable EA-MIDAS attention features (velocity+acceleration)")
    parser.add_argument('--assets_13', action='store_true',
                        help="Expand asset universe 10 -> 13 (add TIPS, DBC, ACWX)")
    parser.add_argument('--start-date', type=str, default='2007-07-01',
                        help="Training data start date (inclusive, YYYY-MM-DD).")
    parser.add_argument('--end-date', type=str, default='2026-01-01',
                        help="Training data end date (exclusive, YYYY-MM-DD). Use 2026-01-01 to include Dec 2025.")
    parser.add_argument('--oos-start', type=str, default='2016-07-31',
                        help="Walk-forward OOS start date (YYYY-MM-DD).")
    parser.add_argument('--test-window-months', type=int, default=24,
                        help="Walk-forward test window length in months.")
    parser.add_argument('--n-seeds', type=int, default=3,
                        help="Number of random seeds for ensemble averaging.")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Training epochs per fold.")
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                        help="Early stopping patience per fold.")
    parser.add_argument('--lambda-dd', type=float, default=2.0,
                        help="Base drawdown penalty weight.")
    parser.add_argument('--lambda-risk', type=float, default=None,
                        help="BL-CVaR risk aversion (default: from CONFIG). "
                             "Lower → more aggressive (return↑, MDD↓).")
    parser.add_argument('--max-bil-floor', type=float, default=None,
                        help="Crisis Overlay max BIL allocation (default: from CONFIG). "
                             "Lower → less forced cash → higher return.")
    parser.add_argument('--realized-cost-bps', type=float, default=0.0,
                        help="Realized trading cost charged directly in the loss (bps).")
    parser.add_argument('--use-path-mdd', action='store_true',
                        help="Enable PathAwareMDDLoss-driven adaptive lambda_dd updates.")
    parser.add_argument('--train-verbose', action='store_true',
                        help="Print epoch-level training logs.")
    parser.add_argument('--output-dir', type=str, default='results/walkforward',
                        help="Directory to save metrics, fold results, and OOS artifacts.")
    parser.add_argument('--exp', type=str, default='baseline',
                        choices=['baseline', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
                        help=(
                            "Experiment preset. "
                            "baseline: original DecisionAwareLoss. "
                            "x1-x3: v6a round. "
                            "x4: GRU+SharpeLoss+lambda_risk=1.0+bil_floor=0.3+very loose overlay. "
                            "x5: DualShock+same. "
                            "x6: GRU+SharpeLoss+lambda_risk=0.5+no overlay (raw output)."
                        ))
    return parser.parse_args()

# =============================================================================
# Configuration  
# =============================================================================

CONFIG = {
    # Model (v4: GRU + CVaR base, regime conditioning)
    'model_type': 'gru',
    'hidden_dim': 32,
    'omega_mode': 'learnable',
    'sigma_mode': 'prior',
    'lambda_risk': 2.0,
    'regime_dim': 4,
    'max_bil_floor': 0.7,
    
    # Training
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 100,
    'early_stopping_patience': 15,
    'seq_length': 12,
    
    # Loss
    'loss_type': 'decision_aware',    # 'decision_aware' | 'sharpe'  — --exp로 덮어씀
    'eta': 1.0,
    'kappa_base': 0.001,
    'kappa_vix_scale': 0.0001,
    'lambda_dd': 2.0,
    'regime_dd_scale': 3.0,
    'realized_cost_bps': 0.0,
    'use_path_mdd': False,
    'path_mdd_grad_scale': 0.15,
    'return_target_monthly': 0.0,     # SharpeLoss 月 수익 하한 (0 = 비활성, 0.0083 ≈ 연10%)
    'return_hinge_weight': 2.0,       # 미달 패널티 강도
    
    # Vol Targeting + Drawdown Control (Phase1R + Phase2B)
    'vol_targeting': True,
    'target_vol': 0.10,
    'vol_lookback': 3,
    'max_leverage': 1.5,
    'dd_threshold_1': 0.03,    # Phase1R: 5% -> 3% (earlier trigger)
    'dd_threshold_2': 0.05,    # Phase1R: 8% -> 5% (earlier crisis)
    'dd_recovery_months': 2,
    'bull_leverage': 2.0,      # Phase2B: Bull regime max leverage
    'crisis_leverage': 1.0,    # Phase2B: Crisis regime max leverage
    
    # Features
    'use_momentum': True,      # Phase2A: 12-month momentum features
    'use_correlation': True,   # Phase7 STEP2: rolling correlation features
    'use_daily_panic': True,   # Phase7 STEP3: daily VIX/SPY panic features
    'use_nfci': True,          # Phase7 STEP3: NFCI financial conditions
    'e2e_regime': False,       # Phase7 STEP4: End-to-End regime learning
    'use_ea_midas': False,     # Phase7: EA-MIDAS attention features
    'assets_13': False,        # Phase7: 13-asset universe (TIPS, DBC, ACWX)
    'use_macro_regime': True,  # Phase5: macro → RegimeHead
    'macro_dim': 2,            # T10Y3M + BAA10Y
    
    # Walk-Forward
    'start_date': '2007-07-01',
    'end_date': '2026-01-01',
    'n_seeds': 3,
    'oos_start': '2016-07-31',
    'test_window_months': 24,
    'train_verbose': False,
    'output_dir': 'results/walkforward',
    
    'device': get_device(),
}


# =============================================================================
# Walk-Forward Folds
# =============================================================================

def define_folds_dynamic(dates, seq_length=12, oos_start='2016-07-31', test_window_months=24):
    """Expanding-window walk-forward folds with configurable OOS start and test span."""
    if test_window_months <= 0:
        raise ValueError(f"test_window_months must be positive, got {test_window_months}")

    dates_idx = pd.DatetimeIndex(pd.to_datetime(dates))
    last_date = dates_idx.max()
    current_start = pd.Timestamp(oos_start)

    folds = []
    while current_start <= last_date:
        ts = current_start
        te = current_start + pd.DateOffset(months=test_window_months)

        train_mask = dates_idx < ts
        test_mask = (dates_idx >= ts) & (dates_idx < te)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) > seq_length and len(test_idx) > 0:
            folds.append({
                'train_idx': train_idx,
                'test_idx': test_idx,
                'test_start': pd.Timestamp(dates_idx[test_idx[0]]),
                'test_end': pd.Timestamp(dates_idx[test_idx[-1]]),
            })

        current_start = te

    return folds


# =============================================================================
# Single Fold Training (v4)
# =============================================================================

def train_fold(X, y, y_raw, vix, regime_probs, fold, config, seed,
               macro_tensor=None):
    """
    단일 fold 학습 + 예측 (v4: regime + cosine scheduler, v5: macro).
    """
    set_seed(seed)
    device = config['device']
    
    train_idx = fold['train_idx']
    test_idx = fold['test_idx']
    
    # 데이터 분할
    X_train = X[train_idx]
    y_train = y[train_idx]
    vix_train = vix[train_idx]
    regime_train = regime_probs[train_idx]
    
    X_test = X[test_idx]
    regime_test = regime_probs[test_idx]
    
    # Macro features slicing
    macro_train = None
    macro_test = None
    if macro_tensor is not None:
        macro_train = macro_tensor[train_idx]
        macro_test = macro_tensor[test_idx]
    
    # val split (마지막 15%)
    n_train = len(X_train)
    n_val = max(int(n_train * 0.15), 1)
    n_actual_train = n_train - n_val
    
    # DataLoader (4-tuple or 5-tuple)
    if macro_train is not None:
        train_ds = TensorDataset(
            X_train[:n_actual_train], 
            y_train[:n_actual_train],
            vix_train[:n_actual_train],
            regime_train[:n_actual_train],
            macro_train[:n_actual_train],
        )
        val_ds = TensorDataset(
            X_train[n_actual_train:],
            y_train[n_actual_train:],
            vix_train[n_actual_train:],
            regime_train[n_actual_train:],
            macro_train[n_actual_train:],
        )
    else:
        train_ds = TensorDataset(
            X_train[:n_actual_train], 
            y_train[:n_actual_train],
            vix_train[:n_actual_train],
            regime_train[:n_actual_train],
        )
        val_ds = TensorDataset(
            X_train[n_actual_train:],
            y_train[n_actual_train:],
            vix_train[n_actual_train:],
            regime_train[n_actual_train:],
        )
    
    train_sampler = TrajectoryBatchSampler(
        n_actual_train, config['batch_size'], shuffle_chunks=True)
    
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    
    # Model — GRU + CVaR (v1 base) with regime_dim (v4) + macro_dim (v5)
    input_dim = X.shape[-1]
    num_assets = y.shape[-1]  # 동적: 10 (기본) or 13 (--assets_13)
    
    macro_dim = config.get('macro_dim', 0) if macro_tensor is not None else 0
    
    model = get_model(
        model_type=config['model_type'],
        input_dim=input_dim,
        num_assets=num_assets,
        device=device,
        omega_mode=config['omega_mode'],
        sigma_mode=config['sigma_mode'],
        lambda_risk=config['lambda_risk'],
        hidden_dim=config['hidden_dim'],
        regime_dim=config['regime_dim'],
        macro_dim=macro_dim,
        max_bil_floor=config.get('max_bil_floor', 0.5),
        dist_type=config.get('dist_type', 't'),
        t_df=config.get('t_df', 5.0),
        e2e_regime=config.get('e2e_regime', False),
    )
    
    # Loss — SharpeLoss (x1/x3) or DecisionAwareLoss (baseline/x2)
    if config.get('loss_type', 'decision_aware') == 'sharpe':
        loss_fn = SharpeLoss(
            lambda_dd=config['lambda_dd'],
            kappa_base=config['kappa_base'],
            kappa_vix_scale=config['kappa_vix_scale'],
            regime_dd_scale=config['regime_dd_scale'],
            return_target_monthly=config.get('return_target_monthly', 0.0),
            return_hinge_weight=config.get('return_hinge_weight', 2.0),
        )
    else:
        loss_fn = DecisionAwareLoss(
            eta=config['eta'],
            kappa_base=config['kappa_base'],
            kappa_vix_scale=config['kappa_vix_scale'],
            lambda_dd=config['lambda_dd'],
            regime_dd_scale=config['regime_dd_scale'],
            realized_cost_bps=config.get('realized_cost_bps', 0.0),
        )

    path_mdd_fn = None
    if config.get('use_path_mdd', False):
        path_mdd_fn = PathAwareMDDLoss(
            mdd_target=-0.10,
            mdd_lambda=5.0,
            soft_margin=0.02,
        )
    
    # Optimizer — v4: weight_decay 추가
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    
    # Scheduler — v4: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-5)
    
    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        optimizer=optimizer, device=device,
        path_mdd_loss_fn=path_mdd_fn,
        path_mdd_grad_scale=config.get('path_mdd_grad_scale', 0.15),
    )
    
    # Train with scheduler
    history = trainer.fit(
        train_loader, val_loader,
        epochs=config['epochs'],
        verbose=config.get('train_verbose', False) or config.get('use_path_mdd', False),
        early_stopping_patience=config['early_stopping_patience'],
        scheduler=scheduler,
    )
    
    # Predict on test set (v4: regime_probs, v5: macro_features)
    predict_kwargs = {'regime_probs': regime_test.to(device)}
    if macro_test is not None:
        predict_kwargs['macro_features'] = macro_test.to(device)
    
    weights = trainer.predict(
        X_test.to(device),
        **predict_kwargs,
    ).cpu().numpy()
    
    # --- 비중 진단 (13자산 모드 디버깅) ---
    w_sum = weights.sum(axis=1)
    if np.abs(w_sum - 1.0).max() > 0.01:
        print(f"  [WARNING] Weight sum deviation: mean={w_sum.mean():.4f}, "
              f"max_dev={np.abs(w_sum - 1.0).max():.4f}")
        # 강제 정규화: sum(w) != 1 이면 보정
        weights = weights / w_sum[:, np.newaxis]
    
    if weights.min() < -0.01:
        print(f"  [WARNING] Negative weights detected: min={weights.min():.4f}")
        weights = np.maximum(weights, 0.0)
        weights = weights / weights.sum(axis=1, keepdims=True)
    
    return weights, history


# =============================================================================
# Volatility Targeting
# =============================================================================

def apply_vol_targeting(weights, returns, target_vol=0.10, 
                        lookback=3, max_leverage=1.5,
                        dd_threshold_1=0.05, dd_threshold_2=0.08,
                        dd_recovery_months=2,
                        regime_probs=None,
                        bull_leverage=2.0, crisis_leverage=1.0):
    """
    Drawdown-Triggered Adaptive Vol Targeting + Regime Leverage (Phase 1R + 2B).
    
    Logic:
      - Normal: target_vol (base), max_leverage (base)
      - DD > 3%: target_vol * 0.5, max_leverage = min(base, 1.5)
      - DD > 5%: target_vol * 0.3, max_leverage = crisis_leverage
      - Recovery: 2 consecutive positive months -> restore
      - Bull regime (p_bull > 0.5): max_leverage = bull_leverage 
      - Crisis regime (p_crisis > 0.5): max_leverage = crisis_leverage
    """
    N, A = weights.shape
    adjusted_weights = weights.copy()
    scalars = np.ones(N)
    bil_index = A - 1
    
    # Track cumulative portfolio return for drawdown
    cum_return = np.zeros(N)
    port_rets = np.array([(weights[t] * returns[t]).sum() for t in range(N)])
    
    for t in range(N):
        cum_return[t] = (1 + port_rets[:t+1]).prod() - 1 if t > 0 else port_rets[0]
    
    # Rolling peak (look-back only)
    peak = np.zeros(N)
    peak[0] = max(0, cum_return[0])
    for t in range(1, N):
        peak[t] = max(peak[t-1], cum_return[t])
    
    # Drawdown series
    drawdown = np.zeros(N)
    for t in range(N):
        if peak[t] > 0:
            drawdown[t] = (peak[t] - cum_return[t]) / (1 + peak[t])
        elif cum_return[t] < 0:
            drawdown[t] = -cum_return[t]
    
    # Consecutive positive months counter
    consec_positive = 0
    in_defensive = False
    
    for t in range(lookback, N):
        # --- Phase 2B: Regime-adaptive leverage ---
        effective_max_lev = max_leverage
        if regime_probs is not None and t < len(regime_probs):
            rp = regime_probs[t]
            p_bull = rp[0] if len(rp) > 0 else 0.0  # index 0 = Bull
            p_crisis = rp[-1] if len(rp) > 0 else 0.0  # last = Crisis
            
            if p_bull > 0.5:
                effective_max_lev = bull_leverage
            elif p_crisis > 0.5:
                effective_max_lev = crisis_leverage
        
        # --- Drawdown-triggered vol target ---
        current_dd = drawdown[t-1]  # use PREVIOUS month's DD (no look-ahead)
        
        if current_dd > dd_threshold_2:
            effective_target = target_vol * 0.3  # Crisis mode
            effective_max_lev = min(effective_max_lev, crisis_leverage)  # cap
            in_defensive = True
            consec_positive = 0
        elif current_dd > dd_threshold_1:
            effective_target = target_vol * 0.5  # Defensive
            effective_max_lev = min(effective_max_lev, max_leverage)  # cap at normal
            in_defensive = True
            consec_positive = 0
        elif in_defensive:
            # Recovery check: consecutive positive months
            if t > 0 and port_rets[t-1] > 0:
                consec_positive += 1
            else:
                consec_positive = 0
            
            if consec_positive >= dd_recovery_months:
                effective_target = target_vol  # Restored
                in_defensive = False
                consec_positive = 0
            else:
                effective_target = target_vol * 0.5  # Still defensive
        else:
            effective_target = target_vol
        
        # --- Vol targeting ---
        past_port_ret = np.array([
            (weights[k] * returns[k]).sum() for k in range(t - lookback, t)
        ])
        realized_vol = past_port_ret.std() * np.sqrt(12)
        
        if realized_vol > 1e-6:
            scalar = min(effective_target / realized_vol, effective_max_lev)
        else:
            scalar = 1.0
        
        # Phase1R: Cap scalar during drawdown defense (regardless of vol)
        if current_dd > dd_threshold_2:
            scalar = min(scalar, 0.5)   # Crisis: never exceed 50% exposure
        elif current_dd > dd_threshold_1:
            scalar = min(scalar, 0.8)   # Defensive: never exceed 80% exposure
        
        scalars[t] = scalar
        
        risk_weights = weights[t] * scalar
        total_risk = risk_weights.sum()
        
        if total_risk > 1.0:
            risk_weights = risk_weights / total_risk
            cash = 0.0
        else:
            cash = 1.0 - total_risk
        
        adjusted_weights[t] = risk_weights
        adjusted_weights[t, bil_index] += cash

    return adjusted_weights, scalars


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_portfolio(weights, returns, label=""):
    """포트폴리오 성과 평가."""
    port_ret = (weights * returns).sum(axis=1)
    
    mean_ret = port_ret.mean() * 12
    std_ret = port_ret.std() * np.sqrt(12)
    sharpe = mean_ret / std_ret if std_ret > 1e-8 else 0.0
    mdd = -calculate_mdd(port_ret)
    
    result = {
        'sharpe': sharpe,
        'annual_return': mean_ret,
        'annual_vol': std_ret,
        'mdd': mdd,
        'n_months': len(port_ret),
    }
    
    if label:
        print(f"  [{label}]")
        print(f"    Sharpe:        {sharpe:.4f}")
        print(f"    Annual Return: {mean_ret:.2%}")
        print(f"    Annual Vol:    {std_ret:.2%}")
        print(f"    MDD:           {mdd:.2%}")
        print(f"    Test months:   {len(port_ret)}")
    
    return result


def is_triple(metrics):
    """Triple target check: Sharpe >= 1.0, Return >= 10%, MDD >= -10%."""
    return (
        metrics['sharpe'] >= 1.0
        and metrics['annual_return'] >= 0.10
        and metrics['mdd'] >= -0.10
    )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    
    # Update CONFIG based on args
    CONFIG['model_type'] = args.model_type
    if hasattr(args, 'no_correlation') and args.no_correlation:
        CONFIG['use_correlation'] = False
    if hasattr(args, 'no_daily_panic') and args.no_daily_panic:
        CONFIG['use_daily_panic'] = False
    if hasattr(args, 'no_nfci') and args.no_nfci:
        CONFIG['use_nfci'] = False
    if hasattr(args, 'e2e_regime') and args.e2e_regime:
        CONFIG['e2e_regime'] = True
    if hasattr(args, 'ea_midas') and args.ea_midas:
        CONFIG['use_ea_midas'] = True
    if hasattr(args, 'assets_13') and args.assets_13:
        CONFIG['assets_13'] = True
    CONFIG['dist_type'] = args.dist_type
    CONFIG['t_df'] = args.t_df
    CONFIG['start_date'] = args.start_date
    CONFIG['end_date'] = args.end_date
    CONFIG['oos_start'] = args.oos_start
    CONFIG['test_window_months'] = args.test_window_months
    CONFIG['n_seeds'] = args.n_seeds
    CONFIG['epochs'] = args.epochs
    CONFIG['early_stopping_patience'] = args.early_stopping_patience
    CONFIG['lambda_dd'] = args.lambda_dd
    if args.lambda_risk is not None:
        CONFIG['lambda_risk'] = args.lambda_risk
    if args.max_bil_floor is not None:
        CONFIG['max_bil_floor'] = args.max_bil_floor
    CONFIG['realized_cost_bps'] = args.realized_cost_bps
    CONFIG['use_path_mdd'] = args.use_path_mdd
    CONFIG['train_verbose'] = args.train_verbose
    CONFIG['output_dir'] = args.output_dir

    # -------------------------------------------------------------------------
    # Experiment Presets (--exp)
    # args 기반 CONFIG를 덮어씀 — 순서 중요: 이 블록은 항상 args 매핑 후에 위치
    # -------------------------------------------------------------------------
    _exp = getattr(args, 'exp', 'baseline')

    # 공통 "Loose Overlay" 설정 (x1/x2/x3 공유)
    _LOOSE_OVERLAY = dict(
        target_vol=0.13,         # 0.10 → 0.13: equity 50~65% 허용
        max_leverage=1.8,        # 1.5  → 1.8
        dd_threshold_1=0.05,     # 0.03 → 0.05: 빈번한 de-leverage 방지
        dd_threshold_2=0.08,     # 0.05 → 0.08
        use_path_mdd=False,      # Adaptive λ_dd 즉시포화 버그 방지
    )

    if _exp == 'x1':
        # X1: SharpeLoss (Sharpe 직접 최대화) + Return Hinge + Loose Overlay
        CONFIG.update(_LOOSE_OVERLAY)
        CONFIG['loss_type'] = 'sharpe'
        CONFIG['lambda_dd'] = 0.3           # MDD 이미 -7.4%로 여유 있으므로 완화
        CONFIG['regime_dd_scale'] = 2.0
        CONFIG['return_target_monthly'] = 0.0083   # 연 10% ≈ 월 0.83%
        CONFIG['return_hinge_weight'] = 2.0
        if CONFIG['output_dir'] == 'results/walkforward':
            CONFIG['output_dir'] = 'results/exp_x1'

    elif _exp == 'x2':
        # X2: DecisionAwareLoss(eta=0.3, λ_dd=0.3) + Loose Overlay
        # Risk/DD 패널티를 대폭 낮춰 Return term이 실질적으로 작동하도록 함
        CONFIG.update(_LOOSE_OVERLAY)
        CONFIG['loss_type'] = 'decision_aware'
        CONFIG['eta'] = 0.3                # 1.0 → 0.3
        CONFIG['lambda_dd'] = 0.3          # 2.0 → 0.3
        CONFIG['regime_dd_scale'] = 2.0    # 3.0 → 2.0
        if CONFIG['output_dir'] == 'results/walkforward':
            CONFIG['output_dir'] = 'results/exp_x2'

    elif _exp == 'x3':
        # X3: SharpeLoss + DualShockExpert + Loose Overlay
        CONFIG.update(_LOOSE_OVERLAY)
        CONFIG['loss_type'] = 'sharpe'
        CONFIG['model_type'] = 'dual_shock_expert'
        CONFIG['lambda_dd'] = 0.3
        CONFIG['regime_dd_scale'] = 2.0
        CONFIG['return_target_monthly'] = 0.0083
        CONFIG['return_hinge_weight'] = 2.0
        if CONFIG['output_dir'] == 'results/walkforward':
            CONFIG['output_dir'] = 'results/exp_x3'

    # =========================================================================
    # v6b: Return Drought 해결 — lambda_risk + max_bil_floor + overlay 완화
    # =========================================================================
    # 공통 "Very Loose Overlay" (x4/x5 공유)
    _VERY_LOOSE_OVERLAY = dict(
        target_vol=0.16,         # 0.13 → 0.16: equity 60~75% 허용
        max_leverage=2.0,        # 1.8  → 2.0
        dd_threshold_1=0.07,     # 0.05 → 0.07
        dd_threshold_2=0.10,     # 0.08 → 0.10
        bull_leverage=2.5,       # 2.0  → 2.5
        use_path_mdd=False,
    )

    if _exp == 'x4':
        # X4: GRU + SharpeLoss + lambda_risk↓ + bil_floor↓ + return hinge 강화
        # 핵심 변경: lambda_risk 2.0→1.0 (CVaR 보수성 반감)
        #           max_bil_floor 0.7→0.3 (위기 시 현금 비중 70%→30%)
        CONFIG.update(_VERY_LOOSE_OVERLAY)
        CONFIG['loss_type'] = 'sharpe'
        CONFIG['lambda_risk'] = 1.0            # ★ 2.0 → 1.0
        CONFIG['max_bil_floor'] = 0.3          # ★ 0.7 → 0.3
        CONFIG['lambda_dd'] = 0.1              # 0.3 → 0.1 (MDD 이미 -9.7% 여유)
        CONFIG['regime_dd_scale'] = 1.0        # 2.0 → 1.0
        CONFIG['return_target_monthly'] = 0.01 # 연 12% 목표 (오버슈팅)
        CONFIG['return_hinge_weight'] = 5.0    # 2.0 → 5.0 (강력 추진)
        if CONFIG['output_dir'] == 'results/walkforward':
            CONFIG['output_dir'] = 'results/exp_x4'

    elif _exp == 'x5':
        # X5: DualShockExpert + X4와 동일한 loss/overlay
        # lambda_risk=1.0 → expert_lambda_scale = 1.0/2.0 = 0.5
        #   → expert lambdas [1,3,5] × 0.5 = [0.5, 1.5, 2.5]
        CONFIG.update(_VERY_LOOSE_OVERLAY)
        CONFIG['loss_type'] = 'sharpe'
        CONFIG['model_type'] = 'dual_shock_expert'
        CONFIG['lambda_risk'] = 1.0            # ★ expert scale 0.5
        CONFIG['max_bil_floor'] = 0.3          # ★
        CONFIG['lambda_dd'] = 0.1
        CONFIG['regime_dd_scale'] = 1.0
        CONFIG['return_target_monthly'] = 0.01
        CONFIG['return_hinge_weight'] = 5.0
        if CONFIG['output_dir'] == 'results/walkforward':
            CONFIG['output_dir'] = 'results/exp_x5'

    elif _exp == 'x6':
        # X6: GRU + SharpeLoss + lambda_risk=0.5 + Vol Targeting 완전 해제
        # 모델 자체의 Return 상한선을 확인하는 ablation 실험
        CONFIG['loss_type'] = 'sharpe'
        CONFIG['lambda_risk'] = 0.5            # ★★ 극단적 공격
        CONFIG['max_bil_floor'] = 0.2          # ★★
        CONFIG['lambda_dd'] = 0.05             # 최소
        CONFIG['regime_dd_scale'] = 1.0
        CONFIG['return_target_monthly'] = 0.012  # 연 14.4% 목표
        CONFIG['return_hinge_weight'] = 5.0
        CONFIG['vol_targeting'] = False        # ★★ overlay 완전 해제
        CONFIG['use_path_mdd'] = False
        if CONFIG['output_dir'] == 'results/walkforward':
            CONFIG['output_dir'] = 'results/exp_x6'

    # baseline: 기존 설정 유지 (아무것도 덮어쓰지 않음)

    print("=" * 70)
    print("  v5 Walk-Forward: Decision-Aware Tail Risk Optimization")
    print("  GRU / DualShock + CVaR + DD-Conditioned Q + PathMDD")
    print(f"  Distribution:  {args.dist_type.upper()} {f'(df={args.t_df})' if args.dist_type == 't' else ''}")
    print(f"  Correlation:   {'OFF' if args.no_correlation else 'ON'}")
    print(f"  Daily Panic:   {'OFF' if args.no_daily_panic else 'ON'}")
    print(f"  NFCI:          {'OFF' if args.no_nfci else 'ON'}")
    print(f"  E2E Regime:    {'ON' if args.e2e_regime else 'OFF'}")
    print(f"  EA-MIDAS:      {'ON' if CONFIG['use_ea_midas'] else 'OFF'}")
    print(f"  Assets:        {'13' if CONFIG['assets_13'] else '10'}")
    print("=" * 70)
    
    # --- Config summary ---
    print(f"\n  exp:           {_exp.upper()}")
    print(f"  model:         {CONFIG['model_type'].upper()} (hidden={CONFIG['hidden_dim']})")
    print(f"  loss_type:     {CONFIG['loss_type'].upper()}")
    print(f"  regime_dim:    {CONFIG['regime_dim']}")
    print(f"  lambda_risk:   {CONFIG['lambda_risk']} (Mean-CVaR)")
    print(f"  max_bil_floor: {CONFIG['max_bil_floor']} (Crisis cash cap)")
    print(f"  lambda_dd:     {CONFIG['lambda_dd']} (Drawdown)")
    print(f"  eta:           {CONFIG['eta']} (Risk penalty)")
    print(f"  return_hinge:  {CONFIG['return_target_monthly']:.4f}/mo × {CONFIG['return_hinge_weight']:.1f}")
    print(f"  realized_cost: {CONFIG['realized_cost_bps']:.1f} bps")
    print(f"  path_mdd:      {'ON' if CONFIG['use_path_mdd'] else 'OFF'}")
    print(f"  regime_dd:     ×{CONFIG['regime_dd_scale']} (Crisis boost)")
    print(f"  weight_decay:  {CONFIG['weight_decay']}")
    print(f"  vol_target:    {CONFIG['target_vol']:.0%}" if CONFIG['vol_targeting'] else "  vol_target:  OFF")
    print(f"  n_seeds:       {CONFIG['n_seeds']}")
    print(f"  sample:        OOS {CONFIG['oos_start']} -> 2025-12-31 (download end {CONFIG['end_date']})")
    print(f"  output_dir:    {CONFIG['output_dir']}")
    
    # --- Data Loading ---
    print("\n[Step 1] Loading Data...")
    
    from src.data_loader import ASSETS_13
    active_tickers = ASSETS_13 if CONFIG.get('assets_13', False) else ASSET_TICKERS
    
    _, asset_returns_df = get_monthly_asset_data(active_tickers, 
                                                  CONFIG['start_date'], CONFIG['end_date'])
    
    X, y, vix, scaler, asset_names, y_dates, macro_tensor = prepare_training_data(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        seq_length=CONFIG['seq_length'],
        normalize=True,
        use_momentum=CONFIG.get('use_momentum', False),
        use_correlation=CONFIG.get('use_correlation', False),
        use_macro_regime=CONFIG.get('use_macro_regime', False),
        use_daily_panic=CONFIG.get('use_daily_panic', False),
        use_nfci=CONFIG.get('use_nfci', False),
        use_ea_midas=CONFIG.get('use_ea_midas', False),
        assets_13=CONFIG.get('assets_13', False),
    )
    
    y_raw = asset_returns_df[asset_names].reindex(y_dates).values
    nan_mask = np.isnan(y_raw).any(axis=1)
    if nan_mask.any():
        print(f"  [WARNING] {nan_mask.sum()} NaN rows in y_raw, filling with 0.0")
        y_raw = np.nan_to_num(y_raw, nan=0.0)
    
    # --- y_raw 진단 (13자산 수익률 검증) ---
    print(f"\n  [DIAG] y_raw shape: {y_raw.shape}")
    print(f"  [DIAG] asset_returns_df columns: {list(asset_returns_df.columns)}")
    print(f"  [DIAG] asset_names from prepare: {asset_names}")
    for i, col in enumerate(asset_returns_df.columns):
        col_data = y_raw[:, i]
        print(f"  [DIAG] {col:>6s}: mean={col_data.mean():.5f}, std={col_data.std():.5f}, "
              f"min={col_data.min():.5f}, max={col_data.max():.5f}")
    # 이상 감지: 월 평균 수익률 > 5%면 데이터 오류
    mean_monthly = np.abs(y_raw).mean()
    if mean_monthly > 0.05:
        print(f"\n  [CRITICAL] y_raw 월평균 |수익률| = {mean_monthly:.4f} — 비정상적으로 높음!")
        print(f"  [CRITICAL] asset_returns_df.index[:5] = {list(asset_returns_df.index[:5])}")
        print(f"  [CRITICAL] y_dates[:5] = {list(y_dates[:5])}")
        # 날짜 형식 비교
        if len(asset_returns_df.index) > 0 and len(y_dates) > 0:
            print(f"  [CRITICAL] type(asset_returns_df.index[0]) = {type(asset_returns_df.index[0])}")
            print(f"  [CRITICAL] type(y_dates[0]) = {type(y_dates[0])}")
    
    n_samples = min(len(X), len(y_raw))
    X = X[:n_samples]
    y = y[:n_samples]
    vix = vix[:n_samples]
    y_raw = y_raw[:n_samples]
    dates = y_dates[:n_samples]
    if macro_tensor is not None:
        macro_tensor = macro_tensor[:n_samples]
    
    # --- 4-State Regime 확률 ---
    print("\n[Step 1b] Loading 4-State Regime Probabilities...")
    regime_df = get_regime_4state()
    
    if regime_df.empty:
        print("  [FALLBACK] Uniform regime probs")
        regime_probs_np = np.ones((n_samples, CONFIG['regime_dim'])) / CONFIG['regime_dim']
    else:
        regime_aligned = regime_df.reindex(dates)
        missing = regime_aligned.isna().any(axis=1).sum()
        if missing > 0:
            print(f"  [WARNING] {missing} missing dates, forward-filling")
            regime_aligned = regime_aligned.ffill().bfill()
        regime_probs_np = regime_aligned.values
        if np.isnan(regime_probs_np).any():
            regime_probs_np = np.ones((n_samples, CONFIG['regime_dim'])) / CONFIG['regime_dim']
    
    regime_probs_t = torch.FloatTensor(regime_probs_np)
    
    macro_status = "ON" if macro_tensor is not None else "OFF"
    print(f"  X: {X.shape}, regime: {regime_probs_t.shape}, macro: {macro_status}, dates: {len(dates)}")
    
    # --- Walk-Forward ---
    print("\n[Step 2] Walk-Forward Cross-Validation")
    folds = define_folds_dynamic(
        dates,
        CONFIG['seq_length'],
        CONFIG.get('oos_start', '2016-07-31'),
        CONFIG.get('test_window_months', 24),
    )
    print(f"  Folds: {len(folds)}")
    for i, f in enumerate(folds):
        print(f"    Fold {i+1}: train={len(f['train_idx'])}, "
              f"test={len(f['test_idx'])}, "
              f"period={f['test_start'].date()} ~ {f['test_end'].date()}")
    
    # --- Multi-seed Walk-Forward ---
    all_seed_results = []
    all_seed_weights = []  # v4: seed ensemble용
    
    for seed_i in range(CONFIG['n_seeds']):
        seed = 42 + seed_i
        print(f"\n{'='*50}")
        print(f"  Seed {seed_i + 1} / {CONFIG['n_seeds']} (seed={seed})")
        print(f"{'='*50}")
        
        oos_weights_list = []
        oos_returns_list = []
        oos_regime_list = []  # Phase2B: collect regime for vol targeting
        
        for fold_i, fold in enumerate(folds):
            print(f"\n  Fold {fold_i + 1}/{len(folds)}: ", end="")
            
            weights, history = train_fold(
                X, y, y_raw, vix, regime_probs_t, fold, CONFIG, seed,
                macro_tensor=macro_tensor)
            
            test_returns = y_raw[fold['test_idx']]
            
            best_epoch = len(history['train_losses'])
            print(f"trained {best_epoch} epochs, "
                  f"loss={history['train_losses'][-1]:.4f}")
            
            oos_weights_list.append(weights)
            oos_returns_list.append(test_returns)
            oos_regime_list.append(regime_probs_np[fold['test_idx']])
        
        all_weights = np.vstack(oos_weights_list)
        all_returns = np.vstack(oos_returns_list)
        
        all_seed_weights.append(all_weights)  # v4: ensemble용 저장
        
        print(f"\n  --- OOS Results (Seed {seed}) ---")
        
        before_metrics = evaluate_portfolio(
            all_weights, all_returns, "Before Vol Targeting")
        
        if CONFIG['vol_targeting']:
            adj_weights, scalars = apply_vol_targeting(
                all_weights, all_returns,
                target_vol=CONFIG['target_vol'],
                lookback=CONFIG['vol_lookback'],
                max_leverage=CONFIG['max_leverage'],
                dd_threshold_1=CONFIG['dd_threshold_1'],
                dd_threshold_2=CONFIG['dd_threshold_2'],
                dd_recovery_months=CONFIG['dd_recovery_months'],
                regime_probs=np.vstack(oos_regime_list),
                bull_leverage=CONFIG['bull_leverage'],
                crisis_leverage=CONFIG['crisis_leverage'],
            )
            after_metrics = evaluate_portfolio(
                adj_weights, all_returns, "After Vol Targeting")
        else:
            after_metrics = before_metrics
        
        all_seed_results.append({
            'seed': seed,
            'before': before_metrics,
            'after': after_metrics,
        })
    
    # --- v4: Seed Ensemble (비중 평균) ---
    print(f"\n{'='*50}")
    print("  Seed Ensemble (Weight Averaging)")
    print(f"{'='*50}")
    
    all_test_returns = np.vstack([y_raw[f['test_idx']] for f in folds])
    all_test_regime = np.vstack([regime_probs_np[f['test_idx']] for f in folds])
    ensemble_weights = np.mean(all_seed_weights, axis=0)
    
    ensemble_before = evaluate_portfolio(
        ensemble_weights, all_test_returns, "Ensemble Before VT")
    
    if CONFIG['vol_targeting']:
        adj_ensemble, _ = apply_vol_targeting(
            ensemble_weights, all_test_returns,
            target_vol=CONFIG['target_vol'],
            lookback=CONFIG['vol_lookback'],
            max_leverage=CONFIG['max_leverage'],
            dd_threshold_1=CONFIG['dd_threshold_1'],
            dd_threshold_2=CONFIG['dd_threshold_2'],
            dd_recovery_months=CONFIG['dd_recovery_months'],
            regime_probs=all_test_regime,
            bull_leverage=CONFIG['bull_leverage'],
            crisis_leverage=CONFIG['crisis_leverage'],
        )
        ensemble_after = evaluate_portfolio(
            adj_ensemble, all_test_returns, "Ensemble After VT")
    else:
        ensemble_after = ensemble_before
    
    # --- 1/N Baseline ---
    print(f"\n{'='*50}")
    print("  1/N Baseline (전체 OOS)")
    print(f"{'='*50}")
    
    num_assets = all_test_returns.shape[1]  # 동적: 10 or 13
    equal_w = np.ones((len(all_test_returns), num_assets)) / num_assets
    baseline_metrics = evaluate_portfolio(equal_w, all_test_returns, "1/N Equal Weight")
    
    # --- Aggregate ---
    print("\n" + "=" * 70)
    print("  Final Results (Mean ± Std across seeds)")
    print("=" * 70)
    
    def agg(key, metric):
        vals = [r[key][metric] for r in all_seed_results]
        return np.mean(vals), np.std(vals)
    
    print(f"\n  {'Metric':<20s} {'Before VT':<22s} {'After VT':<22s} {'Ensemble':<15s} {'1/N':<10s}")
    print(f"  {'-'*85}")
    
    for metric, fmt in [('sharpe', '.4f'), ('annual_return', '.2%'), ('mdd', '.2%')]:
        bm, bs = agg('before', metric)
        am, as_ = agg('after', metric)
        em = ensemble_after[metric]
        bl = baseline_metrics[metric]
        print(f"  {metric:<20s} {bm:{fmt}} ± {bs:{fmt}}   {am:{fmt}} ± {as_:{fmt}}   {em:{fmt}}   {bl:{fmt}}")
    
    # --- Save OOS artifacts ---
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    all_test_dates = np.concatenate([dates[f['test_idx']] for f in folds])
    final_weights = final_ensemble if 'final_ensemble' in locals() else (
        adj_ensemble if CONFIG['vol_targeting'] else ensemble_weights
    )
    port_ret_series = (final_weights * all_test_returns).sum(axis=1)

    port_ret_df = pd.DataFrame(
        {'return': port_ret_series},
        index=pd.DatetimeIndex(all_test_dates, name='date'),
    )
    port_ret_path = os.path.join(output_dir, 'port_returns.csv')
    port_ret_df.to_csv(port_ret_path)
    print(f"\n  OOS returns saved to {port_ret_path} ({len(port_ret_df)} months)")

    port_weights_df = pd.DataFrame(
        final_weights,
        index=pd.DatetimeIndex(all_test_dates, name='date'),
        columns=asset_names,
    )
    port_weights_path = os.path.join(output_dir, 'port_weights.csv')
    port_weights_df.to_csv(port_weights_path)
    print(f"  OOS weights saved to {port_weights_path}")

    raw_weights_df = pd.DataFrame(
        ensemble_weights,
        index=pd.DatetimeIndex(all_test_dates, name='date'),
        columns=asset_names,
    )
    raw_weights_path = os.path.join(output_dir, 'raw_weights.csv')
    raw_weights_df.to_csv(raw_weights_path)
    print(f"  OOS raw weights saved to {raw_weights_path}")

    asset_ret_df = pd.DataFrame(
        all_test_returns,
        index=pd.DatetimeIndex(all_test_dates, name='date'),
        columns=asset_names,
    )
    asset_returns_path = os.path.join(output_dir, 'asset_returns.csv')
    asset_ret_df.to_csv(asset_returns_path)
    print(f"  OOS asset returns saved to {asset_returns_path}")

    fold_rows = []
    offset = 0
    for fold_i, fold in enumerate(folds, start=1):
        fold_len = len(fold['test_idx'])
        fold_slice = slice(offset, offset + fold_len)
        fold_metrics = evaluate_portfolio(final_weights[fold_slice], all_test_returns[fold_slice])
        fold_rows.append({
            'fold': fold_i,
            'test_start': str(pd.Timestamp(fold['test_start']).date()),
            'test_end': str(pd.Timestamp(fold['test_end']).date()),
            'n_months': int(fold_len),
            'sharpe': float(fold_metrics['sharpe']),
            'annual_return': float(fold_metrics['annual_return']),
            'mdd': float(fold_metrics['mdd']),
            'triple': bool(is_triple(fold_metrics)),
        })
        offset += fold_len

    fold_results_path = os.path.join(output_dir, 'fold_results.csv')
    pd.DataFrame(fold_rows).to_csv(fold_results_path, index=False)
    print(f"  Fold-level results saved to {fold_results_path}")

    metrics_payload = {
        'sharpe': float(ensemble_after['sharpe']),
        'return': float(ensemble_after['annual_return']),
        'mdd': float(ensemble_after['mdd']),
        'triple': bool(is_triple(ensemble_after)),
        'thresholds': {
            'sharpe_min': 1.0,
            'return_min': 0.10,
            'mdd_min': -0.10,
        },
        'experiment': {
            'exp': _exp,
            'model_type': CONFIG['model_type'],
            'loss_type': CONFIG.get('loss_type', 'decision_aware'),
            'lambda_dd': float(CONFIG['lambda_dd']),
            'lambda_risk': float(CONFIG['lambda_risk']),
            'max_bil_floor': float(CONFIG['max_bil_floor']),
            'eta': float(CONFIG['eta']),
            'target_vol': float(CONFIG['target_vol']),
            'dd_threshold_1': float(CONFIG['dd_threshold_1']),
            'dd_threshold_2': float(CONFIG['dd_threshold_2']),
            'return_target_monthly': float(CONFIG.get('return_target_monthly', 0.0)),
            'realized_cost_bps': float(CONFIG.get('realized_cost_bps', 0.0)),
            'use_path_mdd': bool(CONFIG.get('use_path_mdd', False)),
            'oos_start': CONFIG.get('oos_start'),
            'oos_end': '2025-12-31',
        },
    }
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)

    summary = {
        'version': 'v5',
        'dist_type': args.dist_type,
        't_df': args.t_df,
        'config': {k: str(v) if not isinstance(v, (int, float, bool, str)) else v
                   for k, v in CONFIG.items()},
        'n_folds': len(folds),
        'before_vt': {
            'sharpe': float(agg('before', 'sharpe')[0]),
            'annual_return': float(agg('before', 'annual_return')[0]),
            'mdd': float(agg('before', 'mdd')[0]),
        },
        'after_vt': {
            'sharpe': float(agg('after', 'sharpe')[0]),
            'annual_return': float(agg('after', 'annual_return')[0]),
            'mdd': float(agg('after', 'mdd')[0]),
        },
        'ensemble': {
            'sharpe': float(ensemble_after['sharpe']),
            'annual_return': float(ensemble_after['annual_return']),
            'mdd': float(ensemble_after['mdd']),
            'triple': bool(is_triple(ensemble_after)),
        },
        'baseline': {
            'sharpe': float(baseline_metrics['sharpe']),
            'annual_return': float(baseline_metrics['annual_return']),
            'mdd': float(baseline_metrics['mdd']),
        },
        'fold_results': fold_rows,
    }

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"  Metrics saved to {metrics_path}")
    print(f"  Summary saved to {summary_path}")
    print(f"  Triple target: {'PASS' if metrics_payload['triple'] else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
