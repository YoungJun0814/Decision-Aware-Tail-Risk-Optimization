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
from src.loss import DecisionAwareLoss
from src.trainer import Trainer, TrajectoryBatchSampler
from src.utils import set_seed, get_device, calculate_mdd


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
    'eta': 1.0,
    'kappa_base': 0.001,
    'kappa_vix_scale': 0.0001,
    'lambda_dd': 2.0,
    'regime_dd_scale': 3.0,
    
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
    'use_macro_regime': True,  # Phase5: macro → RegimeHead
    'macro_dim': 2,            # T10Y3M + BAA10Y
    
    # Walk-Forward
    'start_date': '2007-07-01',
    'end_date': '2024-01-01',
    'n_seeds': 3,
    
    'device': get_device(),
}


# =============================================================================
# Walk-Forward Folds
# =============================================================================

def define_folds(dates, seq_length=12):
    """Expanding Window Walk-Forward Fold 정의."""
    fold_boundaries = [
        ('2016-07-01', '2018-07-01'),
        ('2018-07-01', '2020-07-01'),
        ('2020-07-01', '2022-07-01'),
        ('2022-07-01', '2025-01-01'),
    ]
    
    folds = []
    for test_start, test_end in fold_boundaries:
        ts = pd.Timestamp(test_start)
        te = pd.Timestamp(test_end)
        
        train_mask = dates < ts
        test_mask = (dates >= ts) & (dates < te)
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        if len(train_idx) > seq_length and len(test_idx) > 0:
            folds.append({
                'train_idx': train_idx,
                'test_idx': test_idx,
                'test_start': ts,
                'test_end': te,
            })
    
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
    num_assets = len(ASSET_TICKERS)
    
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
    )
    
    # Loss — DecisionAwareLoss (v1) with regime-conditional λ_dd (v4)
    loss_fn = DecisionAwareLoss(
        eta=config['eta'],
        kappa_base=config['kappa_base'],
        kappa_vix_scale=config['kappa_vix_scale'],
        lambda_dd=config['lambda_dd'],
        regime_dd_scale=config['regime_dd_scale'],
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
        optimizer=optimizer, device=device)
    
    # Train with scheduler
    history = trainer.fit(
        train_loader, val_loader,
        epochs=config['epochs'],
        verbose=False,
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
    mdd = calculate_mdd(port_ret)
    
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


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  v4 Walk-Forward: Regime-Conditional CVaR")
    print("  GRU + CVaR + Regime Q-Concat + Crisis Overlay")
    print("=" * 70)
    
    # --- Config summary ---
    print(f"\n  model:         {CONFIG['model_type'].upper()} (hidden={CONFIG['hidden_dim']})")
    print(f"  regime_dim:    {CONFIG['regime_dim']}")
    print(f"  lambda_risk:   {CONFIG['lambda_risk']} (Mean-CVaR)")
    print(f"  lambda_dd:     {CONFIG['lambda_dd']} (Drawdown)")
    print(f"  regime_dd:     ×{CONFIG['regime_dd_scale']} (Crisis boost)")
    print(f"  weight_decay:  {CONFIG['weight_decay']}")
    print(f"  vol_target:    {CONFIG['target_vol']:.0%}" if CONFIG['vol_targeting'] else "  vol_target:  OFF")
    print(f"  n_seeds:       {CONFIG['n_seeds']}")
    
    # --- Data Loading ---
    print("\n[Step 1] Loading Data...")
    
    _, asset_returns_df = get_monthly_asset_data(ASSET_TICKERS, 
                                                  CONFIG['start_date'], CONFIG['end_date'])
    
    X, y, vix, scaler, asset_names, y_dates, macro_tensor = prepare_training_data(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        seq_length=CONFIG['seq_length'],
        normalize=True,
        use_momentum=CONFIG.get('use_momentum', False),
        use_macro_regime=CONFIG.get('use_macro_regime', False),
    )
    
    y_raw = asset_returns_df.reindex(y_dates).values
    nan_mask = np.isnan(y_raw).any(axis=1)
    if nan_mask.any():
        print(f"  [WARNING] {nan_mask.sum()} NaN rows in y_raw, filling with 0.0")
        y_raw = np.nan_to_num(y_raw, nan=0.0)
    
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
    folds = define_folds(dates, CONFIG['seq_length'])
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
    
    num_assets = len(ASSET_TICKERS)
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
    
    # --- Save ---
    os.makedirs('results/walkforward', exist_ok=True)
    summary = {
        'version': 'v4',
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
        },
        'baseline': {
            'sharpe': float(baseline_metrics['sharpe']),
            'annual_return': float(baseline_metrics['annual_return']),
            'mdd': float(baseline_metrics['mdd']),
        },
    }
    
    with open('results/walkforward/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Results saved to results/walkforward/summary.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
