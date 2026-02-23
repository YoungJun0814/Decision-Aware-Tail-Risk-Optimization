"""
Phase 2: Regime-Conditioned MIDAS-BL Training
=============================================
End-to-End 학습 + 1/N 벤치마크.

Usage:
    python run_phase2.py
"""

import torch
import torch.optim as optim
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

from src.data_loader import prepare_phase2_data, ASSET_TICKERS
from src.models import RegimeConditionedBLModel
from src.loss import DecisionAwareLoss, RegimeAwareLoss
from src.trainer import Phase2Trainer, create_phase2_dataloaders
from src.utils import set_seed, get_device, calculate_mdd


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # 데이터
    'start_date': '2007-07-01',
    'end_date': '2024-01-01',
    'seq_length': 12,
    'K_midas': 66,
    
    # 모델
    'hidden_dim': 32,
    'n_regimes': 3,
    'dropout': 0.3,
    'num_scenarios': 200,
    'confidence_level': 0.95,
    'bil_index': 9,   # BIL = ASSET_TICKERS의 마지막 자산
    
    # 학습
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 100,
    'train_ratio': 0.8,
    'early_stopping_patience': 15,
    
    # Loss
    'eta': 1.0,
    'kappa_base': 0.001,
    'kappa_vix_scale': 0.0001,
    'lambda_stability': 0.1,
    'lambda_kl_max': 1.0,
    'lambda_kl_min': 0.05,
    
    # 기타
    'n_seeds': 5,
    'device': get_device(),
}


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_portfolio(weights, returns, label=""):
    """
    포트폴리오 성과 평가.
    
    Args:
        weights: (N, A) 포트폴리오 비중
        returns: (N, A) 실현 수익률
        label: 라벨
        
    Returns:
        dict with Sharpe, Annual Return, MDD
    """
    # 포트폴리오 수익률
    port_ret = (weights * returns).sum(axis=1)
    
    # 연율화
    mean_ret = port_ret.mean() * 12
    std_ret = port_ret.std() * np.sqrt(12)
    sharpe = mean_ret / std_ret if std_ret > 1e-8 else 0.0
    mdd = calculate_mdd(port_ret)
    
    result = {
        'sharpe': sharpe,
        'annual_return': mean_ret,
        'annual_vol': std_ret,
        'mdd': mdd,
    }
    
    if label:
        print(f"\n  [{label}]")
        print(f"    Sharpe:        {sharpe:.4f}")
        print(f"    Annual Return: {mean_ret:.2%}")
        print(f"    Annual Vol:    {std_ret:.2%}")
        print(f"    MDD:           {mdd:.2%}")
    
    return result


# =============================================================================
# Single Seed Training
# =============================================================================

def train_single_seed(data, config, seed):
    """단일 seed로 학습 실행."""
    set_seed(seed)
    device = config['device']
    
    # DataLoaders
    train_loader, val_loader = create_phase2_dataloaders(
        data['x_monthly'], data['x_daily'],
        data['y'], data['vix'], data['hmm_probs'],
        batch_size=config['batch_size'],
        train_ratio=config['train_ratio'],
    )
    
    # Model
    monthly_dim = data['x_monthly'].shape[-1]
    model = RegimeConditionedBLModel(
        monthly_input_dim=monthly_dim,
        n_daily_vars=data['x_daily'].shape[-1],
        K_midas=config['K_midas'],
        num_assets=len(ASSET_TICKERS),
        hidden_dim=config['hidden_dim'],
        n_regimes=config['n_regimes'],
        dropout=config['dropout'],
        num_scenarios=config['num_scenarios'],
        confidence_level=config['confidence_level'],
        bil_index=config['bil_index'],
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {n_params:,}")
    
    # Loss
    base_loss = DecisionAwareLoss(
        eta=config['eta'],
        kappa_base=config['kappa_base'],
        kappa_vix_scale=config['kappa_vix_scale'],
    )
    loss_fn = RegimeAwareLoss(
        base_loss,
        lambda_stability=config['lambda_stability'],
        lambda_kl_max=config['lambda_kl_max'],
        lambda_kl_min=config['lambda_kl_min'],
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Trainer
    trainer = Phase2Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )
    
    # Train
    history = trainer.fit(
        train_loader, val_loader,
        epochs=config['epochs'],
        verbose=True,
        early_stopping_patience=config['early_stopping_patience'],
    )
    
    # Evaluate (validation set)
    n_total = len(data['x_monthly'])
    n_train = int(n_total * config['train_ratio'])
    
    with torch.no_grad():
        model.eval()
        x_m_val = data['x_monthly'][n_train:].to(device)
        x_d_val = data['x_daily'][n_train:].to(device)
        val_weights, val_regime = model(x_m_val, x_d_val)
        val_weights = val_weights.cpu().numpy()
        val_regime = val_regime.cpu().numpy()
    
    val_returns_raw = data['y_raw'][n_train:].numpy()  # 원본 수익률로 평가
    
    metrics = evaluate_portfolio(val_weights, val_returns_raw, f"Seed {seed}")
    metrics['history'] = history
    metrics['val_regime'] = val_regime
    
    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  Phase 2: Regime-Conditioned MIDAS-BL Training")
    print("=" * 70)
    
    # --- Data ---
    print("\n[Step 1] Loading Data...")
    data = prepare_phase2_data(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        seq_length=CONFIG['seq_length'],
        K_midas=CONFIG['K_midas'],
        normalize=True,
        train_ratio=CONFIG['train_ratio'],
    )
    
    # --- 1/N Baseline ---
    print("\n[Step 2] 1/N Baseline")
    n_total = len(data['y'])
    n_train = int(n_total * CONFIG['train_ratio'])
    val_returns_raw = data['y_raw'][n_train:].numpy()  # 원본 수익률로 평가
    
    num_assets = len(ASSET_TICKERS)
    equal_weights = np.ones((len(val_returns_raw), num_assets)) / num_assets
    baseline_metrics = evaluate_portfolio(equal_weights, val_returns_raw, "1/N Equal Weight")
    
    # --- Multi-Seed Training ---
    print("\n[Step 3] Multi-Seed Training")
    all_metrics = []
    
    for seed in range(CONFIG['n_seeds']):
        print(f"\n{'='*50}")
        print(f"  Seed {seed + 1} / {CONFIG['n_seeds']}")
        print(f"{'='*50}")
        
        try:
            metrics = train_single_seed(data, CONFIG, seed=42 + seed)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  [FAIL] Seed {seed}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_metrics:
        print("\n[ERROR] No seeds completed successfully!")
        return
    
    # --- Aggregate Results ---
    print("\n" + "=" * 70)
    print("  Final Results")
    print("=" * 70)
    
    sharpes = [m['sharpe'] for m in all_metrics]
    returns_arr = [m['annual_return'] for m in all_metrics]
    mdds = [m['mdd'] for m in all_metrics]
    
    print(f"\n  {'Metric':<20s} {'Phase2 (mean±std)':<25s} {'1/N Baseline':<15s}")
    print(f"  {'-'*60}")
    print(f"  {'Sharpe':<20s} {np.mean(sharpes):.4f} ± {np.std(sharpes):.4f}     {baseline_metrics['sharpe']:.4f}")
    print(f"  {'Annual Return':<20s} {np.mean(returns_arr):.2%} ± {np.std(returns_arr):.2%}  {baseline_metrics['annual_return']:.2%}")
    print(f"  {'MDD':<20s} {np.mean(mdds):.2%} ± {np.std(mdds):.2%}  {baseline_metrics['mdd']:.2%}")
    
    # Beat 1/N?
    mean_sharpe = np.mean(sharpes)
    if mean_sharpe > baseline_metrics['sharpe']:
        print(f"\n  ✅ Phase 2 beats 1/N by {mean_sharpe - baseline_metrics['sharpe']:.4f} Sharpe")
    else:
        print(f"\n  ⚠️ Phase 2 underperforms 1/N by {baseline_metrics['sharpe'] - mean_sharpe:.4f} Sharpe")
    
    # --- Save Results ---
    os.makedirs('results/phase2', exist_ok=True)
    import json
    
    summary = {
        'model': 'RegimeConditionedBLModel',
        'params': sum(p.numel() for p in RegimeConditionedBLModel(
            monthly_input_dim=data['x_monthly'].shape[-1],
            num_assets=num_assets, hidden_dim=CONFIG['hidden_dim'],
        ).parameters()),
        'n_seeds': CONFIG['n_seeds'],
        'sharpe_mean': float(np.mean(sharpes)),
        'sharpe_std': float(np.std(sharpes)),
        'annual_return_mean': float(np.mean(returns_arr)),
        'mdd_mean': float(np.mean(mdds)),
        'baseline_sharpe': float(baseline_metrics['sharpe']),
        'baseline_annual_return': float(baseline_metrics['annual_return']),
        'baseline_mdd': float(baseline_metrics['mdd']),
    }
    
    with open('results/phase2/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Results saved to results/phase2/summary.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
