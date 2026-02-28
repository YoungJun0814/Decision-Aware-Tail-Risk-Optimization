"""
Proper Baselines — Traditional Strategy Comparison
====================================================
AI 모델과 동일한 OOS 기간에서 전통적 전략을 평가합니다.

Baselines:
  1. 60/40      — 60% SPY + 40% TLT (monthly rebalance)
  2. Risk Parity — Inverse-volatility weighting
  3. Min Variance — Minimum variance portfolio (scipy)
  4. HRP         — Hierarchical Risk Parity (Lopez de Prado)
  5. Momentum    — Top-5 12-month momentum, equal weight

Usage:
  python run_baselines.py
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from src.data_loader import get_monthly_asset_data, ASSET_TICKERS
from src.utils import calculate_mdd
from run_walkforward import CONFIG, define_folds, evaluate_portfolio
from src.data_loader import prepare_training_data


# =============================================================================
# Baseline Strategies
# =============================================================================

def strategy_60_40(returns, asset_names, **kwargs):
    """60% SPY + 40% TLT, monthly rebalance."""
    N, A = returns.shape
    weights = np.zeros((N, A))
    
    spy_idx = list(asset_names).index('SPY') if 'SPY' in asset_names else 0
    tlt_idx = list(asset_names).index('TLT') if 'TLT' in asset_names else 5
    
    weights[:, spy_idx] = 0.60
    weights[:, tlt_idx] = 0.40
    return weights


def strategy_risk_parity(returns, lookback=12, **kwargs):
    """Inverse-volatility weighting (rolling lookback, 20% cap per asset)."""
    N, A = returns.shape
    weights = np.ones((N, A)) / A  # fallback
    max_single = 0.20  # cap per asset to prevent BIL domination
    
    for t in range(lookback, N):
        past = returns[t-lookback:t]
        vols = past.std(axis=0)
        vols = np.maximum(vols, 1e-8)
        
        inv_vol = 1.0 / vols
        w = inv_vol / inv_vol.sum()
        
        # Cap individual weights, redistribute excess
        for _ in range(5):
            excess = np.maximum(w - max_single, 0)
            if excess.sum() < 1e-8:
                break
            w = np.minimum(w, max_single)
            uncapped = w < max_single
            if uncapped.any():
                w[uncapped] += excess.sum() * (w[uncapped] / w[uncapped].sum())
        w = w / w.sum()
        weights[t] = w
    
    return weights


def strategy_min_variance(returns, lookback=24, **kwargs):
    """Minimum variance portfolio (rolling, no short-sale)."""
    N, A = returns.shape
    weights = np.ones((N, A)) / A
    
    for t in range(lookback, N):
        past = returns[t-lookback:t]
        cov = np.cov(past, rowvar=False)
        cov = cov + np.eye(A) * 1e-6  # regularize
        
        def port_var(w):
            return w @ cov @ w
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        ]
        bounds = [(0.0, 1.0)] * A
        x0 = np.ones(A) / A
        
        try:
            result = minimize(port_var, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 200})
            if result.success:
                weights[t] = result.x
        except Exception:
            pass
    
    return weights



def _get_quasi_diag(link):
    """Quasi-diagonal ordering from linkage (HRP helper)."""
    link = link.astype(int)
    sort_ix = leaves_list(link)
    return sort_ix.tolist()


def _get_rec_bipart(cov, sort_ix):
    """Recursive bisection for HRP weights."""
    w = np.ones(len(sort_ix))
    
    def _recurse(items):
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]
        
        cov_left = cov[np.ix_(left, left)]
        cov_right = cov[np.ix_(right, right)]
        
        inv_diag_left = 1.0 / np.diag(cov_left)
        w_left = inv_diag_left / inv_diag_left.sum()
        var_left = w_left @ cov_left @ w_left
        
        inv_diag_right = 1.0 / np.diag(cov_right)
        w_right = inv_diag_right / inv_diag_right.sum()
        var_right = w_right @ cov_right @ w_right
        
        alpha = 1 - var_left / (var_left + var_right + 1e-10)
        
        for i in left:
            w[i] *= alpha
        for i in right:
            w[i] *= (1 - alpha)
        
        _recurse(left)
        _recurse(right)
    
    _recurse(sort_ix)
    return w / (w.sum() + 1e-10)


def strategy_hrp(returns, lookback=24, **kwargs):
    """Hierarchical Risk Parity (Lopez de Prado, 2016)."""
    N, A = returns.shape
    weights = np.ones((N, A)) / A
    
    for t in range(lookback, N):
        past = returns[t-lookback:t]
        cov = np.cov(past, rowvar=False)
        cov = cov + np.eye(A) * 1e-6
        
        corr = np.corrcoef(past, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        
        # Distance matrix
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0.0)
        dist = np.maximum(dist, 0.0)
        
        try:
            condensed = squareform(dist, checks=False)
            link = linkage(condensed, method='single')
            sort_ix = _get_quasi_diag(link)
            w = _get_rec_bipart(cov, sort_ix)
            w = np.maximum(w, 0.0)
            w = w / (w.sum() + 1e-10)
            weights[t] = w
        except Exception as e:
            print(f"[ERROR] HRP failed at step {t}: {e}")
            pass
    
    return weights


def strategy_momentum(returns, lookback_mom=12, top_k=5, **kwargs):
    """Cross-sectional momentum: top-K 12M return, equal weight."""
    N, A = returns.shape
    weights = np.zeros((N, A))
    
    for t in range(lookback_mom, N):
        # 12-month cumulative return per asset
        past = returns[t-lookback_mom:t]
        cum_ret = (1 + past).prod(axis=0) - 1
        
        # Top K assets
        top_idx = np.argsort(cum_ret)[-top_k:]
        w = np.zeros(A)
        w[top_idx] = 1.0 / top_k
        weights[t] = w
    
    return weights


# =============================================================================
# Main
# =============================================================================

STRATEGIES = {
    '60_40': {
        'desc': '60% SPY + 40% TLT',
        'func': strategy_60_40,
    },
    'risk_parity': {
        'desc': 'Inverse-Volatility Risk Parity',
        'func': strategy_risk_parity,
    },
    'min_variance': {
        'desc': 'Minimum Variance Portfolio',
        'func': strategy_min_variance,
    },
    'hrp': {
        'desc': 'Hierarchical Risk Parity (HRP)',
        'func': strategy_hrp,
    },
    'momentum': {
        'desc': 'Top-5 Cross-Sectional Momentum',
        'func': strategy_momentum,
    },
}


def run_baselines(config=None):
    """
    전통 전략 6종의 OOS 월간 수익률을 {전략명: pd.Series} 딕셔너리로 반환.
    run_compare.py에서 호출됨.
    """
    if config is None:
        config = CONFIG
    
    _, asset_returns_df = get_monthly_asset_data(
        ASSET_TICKERS, config['start_date'], config['end_date'])
    
    X, y, vix, scaler, asset_names, y_dates, _ = prepare_training_data(
        start_date=config['start_date'],
        end_date=config['end_date'],
        seq_length=config['seq_length'],
        normalize=True,
        use_momentum=False,
    )
    
    y_raw = asset_returns_df.reindex(y_dates).values
    y_raw = np.nan_to_num(y_raw, nan=0.0)
    
    n_samples = min(len(X), len(y_raw))
    y_raw = y_raw[:n_samples]
    dates = y_dates[:n_samples]
    
    folds = define_folds(dates, config['seq_length'])
    all_test_idx = np.concatenate([f['test_idx'] for f in folds])
    all_test_returns = y_raw[all_test_idx]
    all_test_dates = dates[all_test_idx]
    
    full_returns = y_raw
    n_assets = len(ASSET_TICKERS)
    
    # 1/N Equal Weight
    equal_w = np.ones((len(all_test_returns), n_assets)) / n_assets
    
    all_strategies = {'1/N': equal_w}
    
    for strat_name, strat_info in STRATEGIES.items():
        full_weights = strat_info['func'](full_returns, asset_names=asset_names)
        oos_weights = full_weights[all_test_idx]
        all_strategies[strat_info['desc']] = oos_weights
    
    # 월간 수익률 Series로 변환
    result = {}
    dt_index = pd.DatetimeIndex(all_test_dates, name='date')
    for name, weights in all_strategies.items():
        port_ret = (weights * all_test_returns).sum(axis=1)
        result[name] = pd.Series(port_ret, index=dt_index, name=name)
    
    return result


def main():
    print("=" * 70)
    print("  Proper Baselines: Traditional Strategy Comparison")
    print("=" * 70)
    
    # --- Load same data as walk-forward ---
    print("\n[Step 1] Loading Data (same period as walk-forward)...")
    
    _, asset_returns_df = get_monthly_asset_data(
        ASSET_TICKERS, CONFIG['start_date'], CONFIG['end_date'])
    
    X, y, vix, scaler, asset_names, y_dates, _ = prepare_training_data(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        seq_length=CONFIG['seq_length'],
        normalize=True,
        use_momentum=False,  # baselines don't need momentum features
    )
    
    y_raw = asset_returns_df.reindex(y_dates).values
    y_raw = np.nan_to_num(y_raw, nan=0.0)
    
    n_samples = min(len(X), len(y_raw))
    y_raw = y_raw[:n_samples]
    dates = y_dates[:n_samples]
    
    # --- Same folds as walk-forward ---
    folds = define_folds(dates, CONFIG['seq_length'])
    
    # Concatenate all OOS returns (same as walk-forward)
    all_test_idx = np.concatenate([f['test_idx'] for f in folds])
    all_test_returns = y_raw[all_test_idx]
    
    # For strategies that need full history, use ALL returns
    # but only evaluate on OOS indices
    full_returns = y_raw  # (n_samples, 10)
    
    print(f"  Total months: {n_samples}, OOS months: {len(all_test_returns)}")
    print(f"  Assets: {list(asset_names)}")
    
    # Also compute 1/N for reference
    n_assets = len(ASSET_TICKERS)
    equal_w = np.ones((len(all_test_returns), n_assets)) / n_assets
    baseline_1n = evaluate_portfolio(equal_w, all_test_returns, "1/N Equal Weight")
    
    # --- Run Strategies ---
    results = {
        '1_n': {
            'description': '1/N Equal Weight',
            'metrics': {
                'sharpe': float(baseline_1n['sharpe']),
                'annual_return': float(baseline_1n['annual_return']),
                'mdd': float(baseline_1n['mdd']),
            },
        },
    }
    
    for strat_name, strat_info in STRATEGIES.items():
        print(f"\n{'='*50}")
        print(f"  {strat_name}: {strat_info['desc']}")
        print(f"{'='*50}")
        
        # Compute weights on FULL history
        full_weights = strat_info['func'](
            full_returns, asset_names=asset_names)
        
        # Extract OOS-only weights
        oos_weights = full_weights[all_test_idx]
        
        # Evaluate
        metrics = evaluate_portfolio(oos_weights, all_test_returns, strat_info['desc'])
        
        results[strat_name] = {
            'description': strat_info['desc'],
            'metrics': {
                'sharpe': float(metrics['sharpe']),
                'annual_return': float(metrics['annual_return']),
                'mdd': float(metrics['mdd']),
            },
        }
    
    # --- Summary Table ---
    print("\n\n" + "=" * 70)
    print("  BASELINE SUMMARY")
    print("=" * 70)
    
    print(f"\n  {'Strategy':<22s} {'Sharpe':>8s} {'Return':>8s} {'MDD':>8s}")
    print(f"  {'-'*50}")
    
    for name, data in results.items():
        m = data['metrics']
        print(f"  {name:<22s} {m['sharpe']:>8.4f} {m['annual_return']:>8.2%} {m['mdd']:>8.2%}")
    
    # --- Save ---
    os.makedirs('results/baselines', exist_ok=True)
    with open('results/baselines/summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to results/baselines/summary.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
