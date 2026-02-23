"""
Crisis Period Deep Dive — 위기 구간 상세 분석
================================================
특정 위기 구간(COVID-19, 금리인상기)에서 모델의 방어력을 시각화합니다.

3-패널 시계열 차트:
  Panel 1: Regime 확률 (4-state 면적 차트)
  Panel 2: BIL(안전자산) 비중 — 우리 모델 vs 벤치마크
  Panel 3: 누적 수익률 비교

Usage:
    python run_crisis_deepdive.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import os
import warnings

warnings.filterwarnings('ignore')

from src.data_loader import (
    prepare_training_data, get_monthly_asset_data,
    get_regime_4state, ASSET_TICKERS,
)
from src.utils import get_device
from run_walkforward import CONFIG, define_folds, apply_vol_targeting
from run_baselines import (
    strategy_60_40, strategy_risk_parity,
    strategy_min_variance, strategy_momentum,
)


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = 'results/crisis_deepdive'

CRISIS_PERIODS = {
    'covid_crisis': {
        'label': 'COVID-19 Pandemic Crisis',
        'start': '2019-10-01',
        'end': '2020-10-01',
        'event_date': '2020-03-01',
        'event_label': 'WHO Pandemic\nDeclaration',
    },
    'rate_hike': {
        'label': '2022 Aggressive Rate Hike Period',
        'start': '2021-10-01',
        'end': '2023-04-01',
        'event_date': '2022-03-01',
        'event_label': 'Fed Rate\nHike Begins',
    },
}

# Color palette
COLORS = {
    'Bull': '#4CAF50',
    'Sideways': '#FFC107',
    'Correction': '#FF9800',
    'Crisis': '#F44336',
    'our_model': '#2196F3',
    '1/N Equal': '#FF5722',
    'Risk Parity': '#FF9800',
    'Min Variance': '#4CAF50',
    'Momentum': '#607D8B',
    '60/40': '#9C27B0',
}


# =============================================================================
# Data Loading
# =============================================================================

def load_all_data():
    """
    Walk-forward OOS data와 baseline weights를 로드합니다.
    캐시가 있으면 캐시를 사용합니다.
    """
    cache_file = 'results/vol_sweep/cache/oos_data.npz'
    dates_file = 'results/vol_sweep/cache/oos_dates.csv'
    
    if os.path.exists(cache_file) and os.path.exists(dates_file):
        print("[Cache] Vol Sweep 캐시에서 OOS 데이터 로드...")
        data = np.load(cache_file)
        dates_df = pd.read_csv(dates_file, parse_dates=['date'])
        raw_weights = data['raw_weights']
        oos_returns = data['oos_returns']
        oos_regime = data['oos_regime']
        oos_dates = pd.DatetimeIndex(dates_df['date'])
    else:
        print("[ERROR] OOS 캐시가 없습니다. 먼저 run_vol_sweep.py를 실행하세요.")
        print("        python run_vol_sweep.py")
        raise FileNotFoundError("OOS cache not found. Run run_vol_sweep.py first.")
    
    # Apply vol targeting (default target_vol=10%)
    adj_weights, scalars = apply_vol_targeting(
        raw_weights, oos_returns,
        target_vol=CONFIG['target_vol'],
        lookback=CONFIG['vol_lookback'],
        max_leverage=CONFIG['max_leverage'],
        dd_threshold_1=CONFIG['dd_threshold_1'],
        dd_threshold_2=CONFIG['dd_threshold_2'],
        dd_recovery_months=CONFIG['dd_recovery_months'],
        regime_probs=oos_regime,
        bull_leverage=CONFIG['bull_leverage'],
        crisis_leverage=CONFIG['crisis_leverage'],
    )
    
    # Regime 4-state probabilities (from HMM, not model output)
    regime_df = get_regime_4state()
    
    # Full returns for baselines
    _, asset_returns_df = get_monthly_asset_data(
        ASSET_TICKERS, CONFIG['start_date'], CONFIG['end_date'])
    
    X_tmp, _, _, _, asset_names, y_dates_tmp, _ = prepare_training_data(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        seq_length=CONFIG['seq_length'],
        normalize=True,
        use_momentum=False,
    )
    y_raw_full = asset_returns_df.reindex(y_dates_tmp)         # row align by date
    y_raw_full = y_raw_full.reindex(columns=ASSET_TICKERS)    # col align: ASSET_TICKERS order!
    y_raw_full = y_raw_full.values
    y_raw_full = np.nan_to_num(y_raw_full, nan=0.0)
    n_full = min(len(X_tmp), len(y_raw_full))
    y_raw_full = y_raw_full[:n_full]
    dates_full = y_dates_tmp[:n_full]
    
    # Compute baseline weights over full period
    baseline_weights = {}
    baseline_weights['1/N Equal'] = np.ones_like(y_raw_full) / len(ASSET_TICKERS)
    baseline_weights['Risk Parity'] = strategy_risk_parity(
        y_raw_full, asset_names=asset_names)
    baseline_weights['Min Variance'] = strategy_min_variance(
        y_raw_full, asset_names=asset_names)
    baseline_weights['60/40'] = strategy_60_40(
        y_raw_full, asset_names=asset_names)
    
    return {
        'adj_weights': adj_weights,
        'raw_weights': raw_weights,
        'oos_returns': oos_returns,
        'oos_regime': oos_regime,
        'oos_dates': oos_dates,
        'regime_df': regime_df,
        'full_returns': y_raw_full,
        'full_dates': dates_full,
        'baseline_weights': baseline_weights,
        'asset_names': list(asset_names),
    }


# =============================================================================
# 3-Panel Crisis Chart
# =============================================================================

def plot_crisis_deepdive(data, period_key, period_info, save_path):
    """
    3-패널 위기 구간 심층 분석 차트.
    
    Panel 1: Regime Probability (4-state stacked area)
    Panel 2: BIL Weight (safety asset allocation)
    Panel 3: Cumulative Return comparison
    """
    start = pd.Timestamp(period_info['start'])
    end = pd.Timestamp(period_info['end'])
    event_date = pd.Timestamp(period_info['event_date'])
    
    oos_dates = data['oos_dates']
    full_dates = data['full_dates']
    
    # --- Filter OOS data for the crisis period ---
    oos_mask = (oos_dates >= start) & (oos_dates < end)
    crisis_dates_oos = oos_dates[oos_mask]
    crisis_weights = data['adj_weights'][oos_mask]
    crisis_returns = data['oos_returns'][oos_mask]
    
    if len(crisis_dates_oos) == 0:
        print(f"  [SKIP] {period_info['label']}: OOS 기간에 해당 데이터 없음")
        return
    
    # --- Filter full data for baselines ---
    full_mask = (full_dates >= start) & (full_dates < end)
    crisis_dates_full = full_dates[full_mask]
    crisis_returns_full = data['full_returns'][full_mask]
    
    # --- Regime probabilities ---
    regime_df = data['regime_df']
    if not regime_df.empty:
        regime_crisis = regime_df.loc[
            (regime_df.index >= start) & (regime_df.index < end)]
    else:
        regime_crisis = pd.DataFrame()
    
    # --- BIL index ---
    bil_idx = data['asset_names'].index('BIL') if 'BIL' in data['asset_names'] else -1
    
    # =============================================
    # Create figure
    # =============================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), 
                              gridspec_kw={'height_ratios': [1, 1, 1.3]},
                              sharex=True)
    
    fig.suptitle(f"Crisis Deep Dive: {period_info['label']}",
                 fontsize=16, fontweight='bold', y=0.98)
    
    # =============================================
    # Panel 1: Regime Probabilities (stacked area)
    # =============================================
    ax1 = axes[0]
    
    if not regime_crisis.empty:
        regime_cols = ['Prob_Bull', 'Prob_Sideways', 'Prob_Correction', 'Prob_Crisis']
        regime_labels = ['Bull', 'Sideways', 'Correction', 'Crisis']
        regime_colors = [COLORS[l] for l in regime_labels]
        
        ax1.stackplot(regime_crisis.index,
                      [regime_crisis[col].values for col in regime_cols],
                      labels=regime_labels,
                      colors=regime_colors,
                      alpha=0.8)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Regime Probability', fontsize=11, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9, ncol=4, framealpha=0.9)
    else:
        ax1.text(0.5, 0.5, 'Regime data not available',
                transform=ax1.transAxes, ha='center', va='center',
                fontsize=12, color='gray')
    
    ax1.set_title('① HMM Regime State Probabilities', fontsize=12, 
                  fontweight='bold', loc='left')
    
    # Event line
    if start <= event_date < end:
        ax1.axvline(x=event_date, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax1.annotate(period_info['event_label'], xy=(event_date, 0.95),
                    fontsize=8, fontweight='bold', color='red',
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # =============================================
    # Panel 2: BIL (Safety Asset) Weight
    # =============================================
    ax2 = axes[1]
    
    # Our model BIL weight
    if bil_idx >= 0:
        our_bil = crisis_weights[:, bil_idx]
        ax2.plot(crisis_dates_oos, our_bil, '-', 
                color=COLORS['our_model'], linewidth=2.5,
                label='Our Model (GRU+BL+CVaR)', zorder=5)
        ax2.fill_between(crisis_dates_oos, 0, our_bil,
                        color=COLORS['our_model'], alpha=0.15)
        
        # Baseline BIL weights
        for bname in ['1/N Equal', 'Risk Parity', 'Min Variance']:
            if bname in data['baseline_weights']:
                bw = data['baseline_weights'][bname]
                bw_crisis = bw[full_mask]
                if len(bw_crisis) > 0:
                    bl_bil = bw_crisis[:, bil_idx]
                    ax2.plot(crisis_dates_full[:len(bl_bil)], bl_bil, '--',
                            color=COLORS.get(bname, 'gray'), linewidth=1.5,
                            alpha=0.8, label=bname)
    
    ax2.set_ylabel('BIL Weight', fontsize=11, fontweight='bold')
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax2.set_title('② Safety Asset (BIL) Allocation — CrisisOverlay Response',
                  fontsize=12, fontweight='bold', loc='left')
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # Event line
    if start <= event_date < end:
        ax2.axvline(x=event_date, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # =============================================
    # Panel 3: Cumulative Return
    # =============================================
    ax3 = axes[2]
    
    # Our model cumulative return
    our_port_ret = (crisis_weights * crisis_returns).sum(axis=1)
    our_cum = (1 + our_port_ret).cumprod()
    ax3.plot(crisis_dates_oos, our_cum, '-',
            color=COLORS['our_model'], linewidth=3.0,
            label=f'Our Model ({(our_cum[-1]-1)*100:+.1f}%)', zorder=5)
    
    # Baseline cumulative returns
    for bname in ['1/N Equal', 'Risk Parity', 'Min Variance', '60/40']:
        if bname in data['baseline_weights']:
            bw = data['baseline_weights'][bname]
            bw_crisis = bw[full_mask]
            if len(bw_crisis) > 0:
                n_common = min(len(bw_crisis), len(crisis_returns_full))
                bl_ret = (bw_crisis[:n_common] * crisis_returns_full[:n_common]).sum(axis=1)
                bl_cum = (1 + bl_ret).cumprod()
                total_ret = (bl_cum[-1] - 1) * 100
                ax3.plot(crisis_dates_full[:n_common], bl_cum, '--',
                        color=COLORS.get(bname, 'gray'), linewidth=1.5,
                        alpha=0.8, label=f'{bname} ({total_ret:+.1f}%)')
    
    # SPY reference (raw)
    spy_idx = data['asset_names'].index('SPY') if 'SPY' in data['asset_names'] else 0
    spy_ret = crisis_returns_full[:, spy_idx]
    spy_cum = (1 + spy_ret).cumprod()
    ax3.plot(crisis_dates_full[:len(spy_cum)], spy_cum, ':',
            color='#E91E63', linewidth=1.5, alpha=0.7,
            label=f'SPY Only ({(spy_cum[-1]-1)*100:+.1f}%)')
    
    ax3.axhline(y=1.0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    ax3.set_ylabel('Cumulative Return ($1)', fontsize=11, fontweight='bold')
    ax3.set_title('③ Portfolio Value — Defensive Performance',
                  fontsize=12, fontweight='bold', loc='left')
    ax3.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # Event line
    if start <= event_date < end:
        ax3.axvline(x=event_date, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # X-axis formatting
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [Saved] {save_path}")
    plt.close()


# =============================================================================
# Summary Statistics Table
# =============================================================================

def _calc_mdd(returns: np.ndarray) -> float:
    """최대 낙폭(MDD) 계산. returns는 월별 수익률 배열."""
    if len(returns) == 0:
        return 0.0
    cumulative = (1 + returns).cumprod()
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / rolling_max - 1
    return float(np.min(drawdowns))


def print_crisis_stats(data, period_info):
    """위기 구간 핵심 통계 출력."""
    start = pd.Timestamp(period_info['start'])
    end = pd.Timestamp(period_info['end'])

    # OOS 기간 기준 마스크 (우리 모델용)
    oos_mask = (data['oos_dates'] >= start) & (data['oos_dates'] < end)
    crisis_weights = data['adj_weights'][oos_mask]
    crisis_returns = data['oos_returns'][oos_mask]   # shape: (T, n_assets)

    if len(crisis_weights) == 0:
        print(f"  [SKIP] OOS 기간({data['oos_dates'][0].date()} ~ {data['oos_dates'][-1].date()})에"
              f" 해당 위기 구간이 없습니다.")
        return

    n_months = len(crisis_weights)
    bil_idx  = data['asset_names'].index('BIL') if 'BIL' in data['asset_names'] else -1
    spy_idx  = data['asset_names'].index('SPY') if 'SPY' in data['asset_names'] else 0

    # ── 우리 모델 통계 ──────────────────────────────────────
    our_ret     = (crisis_weights * crisis_returns).sum(axis=1)
    our_cum     = (1 + our_ret).cumprod()[-1] - 1
    our_mdd     = _calc_mdd(our_ret)
    our_avg_bil = crisis_weights[:, bil_idx].mean() if bil_idx >= 0 else 0.0
    our_vol     = float(np.std(our_ret, ddof=1) * np.sqrt(12)) if len(our_ret) > 1 else 0.0

    # ── SPY 통계 (oos_returns 사용 → 날짜 정합성 보장) ──────
    spy_ret = crisis_returns[:, spy_idx]            # oos 기간 SPY 월수익률
    spy_cum = (1 + spy_ret).cumprod()[-1] - 1
    spy_mdd = _calc_mdd(spy_ret)
    spy_vol = float(np.std(spy_ret, ddof=1) * np.sqrt(12)) if len(spy_ret) > 1 else 0.0

    # ── 출력 ────────────────────────────────────────────────
    print(f"\n  {'Metric':<28s} {'Our Model':>12s} {'SPY Only':>12s}")
    print(f"  {'-'*55}")
    print(f"  {'Period (months)':<28s} {n_months:>12d} {n_months:>12d}")
    print(f"  {'Cumulative Return':<28s} {our_cum:>12.2%} {spy_cum:>12.2%}")
    print(f"  {'Max Drawdown (MDD)':<28s} {our_mdd:>12.2%} {spy_mdd:>12.2%}")
    print(f"  {'Annualized Volatility':<28s} {our_vol:>12.2%} {spy_vol:>12.2%}")
    print(f"  {'Avg BIL Weight':<28s} {our_avg_bil:>12.2%} {'N/A':>12s}")

    if spy_mdd < -1e-6 and our_mdd < -1e-6:
        # MDD 절대값 차이 (포인트) — "우리가 얼마나 덜 빠졌나"
        mdd_saved = abs(spy_mdd) - abs(our_mdd)
        print(f"  {'MDD Saved vs SPY':<28s} {mdd_saved:>+12.2%}")
    elif spy_mdd >= -1e-6:
        print(f"  {'MDD Saved vs SPY':<28s} {'(SPY MDD≈0, N/A)':>12s}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  Crisis Period Deep Dive — Defensive Performance Analysis")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Load data ---
    print("\n[Step 1] 데이터 로드")
    data = load_all_data()
    print(f"  OOS period: {data['oos_dates'][0].date()} ~ {data['oos_dates'][-1].date()}")
    print(f"  Full period: {data['full_dates'][0].date()} ~ {data['full_dates'][-1].date()}")
    
    # --- Generate crisis charts ---
    for period_key, period_info in CRISIS_PERIODS.items():
        print(f"\n{'='*50}")
        print(f"  {period_info['label']}")
        print(f"  {period_info['start']} ~ {period_info['end']}")
        print(f"{'='*50}")
        
        save_path = os.path.join(OUTPUT_DIR, f'{period_key}.png')
        
        plot_crisis_deepdive(data, period_key, period_info, save_path)
        print_crisis_stats(data, period_info)
    
    print(f"\n{'='*70}")
    print(f"  모든 위기 구간 분석 완료")
    print(f"  Outputs: {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
