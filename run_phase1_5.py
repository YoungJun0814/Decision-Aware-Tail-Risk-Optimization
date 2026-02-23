"""
Phase 1.5: MIDAS Hyperparameter Optimization & Statistical Validation
======================================================================
Phase 1 결과 (ρ=0.903, Sharpe +0.018) 를 개선하기 위한 후속 실험입니다.

실행 내용:
  Part 1: Grid Search — K × poly_degree × window (27 조합)
    → 각 조합의 상관계수(ρ)를 계산하여 최적 설정을 탐색합니다.
  Part 2: Multi-Seed Benchmark — 최적 설정으로 5개 seed GRU 학습
    → Sharpe Ratio의 통계적 유의성을 검증합니다 (paired t-test).
  Part 3: 시각화 및 결과 저장

산출물: results/phase1_5/ 디렉토리
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import os
import warnings
import itertools
import time
from scipy import stats

from src.midas import (
    MIDASFeatureExtractor,
    download_daily_vix,
    download_daily_spy_returns,
    get_monthly_vix_snapshot,
)
from src.data_loader import prepare_training_data, ASSET_TICKERS
from src.models import get_model
from src.loss import DecisionAwareLoss
from src.utils import set_seed, get_device, calculate_mdd

# =============================================================================
# 설정
# =============================================================================

# 데이터 기간
DATA_START = '2007-01-01'
DATA_END = '2024-01-01'

# Grid Search 범위
GRID_K = [22, 44, 66]           # 1, 2, 3 거래월
GRID_POLY_DEGREE = [1, 2, 3]    # 1차~3차 Almon
GRID_WINDOW = [36, 60, 84]      # 3년, 5년, 7년 롤링

# Multi-Seed 설정
N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 1024]

# GRU 모델 설정
MODEL_CONFIG = {
    'start_date': '2007-07-01',
    'end_date': '2024-01-01',
    'seq_length': 12,
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'train_ratio': 0.8,
    'eta': 1.0,
    'kappa_base': 0.001,
    'kappa_vix_scale': 0.0001,
    'device': get_device(),
}

OUTPUT_DIR = os.path.join('results', 'phase1_5')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


# =============================================================================
# Helper: GRU 학습 + 평가
# =============================================================================

def train_gru_single_seed(X_tensor, y_tensor, vix_tensor, scaler, config, seed, label=""):
    """
    단일 seed로 GRU 모델을 학습하고 성과 지표를 반환합니다.

    Parameters
    ----------
    X_tensor, y_tensor, vix_tensor : torch.Tensor
        입력/타겟/VIX 텐서
    scaler : StandardScaler
        역변환용 스케일러
    config : dict
        학습 설정
    seed : int
        랜덤 시드
    label : str
        로깅 라벨

    Returns
    -------
    dict
        sharpe, annual_return, mdd, val_loss, val_returns
    """
    device = config['device']
    input_dim = X_tensor.shape[-1]
    num_assets = y_tensor.shape[-1]

    # 시드 설정
    set_seed(seed)

    # 1. 모델 생성
    model = get_model(
        'gru', input_dim, num_assets, device=device,
        omega_mode='learnable', sigma_mode='prior'
    )

    # 2. 손실 함수 및 옵티마이저
    loss_fn = DecisionAwareLoss(
        eta=config['eta'],
        kappa_base=config['kappa_base'],
        kappa_vix_scale=config['kappa_vix_scale'],
        risk_type='cvar'
    )
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 3. 데이터 분할
    X_dev = X_tensor.to(device).float()
    y_dev = y_tensor.to(device).float()
    vix_dev = vix_tensor.to(device).float()

    train_size = int(len(X_dev) * config['train_ratio'])
    X_train, X_val = X_dev[:train_size], X_dev[train_size:]
    y_train, y_val = y_dev[:train_size], y_dev[train_size:]
    vix_train, vix_val = vix_dev[:train_size], vix_dev[train_size:]

    # 4. 학습 루프
    best_val_loss = float('inf')
    patience = 20
    no_improve = 0
    val_weights = None

    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()

        weights = model(X_train)
        loss = loss_fn(weights, y_train, vix_train)

        loss.backward()
        optimizer.step()

        # 검증
        model.eval()
        with torch.no_grad():
            current_val_weights = model(X_val)
            val_loss = loss_fn(current_val_weights, y_val, vix_val)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            no_improve = 0
            val_weights = current_val_weights
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # 5. 지표 계산
    if val_weights is not None:
        if scaler is not None:
            mean = torch.tensor(scaler.mean_[:num_assets], device=device).float()
            scale = torch.tensor(scaler.scale_[:num_assets], device=device).float()
            y_val_real = y_val * scale + mean
        else:
            y_val_real = y_val

        portfolio_returns = (val_weights * y_val_real).sum(dim=1).cpu().numpy()
        mean_ret = np.mean(portfolio_returns) * 12
        std_ret = np.std(portfolio_returns) * np.sqrt(12)
        sharpe = mean_ret / (std_ret + 1e-6)
        mdd = calculate_mdd(portfolio_returns)
    else:
        mean_ret = 0.0
        sharpe = 0.0
        mdd = 0.0
        portfolio_returns = np.zeros(len(y_val))

    return {
        'seed': seed,
        'label': label,
        'val_loss': best_val_loss,
        'sharpe': sharpe,
        'annual_return': mean_ret,
        'mdd': mdd,
        'val_returns': portfolio_returns,
    }


# =============================================================================
# Part 1: Grid Search
# =============================================================================

def run_grid_search(daily_vix, monthly_spy_returns, monthly_vix_snapshot):
    """
    K × poly_degree × window 그리드 서치를 수행합니다.
    각 조합에 대해 MIDAS Feature를 생성하고 월말 VIX와의 상관계수를 계산합니다.

    Returns
    -------
    pd.DataFrame
        그리드 서치 결과 테이블
    """
    print("\n" + "=" * 70)
    print("[Part 1] Grid Search: K × poly_degree × window")
    print("=" * 70)

    combinations = list(itertools.product(GRID_K, GRID_POLY_DEGREE, GRID_WINDOW))
    total = len(combinations)
    print(f"  총 {total}개 조합 탐색")

    results = []

    for i, (K, poly_deg, window) in enumerate(combinations):
        print(f"\n  [{i+1}/{total}] K={K}, poly_degree={poly_deg}, window={window}")
        t_start = time.time()

        try:
            # MIDAS 추출기 생성 및 실행
            extractor = MIDASFeatureExtractor(K=K, poly_degree=poly_deg)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                midas_vix = extractor.fit_transform(
                    daily_vix, monthly_spy_returns, window=window
                )

            # 상관계수 계산
            common_idx = midas_vix.index.intersection(monthly_vix_snapshot.index)
            if len(common_idx) < 10:
                print(f"    ⚠️ 공통 데이터 부족 ({len(common_idx)}개)")
                continue

            midas_aligned = midas_vix.loc[common_idx]
            monthly_aligned = monthly_vix_snapshot.loc[common_idx]

            correlation = midas_aligned.corr(monthly_aligned)
            n_features = len(midas_vix)

            # Almon 가중치 정보
            latest_weights = extractor.get_latest_weights()
            peak_lag = np.argmax(latest_weights) + 1  # 1-indexed
            peak_weight = latest_weights[peak_lag - 1]
            weight_entropy = -np.sum(latest_weights * np.log(latest_weights + 1e-10))

            elapsed = time.time() - t_start
            print(f"    ρ = {correlation:.4f} | peak_lag = {peak_lag} | "
                  f"n = {n_features}개월 | {elapsed:.1f}s")

            results.append({
                'K': K,
                'poly_degree': poly_deg,
                'window': window,
                'correlation': correlation,
                'n_features': n_features,
                'peak_lag': peak_lag,
                'peak_weight': peak_weight,
                'weight_entropy': weight_entropy,
            })

        except Exception as e:
            print(f"    ❌ 실패: {e}")
            results.append({
                'K': K,
                'poly_degree': poly_deg,
                'window': window,
                'correlation': np.nan,
                'n_features': 0,
                'peak_lag': 0,
                'peak_weight': 0,
                'weight_entropy': 0,
            })

    df = pd.DataFrame(results)

    # 결과 정렬 (ρ가 낮을수록 좋음)
    df = df.sort_values('correlation', ascending=True).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("  Grid Search 결과 (상관계수 낮은 순)")
    print("=" * 70)
    print(df.to_string(index=False))

    return df


# =============================================================================
# Part 2: Multi-Seed Benchmark
# =============================================================================

def run_multi_seed_benchmark(best_config, config):
    """
    최적 MIDAS 설정으로 5개 seed GRU 학습 후 통계적 유의성을 검증합니다.

    Parameters
    ----------
    best_config : dict
        최적 MIDAS 하이퍼파라미터 {K, poly_degree, window}
    config : dict
        GRU 학습 설정

    Returns
    -------
    baseline_results : list of dict
    midas_results : list of dict
    test_result : dict (t-test 결과)
    """
    print("\n" + "=" * 70)
    print("[Part 2] Multi-Seed Benchmark (5 seeds)")
    print("=" * 70)
    print(f"  최적 MIDAS 설정: K={best_config['K']}, "
          f"poly_degree={best_config['poly_degree']}, "
          f"window={best_config['window']}")
    print(f"  Seeds: {SEEDS}")

    baseline_results = []
    midas_results = []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'─'*60}")
        print(f"  Seed {seed} ({i+1}/{N_SEEDS})")
        print(f"{'─'*60}")

        # --- A: Baseline (월간 Feature) ---
        print(f"\n  [A] Baseline 데이터 로딩...")
        set_seed(seed)
        X_base, y_base, vix_base, scaler_base, _, _, _ = prepare_training_data(
            start_date=config['start_date'],
            end_date=config['end_date'],
            seq_length=config['seq_length'],
            normalize=True,
            train_ratio=config['train_ratio'],
            use_midas=False,
        )

        print(f"  [A] Baseline GRU 학습 중 (seed={seed})...")
        result_base = train_gru_single_seed(
            X_base, y_base, vix_base, scaler_base,
            config, seed, label=f"Baseline"
        )
        baseline_results.append(result_base)
        print(f"    → Sharpe: {result_base['sharpe']:.4f}, "
              f"MDD: {result_base['mdd']:.2%}, "
              f"Annual Ret: {result_base['annual_return']:.2%}")

        # --- B: MIDAS (최적 설정) ---
        print(f"\n  [B] MIDAS 데이터 로딩...")
        set_seed(seed)
        X_midas, y_midas, vix_midas, scaler_midas, _, _, _ = prepare_training_data(
            start_date=config['start_date'],
            end_date=config['end_date'],
            seq_length=config['seq_length'],
            normalize=True,
            train_ratio=config['train_ratio'],
            use_midas=True,
            midas_K=best_config['K'],
            midas_poly_degree=best_config['poly_degree'],
            midas_window=best_config['window'],
        )

        print(f"  [B] MIDAS GRU 학습 중 (seed={seed})...")
        result_midas = train_gru_single_seed(
            X_midas, y_midas, vix_midas, scaler_midas,
            config, seed, label=f"MIDAS"
        )
        midas_results.append(result_midas)
        print(f"    → Sharpe: {result_midas['sharpe']:.4f}, "
              f"MDD: {result_midas['mdd']:.2%}, "
              f"Annual Ret: {result_midas['annual_return']:.2%}")

    # --- 통계적 검증 ---
    print("\n" + "=" * 70)
    print("  통계적 유의성 검증")
    print("=" * 70)

    sharpe_baseline = np.array([r['sharpe'] for r in baseline_results])
    sharpe_midas = np.array([r['sharpe'] for r in midas_results])
    sharpe_diff = sharpe_midas - sharpe_baseline

    ret_baseline = np.array([r['annual_return'] for r in baseline_results])
    ret_midas = np.array([r['annual_return'] for r in midas_results])

    mdd_baseline = np.array([r['mdd'] for r in baseline_results])
    mdd_midas = np.array([r['mdd'] for r in midas_results])

    loss_baseline = np.array([r['val_loss'] for r in baseline_results])
    loss_midas = np.array([r['val_loss'] for r in midas_results])

    # Paired t-test on Sharpe
    t_stat_sharpe, p_value_sharpe = stats.ttest_rel(sharpe_midas, sharpe_baseline)

    # Paired t-test on Annual Return
    t_stat_ret, p_value_ret = stats.ttest_rel(ret_midas, ret_baseline)

    # Wilcoxon signed-rank test (비모수 검정, 소표본에 더 적합)
    try:
        w_stat, p_value_wilcoxon = stats.wilcoxon(sharpe_diff)
    except ValueError:
        # 모든 차이가 0인 경우
        w_stat, p_value_wilcoxon = np.nan, np.nan

    # Win rate
    win_rate = np.mean(sharpe_diff > 0)

    test_result = {
        't_stat_sharpe': t_stat_sharpe,
        'p_value_sharpe': p_value_sharpe,
        't_stat_ret': t_stat_ret,
        'p_value_ret': p_value_ret,
        'w_stat': w_stat,
        'p_value_wilcoxon': p_value_wilcoxon,
        'win_rate': win_rate,
        'sharpe_baseline_mean': np.mean(sharpe_baseline),
        'sharpe_baseline_std': np.std(sharpe_baseline),
        'sharpe_midas_mean': np.mean(sharpe_midas),
        'sharpe_midas_std': np.std(sharpe_midas),
        'sharpe_diff_mean': np.mean(sharpe_diff),
        'sharpe_diff_std': np.std(sharpe_diff),
        'ret_baseline_mean': np.mean(ret_baseline),
        'ret_midas_mean': np.mean(ret_midas),
        'mdd_baseline_mean': np.mean(mdd_baseline),
        'mdd_midas_mean': np.mean(mdd_midas),
        'loss_baseline_mean': np.mean(loss_baseline),
        'loss_midas_mean': np.mean(loss_midas),
    }

    # 결과 출력
    print(f"\n  {'':>20} {'Baseline':>20} {'MIDAS':>20}")
    print(f"  {'─'*62}")
    print(f"  {'Sharpe (mean±std)':>20} {np.mean(sharpe_baseline):>10.4f}±{np.std(sharpe_baseline):.4f} {np.mean(sharpe_midas):>10.4f}±{np.std(sharpe_midas):.4f}")
    print(f"  {'Annual Return':>20} {np.mean(ret_baseline):>19.2%} {np.mean(ret_midas):>19.2%}")
    print(f"  {'MDD':>20} {np.mean(mdd_baseline):>19.2%} {np.mean(mdd_midas):>19.2%}")
    print(f"  {'Val Loss':>20} {np.mean(loss_baseline):>20.4f} {np.mean(loss_midas):>20.4f}")
    print(f"  {'─'*62}")
    print(f"  Sharpe 차이: {np.mean(sharpe_diff):+.4f} ± {np.std(sharpe_diff):.4f}")
    print(f"  Win Rate: {win_rate:.0%} ({int(np.sum(sharpe_diff > 0))}/{N_SEEDS})")
    print(f"\n  Paired t-test (Sharpe): t={t_stat_sharpe:.4f}, p={p_value_sharpe:.4f}")
    print(f"  Paired t-test (Return): t={t_stat_ret:.4f}, p={p_value_ret:.4f}")
    print(f"  Wilcoxon test (Sharpe): W={w_stat}, p={p_value_wilcoxon:.4f}" if not np.isnan(w_stat) else "  Wilcoxon test: N/A")

    if p_value_sharpe < 0.05:
        print(f"\n  ✅ Sharpe 차이가 통계적으로 유의합니다 (p < 0.05)")
    elif p_value_sharpe < 0.10:
        print(f"\n  △ Sharpe 차이가 약한 유의성을 보입니다 (p < 0.10)")
    else:
        print(f"\n  ⚠️ Sharpe 차이가 통계적으로 유의하지 않습니다 (p ≥ 0.10)")

    return baseline_results, midas_results, test_result


# =============================================================================
# Part 3: 시각화 및 결과 저장
# =============================================================================

def save_grid_search_results(grid_df):
    """Grid Search 결과를 파일과 히트맵으로 저장합니다."""

    # CSV 저장
    csv_path = os.path.join(OUTPUT_DIR, 'grid_search_results.csv')
    grid_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  Grid Search CSV 저장: {csv_path}")

    # 텍스트 보고서
    txt_path = os.path.join(OUTPUT_DIR, 'grid_search_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Phase 1.5: MIDAS Hyperparameter Grid Search\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Grid 범위:\n")
        f.write(f"  K: {GRID_K}\n")
        f.write(f"  poly_degree: {GRID_POLY_DEGREE}\n")
        f.write(f"  window: {GRID_WINDOW}\n")
        f.write(f"  총 조합: {len(grid_df)}\n\n")
        f.write("결과 (상관계수 낮은 순):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'K':>4} {'poly':>5} {'window':>7} {'ρ':>8} {'n':>5} {'peak_lag':>9} {'entropy':>9}\n")
        f.write("-" * 80 + "\n")
        for _, row in grid_df.iterrows():
            f.write(f"{int(row['K']):>4} {int(row['poly_degree']):>5} "
                    f"{int(row['window']):>7} {row['correlation']:>8.4f} "
                    f"{int(row['n_features']):>5} {int(row['peak_lag']):>9} "
                    f"{row['weight_entropy']:>9.4f}\n")
        f.write("-" * 80 + "\n")

        best = grid_df.iloc[0]
        f.write(f"\n최적 설정: K={int(best['K'])}, poly_degree={int(best['poly_degree'])}, "
                f"window={int(best['window'])}\n")
        f.write(f"최적 상관계수: {best['correlation']:.4f}\n")
        f.write(f"ρ < 0.9 달성: {'예' if best['correlation'] < 0.9 else '아니오'}\n")
        f.write(f"ρ < 0.85 달성: {'예' if best['correlation'] < 0.85 else '아니오'}\n")
    print(f"  Grid Search 보고서 저장: {txt_path}")

    # 히트맵 — poly_degree 별로 K × window 히트맵 생성
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for idx, poly_deg in enumerate(GRID_POLY_DEGREE):
        ax = axes[idx]
        subset = grid_df[grid_df['poly_degree'] == poly_deg]

        if subset.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'poly_degree = {poly_deg}')
            continue

        # 피벗 테이블 생성
        pivot = subset.pivot_table(
            values='correlation', index='K', columns='window', aggfunc='first'
        )

        # 히트맵
        im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto',
                       vmin=grid_df['correlation'].min() - 0.01,
                       vmax=grid_df['correlation'].max() + 0.01)

        # 축 라벨
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.astype(int))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(int))
        ax.set_xlabel('Window (months)')
        ax.set_ylabel('K (daily lags)')
        ax.set_title(f'poly_degree = {poly_deg}')

        # 셀에 값 표시
        for row_idx in range(len(pivot.index)):
            for col_idx in range(len(pivot.columns)):
                val = pivot.values[row_idx, col_idx]
                if not np.isnan(val):
                    color = 'white' if val > 0.9 else 'black'
                    ax.text(col_idx, row_idx, f'{val:.3f}',
                            ha='center', va='center', color=color, fontsize=10)

    plt.colorbar(im, ax=axes, shrink=0.8, label='Correlation (ρ)')
    plt.suptitle('MIDAS Grid Search: Correlation with Monthly VIX', fontsize=14, y=1.02)
    plt.tight_layout()

    heatmap_path = os.path.join(OUTPUT_DIR, 'grid_search_heatmap.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grid Search 히트맵 저장: {heatmap_path}")


def save_multi_seed_results(baseline_results, midas_results, test_result, best_config):
    """Multi-Seed Benchmark 결과를 저장합니다."""

    # 텍스트 보고서
    txt_path = os.path.join(OUTPUT_DIR, 'multi_seed_benchmark.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Phase 1.5: Multi-Seed GRU Benchmark\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"MIDAS 설정 (Grid Search 최적):\n")
        f.write(f"  K: {best_config['K']}\n")
        f.write(f"  poly_degree: {best_config['poly_degree']}\n")
        f.write(f"  window: {best_config['window']}\n\n")
        f.write(f"Seeds: {SEEDS}\n\n")

        # 개별 결과
        f.write("개별 결과:\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Seed':>6} {'Baseline Sharpe':>16} {'MIDAS Sharpe':>14} {'차이':>10}\n")
        f.write("-" * 65 + "\n")
        for b, m in zip(baseline_results, midas_results):
            diff = m['sharpe'] - b['sharpe']
            f.write(f"{b['seed']:>6} {b['sharpe']:>16.4f} {m['sharpe']:>14.4f} {diff:>+10.4f}\n")
        f.write("-" * 65 + "\n\n")

        # 요약 통계
        f.write("요약 통계:\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'':>20} {'Baseline':>20} {'MIDAS':>20}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Sharpe (mean±std)':>20} "
                f"{test_result['sharpe_baseline_mean']:>10.4f}±{test_result['sharpe_baseline_std']:.4f} "
                f"{test_result['sharpe_midas_mean']:>10.4f}±{test_result['sharpe_midas_std']:.4f}\n")
        f.write(f"{'Annual Return':>20} "
                f"{test_result['ret_baseline_mean']:>19.2%} "
                f"{test_result['ret_midas_mean']:>19.2%}\n")
        f.write(f"{'MDD':>20} "
                f"{test_result['mdd_baseline_mean']:>19.2%} "
                f"{test_result['mdd_midas_mean']:>19.2%}\n")
        f.write(f"{'Val Loss':>20} "
                f"{test_result['loss_baseline_mean']:>20.4f} "
                f"{test_result['loss_midas_mean']:>20.4f}\n")
        f.write("-" * 65 + "\n\n")

        # 통계적 검증
        f.write("통계적 검증:\n")
        f.write(f"  Sharpe 차이: {test_result['sharpe_diff_mean']:+.4f} ± {test_result['sharpe_diff_std']:.4f}\n")
        f.write(f"  Win Rate: {test_result['win_rate']:.0%}\n")
        f.write(f"  Paired t-test (Sharpe): t={test_result['t_stat_sharpe']:.4f}, p={test_result['p_value_sharpe']:.4f}\n")
        f.write(f"  Paired t-test (Return): t={test_result['t_stat_ret']:.4f}, p={test_result['p_value_ret']:.4f}\n")
        if not np.isnan(test_result['w_stat']):
            f.write(f"  Wilcoxon test (Sharpe): W={test_result['w_stat']}, p={test_result['p_value_wilcoxon']:.4f}\n")
        f.write(f"\n")

        # 판정
        p = test_result['p_value_sharpe']
        if p < 0.01:
            f.write(f"판정: ✅ 매우 유의 (p < 0.01)\n")
        elif p < 0.05:
            f.write(f"판정: ✅ 유의 (p < 0.05)\n")
        elif p < 0.10:
            f.write(f"판정: △ 약한 유의성 (p < 0.10)\n")
        else:
            f.write(f"판정: ⚠️ 유의하지 않음 (p ≥ 0.10)\n")

    print(f"\n  Multi-Seed 보고서 저장: {txt_path}")

    # Box Plot — Sharpe 분포 비교
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sharpe_b = [r['sharpe'] for r in baseline_results]
    sharpe_m = [r['sharpe'] for r in midas_results]
    ret_b = [r['annual_return'] for r in baseline_results]
    ret_m = [r['annual_return'] for r in midas_results]
    mdd_b = [r['mdd'] for r in baseline_results]
    mdd_m = [r['mdd'] for r in midas_results]

    # Sharpe Box Plot
    bp1 = axes[0].boxplot([sharpe_b, sharpe_m], labels=['Baseline', 'MIDAS'],
                           patch_artist=True, widths=0.5)
    bp1['boxes'][0].set_facecolor('coral')
    bp1['boxes'][1].set_facecolor('steelblue')
    for i, vals in enumerate([sharpe_b, sharpe_m]):
        x = np.ones(len(vals)) * (i + 1) + np.random.normal(0, 0.04, len(vals))
        axes[0].scatter(x, vals, color='black', alpha=0.6, s=40, zorder=3)
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].set_title(f'Sharpe Ratio (p={test_result["p_value_sharpe"]:.3f})')

    # Annual Return Box Plot
    bp2 = axes[1].boxplot([ret_b, ret_m], labels=['Baseline', 'MIDAS'],
                           patch_artist=True, widths=0.5)
    bp2['boxes'][0].set_facecolor('coral')
    bp2['boxes'][1].set_facecolor('steelblue')
    for i, vals in enumerate([ret_b, ret_m]):
        x = np.ones(len(vals)) * (i + 1) + np.random.normal(0, 0.04, len(vals))
        axes[1].scatter(x, vals, color='black', alpha=0.6, s=40, zorder=3)
    axes[1].set_ylabel('Annual Return')
    axes[1].set_title('Annual Return')

    # MDD Box Plot
    bp3 = axes[2].boxplot([mdd_b, mdd_m], labels=['Baseline', 'MIDAS'],
                           patch_artist=True, widths=0.5)
    bp3['boxes'][0].set_facecolor('coral')
    bp3['boxes'][1].set_facecolor('steelblue')
    for i, vals in enumerate([mdd_b, mdd_m]):
        x = np.ones(len(vals)) * (i + 1) + np.random.normal(0, 0.04, len(vals))
        axes[2].scatter(x, vals, color='black', alpha=0.6, s=40, zorder=3)
    axes[2].set_ylabel('MDD')
    axes[2].set_title('Maximum Drawdown')

    plt.suptitle(f'Multi-Seed GRU Benchmark ({N_SEEDS} seeds)', fontsize=14)
    plt.tight_layout()

    boxplot_path = os.path.join(OUTPUT_DIR, 'multi_seed_boxplot.png')
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Box Plot 저장: {boxplot_path}")

    # Seed별 비교 Bar Chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    x = np.arange(N_SEEDS)
    width = 0.35

    ax.bar(x - width/2, sharpe_b, width, label='Baseline', color='coral', alpha=0.8)
    ax.bar(x + width/2, sharpe_m, width, label='MIDAS', color='steelblue', alpha=0.8)

    ax.set_xlabel('Seed')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Seed-by-Seed Sharpe Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.legend()

    # 평균선
    ax.axhline(y=np.mean(sharpe_b), color='coral', linestyle='--', alpha=0.5,
               label=f'Baseline mean ({np.mean(sharpe_b):.4f})')
    ax.axhline(y=np.mean(sharpe_m), color='steelblue', linestyle='--', alpha=0.5,
               label=f'MIDAS mean ({np.mean(sharpe_m):.4f})')
    ax.legend()

    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, 'seed_comparison_bar.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Seed별 비교 차트 저장: {bar_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    total_start = time.time()

    print("=" * 70)
    print("Phase 1.5: MIDAS Hyperparameter Optimization & Statistical Validation")
    print("=" * 70)

    # =========================================================================
    # 데이터 다운로드 (1회)
    # =========================================================================
    print("\n[데이터 준비]")
    daily_vix = download_daily_vix(DATA_START, DATA_END)
    monthly_spy_returns = download_daily_spy_returns(DATA_START, DATA_END)
    monthly_vix_snapshot = get_monthly_vix_snapshot(DATA_START, DATA_END)

    # =========================================================================
    # Part 1: Grid Search
    # =========================================================================
    grid_df = run_grid_search(daily_vix, monthly_spy_returns, monthly_vix_snapshot)
    save_grid_search_results(grid_df)

    # 최적 설정 추출
    best_row = grid_df.iloc[0]
    best_config = {
        'K': int(best_row['K']),
        'poly_degree': int(best_row['poly_degree']),
        'window': int(best_row['window']),
        'correlation': best_row['correlation'],
    }

    print(f"\n{'='*50}")
    print(f"  최적 설정: K={best_config['K']}, "
          f"poly_degree={best_config['poly_degree']}, "
          f"window={best_config['window']}")
    print(f"  최적 상관계수: {best_config['correlation']:.4f}")
    print(f"  ρ < 0.9 달성: {'✅ 예' if best_config['correlation'] < 0.9 else '⚠️ 아니오'}")
    print(f"  ρ < 0.85 달성: {'✅ 예' if best_config['correlation'] < 0.85 else '⚠️ 아니오'}")
    print(f"{'='*50}")

    # =========================================================================
    # Part 2: Multi-Seed Benchmark
    # =========================================================================
    baseline_results, midas_results, test_result = run_multi_seed_benchmark(
        best_config, MODEL_CONFIG
    )
    save_multi_seed_results(baseline_results, midas_results, test_result, best_config)

    # =========================================================================
    # 최종 요약
    # =========================================================================
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("Phase 1.5 전체 결과 요약")
    print("=" * 70)
    print(f"\n  [Grid Search]")
    print(f"    탐색 조합: {len(grid_df)}개")
    print(f"    최적: K={best_config['K']}, poly_degree={best_config['poly_degree']}, "
          f"window={best_config['window']}")
    print(f"    최적 ρ: {best_config['correlation']:.4f} "
          f"(Phase 1: 0.9028 → Phase 1.5: {best_config['correlation']:.4f})")

    print(f"\n  [Multi-Seed Benchmark]")
    print(f"    Baseline Sharpe: {test_result['sharpe_baseline_mean']:.4f} ± "
          f"{test_result['sharpe_baseline_std']:.4f}")
    print(f"    MIDAS Sharpe: {test_result['sharpe_midas_mean']:.4f} ± "
          f"{test_result['sharpe_midas_std']:.4f}")
    print(f"    차이: {test_result['sharpe_diff_mean']:+.4f} ± {test_result['sharpe_diff_std']:.4f}")
    print(f"    Win Rate: {test_result['win_rate']:.0%}")
    print(f"    Paired t-test p-value: {test_result['p_value_sharpe']:.4f}")

    p = test_result['p_value_sharpe']
    if p < 0.05:
        print(f"\n  ✅ MIDAS Feature가 통계적으로 유의한 개선을 보입니다 (p={p:.4f})")
    elif p < 0.10:
        print(f"\n  △ MIDAS Feature가 약한 유의성을 보입니다 (p={p:.4f})")
    else:
        print(f"\n  ⚠️ MIDAS Feature 개선이 통계적으로 유의하지 않습니다 (p={p:.4f})")

    print(f"\n  산출물 ({OUTPUT_DIR}/):")
    print(f"    grid_search_results.csv")
    print(f"    grid_search_report.txt")
    print(f"    grid_search_heatmap.png")
    print(f"    multi_seed_benchmark.txt")
    print(f"    multi_seed_boxplot.png")
    print(f"    seed_comparison_bar.png")

    print(f"\n  총 소요 시간: {total_elapsed/60:.1f}분")
    print("\n" + "=" * 70)
    print("Phase 1.5 완료! ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()
