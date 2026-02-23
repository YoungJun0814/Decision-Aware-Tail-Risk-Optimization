"""
Phase 1 Benchmark: MIDAS Feature Engineering 검증
==================================================
이 스크립트는 MIDAS Feature의 정보량 증가를 검증합니다.

검증 항목:
1. MIDAS VIX vs 기존 월말 VIX 상관계수 (목표: < 0.9)
2. HMM Log-Likelihood 비교 (목표: MIDAS >= 월간)
3. Almon Polynomial 가중치 시각화
4. Feature 비교 시계열 시각화
5. 상관관계 산점도
6. GRU 모델 벤치마크: MIDAS Feature vs 기존 월간 Feature

산출물: results/phase1/ 디렉토리
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import os
import warnings

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

START_DATE = '2007-01-01'
END_DATE = '2024-01-01'
K = 22             # 1거래월
POLY_DEGREE = 2    # 2차 Almon
WINDOW = 60        # 5년 롤링 윈도우

OUTPUT_DIR = os.path.join('results', 'phase1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 모델 벤치마크 설정
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
    'seed': 42,
}


def train_gru_and_evaluate(X_tensor, y_tensor, vix_tensor, scaler, config, label=""):
    """
    GRU 모델을 학습하고 성과 지표를 반환합니다.

    Parameters
    ----------
    X_tensor : torch.Tensor
        입력 텐서 (Batch, Seq, Features)
    y_tensor : torch.Tensor
        타겟 텐서 (Batch, Num_assets)
    vix_tensor : torch.Tensor
        VIX 텐서 (Batch,)
    scaler : StandardScaler
        정규화 스케일러 (역변환용)
    config : dict
        학습 설정
    label : str
        실험 라벨 (로깅용)

    Returns
    -------
    dict
        성과 지표: sharpe, annual_return, mdd, val_loss, val_returns
    """
    device = config['device']
    input_dim = X_tensor.shape[-1]
    num_assets = y_tensor.shape[-1]

    print(f"\n  [{label}] GRU 모델 학습 시작 (input_dim={input_dim})...")

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
    X_tensor = X_tensor.to(device).float()
    y_tensor = y_tensor.to(device).float()
    vix_tensor = vix_tensor.to(device).float()

    train_size = int(len(X_tensor) * config['train_ratio'])
    X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
    vix_train, vix_val = vix_tensor[:train_size], vix_tensor[train_size:]

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

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch [{epoch+1}/{config['epochs']}], Loss: {loss.item():.4f}")

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
                print(f"    Early Stopping at epoch {epoch+1}")
                break

    print(f"    Best Val Loss: {best_val_loss:.4f}")

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
        'label': label,
        'val_loss': best_val_loss,
        'sharpe': sharpe,
        'annual_return': mean_ret,
        'mdd': mdd,
        'val_returns': portfolio_returns,
    }


def main():
    print("=" * 70)
    print("Phase 1 Benchmark: MIDAS Feature Engineering")
    print("=" * 70)

    # =========================================================================
    # Step 1: 데이터 수집
    # =========================================================================
    print("\n[Step 1] 데이터 수집")
    daily_vix = download_daily_vix(START_DATE, END_DATE)
    monthly_spy_returns = download_daily_spy_returns(START_DATE, END_DATE)
    monthly_vix_snapshot = get_monthly_vix_snapshot(START_DATE, END_DATE)

    # =========================================================================
    # Step 2: MIDAS Feature 생성
    # =========================================================================
    print("\n[Step 2] MIDAS Feature 생성")
    extractor = MIDASFeatureExtractor(K=K, poly_degree=POLY_DEGREE)
    midas_vix = extractor.fit_transform(daily_vix, monthly_spy_returns, window=WINDOW)

    # =========================================================================
    # Step 3: 상관계수 분석
    # =========================================================================
    print("\n[Step 3] 상관계수 분석")

    # 공통 인덱스 추출
    common_idx = midas_vix.index.intersection(monthly_vix_snapshot.index)
    midas_aligned = midas_vix.loc[common_idx]
    monthly_aligned = monthly_vix_snapshot.loc[common_idx]

    correlation = midas_aligned.corr(monthly_aligned)

    print(f"\n{'='*50}")
    print(f"  MIDAS VIX vs 월말 VIX 상관계수: {correlation:.4f}")
    print(f"  성공 기준: < 0.9")
    if correlation < 0.9:
        print(f"  ✅ 성공 — MIDAS가 추가 정보를 포착합니다")
    else:
        print(f"  ⚠️ 상관계수가 높음 — MIDAS 설정 조정 필요")
    print(f"{'='*50}")

    # 결과 저장
    corr_path = os.path.join(OUTPUT_DIR, 'midas_vs_monthly_correlation.txt')
    with open(corr_path, 'w', encoding='utf-8') as f:
        f.write("MIDAS Feature Engineering — 상관계수 분석\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"설정:\n")
        f.write(f"  K (일간 래그): {K}\n")
        f.write(f"  Polynomial Degree: {POLY_DEGREE}\n")
        f.write(f"  Rolling Window: {WINDOW}개월\n")
        f.write(f"  기간: {START_DATE} ~ {END_DATE}\n\n")
        f.write(f"결과:\n")
        f.write(f"  공통 데이터 수: {len(common_idx)}개월\n")
        f.write(f"  Pearson 상관계수: {correlation:.6f}\n")
        f.write(f"  성공 기준 (< 0.9): {'충족' if correlation < 0.9 else '미충족'}\n\n")
        f.write(f"해석:\n")
        if correlation < 0.9:
            f.write(f"  MIDAS VIX는 기존 월말 VIX와 상관계수 {correlation:.4f}로,\n")
            f.write(f"  월말 스냅샷이 포착하지 못하는 추가적인 정보를 담고 있습니다.\n")
        else:
            f.write(f"  MIDAS VIX와 기존 월말 VIX 간 상관계수가 {correlation:.4f}로 높습니다.\n")
            f.write(f"  K, polynomial degree, window 등의 하이퍼파라미터 조정이 필요할 수 있습니다.\n")
    print(f"  결과 저장: {corr_path}")

    # =========================================================================
    # Step 4: HMM Log-Likelihood 비교
    # =========================================================================
    print("\n[Step 4] HMM Log-Likelihood 비교")

    try:
        from hmmlearn.hmm import GaussianHMM

        # 공통 인덱스 데이터 준비
        features_midas = midas_aligned.values.reshape(-1, 1)
        features_monthly = monthly_aligned.values.reshape(-1, 1)

        # 동일 조건에서 HMM 학습 (2-state, 여러 시드로 best 선택)
        n_states = 2
        n_restarts = 5
        max_iter = 200

        best_ll_midas = -np.inf
        best_ll_monthly = -np.inf

        for seed in range(n_restarts):
            # MIDAS Feature HMM
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    hmm_midas = GaussianHMM(
                        n_components=n_states,
                        covariance_type='diag',
                        n_iter=max_iter,
                        random_state=seed * 42,
                    )
                    hmm_midas.fit(features_midas)
                    ll_midas = hmm_midas.score(features_midas)
                    best_ll_midas = max(best_ll_midas, ll_midas)
                except Exception:
                    pass

            # 월말 Feature HMM
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    hmm_monthly = GaussianHMM(
                        n_components=n_states,
                        covariance_type='diag',
                        n_iter=max_iter,
                        random_state=seed * 42,
                    )
                    hmm_monthly.fit(features_monthly)
                    ll_monthly = hmm_monthly.score(features_monthly)
                    best_ll_monthly = max(best_ll_monthly, ll_monthly)
                except Exception:
                    pass

        # 샘플당 평균 log-likelihood
        ll_midas_per_sample = best_ll_midas / len(features_midas)
        ll_monthly_per_sample = best_ll_monthly / len(features_monthly)

        print(f"\n{'='*50}")
        print(f"  HMM Log-Likelihood (총합):")
        print(f"    MIDAS VIX:  {best_ll_midas:.2f}")
        print(f"    월말 VIX:   {best_ll_monthly:.2f}")
        print(f"  HMM Log-Likelihood (샘플당):")
        print(f"    MIDAS VIX:  {ll_midas_per_sample:.4f}")
        print(f"    월말 VIX:   {ll_monthly_per_sample:.4f}")
        improvement = best_ll_midas - best_ll_monthly
        print(f"  차이: {improvement:+.2f}")
        if improvement >= 0:
            print(f"  ✅ MIDAS Feature가 HMM 학습에 더 적합합니다")
        else:
            print(f"  ⚠️ 월말 Feature가 더 높은 Log-Likelihood를 보입니다")
        print(f"{'='*50}")

        # 결과 저장
        ll_path = os.path.join(OUTPUT_DIR, 'hmm_loglikelihood_comparison.txt')
        with open(ll_path, 'w', encoding='utf-8') as f:
            f.write("HMM Log-Likelihood 비교\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"HMM 설정:\n")
            f.write(f"  States: {n_states}\n")
            f.write(f"  Covariance Type: diag\n")
            f.write(f"  Max Iterations: {max_iter}\n")
            f.write(f"  Random Restarts: {n_restarts}\n\n")
            f.write(f"Log-Likelihood (총합):\n")
            f.write(f"  MIDAS VIX:  {best_ll_midas:.6f}\n")
            f.write(f"  월말 VIX:   {best_ll_monthly:.6f}\n")
            f.write(f"  차이:       {improvement:+.6f}\n\n")
            f.write(f"Log-Likelihood (샘플당):\n")
            f.write(f"  MIDAS VIX:  {ll_midas_per_sample:.6f}\n")
            f.write(f"  월말 VIX:   {ll_monthly_per_sample:.6f}\n\n")
            f.write(f"결과: {'MIDAS 우세' if improvement >= 0 else '월말 우세'}\n")
        print(f"  결과 저장: {ll_path}")

        hmm_available = True

    except ImportError:
        print("  [WARNING] hmmlearn이 설치되지 않았습니다.")
        print("  'pip install hmmlearn>=0.3.0' 실행 후 다시 시도하세요.")
        hmm_available = False

    # =========================================================================
    # Step 5: 시각화
    # =========================================================================
    print("\n[Step 5] 시각화 생성")

    # 5-1. Almon 가중치 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 최근 가중치
    latest_weights = extractor.get_latest_weights()
    lags = np.arange(1, K + 1)
    axes[0].bar(lags, latest_weights, color='steelblue', alpha=0.8, edgecolor='navy')
    axes[0].set_xlabel('Lag (k = 1: 가장 최근)')
    axes[0].set_ylabel('Weight')
    axes[0].set_title('Almon Polynomial Weights (최근 추정)')
    axes[0].axhline(y=1/K, color='red', linestyle='--', alpha=0.5, label=f'균등 가중 (1/{K})')
    axes[0].legend()

    # 가중치 변화 (시간에 따른)
    if extractor.theta_history_ and len(extractor.theta_history_) > 1:
        n_history = len(extractor.theta_history_)
        sample_indices = np.linspace(0, n_history - 1, min(5, n_history), dtype=int)
        for idx in sample_indices:
            w = extractor.get_weights_at(idx)
            axes[1].plot(lags, w, alpha=0.7, label=f'Month {idx}')
        axes[1].set_xlabel('Lag (k = 1: 가장 최근)')
        axes[1].set_ylabel('Weight')
        axes[1].set_title('Almon Weights 시간 변화')
        axes[1].legend(fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'Not enough data for time evolution',
                     ha='center', va='center', transform=axes[1].transAxes)

    plt.tight_layout()
    weight_path = os.path.join(OUTPUT_DIR, 'almon_weights.png')
    plt.savefig(weight_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Almon 가중치 시각화 저장: {weight_path}")

    # 5-2. Feature 비교 시계열
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # MIDAS vs 월말 VIX 시계열 (정규화하여 비교)
    midas_norm = (midas_aligned - midas_aligned.mean()) / midas_aligned.std()
    monthly_norm = (monthly_aligned - monthly_aligned.mean()) / monthly_aligned.std()

    axes[0].plot(midas_norm.index, midas_norm.values, label='MIDAS VIX (정규화)',
                 color='steelblue', linewidth=1.5, alpha=0.9)
    axes[0].plot(monthly_norm.index, monthly_norm.values, label='월말 VIX (정규화)',
                 color='coral', linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Standardized Value')
    axes[0].set_title('MIDAS VIX vs 월말 VIX (정규화 비교)')
    axes[0].legend()

    # 차이 시계열
    diff = midas_norm - monthly_norm
    axes[1].fill_between(diff.index, diff.values, alpha=0.5, color='green', label='차이 (MIDAS - 월말)')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Difference')
    axes[1].set_title('MIDAS vs 월말 VIX 차이 (정규화)')
    axes[1].legend()

    plt.tight_layout()
    feat_path = os.path.join(OUTPUT_DIR, 'feature_comparison.png')
    plt.savefig(feat_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Feature 비교 시각화 저장: {feat_path}")

    # 5-3. 상관관계 산점도
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(monthly_aligned.values, midas_aligned.values,
               alpha=0.5, color='steelblue', edgecolors='navy', s=30)
    ax.set_xlabel('Monthly VIX (월말 스냅샷)')
    ax.set_ylabel('MIDAS VIX (Almon 가중)')
    ax.set_title(f'MIDAS vs Monthly VIX (ρ = {correlation:.4f})')

    # 45도 참조선
    vmin = min(monthly_aligned.min(), midas_aligned.min())
    vmax = max(monthly_aligned.max(), midas_aligned.max())
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', alpha=0.5, label='y = x')

    # 회귀선
    z = np.polyfit(monthly_aligned.values, midas_aligned.values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(vmin, vmax, 100)
    ax.plot(x_line, p(x_line), 'g-', alpha=0.7,
            label=f'regression (y = {z[0]:.2f}x + {z[1]:.2f})')
    ax.legend()

    plt.tight_layout()
    scatter_path = os.path.join(OUTPUT_DIR, 'correlation_scatter.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  상관관계 산점도 저장: {scatter_path}")

    # =========================================================================
    # Step 6: GRU 모델 벤치마크 (MIDAS vs 기존 월간 Feature)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 6] GRU 모델 벤치마크: MIDAS vs 기존 월간 Feature")
    print("=" * 70)

    config = MODEL_CONFIG.copy()

    # --- A: 기존 방식 (월말 Feature만 사용) ---
    print("\n[A] 기존 방식 (월말 Feature)")
    set_seed(config['seed'])

    X_baseline, y_baseline, vix_baseline, scaler_baseline, _, _, _ = prepare_training_data(
        start_date=config['start_date'],
        end_date=config['end_date'],
        seq_length=config['seq_length'],
        normalize=True,
        train_ratio=config['train_ratio'],
        use_midas=False,
    )

    result_baseline = train_gru_and_evaluate(
        X_baseline, y_baseline, vix_baseline, scaler_baseline,
        config, label="Baseline (월간)"
    )

    # --- B: MIDAS 방식 (MIDAS Feature 추가) ---
    print("\n[B] MIDAS 방식 (MIDAS Feature 추가)")
    set_seed(config['seed'])

    X_midas, y_midas, vix_midas, scaler_midas, _, _, _ = prepare_training_data(
        start_date=config['start_date'],
        end_date=config['end_date'],
        seq_length=config['seq_length'],
        normalize=True,
        train_ratio=config['train_ratio'],
        use_midas=True,
        midas_K=K,
        midas_poly_degree=POLY_DEGREE,
        midas_window=WINDOW,
    )

    result_midas = train_gru_and_evaluate(
        X_midas, y_midas, vix_midas, scaler_midas,
        config, label="MIDAS"
    )

    # --- 비교 출력 ---
    print("\n" + "=" * 70)
    print("  GRU 모델 벤치마크 결과")
    print("=" * 70)
    print(f"{'':>20} {'Baseline (월간)':>18} {'MIDAS':>18} {'차이':>12}")
    print("-" * 70)

    sharpe_diff = result_midas['sharpe'] - result_baseline['sharpe']
    ret_diff = result_midas['annual_return'] - result_baseline['annual_return']
    mdd_diff = result_midas['mdd'] - result_baseline['mdd']
    loss_diff = result_midas['val_loss'] - result_baseline['val_loss']

    print(f"{'Sharpe Ratio':>20} {result_baseline['sharpe']:>18.4f} {result_midas['sharpe']:>18.4f} {sharpe_diff:>+12.4f}")
    print(f"{'Annual Return':>20} {result_baseline['annual_return']:>17.2%} {result_midas['annual_return']:>17.2%} {ret_diff:>+11.2%}")
    print(f"{'MDD':>20} {result_baseline['mdd']:>17.2%} {result_midas['mdd']:>17.2%} {mdd_diff:>+11.2%}")
    print(f"{'Val Loss':>20} {result_baseline['val_loss']:>18.4f} {result_midas['val_loss']:>18.4f} {loss_diff:>+12.4f}")
    print(f"{'Input Dim':>20} {X_baseline.shape[-1]:>18d} {X_midas.shape[-1]:>18d}")
    print("-" * 70)

    if result_midas['sharpe'] > result_baseline['sharpe']:
        print("  ✅ MIDAS Feature가 포트폴리오 성과를 개선했습니다!")
    else:
        print("  ⚠️ 기존 Feature가 더 나은 성과를 보입니다. 하이퍼파라미터 조정이 필요할 수 있습니다.")

    # --- 비교 결과 저장 ---
    model_path = os.path.join(OUTPUT_DIR, 'gru_model_benchmark.txt')
    with open(model_path, 'w', encoding='utf-8') as f:
        f.write("GRU 모델 벤치마크: MIDAS vs 기존 월간 Feature\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"모델: GRU (omega=learnable, sigma=prior)\n")
        f.write(f"학습 설정:\n")
        f.write(f"  Epochs: {config['epochs']}\n")
        f.write(f"  Learning Rate: {config['learning_rate']}\n")
        f.write(f"  Batch Size: {config['batch_size']}\n")
        f.write(f"  Train Ratio: {config['train_ratio']}\n")
        f.write(f"  Device: {config['device']}\n")
        f.write(f"  Seed: {config['seed']}\n\n")
        f.write(f"MIDAS 설정:\n")
        f.write(f"  K: {K}\n")
        f.write(f"  Polynomial Degree: {POLY_DEGREE}\n")
        f.write(f"  Rolling Window: {WINDOW}\n\n")
        f.write(f"{'':>20} {'Baseline':>15} {'MIDAS':>15} {'차이':>12}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Sharpe Ratio':>20} {result_baseline['sharpe']:>15.4f} {result_midas['sharpe']:>15.4f} {sharpe_diff:>+12.4f}\n")
        f.write(f"{'Annual Return':>20} {result_baseline['annual_return']:>14.2%} {result_midas['annual_return']:>14.2%} {ret_diff:>+11.2%}\n")
        f.write(f"{'MDD':>20} {result_baseline['mdd']:>14.2%} {result_midas['mdd']:>14.2%} {mdd_diff:>+11.2%}\n")
        f.write(f"{'Val Loss':>20} {result_baseline['val_loss']:>15.4f} {result_midas['val_loss']:>15.4f} {loss_diff:>+12.4f}\n")
        f.write(f"{'Input Dim':>20} {X_baseline.shape[-1]:>15d} {X_midas.shape[-1]:>15d}\n")
        f.write("-" * 65 + "\n\n")
        f.write(f"결과: {'MIDAS 우세' if result_midas['sharpe'] > result_baseline['sharpe'] else 'Baseline 우세'}\n")
    print(f"\n  결과 저장: {model_path}")

    # --- 누적 수익률 비교 차트 ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    cum_ret_baseline = np.cumprod(1 + result_baseline['val_returns'])
    cum_ret_midas = np.cumprod(1 + result_midas['val_returns'])

    axes[0].plot(cum_ret_baseline, label=f"Baseline (Sharpe: {result_baseline['sharpe']:.4f})",
                 color='coral', linewidth=2)
    axes[0].plot(cum_ret_midas, label=f"MIDAS (Sharpe: {result_midas['sharpe']:.4f})",
                 color='steelblue', linewidth=2)
    axes[0].set_xlabel('Validation Period (months)')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].set_title('GRU 모델: 누적 수익률 비교 (Validation Set)')
    axes[0].legend(fontsize=12)
    axes[0].axhline(y=1.0, color='black', linestyle='--', alpha=0.3)

    # 월별 수익률 비교 (bar chart)
    n_months = min(len(result_baseline['val_returns']), len(result_midas['val_returns']))
    x = np.arange(n_months)
    width = 0.35

    axes[1].bar(x - width/2, result_baseline['val_returns'][:n_months], width,
                label='Baseline', color='coral', alpha=0.7)
    axes[1].bar(x + width/2, result_midas['val_returns'][:n_months], width,
                label='MIDAS', color='steelblue', alpha=0.7)
    axes[1].set_xlabel('Validation Period (months)')
    axes[1].set_ylabel('Monthly Return')
    axes[1].set_title('GRU 모델: 월별 수익률 비교')
    axes[1].legend()
    axes[1].axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    model_chart_path = os.path.join(OUTPUT_DIR, 'gru_model_comparison.png')
    plt.savefig(model_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  차트 저장: {model_chart_path}")

    # =========================================================================
    # 최종 요약
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 1 Benchmark 전체 결과 요약")
    print("=" * 70)
    print(f"\n  MIDAS 설정:")
    print(f"    K = {K} (일간 래그)")
    print(f"    Polynomial Degree = {POLY_DEGREE}")
    print(f"    Rolling Window = {WINDOW}개월")
    print(f"    기간: {START_DATE} ~ {END_DATE}")
    print(f"\n  Feature 레벨 결과:")
    print(f"    1. 상관계수: {correlation:.4f} ({'✅ < 0.9' if correlation < 0.9 else '⚠️ >= 0.9'})")
    if hmm_available:
        print(f"    2. HMM LL 차이: {improvement:+.2f} ({'✅ MIDAS 우세' if improvement >= 0 else '⚠️ 월말 우세'})")
    print(f"\n  모델 레벨 결과:")
    print(f"    3. Sharpe: Baseline {result_baseline['sharpe']:.4f} → MIDAS {result_midas['sharpe']:.4f} ({sharpe_diff:+.4f})")
    print(f"    4. Annual Return: {result_baseline['annual_return']:.2%} → {result_midas['annual_return']:.2%}")
    print(f"    5. MDD: {result_baseline['mdd']:.2%} → {result_midas['mdd']:.2%}")
    print(f"\n  산출물:")
    print(f"    {corr_path}")
    if hmm_available:
        print(f"    {ll_path}")
    print(f"    {weight_path}")
    print(f"    {feat_path}")
    print(f"    {scatter_path}")
    print(f"    {model_path}")
    print(f"    {model_chart_path}")
    print(f"\n" + "=" * 70)
    print(f"Phase 1 Benchmark 완료! ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()
