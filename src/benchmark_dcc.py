"""
DCC-GARCH Benchmark — Dynamic Conditional Correlation Portfolio
================================================================
GARCH(1,1) 기반 개별 변동성 + EWMA 동적 상관계수 결합 모형.

Python arch 패키지의 한계:
  - arch는 단변량 GARCH만 지원, 다변량 DCC를 네이티브로 지원하지 않음
  - R의 rmgarch를 rpy2로 호출하는 방법은 서버 환경에서 불안정

현실적 대안 (본 구현):
  1. 각 자산별 GARCH(1,1)로 시변 변동성(σ_t) 추정
  2. EWMA(λ=0.94, RiskMetrics) 기반 동적 상관계수 추정
  3. H_t = D_t × R_t × D_t (동적 공분산 행렬)
  4. 최소 분산 포트폴리오(Min-Variance MVO) 비중 산출 → OOS 수익률

References:
  - Engle, R. (2002). Dynamic Conditional Correlation. JBES.
  - J.P. Morgan (1996). RiskMetrics Technical Document.

Usage:
    from src.benchmark_dcc import run_dcc_benchmark
    oos_returns = run_dcc_benchmark(config)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("[WARNING] arch 패키지 미설치. pip install arch")


# =============================================================================
# GARCH(1,1) 개별 변동성 추정
# =============================================================================

def _fit_garch_vol(series: pd.Series) -> np.ndarray:
    """
    단일 자산의 GARCH(1,1) 조건부 변동성을 추정합니다.
    
    Parameters:
        series: 수익률 시계열 (pd.Series)
    
    Returns:
        conditional_vol: 조건부 변동성 배열 (annualized)
    """
    if not ARCH_AVAILABLE:
        # fallback: 12-month rolling std
        return series.rolling(12, min_periods=6).std().fillna(series.std()).values
    
    try:
        # arch_model은 퍼센트 단위 수익률을 기대함
        am = arch_model(series * 100, vol='Garch', p=1, q=1, 
                        mean='Zero', dist='Normal')
        res = am.fit(disp='off', show_warning=False)
        # arch의 conditional_volatility는 이미 표준편차(σ) 단위 (% 스케일)
        # → 소수 단위로 변환만 하면 됨 (sqrt 불필요)
        cond_vol = res.conditional_volatility.values / 100.0
        return cond_vol
    except Exception:
        # GARCH 수렴 실패 시 rolling std fallback
        return series.rolling(12, min_periods=6).std().fillna(series.std()).values


# =============================================================================
# EWMA 동적 상관계수 (RiskMetrics 방법론)
# =============================================================================

def _ewma_corr(returns: np.ndarray, decay: float = 0.94) -> np.ndarray:
    """
    RiskMetrics EWMA 방식으로 시변 상관행렬을 추정합니다.
    
    Parameters:
        returns: (T, N) 표준화된 잔차 행렬
        decay: EWMA 감쇠 계수 (λ), RiskMetrics 기본값 0.94
    
    Returns:
        ewma_cov: (T, N, N) 시변 공분산 행렬
    """
    T, N = returns.shape
    ewma_cov = np.zeros((T, N, N))
    
    # 초기값: 처음 12개월의 공분산
    init_window = min(12, T)
    S = np.cov(returns[:init_window], rowvar=False)
    if np.isnan(S).any():
        S = np.eye(N) * 0.01
    ewma_cov[0] = S
    
    for t in range(1, T):
        r = returns[t:t+1].T  # (N, 1)
        S = decay * S + (1 - decay) * (r @ r.T)
        ewma_cov[t] = S
    
    return ewma_cov


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """공분산 행렬을 상관행렬로 변환합니다."""
    vols = np.sqrt(np.diag(cov)).clip(min=1e-10)
    D_inv = np.diag(1.0 / vols)
    corr = D_inv @ cov @ D_inv
    # 대각선을 정확히 1로 보정
    np.fill_diagonal(corr, 1.0)
    return corr


# =============================================================================
# Min-Variance MVO (최소 분산 포트폴리오)
# =============================================================================

def _min_var_weights(cov: np.ndarray, max_weight: float = 0.30) -> np.ndarray:
    """
    최소 분산 포트폴리오 비중을 구합니다 (long-only, 비중 상한 cap).
    
    Parameters:
        cov: (N, N) 공분산 행렬
        max_weight: 단일 자산 최대 비중 (default: 30%)
    
    Returns:
        weights: (N,) 최적 포트폴리오 비중
    """
    N = cov.shape[0]
    
    # 정규화(regularize)
    cov_reg = cov + np.eye(N) * 1e-6
    
    def port_var(w):
        return w @ cov_reg @ w
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight)] * N
    x0 = np.ones(N) / N
    
    try:
        result = minimize(port_var, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 300})
        if result.success:
            w = result.x
            w = np.maximum(w, 0.0)
            return w / w.sum()
    except Exception:
        pass
    
    # fallback: equal weight
    return np.ones(N) / N


# =============================================================================
# Walk-Forward DCC-GARCH Benchmark
# =============================================================================

def run_dcc_benchmark(config: dict) -> pd.Series:
    """
    DCC-GARCH 기반 동적 공분산 포트폴리오 벤치마크 (Walk-Forward).
    
    AI 모델과 동일한 OOS 기간에서 GARCH(1,1) + EWMA 상관계수로
    동적 공분산을 추정하고 Min-Variance MVO로 투자 비중을 산출합니다.
    
    Parameters:
        config: run_walkforward.py의 CONFIG 딕셔너리
    
    Returns:
        pd.Series: OOS 월간 포트폴리오 수익률 (index=날짜)
    """
    from src.data_loader import get_monthly_asset_data, ASSET_TICKERS, prepare_training_data
    from run_walkforward import define_folds
    
    # 1. 데이터 로드
    _, asset_returns_df = get_monthly_asset_data(
        ASSET_TICKERS, config['start_date'], config['end_date'])
    
    X, y, vix, scaler, asset_names, y_dates, _ = prepare_training_data(
        start_date=config['start_date'],
        end_date=config['end_date'],
        seq_length=config['seq_length'],
        normalize=True,
        use_momentum=False,  # 벤치마크는 momentum feature 불필요
    )
    
    y_raw = asset_returns_df.reindex(y_dates).values
    y_raw = np.nan_to_num(y_raw, nan=0.0)
    
    n_samples = min(len(X), len(y_raw))
    y_raw = y_raw[:n_samples]
    dates = y_dates[:n_samples]
    
    # 2. Walk-Forward Folds
    folds = define_folds(dates, config['seq_length'])
    
    all_oos_returns = []
    all_oos_dates = []
    
    for fold_idx, fold in enumerate(folds):
        train_idx = fold['train_idx']
        test_idx = fold['test_idx']
        
        # 훈련 구간 수익률
        train_returns = y_raw[train_idx]  # (T_train, N_assets)
        test_returns = y_raw[test_idx]    # (T_test, N_assets)
        
        n_assets = train_returns.shape[1]
        
        # 3. GARCH(1,1) 개별 변동성 추정 (훈련 구간)
        garch_vols = np.zeros_like(train_returns)
        for j in range(n_assets):
            asset_series = pd.Series(train_returns[:, j])
            garch_vols[:, j] = _fit_garch_vol(asset_series)
        
        # 표준화 잔차: ε_t = r_t / σ_t
        garch_vols_safe = np.maximum(garch_vols, 1e-8)
        std_resids = train_returns / garch_vols_safe
        
        # 4. EWMA 동적 상관계수 추정
        ewma_covs = _ewma_corr(std_resids, decay=0.94)
        
        # 마지막 시점의 상관행렬을 OOS에 적용
        last_cov = ewma_covs[-1]
        last_corr = _cov_to_corr(last_cov)
        
        # 마지막 시점의 GARCH 변동성
        last_vols = garch_vols[-1]
        
        # 5. 동적 공분산 행렬 = D × R × D
        D = np.diag(last_vols)
        H = D @ last_corr @ D
        
        # 6. Min-Variance MVO 비중 산출
        weights = _min_var_weights(H, max_weight=0.30)
        
        # 7. OOS 수익률 계산
        fold_oos_returns = (test_returns * weights).sum(axis=1)
        all_oos_returns.extend(fold_oos_returns.tolist())
        all_oos_dates.extend(dates[test_idx].tolist())
    
    # 8. 최종 Series 반환
    oos_series = pd.Series(
        all_oos_returns, 
        index=pd.DatetimeIndex(all_oos_dates, name='date'),
        name='DCC-GARCH MVO'
    )
    
    print(f"[DCC-GARCH] OOS {len(oos_series)} months, "
          f"Sharpe={oos_series.mean()/oos_series.std()*np.sqrt(12):.4f}")
    
    return oos_series
