"""
Fama-French 5-Factor BL Prior Benchmark (Phase 7 STEP 6 / run_compare 통합)
===========================================================================

현재 BL Prior(Rolling Mean) vs FF 5-Factor 이론 기반 Prior 성능 비교.

FF 5개 팩터 (Kenneth French 라이브러리에서 무료 다운로드):
  Mkt-RF: 시장 프리미엄  SMB: 소형-대형  HML: 가치-성장
  RMW: 고수익성-저수익성  CMA: 보수적-공격적 투자

FF Prior 계산:
  1. 각 자산 i의 팩터 민감도 β_i = [β_Mkt, β_SMB, β_HML, β_RMW, β_CMA]
     (롤링 OLS, window=60개월)
  2. 팩터 프리미엄 λ = 장기 평균 팩터 수익률
  3. π_i(FF) = Σ_k (β_ik × λ_k)

Usage:
    python -m src.benchmark_ff               # 독립 실행
    python run_compare.py --label v4          # run_compare에서 자동 호출
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Kenneth French 데이터 다운로드
try:
    import pandas_datareader.data as web
    PDR_AVAILABLE = True
except ImportError:
    PDR_AVAILABLE = False
    print("[WARNING] pandas_datareader 미설치. pip install pandas_datareader")


# ─── Constants ───────────────────────────────────────────────────────────────

FF_DATASET = 'F-F_Research_Data_5_Factors_2x3'
FACTOR_COLS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
BETA_WINDOW = 60  # 롤링 OLS 윈도우 (60개월 = 5년)


# ─── FF Factor Download ─────────────────────────────────────────────────────

def download_ff5_factors(start='2005-01-01', end='2025-01-01',
                         cache_path='data/cache/ff5_factors.csv') -> pd.DataFrame:
    """
    FF 5팩터 월간 수익률 다운로드 (캐시 지원).
    
    Returns:
        DataFrame: columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
                   index = DatetimeIndex (월말)
                   단위 = 소수 (예: 0.01 = 1%)
    """
    # 1. 캐시 확인
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if len(df) > 100:  # 최소 데이터 수 확인
            print(f"[FF] 캐시 로드: {cache_path} ({len(df)} months)")
            return df

    if not PDR_AVAILABLE:
        raise ImportError("pandas_datareader 필요. pip install pandas_datareader")

    # 2. Kenneth French 서버에서 다운로드
    print(f"[FF] Kenneth French 서버에서 FF 5-Factor 데이터 다운로드 중...")
    raw = web.DataReader(FF_DATASET, 'famafrench', start, end)
    ff = raw[0] / 100.0  # 퍼센트 → 소수
    ff.index = ff.index.to_timestamp(how='end').normalize()  # PeriodIndex → Month-End DatetimeIndex
    
    # 3. 캐시 저장
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    ff.to_csv(cache_path)
    print(f"[FF] 다운로드 완료: {len(ff)} months ({ff.index[0].date()} ~ {ff.index[-1].date()})")
    
    return ff


# ─── Rolling Beta Estimation ────────────────────────────────────────────────

def rolling_ff_betas(asset_rets: pd.Series, ff_factors: pd.DataFrame,
                     window: int = BETA_WINDOW) -> pd.DataFrame:
    """
    단일 자산에 대한 FF 5팩터 노출도(β) 롤링 OLS 추정.
    
    Args:
        asset_rets: 자산의 월간 초과수익률 (총수익률 - RF)
        ff_factors: FF 5팩터 수익률 DataFrame
        window: 롤링 윈도우 크기 (기본 60개월)
        
    Returns:
        DataFrame: (T-window, 5) 크기. 각 행은 해당 시점의 [β_Mkt, β_SMB, ...]
    """
    # 날짜 정렬
    aligned = pd.concat([asset_rets.rename('excess_ret'),
                         ff_factors[FACTOR_COLS]], axis=1).dropna()
    
    betas_list = []
    dates_list = []
    
    for t in range(window, len(aligned)):
        X = aligned.iloc[t - window:t][FACTOR_COLS].values
        y = aligned.iloc[t - window:t]['excess_ret'].values
        
        reg = LinearRegression(fit_intercept=True).fit(X, y)
        betas_list.append(reg.coef_)
        dates_list.append(aligned.index[t])
    
    return pd.DataFrame(betas_list, index=dates_list, columns=FACTOR_COLS)


# ─── FF Prior Calculation ────────────────────────────────────────────────────

def compute_ff_prior(asset_returns_df: pd.DataFrame, ff_factors: pd.DataFrame,
                     window: int = BETA_WINDOW) -> pd.DataFrame:
    """
    모든 자산에 대한 FF 5-Factor 기반 기대수익률(Prior) 계산.
    
    π_i(FF) = Σ_k (β_ik × λ_k)
    
    Args:
        asset_returns_df: (T, N_assets) 월간 수익률 DataFrame
        ff_factors: FF 5팩터 수익률 DataFrame
        window: 롤링 OLS 윈도우
        
    Returns:
        DataFrame: (T-window, N_assets) FF Prior 기대수익률
    """
    rf = ff_factors.get('RF', pd.Series(0, index=ff_factors.index))
    
    # 장기 평균 팩터 프리미엄 (전체 기간)
    lambda_ff = ff_factors[FACTOR_COLS].mean()
    
    all_priors = {}
    
    for col in asset_returns_df.columns:
        # 초과수익률 = 총수익률 - RF
        aligned_rf = rf.reindex(asset_returns_df.index).fillna(0)
        excess_ret = asset_returns_df[col] - aligned_rf
        
        # 롤링 베타 추정
        betas = rolling_ff_betas(excess_ret, ff_factors, window)
        
        # FF Prior = β × λ
        prior = betas.dot(lambda_ff)
        all_priors[col] = prior
    
    return pd.DataFrame(all_priors)


# ─── FF Prior Walk-Forward Benchmark ─────────────────────────────────────────

def run_ff_prior_benchmark(config: dict) -> pd.Series:
    """
    FF 5-Factor Prior 기반 포트폴리오 Walk-forward 벤치마크.
    
    전략: 매월 FF Prior(기대수익률)를 기반으로 단순 MVO 포트폴리오 구성.
    - FF Prior가 높은 자산에 더 많은 비중 배분 (Softmax 방식)
    - 동일한 OOS 기간에서 AI 모델과 비교 가능.
    
    Args:
        config: run_walkforward.py의 CONFIG 딕셔너리
        
    Returns:
        pd.Series: OOS 월간 포트폴리오 수익률
    """
    from src.data_loader import prepare_training_data, get_monthly_asset_data, ASSET_TICKERS
    
    # 데이터 로드 (run_baselines와 동일 구조)
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
    
    # FF 5팩터 다운로드
    ff_factors = download_ff5_factors(
        start=config['start_date'], end=config['end_date'])
    
    # FF Prior 계산 (모든 자산)
    ff_prior_df = compute_ff_prior(asset_returns_df, ff_factors)
    
    # Walk-Forward Folds (run_baselines와 동일)
    import sys
    sys.path.insert(0, '.')
    from run_walkforward import define_folds
    
    folds = define_folds(dates, config['seq_length'])
    
    all_test_returns = []
    all_test_dates = []
    
    for fold in folds:
        test_idx = fold['test_idx']
        
        for idx in test_idx:
            test_date = dates[idx]
            test_ret = y_raw[idx]
            
            # 해당 시점의 FF Prior가 있는지 확인
            if test_date in ff_prior_df.index:
                prior = ff_prior_df.loc[test_date].values
            else:
                # 가장 가까운 이전 날짜의 Prior 사용
                valid_dates = ff_prior_df.index[ff_prior_df.index <= test_date]
                if len(valid_dates) > 0:
                    prior = ff_prior_df.loc[valid_dates[-1]].values
                else:
                    # Prior 없으면 Equal Weight
                    prior = np.ones(len(ASSET_TICKERS)) / len(ASSET_TICKERS)
            
            # FF Prior 기반 비중: Softmax(prior / temperature)
            # temperature = 0.02로 약간의 집중도 적용
            temperature = 0.02
            exp_prior = np.exp(prior / temperature)
            weights = exp_prior / (exp_prior.sum() + 1e-10)
            
            # 비중 클리핑 (극단적 집중 방지, 최대 30%)
            weights = np.clip(weights, 0.02, 0.30)
            weights = weights / weights.sum()
            
            port_ret = (weights * test_ret).sum()
            all_test_returns.append(port_ret)
            all_test_dates.append(test_date)
    
    return pd.Series(all_test_returns,
                     index=pd.DatetimeIndex(all_test_dates, name='date'),
                     name='FF-5Factor Prior')


# ─── Standalone Execution ────────────────────────────────────────────────────

if __name__ == '__main__':
    """독립 실행: FF Prior 벤치마크 결과를 단독으로 출력."""
    import sys
    sys.path.insert(0, '.')
    
    from run_walkforward import CONFIG
    
    print("=" * 60)
    print("  Fama-French 5-Factor BL Prior Benchmark")
    print("=" * 60)
    
    rets = run_ff_prior_benchmark(CONFIG)
    
    ann_ret = rets.mean() * 12
    ann_vol = rets.std() * np.sqrt(12)
    sharpe = ann_ret / (ann_vol + 1e-9)
    cum = (1 + rets).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    
    print(f"\n  FF-5Factor Prior Portfolio:")
    print(f"    Sharpe:        {sharpe:.4f}")
    print(f"    Annual Return: {ann_ret:.2%}")
    print(f"    Annual Vol:    {ann_vol:.2%}")
    print(f"    MDD:           {mdd:.2%}")
    print(f"    Test months:   {len(rets)}")
    
    # 결과 저장
    os.makedirs('results/walkforward', exist_ok=True)
    rets.to_csv('results/walkforward/ff_prior_returns.csv')
    print(f"\n  [SAVED] results/walkforward/ff_prior_returns.csv")
