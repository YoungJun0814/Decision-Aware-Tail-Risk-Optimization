"""
Data Loader Module
==================
[Step 1] 데이터 수집 및 전처리

- Universe: 10개 자산 (자산 클래스 분산)
  - 공격: SPY (S&P500), QQQ (나스닥100)
  - 방어: XLV (헬스케어), XLP (필수소비재)
  - 순환: XLE (에너지)
  - 채권: TLT (장기국채), IEF (중기국채)
  - 대안: GLD (금), VNQ (리츠)
  - 현금: BIL (초단기채, 위기 시 대피처)
- Frequency: 일간 데이터를 받아 월말(Month-end) 기준으로 리샘플링
- Features: 자산 수익률 + 거시변수(Macro, 추후 확장)
- VIX: 손실 함수(Loss Function)의 동적 거래비용 계산을 위해 별도로 수집
- MIDAS: 일간 데이터를 Almon Polynomial로 최적 가중하여 월간 Feature 생성 (Phase 1)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from pathlib import Path
import os

try:
    from src.midas import MIDASFeatureExtractor, download_daily_vix, download_daily_spy_returns
    MIDAS_AVAILABLE = True
except ImportError:
    MIDAS_AVAILABLE = False

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


# =============================================================================
# 설정 (Configuration)
# =============================================================================

# 자산 유니버스 (10개) — BIL은 반드시 마지막 (CVaR 안전장치 인덱스)
ASSET_TICKERS = ['SPY', 'QQQ', 'XLV', 'XLP', 'XLE', 'TLT', 'IEF', 'GLD', 'VNQ', 'BIL']

# VIX 티커 (Loss 함수에서 시장 상황 반영을 위해 사용)
VIX_TICKER = '^VIX'

# 거시경제 지표 (Strategist 선정 시 추가 예정)
MACRO_TICKERS = []

# FRED API Key — 환경변수로 설정 필요 (절대 코드에 직접 입력하지 마세요)
# 설정 방법: $env:FRED_API_KEY = 'your_key_here'  (Windows PowerShell)
#            export FRED_API_KEY='your_key_here'    (Linux/Mac)
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')

# Regime Head 전용 매크로 피처 (FRED Series ID)
MACRO_REGIME_FEATURES = {
    'T10Y3M': 'Term Spread (10Y - 3M)',     # 장단기 금리차 → 경기침체 선행지표
    'BAA10Y': 'Credit Spread (BAA - 10Y)',   # 신용 스프레드 → 위험 선호 지표
}

# Regime 확률 데이터 경로
REGIME_PROB_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'prob_data.csv'
REGIME_4STATE_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'regime_4state.csv'


# =============================================================================
# 데이터 수집 함수 (Data Collection)
# =============================================================================

def get_monthly_asset_data(tickers: list, start_date: str, end_date: str) -> tuple:
    """
    일간 데이터를 다운로드하여 월말(Month-End) 수익률로 변환합니다.
    
    Args:
        tickers: 종목 코드 리스트
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
    
    Returns:
        monthly_prices: 월말 가격 데이터 (DataFrame)
        monthly_returns: 월간 수익률 데이터 (DataFrame)
    """
    print(f"Downloading data for: {tickers}")
    
    # 1. 일간 데이터 다운로드 (수정 주가 기준)
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
    
    # 2. 월말(Month End) 기준으로 리샘플링
    monthly_prices = data.resample('ME').last()
    
    # 3. 월간 수익률 계산 (변화율)
    monthly_returns = monthly_prices.pct_change().dropna()
    
    return monthly_prices, monthly_returns


def get_macro_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    거시경제 변수 데이터를 수집합니다. (현재는 비어있음)
    """
    if not tickers:
        print("[INFO] MACRO_TICKERS가 비어있어 거시변수 수집을 건너뜁니다.")
        return pd.DataFrame()
    
    print(f"Downloading macro data for: {tickers}")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
    monthly_macro = data.resample('ME').last().pct_change().dropna()
    
    return monthly_macro


def get_macro_regime_features(start_date: str, end_date: str) -> pd.DataFrame:
    """
    FRED에서 RegimeHead 전용 매크로 피처를 수집합니다.
    
    - T10Y3M: 장단기 금리차 (10년 국채 - 3개월 T-bill)
      → 역전(음수) 시 경기침체 선행. 레벨 값 직접 사용.
    - BAA10Y: 신용 스프레드 (BAA 회사채 - 10년 국채)
      → 확대 시 신용위험 증가. 레벨 값 직접 사용.
    
    Returns:
        DataFrame (N_months, 2) columns=['T10Y3M', 'BAA10Y']
        인덱스: 월말(ME) DatetimeIndex
    """
    if not FRED_AVAILABLE:
        print("[WARNING] fredapi가 설치되어 있지 않습니다. pip install fredapi")
        return pd.DataFrame()
    
    if not FRED_API_KEY:
        print("[WARNING] FRED_API_KEY가 설정되지 않았습니다.")
        print("  환경변수 FRED_API_KEY를 설정하거나,")
        print("  src/data_loader.py의 FRED_API_KEY 변수에 직접 입력하세요.")
        return pd.DataFrame()
    
    print("[INFO] FRED에서 매크로 Regime 피처 수집 중...")
    fred = Fred(api_key=FRED_API_KEY)
    
    macro_dict = {}
    for series_id, desc in MACRO_REGIME_FEATURES.items():
        try:
            series = fred.get_series(series_id, start_date, end_date)
            # 월말 리샘플 (FRED 데이터는 일간 → 월말 마지막 값)
            monthly = series.resample('ME').last().dropna()
            macro_dict[series_id] = monthly
            print(f"  {series_id} ({desc}): {len(monthly)} months")
        except Exception as e:
            print(f"  [ERROR] {series_id} 수집 실패: {e}")
    
    if not macro_dict:
        return pd.DataFrame()
    
    macro_df = pd.DataFrame(macro_dict)
    macro_df.index.name = 'Date'
    
    # 결측치 처리: forward fill → backward fill
    macro_df = macro_df.ffill().bfill()
    
    print(f"[INFO] 매크로 Regime 피처: {macro_df.shape}, "
          f"기간 {macro_df.index[0].date()} ~ {macro_df.index[-1].date()}")
    
    return macro_df


def get_vix_data(start_date: str, end_date: str) -> pd.Series:
    """
    VIX(변동성 지수) 데이터를 수집합니다.
    Loss 함수에서 동적 거래비용($\\kappa(VIX)$)을 계산할 때 사용됩니다.
    VIX는 수익률이 아닌 '지수 레벨' 자체를 사용합니다.
    """
    print(f"[INFO] Loss 함수용 VIX 데이터를 다운로드합니다...")
    
    try:
        vix_data = yf.download(VIX_TICKER, start=start_date, end=end_date, progress=False)['Close']
        monthly_vix = vix_data.resample('ME').last()
        monthly_vix = monthly_vix.ffill().bfill()
        print(f"[INFO] VIX 데이터 범위: {monthly_vix.index[0]} ~ {monthly_vix.index[-1]}")
        return monthly_vix
    except Exception as e:
        print(f"[WARNING] VIX 다운로드 실패: {e}. 기본값 20을 사용합니다.")
        return pd.Series()


def generate_midas_features(
    start_date: str,
    end_date: str,
    K: int = 22,
    poly_degree: int = 2,
    window: int = 60,
) -> pd.Series:
    """
    MIDAS Feature를 생성합니다.
    일간 VIX 데이터를 Almon Polynomial로 최적 가중하여 월간 Feature로 변환합니다.

    Parameters
    ----------
    start_date : str
        시작일 (YYYY-MM-DD)
    end_date : str
        종료일 (YYYY-MM-DD)
    K : int
        일간 래그 수 (default: 22, 1거래월)
    poly_degree : int
        Almon 다항식 차수 (default: 2)
    window : int
        OLS 롤링 윈도우 크기 (월 단위, default: 60 = 5년)

    Returns
    -------
    pd.Series
        월간 MIDAS VIX Feature
    """
    print(f"[INFO] MIDAS Feature 생성 시작 (K={K}, poly_degree={poly_degree}, window={window})")

    # 일간 데이터 다운로드
    daily_vix = download_daily_vix(start_date, end_date)
    monthly_spy_returns = download_daily_spy_returns(start_date, end_date)

    # MIDAS 추출기 생성 및 실행
    extractor = MIDASFeatureExtractor(K=K, poly_degree=poly_degree)
    midas_features = extractor.fit_transform(daily_vix, monthly_spy_returns, window=window)

    print(f"[INFO] MIDAS Feature 생성 완료: {len(midas_features)}개월")
    return midas_features


def get_regime_proba() -> pd.DataFrame:
    """
    Regime 확률 데이터를 로드합니다.
    
    Returns:
        DataFrame with Prob_Bull, Prob_Uncertain, Prob_Crisis columns
    """
    if not REGIME_PROB_PATH.exists():
        print(f"[WARNING] Regime 데이터가 없습니다: {REGIME_PROB_PATH}")
        print("[INFO] 'python -m src.gen_regime' 명령으로 생성하세요.")
        return pd.DataFrame()
    
    print(f"[INFO] Regime 확률 데이터 로드 중...")
    regime_df = pd.read_csv(REGIME_PROB_PATH, parse_dates=['Date'], index_col='Date')
    
    # 필요한 컬럼만 선택
    prob_cols = ['Prob_Bull', 'Prob_Uncertain', 'Prob_Crisis']
    regime_proba = regime_df[prob_cols]
    
    print(f"[INFO] Regime 데이터 기간: {regime_proba.index[0].date()} ~ {regime_proba.index[-1].date()}")
    return regime_proba


def get_regime_4state() -> pd.DataFrame:
    """
    4-state Hierarchical HMM Regime 확률 로드.
    
    Columns: Prob_Bull, Prob_Sideways, Prob_Correction, Prob_Crisis
    생성: python -m src.gen_regime_4state
    """
    if not REGIME_4STATE_PATH.exists():
        print(f"[WARNING] 4-state Regime 데이터가 없습니다: {REGIME_4STATE_PATH}")
        print("[INFO] 'python -m src.gen_regime_4state' 명령으로 생성하세요.")
        return pd.DataFrame()
    
    print(f"[INFO] 4-state Regime 확률 로드 중...")
    df = pd.read_csv(REGIME_4STATE_PATH, parse_dates=['Date'], index_col='Date')
    
    prob_cols = ['Prob_Bull', 'Prob_Sideways', 'Prob_Correction', 'Prob_Crisis']
    result = df[prob_cols]
    
    print(f"[INFO] 4-state Regime 기간: {result.index[0].date()} ~ {result.index[-1].date()}")
    return result


# =============================================================================
# 데이터 전처리 함수 (Preprocessing)
# =============================================================================

def normalize_data(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
    """
    데이터를 표준화(StandardScaler)합니다.
    
    Data Leakage 방지: train 구간만으로 scaler를 fit하고,
    동일한 scaler로 전체 데이터를 transform합니다.
    
    Args:
        df: 정규화할 데이터프레임
        train_ratio: 학습 데이터 비율 (기본 0.8)
    """
    scaler = StandardScaler()
    n_train = int(len(df) * train_ratio)
    scaler.fit(df.values[:n_train])               # Train만으로 fit
    normalized_values = scaler.transform(df.values) # 전체를 transform
    normalized_df = pd.DataFrame(
        normalized_values, 
        index=df.index, 
        columns=df.columns
    )
    print(f"[INFO] Scaler fit on train({n_train}rows), transform on all({len(df)}rows) - No data leakage")
    return normalized_df, scaler


def create_sequences(data: np.ndarray, seq_length: int) -> tuple:
    """
    시계열 데이터를 LSTM/TFT 입력 형태인 (Batch, Sequence, Features)로 변환합니다.
    과거 12개월(t-12 ~ t-1) 데이터를 보고 다음 달(t)을 예측하도록 구성합니다.
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    
    return np.array(X), np.array(y)


def prepare_training_data(
    start_date: str = '2007-07-01',
    end_date: str = '2024-01-01',
    seq_length: int = 12,
    normalize: bool = True,
    train_ratio: float = 0.8,
    use_midas: bool = False,
    use_momentum: bool = False,
    use_macro_regime: bool = False,
    midas_K: int = 22,
    midas_poly_degree: int = 2,
    midas_window: int = 60,
) -> tuple:
    """
    전체 데이터 파이프라인: 수집 -> 병합 -> 전처리 -> 텐서 변환

    Parameters:
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        seq_length: 시퀀스 길이 (default: 12개월)
        normalize: 정규화 여부
        train_ratio: 학습 데이터 비율
        use_midas: MIDAS Feature 사용 여부 (default: False)
        midas_K: MIDAS 일간 래그 수 (default: 22)
        midas_poly_degree: Almon 다항식 차수 (default: 2)
        midas_window: MIDAS OLS 롤링 윈도우 (월 단위, default: 60)

    Returns:
        X_tensor: (Batch, Seq, Features) 입력 데이터
        y_tensor: (Batch, Num_assets) 정답 데이터 (다음 달 자산 수익률)
        vix_tensor: (Batch,) VIX 데이터
        scaler: 정규화 스케일러
        asset_names: 자산 이름 리스트
    """
    # 1. 자산 데이터 수집
    _, asset_returns = get_monthly_asset_data(ASSET_TICKERS, start_date, end_date)
    
    # 2. VIX 데이터 수집
    vix_data = get_vix_data(start_date, end_date)
    
    # 3. 거시변수 데이터 수집
    macro_returns = get_macro_data(MACRO_TICKERS, start_date, end_date)
    
    # 4. Regime 확률 데이터 수집
    regime_proba = get_regime_proba()
    
    # 5. 데이터 병합
    if not macro_returns.empty:
        combined_data = pd.concat([asset_returns, macro_returns], axis=1).dropna()
    else:
        combined_data = asset_returns.copy()
    
    # 6. Regime 확률 병합 (있을 경우)
    if not regime_proba.empty:
        combined_data = combined_data.join(regime_proba, how='left')
        # 결측치 처리: 데이터가 없는 초기 구간은 중립값으로
        combined_data['Prob_Bull'] = combined_data['Prob_Bull'].fillna(0.33)
        combined_data['Prob_Uncertain'] = combined_data['Prob_Uncertain'].fillna(0.34)
        combined_data['Prob_Crisis'] = combined_data['Prob_Crisis'].fillna(0.33)
        print(f"[INFO] Regime 확률 Feature 추가 완료 (3개 컬럼)")

    # 6b. Momentum Features (Phase 2A)
    if use_momentum:
        n_mom_added = 0
        mom_cols = []
        for ticker in ASSET_TICKERS:
            if ticker in combined_data.columns:
                mom_col = f'{ticker}_MOM12'
                combined_data[mom_col] = (
                    (1 + combined_data[ticker]).rolling(12).apply(
                        lambda x: x.prod(), raw=True) - 1
                )
                combined_data[mom_col] = combined_data[mom_col].fillna(0.0)
                mom_cols.append(mom_col)
                n_mom_added += 1
        print(f"[INFO] Momentum Feature 추가 완료 ({n_mom_added}개 컬럼)")

        # 6c. Cross-Sectional Momentum (Proposal B)
        # 동일 시점에서 10개 자산 간 상대 순위/강도 비교 → look-ahead 없음
        if len(mom_cols) > 1:
            mom_matrix = combined_data[mom_cols]  # (T, 10)
            # Percentile Rank (0~1): 상위 자산일수록 1에 가까움
            cs_rank = mom_matrix.rank(axis=1, pct=True)
            cs_rank.columns = [c.replace('_MOM12', '_CS_RANK') for c in mom_cols]
            # Cross-Sectional Z-score: 해당 시점 평균 대비 표준화
            cs_mean = mom_matrix.mean(axis=1)
            cs_std  = mom_matrix.std(axis=1).clip(lower=1e-8)
            cs_zscore = mom_matrix.subtract(cs_mean, axis=0).divide(cs_std, axis=0)
            cs_zscore.columns = [c.replace('_MOM12', '_CS_Z') for c in mom_cols]
            combined_data = pd.concat([combined_data, cs_rank, cs_zscore], axis=1)
            print(f"[INFO] Cross-Sectional Momentum 추가 완료 ({len(mom_cols)*2}개 컬럼: rank + zscore)")


    # 7. MIDAS Feature 병합 (use_midas=True인 경우)
    if use_midas:
        if not MIDAS_AVAILABLE:
            print("[WARNING] src.midas 모듈을 import할 수 없습니다. MIDAS Feature를 건너뜁니다.")
            print("[INFO] 프로젝트 루트에서 실행하세요: python -m src.data_loader")
        else:
            print(f"\n[INFO] MIDAS Feature를 생성하여 병합합니다...")
            midas_features = generate_midas_features(
                start_date, end_date, K=midas_K,
                poly_degree=midas_poly_degree, window=midas_window
            )
            # MIDAS Feature를 combined_data에 병합
            midas_df = midas_features.to_frame(name='MIDAS_VIX')
            combined_data = combined_data.join(midas_df, how='left')
            # 결측치 처리: MIDAS가 없는 초기 구간은 median으로 (정규화 후 평균에 가까움)
            combined_data['MIDAS_VIX'] = combined_data['MIDAS_VIX'].fillna(
                combined_data['MIDAS_VIX'].median() if combined_data['MIDAS_VIX'].notna().any() else 0.0
            )
            print(f"[INFO] MIDAS Feature 추가 완료 (MIDAS_VIX 컬럼)")
    
    print(f"\n[INFO] 병합된 데이터 크기: {combined_data.shape}")
    print(f"[INFO] 데이터 기간: {combined_data.index[0]} ~ {combined_data.index[-1]}")
    
    # 8. VIX 데이터 인덱스 동기화
    if not vix_data.empty:
        vix_aligned = vix_data.reindex(combined_data.index).ffill().bfill()
    else:
        print("[WARNING] VIX 데이터가 없어 기본값 20을 사용합니다.")
        vix_aligned = pd.Series(20.0, index=combined_data.index)
    
    # 9. 정규화 (Standard Scaling)
    scaler = None
    if normalize:
        combined_data, scaler = normalize_data(combined_data, train_ratio=train_ratio)
    
    # 10. 시퀀스 생성 (Windowing)
    data_values = combined_data.values
    X, y = create_sequences(data_values, seq_length)
    
    # 11. VIX 시퀀스 생성 (타겟 시점의 VIX 사용)
    vix_values = vix_aligned.values
    vix_seq = []
    for i in range(len(vix_values) - seq_length):
        vix_seq.append(vix_values[i + seq_length])
    vix_array = np.array(vix_seq)
    
    # 12. 타겟에서 거시변수는 제외하고 자산 수익률만 남김
    num_assets = len(ASSET_TICKERS)
    y_assets = y[:, :num_assets]
    
    # 13. PyTorch 텐서로 변환
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_assets, dtype=torch.float32)
    vix_tensor = torch.tensor(vix_array.flatten(), dtype=torch.float32)
    
    # 14. 날짜 데이터 (Target 기준)
    y_dates = combined_data.index[seq_length:]
    
    # 15. 매크로 Regime 피처 (RegimeHead 전용, GRU 입력에 섞지 않음)
    macro_tensor = None
    if use_macro_regime:
        macro_df = get_macro_regime_features(start_date, end_date)
        if not macro_df.empty:
            # y_dates(타겟 시점)에 맞춰 정렬
            macro_aligned = macro_df.reindex(y_dates).ffill().bfill()
            # StandardScaler 정규화 (train 구간 기준)
            n_train_macro = int(len(macro_aligned) * train_ratio)
            macro_scaler = StandardScaler()
            macro_scaler.fit(macro_aligned.values[:n_train_macro])
            macro_scaled = macro_scaler.transform(macro_aligned.values)
            macro_tensor = torch.tensor(macro_scaled, dtype=torch.float32)
            print(f"[INFO] macro_tensor shape: {macro_tensor.shape}")
        else:
            print("[WARNING] 매크로 데이터 수집 실패. macro_tensor=None")
    
    print(f"[INFO] X_tensor shape: {X_tensor.shape}")
    print(f"[INFO] y_tensor shape: {y_tensor.shape}")
    print(f"[INFO] vix_tensor shape: {vix_tensor.shape}")
    
    return X_tensor, y_tensor, vix_tensor, scaler, ASSET_TICKERS, y_dates, macro_tensor


# =============================================================================
# Phase 2 — Daily Data & prepare_phase2_data
# =============================================================================

def download_daily_realized_vol(start_date: str, end_date: str, window: int = 22) -> pd.Series:
    """
    일간 Realized Volatility (SPY 22일 롤링 표준편차) 다운로드.
    
    RV_t = std(r_{t-21}, ..., r_t) * sqrt(252)  (연율화)
    """
    import yfinance as yf
    
    print(f"[Phase2] Realized Vol 생성 중 (window={window})...")
    spy = yf.download('SPY', start=start_date, end=end_date, 
                       progress=False, auto_adjust=True)['Close']
    if isinstance(spy, pd.DataFrame):
        spy = spy.squeeze()
    if isinstance(spy.index, pd.MultiIndex):
        spy.index = spy.index.get_level_values(0)
    
    daily_ret = spy.pct_change().dropna()
    rv = daily_ret.rolling(window).std() * np.sqrt(252)
    rv = rv.dropna()
    rv.name = 'RealizedVol'
    
    print(f"[Phase2] RV: {len(rv)}일, {rv.index[0].date()} ~ {rv.index[-1].date()}")
    return rv


def download_daily_credit_spread(start_date: str, end_date: str) -> pd.Series:
    """
    일간 Credit Spread 프록시 (HYG - IEF daily return spread).
    
    HYG: 하이일드 채권 ETF, IEF: 중기 국채 ETF
    스프레드 = HYG 수익률 - IEF 수익률 → 신용위험 지표
    양수 = HY가 국채보다 하락 (스프레드 확대 = 위험 증가)
    """
    import yfinance as yf
    
    print(f"[Phase2] Credit Spread 프록시 생성 중 (HYG - IEF)...")
    data = yf.download(['HYG', 'IEF'], start=start_date, end=end_date,
                        progress=False, auto_adjust=True)['Close']
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    ret = data.pct_change().dropna()
    spread = -(ret['HYG'] - ret['IEF'])  # 음수 반전: 위기 시 양수
    spread = spread.rolling(5).mean().dropna()  # 5일 이동 평균 (노이즈 제거)
    spread.name = 'CreditSpread'
    
    print(f"[Phase2] CreditSpread: {len(spread)}일, "
          f"{spread.index[0].date()} ~ {spread.index[-1].date()}")
    return spread


def prepare_daily_windows(
    daily_series_list: list,
    monthly_dates: pd.DatetimeIndex,
    K: int = 66,
) -> np.ndarray:
    """
    일간 데이터를 월 기준 K일 윈도우로 분할.
    
    각 월말 시점에서 직전 K 거래일의 일간 데이터를 추출.
    
    Args:
        daily_series_list: [VIX, RV, CreditSpread] 각각 pd.Series
        monthly_dates: 월말 날짜 인덱스
        K: 래그 수 (default: 66)
        
    Returns:
        daily_windows: (N_months, K, n_vars) ndarray
    """
    n_vars = len(daily_series_list)
    
    # 모든 일간 데이터를 하나의 DataFrame으로 병합
    daily_df = pd.concat(daily_series_list, axis=1).ffill().bfill()
    
    windows = []
    for date in monthly_dates:
        # 해당 월말까지의 데이터 중 직전 K일
        mask = daily_df.index <= date
        available = daily_df.loc[mask].tail(K)
        
        if len(available) < K:
            # 데이터 부족 시 패딩 (첫 번째 값으로)
            pad = available.iloc[:1].values
            pad_count = K - len(available)
            padded = np.vstack([np.tile(pad, (pad_count, 1)), available.values])
            windows.append(padded)
        else:
            windows.append(available.values)
    
    return np.array(windows, dtype=np.float32)  # (N_months, K, n_vars)


def prepare_hmm_probs(
    regime_proba: pd.DataFrame,
    monthly_dates: pd.DatetimeIndex,
    smooth_alpha: float = 0.8,
) -> np.ndarray:
    """
    HMM regime 확률을 KL-friendly soft target으로 변환.
    
    원본 HMM은 near-degenerate (1.0/0.0) → KL 발산 위험.
    Label smoothing: hard [1,0,0] → soft [0.8, 0.1, 0.1]
    
    Args:
        regime_proba: DataFrame with Prob_Bull, Prob_Uncertain, Prob_Crisis
        monthly_dates: 대상 날짜
        smooth_alpha: smoothing 강도 (0.8 = 주 클래스 80%)
        
    Returns:
        hmm_probs: (N, 3) soft probability array
    """
    n_regimes = 3
    n_dates = len(monthly_dates)
    probs = np.ones((n_dates, n_regimes)) / n_regimes  # 기본: 균등
    
    if regime_proba.empty:
        print("[WARNING] Regime 데이터 없음, 균등 확률 사용")
        return probs.astype(np.float32)
    
    for i, date in enumerate(monthly_dates):
        # 가장 가까운 날짜 찾기
        idx = regime_proba.index.get_indexer([date], method='nearest')
        if idx[0] >= 0 and idx[0] < len(regime_proba):
            row = regime_proba.iloc[idx[0]]
            raw = np.array([row.get('Prob_Bull', 0.33),
                            row.get('Prob_Uncertain', 0.34),
                            row.get('Prob_Crisis', 0.33)])
            
            # Label smoothing: argmax에 alpha, 나머지에 (1-alpha)/(K-1)
            hard_idx = np.argmax(raw)
            smoothed = np.full(n_regimes, (1 - smooth_alpha) / (n_regimes - 1))
            smoothed[hard_idx] = smooth_alpha
            probs[i] = smoothed
    
    return probs.astype(np.float32)


def prepare_phase2_data(
    start_date: str = '2007-07-01',
    end_date: str = '2024-01-01',
    seq_length: int = 12,
    K_midas: int = 66,
    normalize: bool = True,
    train_ratio: float = 0.8,
) -> dict:
    """
    Phase 2 전체 데이터 파이프라인.
    
    Returns:
        dict with:
            x_monthly: (N, Seq, F) 월간 features
            x_daily: (N, Seq, K, 3) 일간 데이터 (VIX, RV, CreditSpread)
            y: (N, 10) 다음 달 자산 수익률
            vix: (N,) VIX 레벨
            hmm_probs: (N, 3) smoothed HMM regime 확률
            asset_names: list
            dates: DatetimeIndex
    """
    print("=" * 60)
    print("[Phase 2] Data Pipeline")
    print("=" * 60)
    
    # 1. 월간 자산 데이터 (Phase 1 재사용)
    _, asset_returns = get_monthly_asset_data(ASSET_TICKERS, start_date, end_date)
    
    # 2. VIX (월간 레벨 — Loss 함수용)
    vix_data = get_vix_data(start_date, end_date)
    
    # 3. Regime 확률 로드
    regime_proba = get_regime_proba()
    
    # 4. 일간 데이터 다운로드
    try:
        from src.midas import download_daily_vix
        daily_vix = download_daily_vix(start_date, end_date)
    except Exception:
        import yfinance as yf
        daily_vix = yf.download('^VIX', start=start_date, end=end_date,
                                 progress=False)['Close']
        if isinstance(daily_vix, pd.DataFrame):
            daily_vix = daily_vix.squeeze()
        daily_vix.name = 'VIX'
    
    daily_rv = download_daily_realized_vol(start_date, end_date)
    daily_cs = download_daily_credit_spread(start_date, end_date)
    
    # 5. 월간 features 구성
    combined = asset_returns.copy()
    if not regime_proba.empty:
        combined = combined.join(regime_proba[['Prob_Bull', 'Prob_Uncertain', 'Prob_Crisis']], how='left')
        combined['Prob_Bull'] = combined['Prob_Bull'].fillna(0.33)
        combined['Prob_Uncertain'] = combined['Prob_Uncertain'].fillna(0.34)
        combined['Prob_Crisis'] = combined['Prob_Crisis'].fillna(0.33)
    
    combined = combined.dropna()
    monthly_dates = combined.index
    
    # 5.5. 정규화 전에 raw 자산 수익률 저장 (평가용)
    num_assets = len(ASSET_TICKERS)
    raw_asset_returns = combined.iloc[:, :num_assets].values.copy()
    
    # 6. VIX 동기화
    if not vix_data.empty:
        vix_aligned = vix_data.reindex(monthly_dates).ffill().bfill()
    else:
        vix_aligned = pd.Series(20.0, index=monthly_dates)
    
    # 7. 정규화 (월간 features만)
    scaler = None
    if normalize:
        combined, scaler = normalize_data(combined, train_ratio=train_ratio)
    
    # 8. 일간 데이터 윈도우 구성 (K일 × 3변수)
    daily_windows = prepare_daily_windows(
        [daily_vix, daily_rv, daily_cs], monthly_dates, K=K_midas)
    
    # 9. 일간 데이터 정규화 (z-score per variable)
    n_months, K, n_vars = daily_windows.shape
    train_n = int(n_months * train_ratio)
    for v in range(n_vars):
        train_mean = daily_windows[:train_n, :, v].mean()
        train_std = daily_windows[:train_n, :, v].std() + 1e-8
        daily_windows[:, :, v] = (daily_windows[:, :, v] - train_mean) / train_std
    
    # 10. HMM 확률 (smoothed)
    hmm_probs = prepare_hmm_probs(regime_proba, monthly_dates, smooth_alpha=0.8)
    
    # 11. 시퀀스 생성
    data_values = combined.values
    X, y = create_sequences(data_values, seq_length)
    
    # 일간 데이터 시퀀스 (각 월의 일간 윈도우를 Seq로 묶기)
    n_samples = len(X)
    x_daily_seq = []
    for i in range(n_samples):
        # i번째 sample: 과거 seq_length개월의 일간 데이터
        x_daily_seq.append(daily_windows[i:i + seq_length])
    x_daily_arr = np.array(x_daily_seq)  # (N, Seq, K, 3)
    
    # VIX 시퀀스 (타겟 시점)
    vix_values = vix_aligned.values
    vix_seq = np.array([vix_values[i + seq_length] for i in range(n_samples)])
    
    # HMM 시퀀스 (타겟 시점)
    hmm_seq = hmm_probs[seq_length:seq_length + n_samples]
    
    # 12. 자산 수익률만 타겟으로
    y_assets = y[:, :num_assets]
    
    # 12.5. 평가용 raw 수익률 시퀀스
    y_raw_list = []
    for i in range(n_samples):
        y_raw_list.append(raw_asset_returns[i + seq_length])
    y_raw_arr = np.array(y_raw_list)
    
    # 13. 텐서 변환
    result = {
        'x_monthly': torch.tensor(X, dtype=torch.float32),
        'x_daily': torch.tensor(x_daily_arr, dtype=torch.float32),
        'y': torch.tensor(y_assets, dtype=torch.float32),
        'y_raw': torch.tensor(y_raw_arr, dtype=torch.float32),
        'vix': torch.tensor(vix_seq.flatten(), dtype=torch.float32),
        'hmm_probs': torch.tensor(hmm_seq, dtype=torch.float32),
        'asset_names': ASSET_TICKERS,
        'dates': monthly_dates[seq_length:seq_length + n_samples],
        'scaler': scaler,
    }
    
    print(f"\n[Phase2] Data Ready:")
    print(f"  x_monthly:  {result['x_monthly'].shape}")
    print(f"  x_daily:    {result['x_daily'].shape}")
    print(f"  y:          {result['y'].shape}")
    print(f"  vix:        {result['vix'].shape}")
    print(f"  hmm_probs:  {result['hmm_probs'].shape}")
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("Data Loader Test")
    print("=" * 60)
    
    X, y, vix, scaler, assets, dates, _ = prepare_training_data(
        start_date='2007-07-01',
        end_date='2024-01-01',
        seq_length=12,
        normalize=True
    )
    
    print(f"\n--- Data Summary ---")
    print(f"Assets: {assets}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Dates shape: {dates.shape}")