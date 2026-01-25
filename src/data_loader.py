"""
Data Loader Module
==================
[Step 1] 데이터 수집 및 전처리

- Universe: SPY, XLV, TLT, GLD, BIL (5개 자산)
  - SPY: S&P 500 (공격 자산)
  - XLV: Healthcare (방어 섹터)
  - TLT: 장기 국채 (안전 자산)
  - GLD: 금 (인플레이션 헷지)
  - BIL: 초단기 채권 (현금성 안전 자산, 위기 시 대피처)
- Frequency: 일간 데이터를 받아 월말(Month-end) 기준으로 리샘플링
- Features: 자산 수익률 + 거시변수(Macro, 추후 확장)
- VIX: 손실 함수(Loss Function)의 동적 거래비용 계산을 위해 별도로 수집
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


# =============================================================================
# 설정 (Configuration)
# =============================================================================

# 자산 유니버스 (5개)
# 기존 SH(인버스)는 Volatility Drag 문제로 제거되고 BIL(현금성 자산)로 대체됨.
ASSET_TICKERS = ['SPY', 'XLV', 'TLT', 'GLD', 'BIL']

# VIX 티커 (Loss 함수에서 시장 상황 반영을 위해 사용)
VIX_TICKER = '^VIX'

# 거시경제 지표 (Strategist 선정 시 추가 예정)
MACRO_TICKERS = []


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


# =============================================================================
# 데이터 전처리 함수 (Preprocessing)
# =============================================================================

def normalize_data(df: pd.DataFrame) -> tuple:
    """
    데이터를 표준화(StandardScaler)합니다.
    """
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(df.values)
    normalized_df = pd.DataFrame(
        normalized_values, 
        index=df.index, 
        columns=df.columns
    )
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
    start_date: str = '2005-01-01',
    end_date: str = '2024-01-01',
    seq_length: int = 12,
    normalize: bool = True
) -> tuple:
    """
    전체 데이터 파이프라인: 수집 -> 병합 -> 전처리 -> 텐서 변환
    
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
    
    # 4. 데이터 병합
    if not macro_returns.empty:
        combined_data = pd.concat([asset_returns, macro_returns], axis=1).dropna()
    else:
        combined_data = asset_returns.copy()
    
    print(f"\n[INFO] 병합된 데이터 크기: {combined_data.shape}")
    print(f"[INFO] 데이터 기간: {combined_data.index[0]} ~ {combined_data.index[-1]}")
    
    # 5. VIX 데이터 인덱스 동기화
    if not vix_data.empty:
        vix_aligned = vix_data.reindex(combined_data.index).ffill().bfill()
    else:
        print("[WARNING] VIX 데이터가 없어 기본값 20을 사용합니다.")
        vix_aligned = pd.Series(20.0, index=combined_data.index)
    
    # 6. 정규화 (Standard Scaling)
    scaler = None
    if normalize:
        combined_data, scaler = normalize_data(combined_data)
    
    # 7. 시퀀스 생성 (Windowing)
    data_values = combined_data.values
    X, y = create_sequences(data_values, seq_length)
    
    # 8. VIX 시퀀스 생성 (타겟 시점의 VIX 사용)
    vix_values = vix_aligned.values
    vix_seq = []
    for i in range(len(vix_values) - seq_length):
        vix_seq.append(vix_values[i + seq_length])
    vix_array = np.array(vix_seq)
    
    # 9. 타겟에서 거시변수는 제외하고 자산 수익률만 남김
    num_assets = len(ASSET_TICKERS)
    y_assets = y[:, :num_assets]
    
    # 10. PyTorch 텐서로 변환
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_assets, dtype=torch.float32)
    vix_tensor = torch.tensor(vix_array.flatten(), dtype=torch.float32)
    
    # 11. 날짜 데이터 (Target 기준)
    y_dates = combined_data.index[seq_length:]
    
    print(f"[INFO] X_tensor shape: {X_tensor.shape}")
    print(f"[INFO] y_tensor shape: {y_tensor.shape}")
    print(f"[INFO] vix_tensor shape: {vix_tensor.shape}")
    
    return X_tensor, y_tensor, vix_tensor, scaler, ASSET_TICKERS, y_dates


if __name__ == "__main__":
    print("=" * 60)
    print("Data Loader Test")
    print("=" * 60)
    
    X, y, vix, scaler, assets = prepare_training_data(
        start_date='2005-01-01',
        end_date='2024-01-01',
        seq_length=12,
        normalize=True
    )
    
    print(f"\n--- Data Summary ---")
    print(f"Assets: {assets}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")