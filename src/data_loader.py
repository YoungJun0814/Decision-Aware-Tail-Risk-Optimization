"""
Data Loader Module
==================
[Step 1] 데이터 수집 및 전처리

역할: 모델이 먹을 수 있는 형태로 데이터를 가공합니다.
- Universe: SPY, TLT, GLD, DBC (4개 자산)
- Frequency: 일간 -> 월말(Month-end) 리샘플링
- Features: 자산 수익률 + 거시변수(Macro)
- Output: (Batch_size, Sequence_length, Num_features) 형태의 텐서
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


# =============================================================================
# Configuration
# =============================================================================

# 자산 유니버스 (주식, 채권, 금, 원자재)
ASSET_TICKERS = ['SPY', 'TLT', 'GLD', 'DBC']

# TODO: Strategist가 선정해주면 채워 넣을 것
# 예시: ['^VIX', '^TNX', 'DX-Y.NYB'] (VIX, 10년물 금리, 달러 인덱스)
MACRO_TICKERS = []


# =============================================================================
# Data Collection Functions
# =============================================================================

def get_monthly_asset_data(tickers: list, start_date: str, end_date: str) -> tuple:
    """
    일간 데이터를 받아서 월간(Month-End) 수익률로 변환하는 함수
    
    Args:
        tickers: 티커 리스트
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
    
    Returns:
        monthly_prices: 월말 가격 DataFrame
        monthly_returns: 월간 수익률 DataFrame
    """
    print(f"Downloading data for: {tickers}")
    
    # 1. 일간 데이터 다운로드
    # yfinance 최신 버전에서는 리턴되는 DataFrame의 컬럼 구조가 다를 수 있음
    # auto_adjust=True로 설정하여 'Close'가 이미 수정 종가가 되도록 함
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
    
    # 2. 월말(Month End) 기준으로 리샘플링 (.last() 사용)
    # 금융에서는 보통 월말 종가를 그 달의 가격으로 봅니다.
    monthly_prices = data.resample('ME').last()
    
    # 3. 월간 수익률 계산 (Percentage Change)
    monthly_returns = monthly_prices.pct_change().dropna()
    
    return monthly_prices, monthly_returns


def get_macro_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    거시변수 데이터를 수집하는 함수 (Placeholder)
    
    TODO: Strategist가 거시변수 리스트를 확정하면 구현 완성
    
    Args:
        tickers: 거시변수 티커 리스트
        start_date: 시작일
        end_date: 종료일
    
    Returns:
        월간 거시변수 DataFrame
    """
    if not tickers:
        print("[INFO] MACRO_TICKERS is empty. Skipping macro data collection.")
        return pd.DataFrame()
    
    print(f"Downloading macro data for: {tickers}")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    monthly_macro = data.resample('ME').last().pct_change().dropna()
    
    return monthly_macro


# =============================================================================
# Data Preprocessing Functions
# =============================================================================

def normalize_data(df: pd.DataFrame) -> tuple:
    """
    StandardScaler를 사용해 데이터를 정규화하는 함수
    
    Args:
        df: 정규화할 DataFrame
    
    Returns:
        normalized_df: 정규화된 DataFrame
        scaler: 학습된 StandardScaler (나중에 역변환용)
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
    시계열 데이터를 (Batch, Sequence, Features) 형태로 변환
    
    주의: 월말 데이터가 다음 달 수익률 예측에 쓰여야 함 -> Lagging 처리
    
    Args:
        data: (Time, Features) 형태의 numpy 배열
        seq_length: 시퀀스 길이 (예: 12개월)
    
    Returns:
        X: (Batch, Sequence, Features) 형태의 입력 데이터
        y: (Batch, Features) 형태의 타겟 수익률 (다음 달)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # 과거 seq_length 개월의 데이터를 입력으로
        X.append(data[i:i + seq_length])
        # 다음 달 수익률을 타겟으로 (Lagging 처리)
        y.append(data[i + seq_length])
    
    return np.array(X), np.array(y)


def prepare_training_data(
    start_date: str = '2005-01-01',
    end_date: str = '2024-01-01',
    seq_length: int = 12,
    normalize: bool = True
) -> tuple:
    """
    전체 데이터 파이프라인: 수집 -> 전처리 -> 텐서 변환
    
    Args:
        start_date: 데이터 시작일
        end_date: 데이터 종료일
        seq_length: 시퀀스 길이 (몇 개월 lookback)
        normalize: 정규화 여부
    
    Returns:
        X_tensor: (Batch, Seq, Features) PyTorch 텐서
        y_tensor: (Batch, Num_assets) PyTorch 텐서 (다음 달 수익률)
        scaler: 정규화에 사용된 Scaler (역변환용)
        asset_names: 자산 이름 리스트
    """
    # 1. 자산 데이터 수집
    _, asset_returns = get_monthly_asset_data(ASSET_TICKERS, start_date, end_date)
    
    # 2. 거시변수 데이터 수집 (비어있으면 스킵)
    macro_returns = get_macro_data(MACRO_TICKERS, start_date, end_date)
    
    # 3. 데이터 병합
    if not macro_returns.empty:
        # 인덱스 맞추기
        combined_data = pd.concat([asset_returns, macro_returns], axis=1).dropna()
    else:
        combined_data = asset_returns.copy()
    
    print(f"\n[INFO] Combined data shape: {combined_data.shape}")
    print(f"[INFO] Date range: {combined_data.index[0]} ~ {combined_data.index[-1]}")
    
    # 4. 정규화
    scaler = None
    if normalize:
        combined_data, scaler = normalize_data(combined_data)
    
    # 5. 시퀀스 생성 (Lagging 포함)
    data_values = combined_data.values
    X, y = create_sequences(data_values, seq_length)
    
    # 6. 타겟은 자산 수익률만 (Macro 제외)
    num_assets = len(ASSET_TICKERS)
    y_assets = y[:, :num_assets]  # 첫 N개 컬럼만 (자산 수익률)
    
    # 7. PyTorch 텐서 변환
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_assets, dtype=torch.float32)
    
    print(f"[INFO] X_tensor shape: {X_tensor.shape}")  # (Batch, Seq, Features)
    print(f"[INFO] y_tensor shape: {y_tensor.shape}")  # (Batch, Num_assets)
    
    return X_tensor, y_tensor, scaler, ASSET_TICKERS


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Data Loader Test")
    print("=" * 60)
    
    # 데이터 준비
    X, y, scaler, assets = prepare_training_data(
        start_date='2005-01-01',
        end_date='2024-01-01',
        seq_length=12,
        normalize=True
    )
    
    print(f"\n--- Data Summary ---")
    print(f"Assets: {assets}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # 저장 (옵션)
    # torch.save({'X': X, 'y': y}, 'data/processed/training_data.pt')
    # print("\n[Success] Training data saved!")