"""
Utils Module
============
Utility functions for reproducibility and device management.
"""

import random
import os
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Random Seed 고정 (Reproducibility)
    
    Args:
        seed: 시드 값
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to {seed}")

def get_device() -> str:
    """
    사용 가능한 디바이스(cuda, mps, cpu) 반환
    
    Returns:
        device string ('cuda', 'mps', 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available(): # For MacOS M1/M2
        return 'mps'
    else:
        return 'cpu'

def calculate_mdd(returns):
    """
    최대 낙폭 (MDD) 계산
    
    Args:
        returns: 수익률 배열 (1D numpy array or torch tensor)
        
    Returns:
        mdd: 최대 낙폭 값 (양수, e.g., 0.2 means 20% drawdown)
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.cpu().numpy()
        
    # 누적 수익률 계산 (Cumulative Return)
    # returns는 simple returns 가정: r_t = (P_t - P_{t-1}) / P_{t-1}
    # cum_ret_t = (1 + r_1) * ... * (1 + r_t)
    cum_ret = np.cumprod(1 + returns)
    
    # Running Max 계산
    peak = np.maximum.accumulate(cum_ret)
    
    # Drawdown 계산
    drawdown = (peak - cum_ret) / peak
    
    # Max Drawdown
    mdd = np.max(drawdown)
    
    return mdd

def backtest():
    """TODO: 백테스트 기능 구현 예정"""
    raise NotImplementedError("Backtest function is not yet implemented")

def plot_results():
    """TODO: 결과 시각화 기능 구현 예정"""
    raise NotImplementedError("Plot results function is not yet implemented")


def compute_dd_state(portfolio_returns: np.ndarray, t: int,
                     max_duration: float = 24.0,
                     max_speed: float = 0.05) -> np.ndarray:
    """
    시점 t에서의 Drawdown 상태 벡터를 계산합니다. (v5 신규)

    포트폴리오 역사적 수익률을 사용하여 현재 DD 상태를 3차원 벡터로 인코딩.
    모델의 Q head에 입력되어 "지금 내 포트폴리오가 얼마나 힘든 상태인가"를
    인식하게 함. 깊은 DD일수록 자동으로 방어적 BL View를 생성하도록 유도.

    Args:
        portfolio_returns: (T,) 시점 0부터 t까지의 포트폴리오 월간 수익률 배열.
        t: 현재 시점 인덱스 (포함).
        max_duration: 정규화 기준 최대 지속기간 (월). 기본 24개월.
        max_speed: 정규화 기준 최대 DD 속도. 기본 5% per month.

    Returns:
        dd_state: (3,) float32 배열
            [0] current_drawdown: 현재 낙폭률 (0 ~ 1, 0이면 신고점)
            [1] dd_duration_norm: 낙폭 지속기간 / max_duration (0 ~ 1)
            [2] dd_speed_norm:    최근 1개월 DD 변화 / max_speed (0 ~ 1, 클수록 악화 빠름)
    """
    if t < 0 or len(portfolio_returns) == 0:
        return np.zeros(3, dtype=np.float32)

    rets = portfolio_returns[:t + 1]
    cum_ret = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(cum_ret)
    drawdowns = (peak - cum_ret) / (peak + 1e-8)

    # (1) 현재 낙폭률
    current_dd = float(drawdowns[-1])

    # (2) 낙폭 지속기간: 마지막으로 신고점이었던 시점부터 현재까지
    if current_dd < 1e-6:
        duration = 0
    else:
        # 마지막 신고점 시점: drawdowns == 0인 가장 마지막 인덱스
        zero_indices = np.where(drawdowns < 1e-6)[0]
        last_peak_t = zero_indices[-1] if len(zero_indices) > 0 else 0
        duration = t - last_peak_t
    duration_norm = min(float(duration) / max_duration, 1.0)

    # (3) DD 속도: 현재와 1개월 전 DD의 차이 (양수 = 악화)
    if len(drawdowns) >= 2:
        dd_speed = float(drawdowns[-1]) - float(drawdowns[-2])
        dd_speed = max(dd_speed, 0.0)  # 개선(음수)은 0으로 처리
    else:
        dd_speed = 0.0
    speed_norm = min(dd_speed / max_speed, 1.0)

    return np.array([current_dd, duration_norm, speed_norm], dtype=np.float32)


def compute_dd_state_sequence(portfolio_returns: np.ndarray,
                              max_duration: float = 24.0,
                              max_speed: float = 0.05) -> np.ndarray:
    """
    전체 수익률 시퀀스에 대해 DD 상태 벡터를 일괄 계산합니다.

    Args:
        portfolio_returns: (T,) 포트폴리오 월간 수익률 배열.
        max_duration: 정규화 최대 지속기간 (월).
        max_speed: 정규화 최대 DD 속도.

    Returns:
        dd_states: (T, 3) float32 배열. 각 시점의 dd_state 벡터.
    """
    T = len(portfolio_returns)
    dd_states = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        dd_states[t] = compute_dd_state(portfolio_returns, t, max_duration, max_speed)
    return dd_states

