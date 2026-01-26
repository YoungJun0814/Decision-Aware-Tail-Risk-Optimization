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

