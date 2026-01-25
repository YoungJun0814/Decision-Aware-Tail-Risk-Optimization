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

def backtest():
    pass

def plot_results():
    pass
