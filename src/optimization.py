"""
Optimization Module
===================
[Step 3] 미분 가능 최적화 레이어 (CVaR Optimization)

역할: Black-Litterman 모델이 산출한 보정된 기대수익률(mu_BL)과 공분산(Sigma_BL)을 바탕으로,
      Tail Risk(CVaR)를 최소화하는 포트폴리오 비중을 계산합니다.
"""

import torch
import torch.nn as nn
import numpy as np

# cvxpy 관련 import
try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("[WARNING] cvxpylayers가 설치되지 않았습니다. Softmax로 대체됩니다.")


class CVaROptimizationLayer(nn.Module):
    """
    CVaR (Conditional Value-at-Risk) 최적화 레이어 (via cvxpylayers)
    
    문제 정의:
        Minimize CVaR(w)
        Subject to:
            sum(w) == 1 (예산 제약)
            w >= 0 (공매도 금지)
            if is_crisis: w[BIL] >= safety_threshold (위기 시 안전자산 강제)
            
    Args:
        num_assets: 자산 개수
        num_scenarios: 몬테카를로 시나리오 개수
        confidence_level: CVaR 베타 (예: 0.95 -> 상위 5% 손실 평균)
        bil_index: 안전자산(BIL) 인덱스
        safety_threshold: 위기 시 최소 확보해야 할 BIL 비중
    """
    def __init__(
        self, 
        num_assets: int, 
        num_scenarios: int = 200, 
        confidence_level: float = 0.95,
        bil_index: int = 4,
        safety_threshold: float = 0.5
    ):
        super(CVaROptimizationLayer, self).__init__()
        self.num_assets = num_assets
        self.num_scenarios = num_scenarios
        self.beta = confidence_level
        self.bil_index = bil_index
        self.safety_threshold = safety_threshold
        
        self.cvxpy_layer = None
        
        if CVXPY_AVAILABLE:
            self._build_cvar_layer()
            
    def _build_cvar_layer(self):
        """
        CVaR 최적화를 위한 CvxpyLayer를 구축합니다.
        """
        n = self.num_assets
        s = self.num_scenarios
        
        # 변수 정의
        w = cp.Variable(n)          # 비중
        alpha = cp.Variable()       # VaR 임계값
        u = cp.Variable(s)          # 보조 변수 (max(0, loss - alpha))
        
        # 파라미터 정의
        # R_scenarios: (S, N) - 몬테카를로 시뮬레이션으로 생성된 수익률 시나리오
        R_scenarios = cp.Parameter((s, n)) 
        
        # 위기 플래그: 스칼라 (0.0 또는 1.0)
        # 1.0일 경우 안전자산 제약조건이 활성화됨
        is_crisis = cp.Parameter() 
        
        # 목적 함수: CVaR 최소화
        # CVaR = alpha + 1/(1-beta) * E[u]
        cvar_term = alpha + (1.0 / (s * (1.0 - self.beta))) * cp.sum(u)
        objective = cp.Minimize(cvar_term)
        
        # 제약 조건
        loss = -R_scenarios @ w  # 손실 = -수익률
        constraints = [
            cp.sum(w) == 1.0,       # 예산 제약
            w >= 0.0,               # 롱 온리
            u >= loss - alpha,      # CVaR 보조식 1
            u >= 0.0                # CVaR 보조식 2
        ]
        
        # 안전망 제약조건 (조건부 활성화)
        # is_crisis 파라미터와 곱하여 플래그가 켜졌을 때만 제약이 걸리도록 함
        constraints.append(w[self.bil_index] >= self.safety_threshold * is_crisis)
        
        # 문제 및 레이어 생성
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        
        self.cvxpy_layer = CvxpyLayer(
            problem, 
            parameters=[R_scenarios, is_crisis], 
            variables=[w]
        )
        print("[INFO] CVaR 최적화 레이어 구축 완료 (안전망 포함)")

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, is_crisis: float = 0.0) -> torch.Tensor:
        """
        Forward Pass:
        1. Reparameterization Trick을 사용하여 N(mu, Sigma)에서 시나리오 샘플링
        2. cvxpylayers를 통해 CVaR 최적화 수행
        
        Args:
            mu: (Batch, N) 기대 수익률
            sigma: (Batch, N, N) 공분산 행렬
            is_crisis: (Batch, ) 또는 스칼라. 1.0이면 안전망 가동.
        
        Returns:
            weights: (Batch, N) 최적 비중
        """
        batch_size = mu.size(0)
        device = mu.device
        
        # cvxpy 미설치 시 
        if not CVXPY_AVAILABLE or self.cvxpy_layer is None:
            return torch.softmax(mu, dim=1)

        # 1. 미분 가능한 샘플링: R = mu + L * eps
        # Cholesky 분해: Sigma = L * L^T
        jit = 1e-6 * torch.eye(self.num_assets, device=device).expand(batch_size, -1, -1)
        try:
            L = torch.linalg.cholesky(sigma + jit)
        except RuntimeError:
            # PSD가 아닐 경우 대각 행렬 근사 사용
            L = torch.diag_embed(torch.sqrt(torch.diagonal(sigma, dim1=1, dim2=2).abs() + 1e-6))
            
        eps = torch.randn(batch_size, self.num_scenarios, self.num_assets, device=device) # (B, S, N)
        
        mu_expanded = mu.unsqueeze(1)
        
        # 시나리오 생성: R = mu + eps @ L^T
        scenarios = mu_expanded + torch.bmm(eps, L.transpose(1, 2))
        
        # 2. 최적화 실행
        if isinstance(is_crisis, float) or isinstance(is_crisis, int):
            crisis_param = torch.full((batch_size,), float(is_crisis), device=device)
        else:
            crisis_param = is_crisis 
            
        try:
            # cvxpylayers 호출
            weights, = self.cvxpy_layer(scenarios, crisis_param)
        except Exception as e:
            # 최적화 실패 시 Softmax로 대체 (그래디언트 단절 방지)
            weights = torch.softmax(mu, dim=1) 
            
        return weights

# =============================================================================
# Helper Functions
# =============================================================================

def estimate_covariance(returns: torch.Tensor, shrinkage: float = 0.1) -> torch.Tensor:
    """
    수익률 데이터에서 공분산 행렬 추정 (Shrinkage 적용)
    
    Args:
        returns: (Batch, Time, Assets) 수익률 텐서
        shrinkage: Shrinkage 강도 (0~1)
    
    Returns:
        Sigma: (Batch, Assets, Assets) 공분산 행렬
    """
    batch_size, seq_len, n_assets = returns.shape
    device = returns.device
    
    # 샘플 공분산 계산
    means = returns.mean(dim=1, keepdim=True)
    centered = returns - means
    sample_cov = torch.bmm(centered.transpose(1, 2), centered) / (seq_len - 1)
    
    # Shrinkage: 대각 행렬 방향으로 당기기
    identity = torch.eye(n_assets, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    
    trace_avg = sample_cov.diagonal(dim1=1, dim2=2).mean(dim=1, keepdim=True).unsqueeze(-1)
    shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * trace_avg * identity
    
    return shrunk_cov

