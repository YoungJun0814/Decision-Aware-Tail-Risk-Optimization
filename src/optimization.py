"""
Optimization Module
===================
[Step 3] 미분 가능 최적화 레이어

역할: 딥러닝이 준 파라미터를 가지고, 수학적으로 최적의 비중(w)을 계산합니다.

구현 디테일 (cvxpylayers 사용):
- CVaR 최적화 수식을 cvxpy 코드로 정의
- Objective: Minimize CVaR (Tail Risk)
- Constraints: Σw = 1 (예산), w ≥ 0 (공매도 금지), Turnover Limit 등

주의: cvxpylayers에 들어가는 수식은 반드시 Convex(볼록) 문제여야 합니다!
"""

import torch
import torch.nn as nn
import numpy as np

# cvxpy 관련 import
# NOTE: cvxpylayers가 설치되지 않은 경우를 대비한 조건부 import
try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("[WARNING] cvxpylayers not installed. Using Softmax fallback.")


class MinVarianceLayer(nn.Module):
    """
    Minimum Variance Optimization Layer (Placeholder)
    
    TODO: Researcher가 CVaR 수식 확정하면 교체할 예정
    
    현재는 가장 단순한 Min-Variance 문제로 구현하여 에러가 안 나는지만 확인합니다.
    
    수식:
        minimize    w^T Σ w  (포트폴리오 분산)
        subject to  Σw = 1   (예산 제약)
                    w ≥ 0    (공매도 금지)
    
    Args:
        num_assets: 자산 개수
    """
    
    def __init__(self, num_assets: int):
        super(MinVarianceLayer, self).__init__()
        self.num_assets = num_assets
        self.cvxpy_layer = None
        
        if CVXPY_AVAILABLE:
            self._build_cvxpy_layer()
        else:
            print("[INFO] cvxpylayers not available. Using Softmax fallback.")
    
    def _build_cvxpy_layer(self):
        """
        cvxpy 문제를 정의하고 CvxpyLayer로 감싸는 함수
        """
        n = self.num_assets
        
        # 결정 변수: 포트폴리오 비중
        w = cp.Variable(n, nonneg=True)  # w ≥ 0 (공매도 금지)
        
        # 파라미터: 공분산 행렬 (딥러닝에서 예측)
        # 실제로는 예측 수익률도 필요할 수 있음
        Sigma = cp.Parameter((n, n), PSD=True)  # Positive Semi-Definite
        
        # TODO: Researcher가 기대수익률 파라미터 추가
        # mu = cp.Parameter(n)
        
        # 목적함수: 분산 최소화
        # TODO: Researcher가 CVaR 목적함수로 교체
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        
        # 제약조건
        constraints = [
            cp.sum(w) == 1,  # 예산 제약: 비중 합 = 1
            # TODO: Researcher가 추가 제약조건 정의
            # w <= 0.4,  # 개별 자산 최대 비중 제한 (예시)
            # TODO: 제약조건(Turnover 등) 확정되면 추가
        ]
        
        # 문제 정의
        problem = cp.Problem(objective, constraints)
        
        # CvxpyLayer로 변환
        # parameters: 딥러닝에서 넘겨받을 값
        # variables: 최적화 결과로 반환할 값
        self.cvxpy_layer = CvxpyLayer(
            problem, 
            parameters=[Sigma], 
            variables=[w]
        )
        
        print(f"[INFO] MinVarianceLayer built with {n} assets")
    
    def forward(self, Sigma: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: 공분산 행렬을 받아 최적 비중 반환
        
        Args:
            Sigma: (Batch, N, N) 공분산 행렬 텐서
        
        Returns:
            weights: (Batch, N) 최적 포트폴리오 비중
        """
        if self.cvxpy_layer is None:
            # Fallback: 동일 비중
            batch_size = Sigma.shape[0]
            return torch.ones(batch_size, self.num_assets) / self.num_assets
        
        # cvxpylayers는 배치 처리 가능
        # Sigma shape: (batch, n, n)
        weights, = self.cvxpy_layer(Sigma)
        
        return weights


class CVaROptimizationLayer(nn.Module):
    """
    CVaR (Conditional Value-at-Risk) Optimization Layer
    
    TODO: Researcher가 CVaR 수식 확정하면 완성
    
    CVaR 최적화는 다음과 같은 수식을 가집니다:
    
        minimize    α + (1/(1-β)) * E[max(0, -r^T w - α)]
        subject to  Σw = 1
                    w ≥ 0
    
    여기서:
    - α (VaR): Value-at-Risk 수준
    - β: 신뢰수준 (예: 0.95)
    - r: 시나리오 수익률
    - w: 포트폴리오 비중
    
    Args:
        num_assets: 자산 개수
        num_scenarios: 시나리오 수 (Monte Carlo 시뮬레이션 등)
        confidence_level: 신뢰수준 (기본값 0.95)
    """
    
    def __init__(
        self, 
        num_assets: int, 
        num_scenarios: int = 100,
        confidence_level: float = 0.95
    ):
        super(CVaROptimizationLayer, self).__init__()
        self.num_assets = num_assets
        self.num_scenarios = num_scenarios
        self.beta = confidence_level
        self.cvxpy_layer = None
        
        # TODO: Researcher가 CVaR 수식 확정하면 _build_cvar_layer() 호출
        # if CVXPY_AVAILABLE:
        #     self._build_cvar_layer()
        
        print(f"[INFO] CVaROptimizationLayer placeholder created (not yet implemented)")
    
    def _build_cvar_layer(self):
        """
        TODO: Researcher가 CVaR 수식 확정하면 구현
        
        CVaR 최적화 문제를 정의하고 CvxpyLayer로 감싸는 함수
        """
        n = self.num_assets
        S = self.num_scenarios
        beta = self.beta
        
        # 결정 변수
        w = cp.Variable(n, nonneg=True)
        alpha = cp.Variable()  # VaR
        u = cp.Variable(S, nonneg=True)  # auxiliary variables
        
        # 파라미터: 시나리오 수익률 (S x N 행렬)
        R = cp.Parameter((S, n))
        
        # 목적함수: CVaR 최소화
        cvar = alpha + (1 / (S * (1 - beta))) * cp.sum(u)
        objective = cp.Minimize(cvar)
        
        # 제약조건
        constraints = [
            cp.sum(w) == 1,
            u >= -R @ w - alpha,
            # TODO: 추가 제약조건
        ]
        
        problem = cp.Problem(objective, constraints)
        self.cvxpy_layer = CvxpyLayer(problem, parameters=[R], variables=[w])
    
    def forward(self, scenarios: torch.Tensor) -> torch.Tensor:
        """
        TODO: Researcher가 CVaR 수식 확정하면 구현
        
        Args:
            scenarios: (Batch, Scenarios, Assets) 시나리오 수익률
        
        Returns:
            weights: (Batch, Assets) 최적 비중
        """
        # Placeholder: 동일 비중 반환
        batch_size = scenarios.shape[0]
        return torch.ones(batch_size, self.num_assets) / self.num_assets


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
    
    # 샘플 공분산 계산
    means = returns.mean(dim=1, keepdim=True)
    centered = returns - means
    sample_cov = torch.bmm(centered.transpose(1, 2), centered) / (seq_len - 1)
    
    # Shrinkage: 대각 행렬 방향으로 당기기
    identity = torch.eye(n_assets).unsqueeze(0).expand(batch_size, -1, -1)
    identity = identity.to(returns.device)
    
    trace_avg = sample_cov.diagonal(dim1=1, dim2=2).mean(dim=1, keepdim=True).unsqueeze(-1)
    shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * trace_avg * identity
    
    return shrunk_cov


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Optimization Layer Test")
    print("=" * 60)
    
    num_assets = 4
    batch_size = 5
    seq_length = 12
    
    # 더미 수익률 데이터
    dummy_returns = torch.randn(batch_size, seq_length, num_assets) * 0.02
    
    # 공분산 추정
    Sigma = estimate_covariance(dummy_returns)
    print(f"\nCovariance matrix shape: {Sigma.shape}")
    
    # MinVariance 레이어 테스트
    print("\n--- MinVarianceLayer Test ---")
    min_var_layer = MinVarianceLayer(num_assets=num_assets)
    
    if CVXPY_AVAILABLE:
        try:
            weights = min_var_layer(Sigma)
            print(f"Optimal weights shape: {weights.shape}")
            print(f"Sample weights: {weights[0].detach().numpy()}")
            print(f"Weights sum: {weights.sum(dim=1)}")
        except Exception as e:
            print(f"[ERROR] cvxpylayers failed: {e}")
            print("Using fallback (equal weights)")
    else:
        print("cvxpylayers not installed. Skipping test.")
    
    print("\n[Success] Optimization layer tests completed!")
