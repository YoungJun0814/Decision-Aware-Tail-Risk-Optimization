"""
MIDAS Layer Module (Phase 2)
============================
End-to-end differentiable MIDAS (Mixed-Data Sampling) Layer.

Phase 1의 오프라인 MIDAS와 달리, Almon polynomial 파라미터(θ)를
포트폴리오 목적함수(Mean-CVaR)에서 직접 역전파하여 학습합니다.

핵심 차이:
  Phase 1: θ = argmin_θ ||y - Σw(θ)·x||²  (OLS, 예측 최적)
  Phase 2: θ = argmin_θ Loss(Portfolio(θ))  (End-to-End, 포트폴리오 최적)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableMIDASLayer(nn.Module):
    """
    Differentiable MIDAS Layer.
    
    각 일간 변수에 독립적인 Almon polynomial weights를 학습합니다.
    Almon weights: w(k; θ) = softmax(θ₁k̃ + θ₂k̃²)  (k̃ = k/K ∈ [1/K, 1])
    
    Args:
        n_daily_vars: 일간 변수 수 (VIX, RealizedVol, CreditSpread)
        K: 일간 래그 수 (default: 66, 약 3거래월)
        poly_degree: Almon 다항식 차수 (default: 2)
    """
    def __init__(self, n_daily_vars: int = 3, K: int = 66, poly_degree: int = 2):
        super().__init__()
        self.n_daily_vars = n_daily_vars
        self.K = K
        self.poly_degree = poly_degree
        
        # 각 변수별 독립 theta: (n_daily_vars, poly_degree)
        # 초기값 0 → 초기에는 균등 가중치 (softmax(0) = 1/K)
        self.theta = nn.Parameter(torch.zeros(n_daily_vars, poly_degree))
    
    def _almon_weights(self, theta_single: torch.Tensor) -> torch.Tensor:
        """
        단일 변수에 대한 Almon polynomial weights 계산.
        
        Args:
            theta_single: (poly_degree,) 해당 변수의 Almon 파라미터
            
        Returns:
            weights: (K,) 정규화된 가중치 (합=1)
        """
        # k를 [1/K, 1]로 정규화 → exp() 발산 방지
        # k²의 최대값이 1.0이므로 θ₂가 커도 안전
        k = torch.arange(1, self.K + 1, dtype=torch.float32,
                         device=self.theta.device) / self.K
        
        # Almon polynomial: θ₁k̃ + θ₂k̃² + ... + θ_d·k̃^d
        exponent = torch.zeros(self.K, device=self.theta.device)
        for d in range(self.poly_degree):
            exponent = exponent + theta_single[d] * k ** (d + 1)
        
        return F.softmax(exponent, dim=0)
    
    def get_all_weights(self) -> torch.Tensor:
        """
        모든 변수의 Almon weights 반환 (시각화용).
        
        Returns:
            (n_daily_vars, K) 각 변수별 가중치
        """
        weights = []
        for v in range(self.n_daily_vars):
            weights.append(self._almon_weights(self.theta[v]))
        return torch.stack(weights, dim=0)
    
    def forward(self, daily_data: torch.Tensor) -> torch.Tensor:
        """
        일간 데이터를 가중 합산하여 월간 features 생성.
        
        Args:
            daily_data: (B, K, n_daily_vars) 일간 데이터
            
        Returns:
            midas_features: (B, n_daily_vars) 가중 합산된 월간 features
        """
        B = daily_data.size(0)
        features = []
        
        for v in range(self.n_daily_vars):
            w = self._almon_weights(self.theta[v])                    # (K,)
            feat = (daily_data[:, :, v] * w.unsqueeze(0)).sum(dim=1)  # (B,)
            features.append(feat)
        
        return torch.stack(features, dim=1)  # (B, n_daily_vars)


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LearnableMIDASLayer Self-Test")
    print("=" * 60)
    
    layer = LearnableMIDASLayer(n_daily_vars=3, K=66, poly_degree=2)
    print(f"Parameters: {sum(p.numel() for p in layer.parameters())} "
          f"(expected: {3 * 2} = 6)")
    
    # Forward pass
    x = torch.randn(8, 66, 3)  # (B=8, K=66, vars=3)
    out = layer(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape} (expected: [8, 3])")
    
    # Gradient flow
    loss = out.sum()
    loss.backward()
    assert layer.theta.grad is not None, "Gradient not flowing to theta!"
    print(f"Theta grad: {layer.theta.grad.shape} ✓")
    
    # Weight properties
    weights = layer.get_all_weights()
    print(f"Almon weights shape: {weights.shape}")
    print(f"Weight sums: {weights.sum(dim=1)} (expected: [1, 1, 1])")
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-5)
    
    # Numerical stability check
    layer.theta.data = torch.tensor([[5.0, -3.0], [0.0, 0.0], [-2.0, 4.0]])
    weights_extreme = layer.get_all_weights()
    assert not torch.any(torch.isnan(weights_extreme)), "NaN in extreme weights!"
    assert not torch.any(torch.isinf(weights_extreme)), "Inf in extreme weights!"
    print(f"Extreme theta stability: ✓ (no NaN/Inf)")
    
    print("\n[SUCCESS] All LearnableMIDASLayer tests passed!")
