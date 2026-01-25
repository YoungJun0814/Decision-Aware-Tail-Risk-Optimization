"""
Loss Module
===========
[Step 4] 손실 함수 (Decision-Aware Loss)

역할: AI가 뱉은 비중(w)대로 투자했을 때, 결과가 좋았는지 나빴는지 판단.

주의: 예측 오차(MSE)가 아닙니다!

Loss = -(Return) + η × (Risk) + κ(VIX) × (Turnover)

- Return: 포트폴리오 수익률 (최대화 → 음수로 변환)
- Risk: 분산/CVaR (Role B가 확정)
- κ(VIX): 변동성 연동 거래비용 계수 (높은 VIX에서는 매매 자제)
"""

import torch
import torch.nn as nn


class DecisionAwareLoss(nn.Module):
    """
    Decision-Aware Loss Function (v2)
    
    프로젝트 정의서의 Loss Function을 구현합니다:
    
    Loss = -Return + η × Risk + κ(VIX) × Turnover
    
    Args:
        eta (float): 리스크(Risk) 페널티 가중치. 기본값 1.0
        kappa_base (float): 거래비용 기본 계수. 기본값 0.001
        kappa_vix_scale (float): VIX에 따른 거래비용 스케일링 계수. 기본값 0.0001
    """
    
    def __init__(
        self,
        eta: float = 1.0,
        kappa_base: float = 0.001,
        kappa_vix_scale: float = 0.0001
    ):
        super(DecisionAwareLoss, self).__init__()
        self.eta = eta
        self.kappa_base = kappa_base
        self.kappa_vix_scale = kappa_vix_scale
    
    def compute_dynamic_cost(self, vix: torch.Tensor) -> torch.Tensor:
        """
        VIX 레벨에 따른 동적 거래비용 계수 계산
        
        κ(VIX) = kappa_base + kappa_vix_scale × VIX
        
        VIX가 높을수록 (시장 불안) 거래비용을 높게 책정하여
        포지션 변경을 억제합니다 (관성 학습).
        
        Args:
            vix: (Batch,) VIX 레벨 텐서
        
        Returns:
            kappa: (Batch,) 동적 거래비용 계수
        """
        return self.kappa_base + self.kappa_vix_scale * vix
    
    def forward(
        self,
        weights: torch.Tensor,
        future_returns: torch.Tensor,
        vix: torch.Tensor,
        prev_weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        손실 계산
        
        Args:
            weights: (Batch, Num_assets) 현재 포트폴리오 비중
            future_returns: (Batch, Num_assets) 다음 기간 실현 수익률
            vix: (Batch,) 해당 시점의 VIX 레벨
            prev_weights: (Batch, Num_assets) 이전 포트폴리오 비중 (Turnover 계산용)
                          None이면 균등 비중으로 초기화
        
        Returns:
            loss: 스칼라 손실 값
        """
        batch_size, num_assets = weights.shape
        
        # ======================================================================
        # Term 1: Return (수익률 최대화 -> 음수로 최소화 문제 변환)
        # ======================================================================
        portfolio_returns = (weights * future_returns).sum(dim=1)
        return_loss = -portfolio_returns.mean()
        
        # ======================================================================
        # Term 2: Risk (분산/표준편차)
        # ======================================================================
        risk_penalty = portfolio_returns.std() + 1e-8  # 수치 안정성
        
        # ======================================================================
        # Term 3: Dynamic Transaction Cost (VIX 연동 거래비용)
        # ======================================================================
        if prev_weights is None:
            # 첫 시점이면 균등 비중에서 시작한다고 가정
            prev_weights = torch.ones_like(weights) / num_assets
        
        # Turnover: L1 norm of weight changes
        turnover = torch.abs(weights - prev_weights).sum(dim=1)  # (Batch,)
        
        # 동적 거래비용
        kappa = self.compute_dynamic_cost(vix)  # (Batch,)
        turnover_cost = (kappa * turnover).mean()
        
        # ======================================================================
        # Total Loss
        # ======================================================================
        total_loss = (
            return_loss
            + self.eta * risk_penalty
            + turnover_cost
        )
        
        return total_loss


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Loss Function Test")
    print("=" * 60)
    
    batch_size = 32
    num_assets = 5  # SPY, XLV, TLT, GLD, BIL
    
    # 더미 데이터
    weights = torch.softmax(torch.randn(batch_size, num_assets), dim=1)
    future_returns = torch.randn(batch_size, num_assets) * 0.02
    vix = torch.rand(batch_size) * 30 + 10  # VIX: 10 ~ 40 범위
    
    # DecisionAwareLoss 테스트 (v2 Main)
    print("\n--- DecisionAwareLoss Test (v2) ---")
    decision_loss_fn = DecisionAwareLoss(
        eta=1.0,
        kappa_base=0.001,
        kappa_vix_scale=0.0001
    )
    decision_loss = decision_loss_fn(weights, future_returns, vix)
    print(f"DecisionAwareLoss: {decision_loss.item():.6f}")
    
    print("\n[Success] Loss function tests passed!")
