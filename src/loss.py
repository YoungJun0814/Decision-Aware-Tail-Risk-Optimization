"""
Loss Module
===========
[Step 4] 손실 함수

역할: AI가 뱉은 비중(w)대로 투자했을 때, 결과가 좋았는지 나빴는지 채점합니다.

주의: 예측 오차(MSE)가 아닙니다!
Loss = -(Portfolio Return) + λ × (Realized Risk)

수익은 높이고(Maximize), 실현된 리스크는 낮추는(Minimize) 방향으로 학습합니다.
"""

import torch
import torch.nn as nn


class TaskLoss(nn.Module):
    """
    Task-Aware Loss Function
    
    Decision-Aware 학습을 위한 손실 함수입니다.
    예측 오차가 아니라, 실제 투자 결과(수익률, 리스크)를 기반으로 손실을 계산합니다.
    
    TODO: Researcher가 Utility Function 정의하면 교체
    
    현재는 가장 단순한 형태로 구현:
    Loss = -mean(portfolio_return) + λ × std(portfolio_return)
    
    Args:
        risk_lambda: 리스크 패널티 가중치 (기본값 1.0)
    """
    
    def __init__(self, risk_lambda: float = 1.0):
        super(TaskLoss, self).__init__()
        self.risk_lambda = risk_lambda
    
    def forward(
        self, 
        weights: torch.Tensor, 
        future_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        손실 계산
        
        Args:
            weights: (Batch, Num_assets) 포트폴리오 비중
            future_returns: (Batch, Num_assets) 다음 기간 실현 수익률
        
        Returns:
            loss: 스칼라 손실 값
        """
        # 포트폴리오 수익률 계산: Σ(w_i × r_i)
        portfolio_returns = (weights * future_returns).sum(dim=1)
        
        # 수익률 최대화 (음수 부호로 최소화 문제로 변환)
        return_loss = -portfolio_returns.mean()
        
        # 리스크 페널티 (분산/표준편차)
        # TODO: Researcher가 CVaR, MDD 등 다른 리스크 척도로 교체 가능
        risk_penalty = portfolio_returns.std()
        
        # 총 손실
        total_loss = return_loss + self.risk_lambda * risk_penalty
        
        return total_loss


class SharpeRatioLoss(nn.Module):
    """
    Sharpe Ratio Loss
    
    샤프 비율을 최대화하는 손실 함수입니다.
    Loss = -Sharpe = -(mean(r) - rf) / std(r)
    
    Args:
        risk_free_rate: 무위험 이자율 (월간)
    """
    
    def __init__(self, risk_free_rate: float = 0.02 / 12):
        super(SharpeRatioLoss, self).__init__()
        self.rf = risk_free_rate
    
    def forward(
        self, 
        weights: torch.Tensor, 
        future_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        손실 계산
        """
        # 포트폴리오 수익률
        portfolio_returns = (weights * future_returns).sum(dim=1)
        
        # 초과 수익률
        excess_returns = portfolio_returns - self.rf
        
        # 샤프 비율 (음수로 최소화)
        mean_excess = excess_returns.mean()
        std_returns = portfolio_returns.std() + 1e-8  # 0으로 나누기 방지
        
        sharpe_loss = -mean_excess / std_returns
        
        return sharpe_loss


class CVaRLoss(nn.Module):
    """
    CVaR (Conditional Value-at-Risk) Loss
    
    TODO: Researcher가 CVaR 수식 확정하면 구현 완성
    
    Tail Risk를 직접 페널티로 부과하는 손실 함수입니다.
    
    Args:
        confidence_level: 신뢰수준 (기본값 0.95)
        return_weight: 수익률 가중치
        cvar_weight: CVaR 가중치
    """
    
    def __init__(
        self, 
        confidence_level: float = 0.95,
        return_weight: float = 1.0,
        cvar_weight: float = 1.0
    ):
        super(CVaRLoss, self).__init__()
        self.beta = confidence_level
        self.return_weight = return_weight
        self.cvar_weight = cvar_weight
    
    def forward(
        self, 
        weights: torch.Tensor, 
        future_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        손실 계산
        
        TODO: Researcher가 CVaR 계산 로직 확정하면 완성
        """
        # 포트폴리오 수익률
        portfolio_returns = (weights * future_returns).sum(dim=1)
        
        # 수익률 최대화
        return_loss = -portfolio_returns.mean()
        
        # CVaR 계산 (Empirical)
        # 하위 (1-β)% 수익률의 평균
        sorted_returns, _ = torch.sort(portfolio_returns)
        cutoff_idx = int(len(sorted_returns) * (1 - self.beta))
        cutoff_idx = max(1, cutoff_idx)  # 최소 1개
        
        tail_returns = sorted_returns[:cutoff_idx]
        cvar = -tail_returns.mean()  # 손실이므로 음수
        
        # 총 손실
        total_loss = self.return_weight * return_loss + self.cvar_weight * cvar
        
        return total_loss


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Loss Function Test")
    print("=" * 60)
    
    batch_size = 32
    num_assets = 4
    
    # 더미 데이터
    weights = torch.softmax(torch.randn(batch_size, num_assets), dim=1)
    future_returns = torch.randn(batch_size, num_assets) * 0.02
    
    # TaskLoss 테스트
    print("\n--- TaskLoss Test ---")
    task_loss_fn = TaskLoss(risk_lambda=1.0)
    task_loss = task_loss_fn(weights, future_returns)
    print(f"TaskLoss: {task_loss.item():.6f}")
    
    # SharpeRatioLoss 테스트
    print("\n--- SharpeRatioLoss Test ---")
    sharpe_loss_fn = SharpeRatioLoss()
    sharpe_loss = sharpe_loss_fn(weights, future_returns)
    print(f"SharpeRatioLoss: {sharpe_loss.item():.6f}")
    
    # CVaRLoss 테스트
    print("\n--- CVaRLoss Test ---")
    cvar_loss_fn = CVaRLoss(confidence_level=0.95)
    cvar_loss = cvar_loss_fn(weights, future_returns)
    print(f"CVaRLoss: {cvar_loss.item():.6f}")
    
    print("\n[Success] Loss function tests passed!")
