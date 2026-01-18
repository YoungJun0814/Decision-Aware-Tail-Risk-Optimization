"""
Loss Module
===========
[Step 4] 손실 함수 (Decision-Aware Loss)

역할: AI가 뱉은 비중(w)대로 투자했을 때, 결과가 좋았는지 나빴는지 판단.

주의: 예측 오차(MSE)가 아닙니다!

Loss = -(Return) + η × (Risk) + κ(VIX) × (Turnover) + ρ × (InverseDecay)

- Return: 포트폴리오 수익률 (최대화 → 음수로 변환)
- Risk: 분산/CVaR (Role B가 확정)
- κ(VIX): 변동성 연동 거래비용 계수
- ρ × InverseDecay: 인버스 ETF(SH) 장기 보유 페널티
"""

import torch
import torch.nn as nn


class TaskLoss(nn.Module):
    """
    Task-Aware Loss Function (Legacy - 참고용)
    
    Decision-Aware 학습을 위한 손실 함수입니다.
    예측 오차가 아니라, 실제 투자 결과(수익률, 리스크)를 기반으로 손실을 계산합니다.
    
    NOTE: DecisionAwareLoss로 대체 권장
    
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
        risk_penalty = portfolio_returns.std()
        
        # 총 손실
        total_loss = return_loss + self.risk_lambda * risk_penalty
        
        return total_loss


class DecisionAwareLoss(nn.Module):
    """
    Decision-Aware Loss Function (v2)
    
    프로젝트 정의서의 Loss Function을 구현합니다:
    
    Loss = -Return + η × Risk + κ(VIX) × Turnover + ρ × InverseDecay
    
    Args:
        eta (float): 리스크(Risk) 페널티 가중치. 기본값 1.0
        kappa_base (float): 거래비용 기본 계수. 기본값 0.001
        kappa_vix_scale (float): VIX에 따른 거래비용 스케일링 계수. 기본값 0.0001
        rho (float): 인버스 ETF 장기 보유 페널티 계수. 기본값 0.01
        inverse_etf_index (int): 인버스 ETF의 인덱스 (SH). 기본값 4
    """
    
    def __init__(
        self,
        eta: float = 1.0,
        kappa_base: float = 0.001,
        kappa_vix_scale: float = 0.0001,
        rho: float = 0.01,
        inverse_etf_index: int = 4
    ):
        super(DecisionAwareLoss, self).__init__()
        self.eta = eta
        self.kappa_base = kappa_base
        self.kappa_vix_scale = kappa_vix_scale
        self.rho = rho
        self.inverse_etf_index = inverse_etf_index
    
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
    
    def compute_inverse_decay_penalty(self, weights: torch.Tensor) -> torch.Tensor:
        """
        인버스 ETF(SH) 장기 보유 페널티 계산
        
        인버스 ETF는 구조적으로 횡보장에서 가치가 소멸(Decay)됩니다.
        따라서 높은 인버스 비중을 페널티로 부과하여
        "짧고 굵게" 사용하도록 학습시킵니다.
        
        Args:
            weights: (Batch, Num_assets) 포트폴리오 비중
        
        Returns:
            penalty: 스칼라 페널티 값
        """
        # SH(인버스) 비중의 평균값에 비례하는 페널티
        inverse_weights = weights[:, self.inverse_etf_index]
        return inverse_weights.mean()
    
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
        # TODO: Role B가 CVaR 등으로 교체 가능
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
        # Term 4: Inverse ETF Decay Penalty
        # ======================================================================
        inverse_decay = self.compute_inverse_decay_penalty(weights)
        
        # ======================================================================
        # Total Loss
        # ======================================================================
        total_loss = (
            return_loss
            + self.eta * risk_penalty
            + turnover_cost
            + self.rho * inverse_decay
        )
        
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
    num_assets = 5  # Updated: SPY, XLV, TLT, GLD, SH
    
    # 더미 데이터
    weights = torch.softmax(torch.randn(batch_size, num_assets), dim=1)
    future_returns = torch.randn(batch_size, num_assets) * 0.02
    vix = torch.rand(batch_size) * 30 + 10  # VIX: 10 ~ 40 범위
    
    # TaskLoss 테스트 (Legacy)
    print("\n--- TaskLoss Test (Legacy) ---")
    task_loss_fn = TaskLoss(risk_lambda=1.0)
    task_loss = task_loss_fn(weights, future_returns)
    print(f"TaskLoss: {task_loss.item():.6f}")
    
    # DecisionAwareLoss 테스트 (v2 Main)
    print("\n--- DecisionAwareLoss Test (v2) ---")
    decision_loss_fn = DecisionAwareLoss(
        eta=1.0,
        kappa_base=0.001,
        kappa_vix_scale=0.0001,
        rho=0.01,
        inverse_etf_index=4  # SH
    )
    decision_loss = decision_loss_fn(weights, future_returns, vix)
    print(f"DecisionAwareLoss: {decision_loss.item():.6f}")
    
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
