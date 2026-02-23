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
        kappa_vix_scale: float = 0.0001,
        risk_type: str = 'downside_deviation',
        lambda_dd: float = 0.0,
        regime_dd_scale: float = 3.0,  # v4: Crisis에서 DD penalty ×(1+3*p_crisis)
    ):
        super(DecisionAwareLoss, self).__init__()
        self.eta = eta
        self.kappa_base = kappa_base
        self.kappa_vix_scale = kappa_vix_scale
        self.risk_type = risk_type
        self.lambda_dd = lambda_dd
        self.regime_dd_scale = regime_dd_scale
    
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
        prev_weights: torch.Tensor = None,
        regime_probs: torch.Tensor = None,
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
        # Term 2: Risk (선택 가능: std, downside_deviation, cvar)
        # ======================================================================
        if self.risk_type == 'std':
            # 1. Standard Deviation (변동성)
            risk_penalty = portfolio_returns.std() + 1e-8
            
        elif self.risk_type == 'cvar':
            # 2. CVaR (Conditional Value at Risk) - 95%
            # 배치 내 수익률을 정렬하여 하위 5%의 평균 손실 계산
            alpha = 0.05
            k = int(batch_size * alpha)
            if k < 1: k = 1 # 최소 1개 샘플 보장
            
            sorted_returns, _ = torch.sort(portfolio_returns)
            worst_returns = sorted_returns[:k]
            cvar = -worst_returns.mean() # 손실이므로 음수 -> 양수로 변환
            risk_penalty = cvar
            
        else: # 'downside_deviation' (Default)
            # 3. Downside Deviation (하방 편차)
            # 수익률 평균보다 낮은 경우(Downside)만 리스크로 간주
            mean_return = portfolio_returns.mean()
            downside_diff = portfolio_returns - mean_return
            # 양수(수익이 평균보다 높음)는 0으로 처리, 음수(손실)는 제곱
            downside_risk = torch.clamp(downside_diff, max=0.0) ** 2
            risk_penalty = torch.sqrt(downside_risk.mean() + 1e-8)
        
        
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
        # Term 4: Drawdown Penalty (배치 내 시계열 기반)
        # ======================================================================
        dd_penalty = torch.tensor(0.0, device=weights.device)
        if self.lambda_dd > 0 and batch_size > 1:
            # 배치가 시간 순서(TrajectoryBatchSampler)일 때 유효
            cum_ret = torch.cumprod(1 + portfolio_returns, dim=0)
            peak = torch.cummax(cum_ret, dim=0).values
            drawdowns = (peak - cum_ret) / (peak + 1e-8)
            dd_penalty = drawdowns.max()
        
        # ======================================================================
        # Total Loss (v4: regime-conditional λ_dd)
        # ======================================================================
        effective_lambda_dd = self.lambda_dd
        if regime_probs is not None and self.lambda_dd > 0:
            p_crisis = regime_probs[:, -1].mean()  # batch 평균 crisis 확률
            effective_lambda_dd = self.lambda_dd * (1 + self.regime_dd_scale * p_crisis)
        
        total_loss = (
            return_loss
            + self.eta * risk_penalty
            + turnover_cost
            + effective_lambda_dd * dd_penalty
        )
        
        return total_loss


# =============================================================================
# Phase 2 — RegimeAwareLoss
# =============================================================================

class RegimeAwareLoss(nn.Module):
    """
    Phase 2 손실 함수: Regime-Aware Decision Loss.
    
    Loss = BaseLoss + λ₁·Stability + λ₃·KL(HMM‖Neural)
    
    - BaseLoss: DecisionAwareLoss (수익률, 리스크, 거래비용)
    - Stability: 연속 시점 regime 급변 방지
    - KL: HMM Teacher의 regime 분류 모방 (cosine annealing)
    
    Args:
        base_loss: DecisionAwareLoss 인스턴스
        lambda_stability: regime 안정성 페널티 가중치
        lambda_entropy: entropy 정규화 (기본 비활성화)
        lambda_kl_max: KL 가중치 최대값 (학습 초기)
        lambda_kl_min: KL 가중치 최소값 (학습 후기)
    """
    def __init__(
        self,
        base_loss: DecisionAwareLoss,
        lambda_stability: float = 0.1,
        lambda_entropy: float = 0.0,
        lambda_kl_max: float = 1.0,
        lambda_kl_min: float = 0.05,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_stability = lambda_stability
        self.lambda_entropy = lambda_entropy
        self.lambda_kl_max = lambda_kl_max
        self.lambda_kl_min = lambda_kl_min
        self._current_lambda_kl = lambda_kl_max
    
    def update_lambda_kl(self, epoch: int, total_epochs: int):
        """Cosine annealing: λ_KL = max → min"""
        if total_epochs <= 0:
            return
        import math
        progress = min(epoch / total_epochs, 1.0)
        self._current_lambda_kl = (
            self.lambda_kl_min
            + (self.lambda_kl_max - self.lambda_kl_min)
            * 0.5 * (1.0 + math.cos(math.pi * progress))
        )
    
    def forward(
        self,
        weights: torch.Tensor,
        future_returns: torch.Tensor,
        vix: torch.Tensor,
        regime_probs: torch.Tensor,
        hmm_probs: torch.Tensor,
        prev_regime_probs: torch.Tensor = None,
        prev_weights: torch.Tensor = None,
    ):
        """
        Args:
            weights: (B, N) 포트폴리오 비중
            future_returns: (B, N) 실현 수익률
            vix: (B,) VIX 레벨
            regime_probs: (B, K) Neural regime 확률
            hmm_probs: (B, K) HMM Teacher regime 확률
            prev_regime_probs: (B, K) 이전 batch의 regime 확률
            prev_weights: (B, N) 이전 포트폴리오 비중
            
        Returns:
            total_loss: 스칼라 손실
            details: dict (각 항의 값, 디버깅용)
        """
        device = weights.device
        
        # Term 1: Base Loss (수익률 + 리스크 + 거래비용)
        base = self.base_loss(weights, future_returns, vix, prev_weights)
        
        # Term 2: Regime Stability (연속 시점 급변 방지)
        if prev_regime_probs is not None:
            stability = ((regime_probs - prev_regime_probs) ** 2).sum(dim=1).mean()
        else:
            stability = torch.tensor(0.0, device=device)
        
        # Term 3: KL(HMM ‖ Neural) — Teacher-Student
        # PyTorch F.kl_div: input=log(student), target=teacher
        # = Σ teacher * (log(teacher) - log(student))
        # = KL(teacher ‖ student) = KL(HMM ‖ Neural) ✓
        kl_loss = torch.nn.functional.kl_div(
            torch.log(regime_probs + 1e-8),
            hmm_probs,
            reduction='batchmean',
        )
        
        # (Optional) Entropy — 기본 비활성화
        entropy_loss = torch.tensor(0.0, device=device)
        if self.lambda_entropy > 0:
            batch_mean = regime_probs.mean(dim=0)
            H_batch = -(batch_mean * torch.log(batch_mean + 1e-8)).sum()
            H_sample = -(regime_probs * torch.log(regime_probs + 1e-8)).sum(dim=1).mean()
            entropy_loss = -H_batch + H_sample
        
        total = (
            base
            + self.lambda_stability * stability
            + self.lambda_entropy * entropy_loss
            + self._current_lambda_kl * kl_loss
        )
        
        details = {
            'base': base.item(),
            'stability': stability.item(),
            'kl': kl_loss.item(),
            'entropy': entropy_loss.item(),
            'lambda_kl': self._current_lambda_kl,
            'total': total.item(),
        }
        
        return total, details

# =============================================================================
# SharpeLoss — Direct Sharpe Maximization (v3)
# =============================================================================

class SharpeLoss(nn.Module):
    """
    Direct Sharpe Ratio Maximization Loss.
    
    Loss = -Sharpe(w, r) + λ_dd · MaxDrawdown(w, r) + κ(VIX) · Turnover
    
    핵심 차이:
        - DecisionAwareLoss: -Return + η·Risk (수익/위험 별도 최적화)
        - SharpeLoss: -(Return/Risk) (Sharpe 비율 직접 최대화)
    
    Args:
        lambda_dd: Drawdown penalty 가중치
        kappa_base: 거래비용 기본 계수
        kappa_vix_scale: VIX 연동 거래비용
        regime_dd_scale: Crisis regime에서 lambda_dd 스케일링 계수
    """
    def __init__(self, lambda_dd: float = 1.0, 
                 kappa_base: float = 0.001,
                 kappa_vix_scale: float = 0.0001,
                 regime_dd_scale: float = 3.0):
        super().__init__()
        self.lambda_dd = lambda_dd
        self.kappa_base = kappa_base
        self.kappa_vix_scale = kappa_vix_scale
        self.regime_dd_scale = regime_dd_scale
    
    def forward(self, weights, returns, vix=None, 
                prev_weights=None, regime_probs=None):
        """
        Args:
            weights: (B, N) 포트폴리오 비중
            returns: (B, N) 자산 수익률
            vix: (B,) VIX 값 (optional)
            prev_weights: (B, N) 이전 비중 (optional, turnover용)
            regime_probs: (B, R) regime 확률 (optional, 조건부 DD 가중)
        
        Returns:
            loss: scalar
        """
        # Portfolio returns
        port_ret = (weights * returns).sum(dim=1)  # (B,)
        
        # --- 1. Sharpe Loss ---
        if port_ret.shape[0] < 2:
            # batch_size=1이면 std 계산 불가 → Return loss로 fallback
            sharpe_loss = -port_ret.mean()
        else:
            mean_ret = port_ret.mean()
            std_ret = port_ret.std(unbiased=False) + 1e-8
            sharpe_loss = -(mean_ret / std_ret)
        
        # --- 2. Drawdown Penalty ---
        dd_penalty = self._drawdown_penalty(port_ret)
        
        # Regime-conditional: Crisis 확률이 높으면 DD 페널티 강화
        effective_lambda_dd = self.lambda_dd
        if regime_probs is not None:
            p_crisis = regime_probs[:, -1].mean()  # 평균 crisis 확률
            effective_lambda_dd = self.lambda_dd * (1 + self.regime_dd_scale * p_crisis)
        
        # --- 3. Turnover Cost ---
        turnover_cost = torch.tensor(0.0, device=weights.device)
        if prev_weights is not None and vix is not None:
            kappa = self.kappa_base + self.kappa_vix_scale * vix
            turnover = torch.abs(weights - prev_weights).sum(dim=1)
            turnover_cost = (kappa * turnover).mean()
        elif vix is not None:
            eq_weights = torch.ones_like(weights) / weights.shape[1]
            kappa = self.kappa_base + self.kappa_vix_scale * vix
            turnover = torch.abs(weights - eq_weights).sum(dim=1)
            turnover_cost = (kappa * turnover).mean()
        
        total = sharpe_loss + effective_lambda_dd * dd_penalty + turnover_cost
        
        return total
    
    def _drawdown_penalty(self, port_ret):
        """배치 내 최대 drawdown 계산 (미분 가능)."""
        if port_ret.shape[0] < 2:
            return torch.tensor(0.0, device=port_ret.device)
        
        cum = torch.cumprod(1 + port_ret, dim=0)
        peak = torch.cummax(cum, dim=0).values
        drawdown = (peak - cum) / (peak + 1e-8)
        
        return drawdown.max()


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
