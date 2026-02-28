"""
Regime Module (Phase 2)
=======================
Gumbel-Softmax 기반 Regime Classification Head.

HMM은 미분 불가능 → Gumbel-Softmax로 미분 가능한 이산 분류 근사.
HMM의 regime 분류 능력을 KL divergence prior로 보존하면서,
포트폴리오 목적함수(Mean-CVaR)에 맞게 미세 조정.

Teacher-Student Framework:
  Teacher: 준상님의 HMM (오프라인, get_regime_proba())
  Student: 이 모듈의 RegimeHead (온라인, end-to-end 학습)
  연결: KL(HMM ‖ Neural) loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimeHead(nn.Module):
    """
    Gumbel-Softmax regime classifier.
    
    HMM prob은 이 모듈의 Input에 포함하지 않음 (Shortcut Learning 방지).
    HMM prob은 Loss 계산 시 KL target으로만 사용됨.
    
    Args:
        hidden_dim: GRU hidden dimension
        n_regimes: regime 수 (default: 3, Bull/Uncertain/Crisis)
        macro_dim: 매크로 피처 차원 (default: 0, 사용 안 함)
            > 0이면 hidden + macro를 concat하여 regime 분류
        tau_init: 초기 Gumbel temperature
        tau_min: 최소 temperature (cosine annealing의 floor)
        hmm_prior_mean: HMM 평균 regime 확률 (bias 초기화용)
            None이면 [0.6, 0.3, 0.1]로 초기화
    """
    def __init__(
        self,
        hidden_dim: int,
        n_regimes: int = 3,
        macro_dim: int = 0,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        hmm_prior_mean: torch.Tensor = None,
    ):
        super().__init__()
        self.n_regimes = n_regimes
        self.macro_dim = macro_dim
        self.tau = tau_init
        self.tau_init = tau_init
        self.tau_min = tau_min
        
        # macro_dim > 0이면 hidden + macro concat → regime logits
        self.regime_logits = nn.Linear(hidden_dim + macro_dim, n_regimes)
        
        # Label Switching 방지: HMM 사전 확률로 bias 초기화
        # idx 0 = Bull (최빈), idx 1 = Uncertain, idx 2 = Crisis (최희귀)
        with torch.no_grad():
            if hmm_prior_mean is not None:
                self.regime_logits.bias.copy_(
                    torch.log(hmm_prior_mean.float() + 1e-8))
            else:
                # Dynamic default priors based on n_regimes
                if n_regimes == 3:
                    default_priors = torch.tensor([0.6, 0.3, 0.1])
                elif n_regimes == 4:
                    default_priors = torch.tensor([0.4, 0.3, 0.2, 0.1])
                else:
                    # Decreasing geometric priors for arbitrary n_regimes
                    priors = torch.arange(n_regimes, 0, -1, dtype=torch.float32)
                    default_priors = priors / priors.sum()
                self.regime_logits.bias.copy_(torch.log(default_priors))
    
    def forward(self, hidden: torch.Tensor, macro_features: torch.Tensor = None,
                hard: bool = False) -> torch.Tensor:
        """
        Args:
            hidden: (B, hidden_dim) GRU 출력
            macro_features: (B, macro_dim) 매크로 피처 (optional)
            hard: True면 argmax (inference용), False면 soft (train용)
            
        Returns:
            regime_probs: (B, n_regimes)
        """
        if macro_features is not None and self.macro_dim > 0:
            hidden = torch.cat([hidden, macro_features], dim=-1)
        logits = self.regime_logits(hidden)
        
        if self.training:
            return F.gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1)
        else:
            return F.softmax(logits / max(self.tau, 1e-8), dim=-1)
    
    def anneal_tau(self, epoch: int, total_epochs: int):
        """
        Cosine annealing for Gumbel temperature.
        
        τ=1.0 → τ=0.1: 초기 탐색 → 후기 확정
        """
        if total_epochs <= 0:
            return
        progress = min(epoch / total_epochs, 1.0)
        self.tau = self.tau_min + (self.tau_init - self.tau_min) * \
                   0.5 * (1.0 + math.cos(math.pi * progress))
    
    def get_lambda_risk(
        self,
        regime_probs: torch.Tensor,
        lambda_anchors: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Regime 확률에서 위험회피 계수 λ를 산출.
        
        Bull → λ=0.5 (공격적), Uncertain → λ=1.0, Crisis → λ=2.0 (보수적)
        Soft interpolation: λ = Σ_k r_k · λ_k
        
        Args:
            regime_probs: (B, n_regimes)
            lambda_anchors: (n_regimes,) 각 regime의 λ 앵커값
            
        Returns:
            lambda_risk: (B,) 배치별 위험회피 계수
        """
        if lambda_anchors is None:
            lambda_anchors = torch.tensor(
                [0.5, 1.0, 2.0], device=regime_probs.device)
        
        return (regime_probs * lambda_anchors.unsqueeze(0)).sum(dim=1)
    
    def get_is_crisis(self, regime_probs: torch.Tensor, 
                       threshold: float = 0.5) -> torch.Tensor:
        """
        Crisis regime 확률에서 안전망 플래그 유도.
        
        Args:
            regime_probs: (B, n_regimes)
            threshold: Crisis 판정 임계값
            
        Returns:
            is_crisis: (B,) float tensor (0.0 or 1.0)
        """
        # idx 2 = Crisis (bias init 기준)
        return (regime_probs[:, -1] > threshold).float()


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RegimeHead Self-Test")
    print("=" * 60)
    
    # --- Test 1: 기본 (macro_dim=0) ---
    print("\n--- Test 1: macro_dim=0 (기존 호환) ---")
    head = RegimeHead(hidden_dim=32, n_regimes=3)
    print(f"Parameters: {sum(p.numel() for p in head.parameters())}")
    print(f"Bias init: {head.regime_logits.bias.data.exp()}")
    
    hidden = torch.randn(8, 32)
    head.train()
    probs = head(hidden)
    print(f"[Train] Output shape: {probs.shape}")
    print(f"[Train] Sum per sample: {probs.sum(dim=1)}")
    assert probs.shape == (8, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)
    
    head.eval()
    probs_eval = head(hidden)
    print(f"[Eval]  Sum per sample: {probs_eval.sum(dim=1)}")
    assert torch.allclose(probs_eval.sum(dim=1), torch.ones(8), atol=1e-5)
    print("✓ macro_dim=0 passed")
    
    # --- Test 2: macro_dim=2 (T10Y3M + BAA10Y) ---
    print("\n--- Test 2: macro_dim=2 (매크로 피처 통합) ---")
    head_macro = RegimeHead(hidden_dim=32, n_regimes=3, macro_dim=2)
    params_macro = sum(p.numel() for p in head_macro.parameters())
    print(f"Parameters: {params_macro} (should be > {sum(p.numel() for p in head.parameters())})")
    assert head_macro.regime_logits.in_features == 34  # 32 + 2
    
    macro = torch.randn(8, 2)
    head_macro.train()
    probs_macro = head_macro(hidden, macro_features=macro)
    print(f"[Train] Output shape: {probs_macro.shape}")
    assert probs_macro.shape == (8, 3)
    assert torch.allclose(probs_macro.sum(dim=1), torch.ones(8), atol=1e-5)
    
    head_macro.eval()
    probs_macro_eval = head_macro(hidden, macro_features=macro)
    assert torch.allclose(probs_macro_eval.sum(dim=1), torch.ones(8), atol=1e-5)
    print("✓ macro_dim=2 passed")
    
    # Temperature annealing
    print(f"\n--- Temperature Annealing ---")
    for epoch in [0, 25, 50, 75, 100]:
        head.anneal_tau(epoch, 100)
        print(f"  Epoch {epoch:3d}: τ = {head.tau:.4f}")
    
    # Lambda risk
    head.train()
    probs = head(hidden)
    lambda_risk = head.get_lambda_risk(probs)
    print(f"\n--- Lambda Risk ---")
    print(f"  Shape: {lambda_risk.shape}, Range: [{lambda_risk.min():.3f}, {lambda_risk.max():.3f}]")
    
    # Is crisis
    is_crisis = head.get_is_crisis(probs)
    print(f"  is_crisis: {is_crisis}")
    
    # Gradient flow
    head.train()
    probs = head(hidden)
    loss = probs.sum()
    loss.backward()
    assert head.regime_logits.weight.grad is not None
    print(f"\nGradient flow: ✓")
    
    print("\n[SUCCESS] All RegimeHead tests passed!")
