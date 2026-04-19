"""
Loss Module
===========
[Step 4] 손실 함수 (Decision-Aware Loss)

역할: AI가 뱉은 비중(w)대로 투자했을 때, 결과가 좋았는지 나빴는지 판단.

주의: 예측 오차(MSE)가 아닙니다!

Loss = -(Return) + η × (Risk) + κ(VIX) × (Turnover) + λ_dd × (PathMDD)

- Return: 포트폴리오 수익률 (최대화 → 음수로 변환)
- Risk: 분산/CVaR (Role B가 확정)
- κ(VIX): 변동성 연동 거래비용 계수 (높은 VIX에서는 매매 자제)
- PathMDD: 전체 경로(fold) 수준의 최대 낙폭 패널티 (Triple 직접 타게팅)

v5 신규 추가:
- PathAwareMDDLoss: 배치 레벨이 아닌 전체 fold 경로의 MDD를 직접 최적화
  - Hinge Loss: MDD가 target(-10%)보다 나쁠 때만 패널티 부과
  - Top-K 에피소드 패널티: 다수의 드로다운 에피소드를 동시 억제 (whack-a-mole 방지)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionAwareLoss(nn.Module):
    """
    Decision-Aware Loss Function (v2)

    Loss = -Return + η × Risk + κ(VIX) × Turnover + λ_dd × DD

    ``risk_type`` options:
      - ``'std'``: portfolio-return standard deviation within the mini-batch.
      - ``'downside_deviation'`` (default): sqrt of mean squared negative
        deviation from the batch mean.
      - ``'batch_tail_mean'``: mean of the worst α=5% portfolio returns
        **within the current mini-batch**. This is NOT the distributional
        CVaR used by ``src/optimization.CVaROptimizationLayer`` (which is a
        Rockafellar-Uryasev LP over sampled scenarios). It is a per-batch
        tail-loss proxy that only converges to distributional CVaR in the
        limit of large, stationary, i.i.d. batches. Reported separately
        from optimizer-level CVaR to avoid confusion.
      - ``'cvar'``: deprecated alias for ``'batch_tail_mean'``; emits a
        DeprecationWarning on first use.
    """
    

    def __init__(
        self,
        eta: float = 1.0,
        kappa_base: float = 0.001,
        kappa_vix_scale: float = 0.0001,
        risk_type: str = 'downside_deviation',
        lambda_dd: float = 0.0,
        regime_dd_scale: float = 3.0,  # v4: Crisis에서 DD penalty ×(1+3*p_crisis)
        realized_cost_bps: float = 0.0,  # v5: 실제 거래비용을 수익률에서 직접 차감 (예: 10.0 = 10bps)
    ):
        super(DecisionAwareLoss, self).__init__()
        self.eta = eta
        self.kappa_base = kappa_base
        self.kappa_vix_scale = kappa_vix_scale
        self.risk_type = risk_type
        self.lambda_dd = lambda_dd
        self.regime_dd_scale = regime_dd_scale
        self.realized_cost_bps = realized_cost_bps / 10000.0  # bps → 소수점
    
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
        # v5: realized_cost_bps > 0이면 실제 거래비용을 수익률에서 직접 차감
        # ======================================================================
        portfolio_returns = (weights * future_returns).sum(dim=1)

        # 실현 거래비용 직접 차감 (Loss와 실제 Triple 평가 간 괴리 축소)
        if self.realized_cost_bps > 0.0 and prev_weights is not None:
            per_sample_turnover = torch.abs(weights - prev_weights).sum(dim=1)
            cost_deduction = self.realized_cost_bps * per_sample_turnover
            portfolio_returns = portfolio_returns - cost_deduction
        elif self.realized_cost_bps > 0.0:
            # prev_weights 없을 때: 균등 비중 기준 turnover
            eq_w = torch.ones_like(weights) / weights.shape[1]
            per_sample_turnover = torch.abs(weights - eq_w).sum(dim=1)
            cost_deduction = self.realized_cost_bps * per_sample_turnover
            portfolio_returns = portfolio_returns - cost_deduction

        return_loss = -portfolio_returns.mean()
        
        # ======================================================================
        # Term 2: Risk (선택 가능: std, downside_deviation, cvar)
        # ======================================================================
        if self.risk_type == 'std':
            # 1. Standard Deviation (변동성)
            risk_penalty = portfolio_returns.std() + 1e-8
            
        elif self.risk_type in ('batch_tail_mean', 'cvar'):
            # Batch tail-loss proxy (NOT distributional CVaR).
            # Mean of the worst α=5% portfolio returns inside the current
            # mini-batch. Only consistent with Rockafellar-Uryasev CVaR
            # in the large-batch, stationary, i.i.d. limit.
            if self.risk_type == 'cvar' and not getattr(
                    self, '_cvar_alias_warned', False):
                import warnings
                warnings.warn(
                    "risk_type='cvar' in DecisionAwareLoss is a batch tail-loss "
                    "proxy, not the distributional CVaR used by the optimizer. "
                    "Use risk_type='batch_tail_mean' to make this explicit.",
                    DeprecationWarning, stacklevel=2,
                )
                self._cvar_alias_warned = True

            alpha = 0.05
            k = int(batch_size * alpha)
            if k < 1:
                k = 1  # guarantee at least one sample

            sorted_returns, _ = torch.sort(portfolio_returns)
            worst_returns = sorted_returns[:k]
            risk_penalty = -worst_returns.mean()  # loss is negative → flip sign
            
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
        # Total Loss (v6: Regime-Aware Dynamic Weighting — Proposal C)
        # ======================================================================
        # regime_probs는 detach()로 gradient 차단:
        # Loss 스케일링 용도이지, regime classifier를 역으로 학습시키면 안 됨.
        # 
        # Scaling 설계:
        #   η (risk):     Crisis↑ → CVaR 패널티 강화 (방어 강화)
        #                 Bull↑   → CVaR 패널티 완화 (공격적 투자)
        #   κ (turnover): Crisis↑ → 거래비용 증가 (포지션 동결)  
        #   λ_dd:         Crisis↑ → DD 패널티 강화 (기존 v4 로직 유지)
        # ======================================================================
        
        effective_eta = self.eta
        effective_lambda_dd = self.lambda_dd
        
        if regime_probs is not None:
            # gradient 차단 — 순수 스케일링 목적
            rp = regime_probs.detach()
            p_crisis = rp[:, -1].mean()    # 마지막 열 = Crisis 확률
            p_bull   = rp[:, 0].mean()     # 첫 번째 열 = Bull 확률
            
            # --- η 동적 조절 ---
            # Crisis → η ×(1 + 2*p_crisis), Bull → η ×(1 - 0.3*p_bull)
            # 클리핑으로 안정성 보장 (0.5 ≤ effective_eta ≤ 3.0 * eta)
            eta_scale = 1.0 + 2.0 * p_crisis - 0.3 * p_bull
            effective_eta = self.eta * torch.clamp(eta_scale, 0.5, 3.0)
            
            # --- κ 동적 조절 (Crisis 시 turnover 추가 억제, implicit) ---
            # Crisis에서 turnover_cost를 1.5배로 스케일링
            turnover_cost = turnover_cost * (1.0 + 0.5 * p_crisis)
            
            # --- λ_dd 동적 조절 (기존 v4 로직 유지) ---
            if self.lambda_dd > 0:
                effective_lambda_dd = self.lambda_dd * (1 + self.regime_dd_scale * p_crisis)
        
        total_loss = (
            return_loss
            + effective_eta * risk_penalty
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
# PathAwareMDDLoss — 전체 경로(Fold) 수준의 MDD 직접 최적화 (v5 신규)
# =============================================================================

class PathAwareMDDLoss(nn.Module):
    """
    Path-Level Maximum Drawdown Loss.

    배치 레벨의 근사 drawdown이 아닌, Trainer에서 에폭 단위로 수집한
    **전체 fold 경로(시간순)**의 누적 수익률에서 MDD를 계산합니다.

    핵심 설계:
    - Hinge Loss: MDD가 목표치(mdd_target)보다 나쁠 때만 패널티. 목표 달성 후엔 0.
    - Top-K 에피소드 패널티: 최악 K개 시점의 drawdown을 동시 억제.
      → "COVID를 고치면 Inflation이 터지는" whack-a-mole 현상 방지.
    - Soft Margin: MDD가 목표보다 soft_margin만큼 더 나쁠 때부터 패널티 시작.
      → MDD = -8% 시점부터 -10% 목표를 향해 점진적 압박.

    Args:
        mdd_target (float): 목표 MDD (기본 -0.10 = -10%). 부호 주의 (음수).
        mdd_lambda (float): 전체 MDD 패널티 가중치.
        soft_margin (float): 사전 경고 마진. mdd_target보다 이 값만큼 더 낫을 때부터 패널티 시작.
                              예: soft_margin=0.02이면 -8%부터 시작.
        top_k (int): Top-K 에피소드 패널티에서 고려할 최악 시점 수.
        top_k_lambda (float): Top-K 에피소드 패널티 가중치 (mdd_lambda 대비 상대적).

    Usage (in Trainer):
        # 에폭 동안 포트폴리오 수익률을 시간순으로 수집
        epoch_returns = torch.cat(all_port_rets_in_order)  # (T,)
        path_loss = path_mdd_loss_fn(epoch_returns)
        path_loss.backward()
    """

    def __init__(
        self,
        mdd_target: float = -0.10,
        mdd_lambda: float = 5.0,
        soft_margin: float = 0.02,
        top_k: int = 5,
        top_k_lambda: float = 0.5,
    ):
        super(PathAwareMDDLoss, self).__init__()
        self.mdd_target = mdd_target          # -0.10 (음수)
        self.mdd_lambda = mdd_lambda
        self.soft_margin = soft_margin        # 0.02 → -8%부터 패널티 시작
        self.top_k = top_k
        self.top_k_lambda = top_k_lambda

    def forward(self, path_returns: torch.Tensor) -> torch.Tensor:
        """
        Args:
            path_returns: (T,) 시간순 포트폴리오 수익률 텐서.
                          T = fold 내 전체 학습 시점 수.

        Returns:
            loss: 스칼라 패널티 (MDD가 목표 이내이면 0).
        """
        if path_returns.shape[0] < 2:
            return torch.tensor(0.0, device=path_returns.device,
                                dtype=path_returns.dtype)

        # 누적 수익률 및 피크-투-트로프 계산
        cum_ret = torch.cumprod(1.0 + path_returns, dim=0)        # (T,)
        peak    = torch.cummax(cum_ret, dim=0).values              # (T,) running peak
        drawdowns = (peak - cum_ret) / (peak + 1e-8)              # (T,) ≥ 0

        # --- (1) 전체 경로 MDD Hinge Loss ---
        # drawdowns는 양수 (낙폭률). mdd_target은 음수 (-0.10).
        # 낙폭률 > (-mdd_target + soft_margin)이면 패널티 발생.
        mdd = drawdowns.max()
        threshold = -self.mdd_target + self.soft_margin   # 0.10 - 0.02 = 0.08
        hinge_mdd = F.relu(mdd - threshold)               # mdd > 0.08이면 양수

        # --- (2) Top-K 에피소드 패널티 ---
        # 상위 K개 최악 drawdown 시점에 대한 추가 패널티.
        # 단일 에피소드만 개선하고 다른 에피소드가 악화되는 현상 방지.
        k = min(self.top_k, drawdowns.shape[0])
        top_k_dd, _ = torch.topk(drawdowns, k, largest=True, sorted=True)
        # 각각에 대해 hinge: 목표치(soft_margin 없이 엄격하게 적용)
        top_k_threshold = -self.mdd_target                # 0.10
        top_k_hinge = F.relu(top_k_dd - top_k_threshold).mean()

        total_loss = self.mdd_lambda * hinge_mdd + self.mdd_lambda * self.top_k_lambda * top_k_hinge
        return total_loss


# =============================================================================
# SoftPathMDDLoss — Differentiable MDD via logsumexp (P1.4)
# =============================================================================

class SoftPathMDDLoss(nn.Module):
    """Fully differentiable path-MDD surrogate.

    ``PathAwareMDDLoss`` above relies on ``cummax`` and ``topk`` which have
    piecewise-zero subgradients: for most inputs only a single time index
    receives gradient. That is fine as a hinge-style regulariser but it
    produces a sparse, biased signal when MDD itself is the quantity we
    want gradients through (cf. P1.4 in the remediation plan).

    This class replaces both the running peak and the final max with a
    soft, everywhere-smooth logsumexp surrogate:

        soft_peak_t = (1/beta) * logsumexp(beta * log_cum_ret_{<=t})
        soft_dd_t   = log_cum_ret_t - soft_peak_t       (<= 0 always)
        soft_mdd    = -(1/beta) * logsumexp(-beta * soft_dd_t)

    All operations are differentiable and every time step contributes a
    non-zero gradient, with weight decaying smoothly away from the actual
    peak/trough. As ``beta -> inf`` the surrogate recovers the hard MDD.

    Working in *log-wealth* space makes the running peak additive and
    numerically stable; the final MDD is reported back as a simple-return
    drawdown via ``1 - exp(soft_dd)``.
    """

    def __init__(
        self,
        mdd_target: float = -0.10,
        mdd_lambda: float = 5.0,
        soft_margin: float = 0.02,
        beta: float = 40.0,
    ):
        super().__init__()
        self.mdd_target = mdd_target
        self.mdd_lambda = mdd_lambda
        self.soft_margin = soft_margin
        self.beta = beta

    def soft_mdd(self, path_returns: torch.Tensor) -> torch.Tensor:
        """Return a differentiable estimate of path MDD as a positive drawdown."""
        if path_returns.shape[0] < 2:
            return torch.zeros((), device=path_returns.device, dtype=path_returns.dtype)
        beta = self.beta
        # log-wealth: stable cumulative sum of log(1 + r)
        log_w = torch.cumsum(torch.log1p(path_returns), dim=0)  # (T,)

        # Soft running peak via cumulative logsumexp.
        # We use torch.logcumsumexp if available (PyTorch >= 1.8); fall back
        # otherwise. Running peak(t) = (1/beta) * logcumsumexp(beta * log_w)[t].
        if hasattr(torch, "logcumsumexp"):
            soft_peak = torch.logcumsumexp(beta * log_w, dim=0) / beta
        else:  # pragma: no cover — modern torch always has this
            cummax_stable = torch.cummax(beta * log_w, dim=0).values
            soft_peak = (
                cummax_stable
                + torch.log(
                    torch.cumsum(torch.exp(beta * log_w - cummax_stable), dim=0)
                )
            ) / beta

        soft_dd_log = log_w - soft_peak   # <= 0 for every t
        # Max of (-soft_dd_log) smoothly via logsumexp.
        soft_mdd_log = torch.logsumexp(-beta * soft_dd_log, dim=0) / beta
        # Back to simple-return space: drawdown = 1 - exp(soft_dd); soft_mdd_log
        # is the smoothed |soft_dd|, so return 1 - exp(-soft_mdd_log).
        return 1.0 - torch.exp(-soft_mdd_log)

    def forward(self, path_returns: torch.Tensor) -> torch.Tensor:
        mdd = self.soft_mdd(path_returns)
        threshold = -self.mdd_target + self.soft_margin  # e.g. 0.08
        hinge = F.relu(mdd - threshold)
        return self.mdd_lambda * hinge


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
                 regime_dd_scale: float = 3.0,
                 return_target_monthly: float = 0.0,
                 return_hinge_weight: float = 2.0):
        super().__init__()
        self.lambda_dd = lambda_dd
        self.kappa_base = kappa_base
        self.kappa_vix_scale = kappa_vix_scale
        self.regime_dd_scale = regime_dd_scale
        self.return_target_monthly = return_target_monthly  # 월 목표수익률 (0이면 비활성)
        self.return_hinge_weight = return_hinge_weight      # 미달 시 패널티 가중치
    
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
        
        # --- 4. Return Target Hinge (선택적) ---
        # 배치 평균 월수익률이 return_target_monthly에 미달하면 패널티 부과.
        # Sharpe 최대화와 병행해 return 하한을 직접 강제한다.
        # 연 10% ≈ 월 0.83% → return_target_monthly=0.0083 권장.
        return_hinge = torch.tensor(0.0, device=weights.device)
        if self.return_target_monthly > 0.0:
            return_hinge = F.relu(self.return_target_monthly - port_ret.mean())

        total = (sharpe_loss
                 + effective_lambda_dd * dd_penalty
                 + turnover_cost
                 + self.return_hinge_weight * return_hinge)

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
