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
from torch.distributions import StudentT

CHOLESKY_JITTER = 1e-4

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
        safety_threshold: float = 0.5,
        dist_type: str = 't',
        t_df: int = 5,
        group_masks=None,
        group_caps=None,
        group_floors=None,
    ):
        """
        Group constraints (P2.1)
        ------------------------
        Previously equity / bond / alternatives caps were applied by the
        Phase-18 overlay *after* the CVaR solve, which produced a
        non-homomorphic policy (upstream ignored constraints the downstream
        then enforced by re-weighting). Internalising the caps makes the
        CVaR solution constraint-consistent.

        ``group_masks`` : (K, N) 0/1 numpy array. Row k selects assets in
            group k (e.g. equities, bonds, alternatives).
        ``group_caps``  : (K,) upper bounds on group weights. ``None`` to
            disable the upper bound on that group.
        ``group_floors``: (K,) lower bounds; ``None`` to disable.
        """
        super(CVaROptimizationLayer, self).__init__()
        self.num_assets = num_assets
        self.num_scenarios = num_scenarios
        self.beta = confidence_level
        self.bil_index = bil_index
        self.safety_threshold = safety_threshold
        self.dist_type = dist_type
        self.t_df = t_df
        self._fallback_count = 0

        self.group_masks = None
        self.group_caps = None
        self.group_floors = None
        if group_masks is not None:
            gm = np.asarray(group_masks, dtype=float)
            if gm.ndim != 2 or gm.shape[1] != num_assets:
                raise ValueError(
                    f"group_masks must have shape (K, {num_assets}); got {gm.shape}"
                )
            if ((gm != 0.0) & (gm != 1.0)).any():
                raise ValueError("group_masks must be 0/1")
            self.group_masks = gm
            K = gm.shape[0]
            if group_caps is not None:
                gc = np.asarray(group_caps, dtype=float)
                if gc.shape != (K,):
                    raise ValueError(f"group_caps must have shape ({K},)")
                self.group_caps = gc
            if group_floors is not None:
                gf = np.asarray(group_floors, dtype=float)
                if gf.shape != (K,):
                    raise ValueError(f"group_floors must have shape ({K},)")
                self.group_floors = gf
        
        self.cvxpy_layer = None
        
        if CVXPY_AVAILABLE:
            self._build_cvar_layer()
            print(f"[INFO] CVaR Layer initialized: {num_assets} assets, {num_scenarios} scenarios, dist={dist_type}(df={t_df})")

    def _sample_noise(self, shape, device):
        """시나리오 노이즈 샘플링: Normal 또는 Student-T."""
        if self.dist_type == 't':
            t_dist = StudentT(df=self.t_df)
            eps = t_dist.sample(shape).to(device)
            # t(df)의 분산 = df/(df-2) → 정규화하여 단위 분산 유지
            if self.t_df > 2:
                eps = eps / (self.t_df / (self.t_df - 2)) ** 0.5
        else:
            eps = torch.randn(*shape, device=device)
        return eps
            
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

        # P2.1: group constraints (constants, so DPP is preserved).
        if self.group_masks is not None:
            A = self.group_masks  # (K, N)
            if self.group_caps is not None:
                constraints.append(A @ w <= self.group_caps)
            if self.group_floors is not None:
                constraints.append(A @ w >= self.group_floors)

        # 문제 및 레이어 생성
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        
        self.cvxpy_layer = CvxpyLayer(
            problem, 
            parameters=[R_scenarios, is_crisis], 
            variables=[w]
        )
        print("[INFO] CVaR 최적화 레이어 구축 완료 (안전망 포함)")

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, 
                is_crisis: float = 0.0, lambda_risk: float = 0.0) -> torch.Tensor:
        """
        Forward Pass:
        1. Reparameterization Trick을 사용하여 N(mu, Sigma)에서 시나리오 샘플링
        2. (선택) Mean-CVaR 시나리오 사전조정: R_adj = R + μ/λ
        3. cvxpylayers를 통해 CVaR 최적화 수행
        
        Args:
            mu: (Batch, N) 기대 수익률
            sigma: (Batch, N, N) 공분산 행렬
            is_crisis: (Batch, ) 또는 스칼라. 1.0이면 안전망 가동.
            lambda_risk: 위험회피 계수. > 0이면 Mean-CVaR 적용.
                         클수록 → 위험 회피 (CVaR 중심)
                         작을수록 → 공격적 (수익 중심)
        
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
        jit = CHOLESKY_JITTER * torch.eye(self.num_assets, device=device).expand(batch_size, -1, -1)
        try:
            L = torch.linalg.cholesky(sigma + jit)
        except RuntimeError:
            # PSD가 아닐 경우 대각 행렬 근사 사용
            L = torch.diag_embed(torch.sqrt(torch.diagonal(sigma, dim1=1, dim2=2).abs() + 1e-6))
            
        eps = self._sample_noise((batch_size, self.num_scenarios, self.num_assets), device)  # (B, S, N)
        
        mu_expanded = mu.unsqueeze(1)
        
        # 시나리오 생성: R = mu + eps @ L^T
        scenarios = mu_expanded + torch.bmm(eps, L.transpose(1, 2))
        
        # 2. Mean-CVaR 시나리오 사전조정 (lambda_risk > 0일 때)
        # R_adj = R + μ/λ → min CVaR(-R_adj·w) ≡ max w'μ - λ·CVaR(w)
        if lambda_risk > 0:
            mu_bonus = mu_expanded / max(lambda_risk, 0.1)
            scenarios = scenarios + mu_bonus
        
        # 3. 최적화 실행
        if isinstance(is_crisis, float) or isinstance(is_crisis, int):
            crisis_param = torch.full((batch_size,), float(is_crisis), device=device)
        else:
            crisis_param = is_crisis 
            
        try:
            # cvxpylayers는 CPU 텐서만 지원 — GPU에서 호출 시 CPU로 이동 후 복원
            if scenarios.is_cuda:
                weights_cpu, = self.cvxpy_layer(
                    scenarios.cpu(), crisis_param.cpu())
                weights = weights_cpu.to(device)
            else:
                weights, = self.cvxpy_layer(scenarios, crisis_param)
        except Exception as e:
            # 최적화 실패 시 안전한 폴백: tempered softmax + equal-weight 블렌드
            self._fallback_count += 1
            if self._fallback_count & (self._fallback_count - 1) == 0:  # 1, 2, 4, 8, ... 회차만 로그
                print(f"  [CVaR FALLBACK #{self._fallback_count}] {str(e)[:80]}")
            
            equal_w = torch.ones_like(mu) / mu.size(1)
            soft_w = torch.softmax(mu / 0.1, dim=1)  # temperature=0.1으로 평탄화
            weights = 0.5 * equal_w + 0.5 * soft_w
        
        # 비중 안전 장치: 음수 방지 + sum=1 보장
        weights = torch.clamp(weights, min=0.0)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            
        return weights

    def get_fallback_stats(self) -> int:
        return self._fallback_count


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


# =============================================================================
# Phase 2 — Mean-CVaR Optimization Layer
# =============================================================================

class MeanCVaROptimizationLayer(nn.Module):
    r"""
    Mean-CVaR 최적화 레이어 (Phase 2).
    
    원래 목적: Maximize  w^T μ_BL - λ · CVaR_α(w)
    
    **구현 전략**: `lambda * f(w)` 혼합은 diffcp 배치 모드와 호환되지 않으므로,
    시나리오를 사전 조정하여 **순수 CVaR 최소화**로 변환:
    
      min_w CVaR_α(-R·w)   where  R_adj[s] = R[s] + (1/λ)·μ_BL
    
    수학적으로:
      min CVaR(-R_adj · w)
      = min CVaR(-(R + (1/λ)μ) · w)
      = min CVaR(-R·w - (1/λ)μ^T w)
    
    μ^T w가 시나리오 무관 상수이므로, 이것은 μ^T w를 최대화하면서 
    CVaR를 최소화하는 것과 동치.
    
    장점:
    - Phase 1의 검증된 CVaR 솔버 구조 재사용 → diffcp 호환 보장
    - λ가 시나리오 스케일링에 흡수 → 목적함수에 Parameter*Variable 없음
    - is_crisis 안전망 제약 그대로 유지
    """
    def __init__(
        self,
        num_assets: int,
        num_scenarios: int = 200,
        confidence_level: float = 0.95,
        bil_index: int = 4,
        safety_threshold: float = 0.5,
        dist_type: str = 't',
        t_df: int = 5,
    ):
        super().__init__()
        self.num_assets = num_assets
        self.num_scenarios = num_scenarios
        self.beta = confidence_level
        self.bil_index = bil_index
        self.safety_threshold = safety_threshold
        self.dist_type = dist_type
        self.t_df = t_df
        self._fallback_count = 0
        
        self.cvxpy_layer = None
        
        if CVXPY_AVAILABLE:
            self._build_mean_cvar_layer()
    
    def _sample_noise(self, shape, device):
        """시나리오 노이즈 샘플링: Normal 또는 Student-T."""
        if self.dist_type == 't':
            t_dist = StudentT(df=self.t_df)
            eps = t_dist.sample(shape).to(device)
            # t(df)의 분산 = df/(df-2) → 정규화하여 단위 분산 유지
            if self.t_df > 2:
                eps = eps / (self.t_df / (self.t_df - 2)) ** 0.5
            else: # For df <= 2, variance is undefined or infinite. Use as is or handle differently.
                pass
        else: # Default to normal distribution
            eps = torch.randn(*shape, device=device)
        return eps

    def _build_mean_cvar_layer(self):
        """
        순수 CVaR 최소화 CvxpyLayer 구축.
        
        Phase 1의 CVaROptimizationLayer와 동일한 구조.
        Mean-CVaR 트레이드오프는 시나리오 사전 조정으로 처리.
        """
        n = self.num_assets
        s = self.num_scenarios
        
        # 변수
        w = cp.Variable(n)
        alpha = cp.Variable()
        u = cp.Variable(s)
        
        # 파라미터: 조정된 시나리오 + 안전망
        R_scenarios = cp.Parameter((s, n))
        is_crisis = cp.Parameter()
        
        # 목적함수: 순수 CVaR 최소화 (Phase 1과 동일)
        cvar_term = alpha + (1.0 / (s * (1.0 - self.beta))) * cp.sum(u)
        objective = cp.Minimize(cvar_term)
        
        # 제약조건
        loss = -R_scenarios @ w
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            u >= loss - alpha,
            u >= 0.0,
            w[self.bil_index] >= self.safety_threshold * is_crisis,
        ]
        
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp(), "Problem is not DPP-compliant!"
        
        self.cvxpy_layer = CvxpyLayer(
            problem,
            parameters=[R_scenarios, is_crisis],
            variables=[w],
        )
        print("[INFO] Mean-CVaR 레이어 구축 완료 (시나리오 사전조정 방식)")
    
    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        lambda_risk: torch.Tensor = None,
        is_crisis: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward Pass:
        1. N(μ, Σ)에서 시나리오 샘플링
        2. 시나리오 사전 조정: R_adj = R + (1/λ)·μ
        3. CVaR 최소화 (조정된 시나리오)
        
        동치: max w'μ - λ·CVaR(w)
        
        Args:
            mu: (B, N) BL 기대수익률
            sigma: (B, N, N) 공분산 행렬
            lambda_risk: (B,) regime-conditioned 위험회피 계수
            is_crisis: (B,) 안전망 플래그
            
        Returns:
            weights: (B, N) 최적 포트폴리오 비중
        """
        batch_size = mu.size(0)
        device = mu.device
        
        if lambda_risk is None:
            lambda_risk = torch.ones(batch_size, device=device)
        if is_crisis is None:
            is_crisis = torch.zeros(batch_size, device=device)
        
        if not CVXPY_AVAILABLE or self.cvxpy_layer is None:
            return torch.softmax(mu, dim=1)
        
        # 1. 시나리오 샘플링 (Reparameterization Trick)
        jit = CHOLESKY_JITTER * torch.eye(self.num_assets, device=device).expand(batch_size, -1, -1)
        try:
            L = torch.linalg.cholesky(sigma + jit)
        except RuntimeError:
            L = torch.diag_embed(
                torch.sqrt(torch.diagonal(sigma, dim1=1, dim2=2).abs() + 1e-6))
        
        eps = self._sample_noise((batch_size, self.num_scenarios, self.num_assets), device)
        scenarios = mu.unsqueeze(1) + torch.bmm(eps, L.transpose(1, 2))  # (B, S, N)
        
        # 2. 시나리오 사전 조정: R_adj = R + (1/λ)·μ
        # λ가 클수록 → (1/λ)μ 작음 → 시나리오 ≈ 원본 → 위험 회피 강조
        # λ가 작을수록 → (1/λ)μ 큼 → 시나리오에 수익 보너스 추가 → 공격적
        mu_bonus = mu.unsqueeze(1) / lambda_risk.unsqueeze(1).unsqueeze(2).clamp(min=0.1)
        adjusted_scenarios = scenarios + mu_bonus  # (B, S, N)
        
        # 3. CVaR 최소화 (배치 모드, Phase 1과 동일 구조)
        if isinstance(is_crisis, float) or isinstance(is_crisis, int):
            crisis_param = torch.full((batch_size,), float(is_crisis), device=device)
        else:
            crisis_param = is_crisis
        
        try:
            if adjusted_scenarios.is_cuda:
                weights_cpu, = self.cvxpy_layer(
                    adjusted_scenarios.cpu(), crisis_param.cpu())
                weights = weights_cpu.to(device)
            else:
                weights, = self.cvxpy_layer(adjusted_scenarios, crisis_param)
        except Exception as e:
            # 안전한 폴백: tempered softmax + equal-weight 블렌드
            self._fallback_count += 1
            if self._fallback_count & (self._fallback_count - 1) == 0:
                print(f"  [MeanCVaR FALLBACK #{self._fallback_count}] {str(e)[:80]}")
            
            equal_w = torch.ones_like(mu) / mu.size(1)
            soft_w = torch.softmax(mu / 0.1, dim=1)
            weights = 0.5 * equal_w + 0.5 * soft_w
        
        # 비중 안전 장치
        weights = torch.clamp(weights, min=0.0)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return weights

    def get_fallback_stats(self) -> int:
        return self._fallback_count
