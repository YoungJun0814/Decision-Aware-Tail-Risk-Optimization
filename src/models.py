"""
Models Module (v2.2)
====================
[2단계] Black-Litterman 통합 딥러닝 모델

업데이트 내역 (v2.2):
- 완전한 Black-Litterman 공식 구현 (P, Q, Omega 행렬 연산)
- 완전한 TFT 아키텍처 구현 (변수 선택 네트워크 포함)
- CVaR 최적화 레이어 통합 (위기 시 안전자산 강제)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

BL_SOLVER_JITTER = 1e-4

# 최적화 레이어 및 헬퍼 함수 임포트
from src.optimization import CVaROptimizationLayer, estimate_covariance

# =============================================================================
# 1. 기본 클래스 (Black-Litterman 로직)
# =============================================================================

class BaseBLModel(nn.Module):
    """
    모든 모델의 공통 부모 클래스입니다.
    구조: Encoder -> BL Views(전망) -> BL Formula(수식) -> CVaR Optimization(최적화) -> Weights(비중)
    
    Args:
        input_dim: 입력 피처 차원
        num_assets: 자산 수
        hidden_dim: 은닉 차원
        dropout: 드롭아웃 비율
        omega_mode: Omega 산출 방식 ('learnable', 'formula', 'hybrid')
            - 'learnable': 기존 방식 (신경망이 직접 출력, hidden_dim × N 파라미터)
            - 'formula': 수식 기반 (Ω_ii = τ · p_i² · Σ_ii, 파라미터 0개)
            - 'hybrid': 수식 + 학습 가능 스케일링 (Ω_ii = α_i · τ · p_i² · Σ_ii, N개 파라미터)
        sigma_mode: Sigma 반환 방식 ('prior', 'residual')
            - 'prior': 사전 공분산(sigma) 그대로 반환 (원기님 방식, 그래디언트 단절)
            - 'residual': sigma + λ * (sigma_bl - sigma.detach()) (잔차 연결, 그래디언트 유지)
    """
    def __init__(self, input_dim: int, num_assets: int, hidden_dim: int, dropout: float = 0.2,
                 omega_mode: str = 'learnable', sigma_mode: str = 'prior',
                 lambda_risk: float = 0.0, regime_dim: int = 0,
                 macro_dim: int = 0, max_bil_floor: float = 0.5,
                 dist_type: str = 't', t_df: float = 5.0,
                 e2e_regime: bool = False,
                 group_masks=None, group_caps=None, group_floors=None,
                 learnable_t_df: bool = False,
                 t_df_prior=None):
        super(BaseBLModel, self).__init__()
        self.num_assets = num_assets
        self.dropout = dropout
        self.omega_mode = omega_mode
        self.sigma_mode = sigma_mode
        self.lambda_risk = lambda_risk
        self.regime_dim = regime_dim
        self.macro_dim = macro_dim
        self.e2e_regime = e2e_regime
        
        # --- 전망(View) 생성 헤드 ---
        # Q head: regime_dim > 0 이면 concat(hidden, regime) → Q
        # v5: dd_state_dim=3 추가 — 현재 DD 상태를 Q head에 전달하여
        #     포트폴리오 상태 인식 포트폴리오 결정 지원.
        # dd_state = [current_drawdown(1), dd_duration_months(1), dd_speed(1)]
        self.dd_state_dim = 3  # Drawdown state 차원 (고정)
        q_input_dim = hidden_dim + regime_dim + self.dd_state_dim
        self.head_q = nn.Linear(q_input_dim, num_assets)
        # P: 자산 선택 행렬의 대각 원소 (0 ~ 1, sigmoid)
        self.head_p_diag = nn.Linear(hidden_dim, num_assets)
        
        # Omega 헤드: omega_mode에 따라 다르게 초기화
        if omega_mode == 'learnable':
            self.head_omega_diag = nn.Linear(hidden_dim, num_assets)
        elif omega_mode == 'hybrid':
            self.log_alpha = nn.Parameter(torch.zeros(num_assets))
        # formula 모드: 추가 파라미터 없음

        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        
        # Residual sigma 모드의 블렌딩 강도
        if sigma_mode == 'residual':
            self.lambda_blend = nn.Parameter(torch.tensor(0.1))
        
        # --- Crisis Overlay (v4: regime_dim > 0 일 때 활성화) ---
        if regime_dim > 0:
            self.crisis_overlay = CrisisOverlay(
                num_assets=num_assets,
                bil_index=num_assets - 1,
                max_bil_floor=max_bil_floor
            )
        
        # --- Internal RegimeHead (macro_dim > 0 시 활성화) ---
        # 매크로 피처(T10Y3M, BAA10Y)를 GRU hidden과 concat하여 regime 분류
        if regime_dim > 0 and macro_dim > 0:
            from src.regime import RegimeHead
            self.macro_regime_head = RegimeHead(
                hidden_dim=hidden_dim,
                n_regimes=regime_dim,
                macro_dim=macro_dim,
            )
        
        # --- E2E RegimeHead (e2e_regime=True 시 활성화) ---
        # GRU hidden state만으로 regime 확률을 End-to-End 학습
        # HMM prob은 KL loss target으로만 사용 (Teacher-Student)
        if e2e_regime and regime_dim > 0:
            from src.regime import RegimeHead
            self.e2e_regime_head = RegimeHead(
                hidden_dim=hidden_dim,
                n_regimes=regime_dim,
                macro_dim=0,  # GRU hidden만 사용
            )
        
        # --- 최적화 레이어 ---
        n_scenarios = max(200, num_assets * 20)  # 자산 수에 비례 (13자산→260)
        self.opt_layer = CVaROptimizationLayer(
            num_assets=num_assets,
            num_scenarios=n_scenarios,
            confidence_level=0.95,
            bil_index=num_assets - 1,  # BIL = 마지막 자산 (10→idx9, 13→idx12)
            safety_threshold=0.5,
            dist_type=dist_type,
            t_df=t_df,
            group_masks=group_masks,
            group_caps=group_caps,
            group_floors=group_floors,
        )

        # --- P2.2: learnable per-regime Student-T df (optional) ------------
        self.regime_t_df = None
        if learnable_t_df and regime_dim > 0 and dist_type == 't':
            from src.regime_t_df import RegimeAdaptiveTDf
            prior = t_df_prior if t_df_prior is not None else [3.0, 5.0, 7.0, 10.0][:regime_dim]
            self.regime_t_df = RegimeAdaptiveTDf(prior_df=prior)

    def get_bl_parameters(self, hidden_features: torch.Tensor, sigma: torch.Tensor = None,
                          regime_probs: torch.Tensor = None,
                          dd_state: torch.Tensor = None):
        """
        BL 파라미터(P, Q, Omega) 생성.

        Args:
            hidden_features: (B, hidden_dim) 인코더 출력
            sigma: (B, N, N) 공분산 행렬 (formula/hybrid 모드에서 필요)
            regime_probs: (B, R) regime 확률 (v4: Q head concat용)
            dd_state: (B, 3) Drawdown 상태 벡터 (v5 신규):
                [current_drawdown, dd_duration_months_norm, dd_speed_norm]
                None이면 영벡터(0)로 처리 → 기존과 동일한 동작.

        Returns:
            p: (B, N, N) Pick 행렬
            q: (B, N, 1) 전망 벡터
            omega: (B, N, N) 불확실성 행렬
        """
        B = hidden_features.shape[0]
        device = hidden_features.device

        # v5: DD state — None이면 영벡터로 대체 (기존 동작 유지)
        if dd_state is None:
            dd_state = torch.zeros(B, self.dd_state_dim, device=device,
                                   dtype=hidden_features.dtype)
        else:
            dd_state = dd_state.to(device=device, dtype=hidden_features.dtype)

        # Q: regime concat (v4) + dd_state concat (v5)
        q_parts = [hidden_features]
        if self.regime_dim > 0 and regime_probs is not None:
            q_parts.append(regime_probs)
        else:
            # regime_probs 없을 때 regime_dim 크기 영벡터로 패딩 유지
            if self.regime_dim > 0:
                q_parts.append(torch.zeros(B, self.regime_dim, device=device,
                                           dtype=hidden_features.dtype))
        q_parts.append(dd_state)
        q_input = torch.cat(q_parts, dim=-1)
        q = self.tanh(self.head_q(q_input)).unsqueeze(-1)
        
        # P: 대각 행렬
        p_diag = torch.sigmoid(self.head_p_diag(hidden_features))
        p = torch.diag_embed(p_diag)
        
        # Omega: 모드에 따라 다르게 계산
        if self.omega_mode == 'learnable':
            omega_diag = self.softplus(self.head_omega_diag(hidden_features)) + 1e-6
            omega = torch.diag_embed(omega_diag)
            
        elif self.omega_mode == 'formula':
            omega = self._compute_omega_formula(p_diag, sigma, tau=0.05)
            
        elif self.omega_mode == 'hybrid':
            omega = self._compute_omega_hybrid(p_diag, sigma, tau=0.05)
        else:
            raise ValueError(f"Unknown omega_mode: {self.omega_mode}")
        
        return p, q, omega
    
    def _compute_omega_formula(self, p_diag: torch.Tensor, sigma: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
        """
        수식 기반 Omega: Ω_ii = τ · p_i² · Σ_ii
        
        P가 대각 행렬일 때 diag(P(τΣ)P^T)를 직접 계산한 결과.
        파라미터 0개 — 시장 공분산에서 직접 유도.
        
        Args:
            p_diag: (B, N) P 행렬의 대각 원소
            sigma: (B, N, N) 공분산 행렬
            tau: 스케일링 인자 (기본 0.05)
        
        Returns:
            omega: (B, N, N) 대각 불확실성 행렬
        """
        sigma_diag = torch.diagonal(sigma, dim1=1, dim2=2)  # (B, N)
        omega_diag = tau * (p_diag ** 2) * sigma_diag + 1e-6
        return torch.diag_embed(omega_diag)
    
    def _compute_omega_hybrid(self, p_diag: torch.Tensor, sigma: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
        """
        하이브리드 Omega: Ω_ii = α_i · τ · p_i² · Σ_ii
        
        수식 기반 + 자산별 학습 가능 스케일링(α).
        - α_i = exp(log_alpha_i): 항상 양수, 초기값 1.0 (표준 BL과 동일)
        - α_i > 1: 해당 자산 뷰에 덜 확신 → BL이 사전확률 쪽으로 기움
        - α_i < 1: 해당 자산 뷰에 더 확신 → BL이 모델 전망 쪽으로 기움
        
        Args:
            p_diag: (B, N) P 행렬의 대각 원소
            sigma: (B, N, N) 공분산 행렬
            tau: 스케일링 인자 (기본 0.05)
        
        Returns:
            omega: (B, N, N) 대각 불확실성 행렬
        """
        alpha = torch.exp(self.log_alpha)  # (N,) 항상 양수
        sigma_diag = torch.diagonal(sigma, dim1=1, dim2=2)  # (B, N)
        omega_diag = alpha * tau * (p_diag ** 2) * sigma_diag + 1e-6
        return torch.diag_embed(omega_diag)

    def black_litterman_formula(self, p, q, omega, pi, sigma, tau=0.05):
        """
        Black-Litterman 공식의 완전한 구현입니다.
        연구 논문의 수식: E[R] = [(tau*Sigma)^-1 + P.T * Omega^-1 * P]^-1 * ...
        
        Returns:
            mu_bl: (B, N) 사후 기대수익률
            sigma_out: (B, N, N) 최적화에 사용할 공분산 (sigma_mode에 따라 다름)
        """
        tau_sigma = tau * sigma
        eye = torch.eye(self.num_assets, device=p.device)
        eye_batch = eye.unsqueeze(0).expand_as(tau_sigma)
        jitter = BL_SOLVER_JITTER * eye
        jitter_batch = BL_SOLVER_JITTER * eye_batch
        
        try:
            inv_tau_sigma = torch.linalg.solve(tau_sigma + jitter, eye_batch)
            inv_omega = torch.linalg.solve(omega + jitter_batch, eye_batch)
        except RuntimeError:
            inv_tau_sigma = torch.linalg.pinv(tau_sigma + jitter_batch)
            inv_omega = torch.linalg.pinv(omega + jitter_batch)

        term_b = torch.bmm(p.transpose(1, 2), torch.bmm(inv_omega, p))
        post_precision = inv_tau_sigma + term_b
        
        try:
            sigma_bl = torch.linalg.solve(post_precision + jitter_batch, eye_batch)
        except RuntimeError:
            sigma_bl = torch.linalg.pinv(post_precision + jitter_batch)

        if pi.dim() == 2:
            pi = pi.unsqueeze(-1)
            
        term_c = torch.bmm(inv_tau_sigma, pi)
        term_d = torch.bmm(p.transpose(1, 2), torch.bmm(inv_omega, q))
        
        mu_bl = torch.bmm(sigma_bl, term_c + term_d)
        
        # Sigma 반환: sigma_mode에 따라 다르게 처리
        if self.sigma_mode == 'prior':
            # 원기님 방식: 사전 공분산 그대로 사용 (그래디언트 단절, 과적합 방지)
            sigma_out = sigma
        elif self.sigma_mode == 'residual':
            # 잔차 연결: 기본은 sigma이지만, sigma_bl 경로로 그래디언트 역전파
            sigma_out = sigma + self.lambda_blend * (sigma_bl - sigma.detach())
        else:
            sigma_out = sigma
        
        return mu_bl.squeeze(-1), sigma_out

    def forward(self, x: torch.Tensor, pi: torch.Tensor = None, sigma: torch.Tensor = None,
                is_crisis: float = 0.0, regime_probs: torch.Tensor = None,
                macro_features: torch.Tensor = None,
                dd_state: torch.Tensor = None) -> torch.Tensor:
        """
        Forward Pass (v4: regime conditioning + macro features 지원, v5: dd_state 추가)

        Args:
            x: (Batch, Seq, Feat) - 자산 수익률 또는 특징 데이터
            is_crisis: 1.0이면 위기 상황(VIX > 30)으로 간주하여 안전망 가동.
            regime_probs: (B, R) regime 확률 (v4, optional. E2E일 때는 HMM target)
            macro_features: (B, macro_dim) 매크로 피처 (optional, RegimeHead 전용)
            dd_state: (B, 3) Drawdown 상태 벡터 (v5, optional):
                [current_drawdown_normalized, dd_duration_normalized, dd_speed_normalized]
                None이면 영벡터로 대체 → 기존 동작과 완전히 동일.

        Returns:
            weights: (B, N) when e2e_regime=False
            (weights, e2e_regime_probs): when e2e_regime=True
        """
        bs, seq_len, _ = x.size()
        device = x.device
        
        # 0. sigma/pi 추정
        if sigma is None:
            asset_returns = x[:, :, :self.num_assets]
            sigma = estimate_covariance(asset_returns, shrinkage=0.1)
            
        if pi is None:
            if 'asset_returns' not in locals():
                asset_returns = x[:, :, :self.num_assets]
            pi = asset_returns.mean(dim=1)

        # 1. 인코딩
        hidden = self.encode(x) 
        
        # 1.5a. E2E RegimeHead (hidden만으로 regime 확률 생성)
        e2e_regime_probs = None
        if self.e2e_regime and hasattr(self, 'e2e_regime_head'):
            e2e_regime_probs = self.e2e_regime_head(hidden)
            # E2E가 생성한 regime_probs로 교체 (외부 HMM은 KL target으로만 사용)
            regime_probs_for_model = e2e_regime_probs
        else:
            regime_probs_for_model = regime_probs
        
        # 1.5b. 매크로 RegimeHead (내부 Regime 생성, macro_dim > 0일 때)
        if hasattr(self, 'macro_regime_head') and macro_features is not None:
            regime_probs_for_model = self.macro_regime_head(hidden, macro_features=macro_features)
        
        # 2. 뷰 생성 (v4: regime_probs를 Q head에 전달, v5: dd_state 추가)
        p, q, omega = self.get_bl_parameters(
            hidden, sigma=sigma,
            regime_probs=regime_probs_for_model,
            dd_state=dd_state,
        )
        
        # 3. Black-Litterman 공식
        mu_bl, sigma_out = self.black_litterman_formula(p, q, omega, pi, sigma)

        # P2.2: regime-adaptive learnable t_df. When enabled, compute the
        # effective df per sample from the regime probabilities and push it
        # into the optimisation layer for gradient-propagating sampling.
        if self.regime_t_df is not None and regime_probs_for_model is not None:
            self.opt_layer.t_df = self.regime_t_df(regime_probs_for_model)

        # 4. CVaR 최적화
        weights = self.opt_layer(mu_bl, sigma_out, is_crisis=is_crisis,
                                  lambda_risk=self.lambda_risk)
        
        # 5. Crisis Overlay (v4: regime_dim > 0 일 때만)
        if self.regime_dim > 0 and regime_probs_for_model is not None and hasattr(self, 'crisis_overlay'):
            weights = self.crisis_overlay(weights, regime_probs_for_model)
        
        # 6. E2E 모드: (weights, regime_probs) 반환하여 KL loss 계산 가능
        if self.e2e_regime and e2e_regime_probs is not None:
            return weights, e2e_regime_probs
        
        return weights
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("각 모델은 encode() 메서드를 구현해야 합니다.")


# =============================================================================
# 2. 5가지 벤치마크 모델 (Encoders)
# =============================================================================

class LSTMModel(BaseBLModel):
    def __init__(self, input_dim, num_assets, hidden_dim=64, num_layers=2, dropout=0.2,
                 omega_mode='learnable', sigma_mode='prior', lambda_risk=0.0,
                 regime_dim=0, macro_dim=0, max_bil_floor=0.5,
                 dist_type='t', t_df=5.0, e2e_regime=False, **kwargs):
        super().__init__(input_dim, num_assets, hidden_dim, dropout, omega_mode, sigma_mode,
                         lambda_risk, regime_dim, macro_dim, max_bil_floor,
                         dist_type=dist_type, t_df=t_df, e2e_regime=e2e_regime, **kwargs)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
    
    def encode(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]

class GRUModel(BaseBLModel):
    def __init__(self, input_dim, num_assets, hidden_dim=64, num_layers=2, dropout=0.2,
                 omega_mode='learnable', sigma_mode='prior', lambda_risk=0.0, regime_dim=0,
                 macro_dim=0, max_bil_floor=0.5, dist_type='t', t_df=5.0, e2e_regime=False,
                 **kwargs):
        super().__init__(input_dim, num_assets, hidden_dim, dropout, omega_mode, sigma_mode,
                         lambda_risk, regime_dim, macro_dim, max_bil_floor, dist_type=dist_type, t_df=t_df,
                         e2e_regime=e2e_regime, **kwargs)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
    
    def encode(self, x):
        _, h_n = self.gru(x)
        return h_n[-1]

class TCNModel(BaseBLModel):
    def __init__(self, input_dim, num_assets, hidden_dim=64, start_kernel_size=3, dropout=0.2,
                 omega_mode='learnable', sigma_mode='prior', lambda_risk=0.0,
                 regime_dim=0, macro_dim=0, max_bil_floor=0.5,
                 dist_type='t', t_df=5.0, e2e_regime=False, **kwargs):
        super().__init__(input_dim, num_assets, hidden_dim, dropout, omega_mode, sigma_mode,
                         lambda_risk, regime_dim, macro_dim, max_bil_floor,
                         dist_type=dist_type, t_df=t_df, e2e_regime=e2e_regime, **kwargs)
        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=start_kernel_size, padding=start_kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    def encode(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x) 
        return out.squeeze(-1)

class TransformerModel(BaseBLModel):
    def __init__(self, input_dim, num_assets, hidden_dim=64, num_layers=2, nhead=4, dropout=0.2,
                 omega_mode='learnable', sigma_mode='prior', lambda_risk=0.0,
                 regime_dim=0, macro_dim=0, max_bil_floor=0.5,
                 dist_type='t', t_df=5.0, e2e_regime=False, **kwargs):
        super().__init__(input_dim, num_assets, hidden_dim, dropout, omega_mode, sigma_mode,
                         lambda_risk, regime_dim, macro_dim, max_bil_floor,
                         dist_type=dist_type, t_df=t_df, e2e_regime=e2e_regime, **kwargs)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def encode(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        return out[:, -1, :] 

# --- TFT (Temporal Fusion Transformer) 컴포넌트 ---
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim * 2) 
        
        # Skip connection을 위한 projection (차원이 다를 경우 사용)
        self.skip_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        gate_out = torch.sigmoid(self.gate(x)[..., :self.fc1.out_features])
        residual = self.fc2(F.elu(self.fc1(x)))
        residual = self.dropout(residual)
        
        # 차원 불일치 시 projection 적용
        x_proj = self.skip_proj(x)
              
        out = self.layernorm(x_proj + gate_out * residual)
        return out

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, num_inputs, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, dropout) for _ in range(num_inputs)
        ])
        self.weight_network = GatedResidualNetwork(input_dim * num_inputs, num_inputs, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        var_outputs = []
        for i in range(len(self.single_variable_grns)):
            # 입력 변수별로 슬라이싱하여 개별 GRN 통과
            feat = x[..., i:i+1] 
            var_outputs.append(self.single_variable_grns[i](feat))
        var_outputs = torch.stack(var_outputs, dim=-1)
        weights = self.softmax(self.weight_network(x)).unsqueeze(-2)
        
        # XAI 분석을 위해 가중치 저장 (Run XAI 스크립트에서 사용됨)
        self.feature_weights = weights.squeeze(-2).detach()
        
        combined = (weights * var_outputs).sum(dim=-1)
        return combined

class TFTModel(BaseBLModel):
    def __init__(self, input_dim, num_assets, hidden_dim=64, num_layers=2, nhead=4, dropout=0.2,
                 omega_mode='learnable', sigma_mode='prior', lambda_risk=0.0,
                 regime_dim=0, macro_dim=0, max_bil_floor=0.5,
                 dist_type='t', t_df=5.0, e2e_regime=False, **kwargs):
        super().__init__(input_dim, num_assets, hidden_dim, dropout, omega_mode, sigma_mode,
                         lambda_risk, regime_dim, macro_dim, max_bil_floor,
                         dist_type=dist_type, t_df=t_df, e2e_regime=e2e_regime, **kwargs)
        
        self.vsn = VariableSelectionNetwork(1, input_dim, hidden_dim, dropout)
        self.lstm_encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.gate_lstm = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
        self.layernorm_lstm = nn.LayerNorm(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.gate_attn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
        self.layernorm_attn = nn.LayerNorm(hidden_dim)
        
    def encode(self, x):
        vsn_out = self.vsn(x) 
        lstm_out, _ = self.lstm_encoder(vsn_out)
        lstm_out = self.layernorm_lstm(self.gate_lstm(lstm_out) + vsn_out) 
        attn_out = self.transformer(lstm_out)
        attn_out = self.layernorm_attn(self.gate_attn(attn_out) + lstm_out)
        return attn_out[:, -1, :] 


# =============================================================================
# 3. Regime-Adaptive Architecture (v3: FiLM + Softmax + Sharpe)
# =============================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    
    Regime 확률이 GRU hidden state를 직접 변조:
        h' = γ(regime) * h + β(regime)
    
    파라미터 효율적 (2 * H * R) + 높은 표현력.
    """
    def __init__(self, regime_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma = nn.Linear(regime_dim, hidden_dim)
        self.beta = nn.Linear(regime_dim, hidden_dim)
        
        # 초기화: γ≈1, β≈0 → 초기에는 regime 무시 (안정적 학습 시작)
        nn.init.ones_(self.gamma.weight.data[:, 0])  # 첫 번째 regime→1
        nn.init.zeros_(self.gamma.bias.data)
        nn.init.zeros_(self.beta.weight.data)
        nn.init.zeros_(self.beta.bias.data)
    
    def forward(self, h: torch.Tensor, regime_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, H) GRU hidden state
            regime_probs: (B, R) regime 확률 [R=4: Bull/Sideways/Correction/Crisis]
        Returns:
            h_modulated: (B, H)
        """
        gamma = 1.0 + self.gamma(regime_probs)  # 1-centered (초기 γ ≈ 1)
        beta = self.beta(regime_probs)
        return gamma * h + beta


class SoftmaxPortfolioLayer(nn.Module):
    """
    Differentiable Portfolio Layer (CVaR 대체).
    
    μ_BL / sqrt(diag(Σ)) → risk-adjusted score → softmax → weights
    
    Temperature는 학습 가능 (초기 τ=1.0, 범위 [0.1, 10.0]).
    """
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(0.0))  # exp(0)=1.0
    
    def forward(self, mu_bl: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu_bl: (B, N) Black-Litterman 사후 기대수익률
            sigma: (B, N, N) 공분산 행렬
        Returns:
            weights: (B, N) 포트폴리오 비중 (합=1, ≥0)
        """
        # Risk-adjusted score: μ / σ (Sharpe-like)
        vol = torch.sqrt(torch.diagonal(sigma, dim1=1, dim2=2).abs() + 1e-8)
        score = mu_bl / vol
        
        temperature = torch.exp(self.log_temperature).clamp(0.1, 10.0)
        weights = torch.softmax(score / temperature, dim=1)
        
        return weights


class CrisisOverlay(nn.Module):
    """
    Differentiable Crisis Safety Net.
    
    Crisis 확률에 비례하여 BIL(안전자산) 비중을 강제 증가.
    나머지 자산은 비례적으로 축소. 합=1 항상 보장.
    """
    def __init__(self, num_assets: int, bil_index: int = -1, 
                 max_bil_floor: float = 0.5):
        super().__init__()
        self.num_assets = num_assets
        # bil_index가 -1이면 마지막 자산
        self.bil_index = bil_index if bil_index >= 0 else num_assets - 1
        self.max_bil_floor = max_bil_floor
    
    def forward(self, weights: torch.Tensor, 
                regime_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weights: (B, N) softmax 출력 (합=1)
            regime_probs: (B, R) regime 확률 [R=4: Bull/Side/Corr/Crisis]
        Returns:
            adjusted: (B, N) crisis-adjusted 비중 (합=1, ≥0)
        """
        # Crisis = 마지막 state (index 3)
        p_crisis = regime_probs[:, -1:]  # (B, 1)
        
        # BIL 최소 비중: crisis 확률에 비례 (최대 max_bil_floor)
        bil_floor = self.max_bil_floor * p_crisis  # (B, 1)
        
        # 현재 BIL 비중
        current_bil = weights[:, self.bil_index:self.bil_index+1]  # (B, 1)
        
        # BIL이 이미 floor 이상이면 변경 없음
        needs_adjustment = (current_bil < bil_floor).float()
        
        # BIL 외 자산을 축소, BIL을 floor로 설정
        non_bil_mask = torch.ones(self.num_assets, device=weights.device)
        non_bil_mask[self.bil_index] = 0.0
        
        non_bil_sum = (weights * non_bil_mask).sum(dim=1, keepdim=True) + 1e-8
        
        # 조정된 BIL
        adjusted_bil = torch.max(current_bil, bil_floor)
        # 나머지 자산은 (1 - adjusted_bil)에 비례 배분
        remaining = 1.0 - adjusted_bil
        scale = remaining / non_bil_sum
        
        adjusted = weights * non_bil_mask * scale
        adjusted[:, self.bil_index:self.bil_index+1] = adjusted_bil
        
        # needs_adjustment = 0이면 원래 weights 유지
        result = needs_adjustment * adjusted + (1 - needs_adjustment) * weights
        
        return result


class RegimeFiLMGRU(nn.Module):
    """
    Regime-Adaptive FiLM-GRU with Direct Sharpe Maximization.
    
    아키텍처:
        Input (X) → GRU Encoder → FiLM(regime_probs) → BL Views 
        → BL Formula → SoftmaxPortfolio → CrisisOverlay → Weights
    
    기존 BaseBLModel의 BL formula를 재사용하면서,
    CVaR optimizer를 Softmax + CrisisOverlay로 대체.
    """
    def __init__(self, input_dim: int, num_assets: int, 
                 hidden_dim: int = 64, n_regimes: int = 4,
                 num_layers: int = 2, dropout: float = 0.2,
                 omega_mode: str = 'learnable', sigma_mode: str = 'prior'):
        super().__init__()
        
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.omega_mode = omega_mode
        self.sigma_mode = sigma_mode
        
        # --- Encoder ---
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # --- FiLM Conditioning ---
        self.film = FiLMLayer(n_regimes, hidden_dim)
        
        # --- BL View Heads (BaseBLModel과 동일 구조) ---
        self.head_q = nn.Linear(hidden_dim, num_assets)
        self.head_p_diag = nn.Linear(hidden_dim, num_assets)
        self.tanh = nn.Tanh()
        
        # Omega head
        if omega_mode == 'learnable':
            self.head_omega_diag = nn.Linear(hidden_dim, num_assets)
            self.softplus = nn.Softplus()
        elif omega_mode == 'hybrid':
            self.log_alpha = nn.Parameter(torch.zeros(num_assets))
        
        # Residual sigma
        if sigma_mode == 'residual':
            self.lambda_blend = nn.Parameter(torch.tensor(0.1))
        
        # --- Portfolio Layer (CVaR 대체) ---
        self.portfolio_layer = SoftmaxPortfolioLayer()
        
        # --- Crisis Safety Net ---
        self.crisis_overlay = CrisisOverlay(
            num_assets=num_assets, 
            bil_index=num_assets - 1,  # BIL = 마지막 자산
            max_bil_floor=0.5
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """GRU Encoding → last hidden state."""
        _, h_n = self.gru(x)
        return h_n[-1]  # (B, H)
    
    def get_bl_parameters(self, hidden: torch.Tensor, sigma=None):
        """BL Views (P, Q, Omega) 생성 — BaseBLModel과 동일 로직."""
        q = self.tanh(self.head_q(hidden)).unsqueeze(-1)
        p_diag = torch.sigmoid(self.head_p_diag(hidden))
        p = torch.diag_embed(p_diag)
        
        if self.omega_mode == 'learnable':
            omega_diag = self.softplus(self.head_omega_diag(hidden)) + 1e-6
            omega = torch.diag_embed(omega_diag)
        elif self.omega_mode == 'formula':
            sigma_diag = torch.diagonal(sigma, dim1=1, dim2=2)
            omega_diag = 0.05 * (p_diag ** 2) * sigma_diag + 1e-6
            omega = torch.diag_embed(omega_diag)
        elif self.omega_mode == 'hybrid':
            alpha = torch.exp(self.log_alpha)
            sigma_diag = torch.diagonal(sigma, dim1=1, dim2=2)
            omega_diag = alpha * 0.05 * (p_diag ** 2) * sigma_diag + 1e-6
            omega = torch.diag_embed(omega_diag)
        else:
            raise ValueError(f"Unknown omega_mode: {self.omega_mode}")
        
        return p, q, omega

    def black_litterman_formula(self, p, q, omega, pi, sigma, tau=0.05):
        """BL 공식 — torch.linalg.solve 사용 (수치 안정성 개선)."""
        N = self.num_assets
        device = p.device
        eye = torch.eye(N, device=device)
        
        tau_sigma = tau * sigma
        eye_batch = eye.unsqueeze(0).expand_as(tau_sigma)
        jitter = BL_SOLVER_JITTER * eye
        jitter_batch = BL_SOLVER_JITTER * eye_batch
        
        # A = (τΣ)^{-1} + P^T Ω^{-1} P
        # b = (τΣ)^{-1} π + P^T Ω^{-1} q
        # μ_BL = solve(A, b)
        
        try:
            # solve(A, b) = A^{-1} b — inverse보다 안정적
            inv_tau_sigma_pi = torch.linalg.solve(
                tau_sigma + jitter, pi.unsqueeze(-1) if pi.dim() == 2 else pi)
            inv_omega_p = torch.linalg.solve(omega + jitter_batch, p)
            inv_omega_q = torch.linalg.solve(omega + jitter_batch, q)
        except RuntimeError:
            # Fallback 
            inv_ts = torch.linalg.pinv(tau_sigma + jitter_batch)
            inv_o = torch.linalg.pinv(omega + jitter_batch)
            inv_tau_sigma_pi = torch.bmm(inv_ts, pi.unsqueeze(-1) if pi.dim() == 2 else pi)
            inv_omega_p = torch.bmm(inv_o, p)
            inv_omega_q = torch.bmm(inv_o, q)
        
        # Precision matrix
        pt_inv_omega_p = torch.bmm(p.transpose(1, 2), inv_omega_p)
        
        try:
            inv_ts_matrix = torch.linalg.solve(
                tau_sigma + jitter, eye_batch)
        except RuntimeError:
            inv_ts_matrix = torch.linalg.pinv(tau_sigma + jitter_batch)
            
        post_precision = inv_ts_matrix + pt_inv_omega_p
        
        # Right-hand side
        pt_inv_omega_q = torch.bmm(p.transpose(1, 2), inv_omega_q)
        rhs = inv_tau_sigma_pi + pt_inv_omega_q
        
        # Solve for mu_bl
        try:
            mu_bl = torch.linalg.solve(post_precision + jitter, rhs)
        except RuntimeError:
            mu_bl = torch.bmm(torch.linalg.pinv(post_precision + jitter_batch), rhs)
        
        # Sigma output
        if self.sigma_mode == 'prior':
            sigma_out = sigma
        elif self.sigma_mode == 'residual':
            try:
                sigma_bl = torch.linalg.solve(
                    post_precision + jitter, eye_batch)
            except RuntimeError:
                sigma_bl = torch.linalg.pinv(post_precision + jitter_batch)
            sigma_out = sigma + self.lambda_blend * (sigma_bl - sigma.detach())
        else:
            sigma_out = sigma
        
        return mu_bl.squeeze(-1), sigma_out

    def forward(self, x: torch.Tensor, 
                regime_probs: torch.Tensor = None) -> torch.Tensor:
        """
        Forward Pass.
        
        Args:
            x: (B, Seq, Feat) 입력 시퀀스
            regime_probs: (B, R) regime 확률 [R=4]. None이면 균등 확률.
        
        Returns:
            weights: (B, N) 포트폴리오 비중 (합=1, ≥0)
        """
        bs = x.size(0)
        device = x.device
        
        # 0. Sigma, Pi 추정
        asset_returns = x[:, :, :self.num_assets]
        sigma = estimate_covariance(asset_returns, shrinkage=0.1)
        pi = asset_returns.mean(dim=1)
        
        # 1. GRU Encoding
        hidden = self.encode(x)  # (B, H)
        
        # 2. FiLM Conditioning
        if regime_probs is None:
            regime_probs = torch.ones(bs, 4, device=device) / 4.0
        
        hidden = self.film(hidden, regime_probs)  # (B, H)
        
        # 3. BL Views
        p, q, omega = self.get_bl_parameters(hidden, sigma=sigma)
        
        # 4. BL Formula
        mu_bl, sigma_out = self.black_litterman_formula(p, q, omega, pi, sigma)
        
        # 5. Softmax Portfolio (CVaR 대체)
        weights = self.portfolio_layer(mu_bl, sigma_out)  # (B, N)
        
        # 6. Crisis Overlay
        weights = self.crisis_overlay(weights, regime_probs)  # (B, N)
        
        return weights


# =============================================================================
# 4. Dual-Shock Expert System (v5 New)
# =============================================================================

class DualShockExpertModel(BaseBLModel):
    """
    Dual-Shock Expert System (v5).

    세 전문가가 레짐 인식 게이팅 네트워크를 통해 혼합됩니다:
      - Trend Expert  : 추세 추종 / 표준 모멘텀  (lambda_risk = 1.0)
      - Inflation Expert: 인플레이션/금리 충격 방어 (lambda_risk = 3.0)
      - Crash Expert  : 급락/꼬리 위험 헤지       (lambda_risk = 5.0)

    아키텍처:
        Input → GRU (공유) → hidden
        [hidden, regime] → Gate → [g_trend, g_infl, g_crash]
        각 전문가: (hidden, regime, dd_state) → Q_i, P, Ω → BL → CVaR → w_i
        w_final = Σ g_i * w_i → CrisisOverlay → output

    Notes:
        - P/Ω 헤드는 기반 클래스(BaseBLModel)의 것을 공유하여 파라미터 절약.
        - 각 전문가는 독립 Q 헤드와 독립 P 헤드를 가짐.
        - Omega 헤드는 공유 (omega_mode에 따라 학습/공식/하이브리드).
        - dd_state는 각 전문가 Q 헤드의 입력에 포함.
    """

    EXPERT_NAMES = ['trend', 'inflation', 'crash']
    _EXPERT_LAMBDA_RISK = [1.0, 3.0, 5.0]  # trend / inflation / crash

    def __init__(self, input_dim: int, num_assets: int,
                 hidden_dim: int = 32, num_layers: int = 2,
                 dropout: float = 0.2,
                 omega_mode: str = 'learnable', sigma_mode: str = 'prior',
                 regime_dim: int = 4, macro_dim: int = 0,
                 max_bil_floor: float = 0.5,
                 dist_type: str = 't', t_df: float = 5.0,
                 e2e_regime: bool = False,
                 gate_hidden: int = 16,
                 lambda_risk_scale: float = 1.0):
        """
        Args:
            lambda_risk_scale: Expert lambda_risk를 일괄 스케일링.
                기본 [1.0, 3.0, 5.0] × scale.
                scale=0.5 → [0.5, 1.5, 2.5] (더 공격적).
                scale=1.0 → 기존 동작.
        """
        super().__init__(
            input_dim=input_dim, num_assets=num_assets,
            hidden_dim=hidden_dim, dropout=dropout,
            omega_mode=omega_mode, sigma_mode=sigma_mode,
            lambda_risk=1.0,          # Trend expert default; overridden per-expert
            regime_dim=regime_dim, macro_dim=macro_dim,
            max_bil_floor=max_bil_floor,
            dist_type=dist_type, t_df=t_df,
            e2e_regime=e2e_regime,
        )

        self._lambda_risk_scale = lambda_risk_scale

        # Shared GRU encoder
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        n_experts = len(self.EXPERT_NAMES)

        # Per-expert Q heads (same input dim as BaseBLModel.head_q)
        q_input_dim = hidden_dim + regime_dim + self.dd_state_dim
        self.expert_q_heads = nn.ModuleDict({
            name: nn.Linear(q_input_dim, num_assets)
            for name in self.EXPERT_NAMES
        })

        # Per-expert P heads
        self.expert_p_heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, num_assets)
            for name in self.EXPERT_NAMES
        })

        # Fixed lambda_risk per expert — scaled by lambda_risk_scale
        scaled_lambdas = [l * lambda_risk_scale for l in self._EXPERT_LAMBDA_RISK]
        self.register_buffer(
            'expert_lambda_risk',
            torch.tensor(scaled_lambdas, dtype=torch.float32),
        )

        # Gating network: [hidden, regime] → n_experts
        gate_in = hidden_dim + regime_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, n_experts),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared GRU encoder → last hidden state."""
        _, h_n = self.gru(x)
        return h_n[-1]  # (B, H)

    def _get_expert_bl_params(self, expert_name: str,
                               hidden: torch.Tensor,
                               sigma: torch.Tensor,
                               regime_probs,
                               dd_state):
        """Compute (P, Q, Ω) for one expert using its dedicated Q/P heads."""
        B = hidden.shape[0]
        device = hidden.device

        # DD state fallback
        if dd_state is None:
            dd_state_t = torch.zeros(B, self.dd_state_dim, device=device,
                                     dtype=hidden.dtype)
        else:
            dd_state_t = dd_state.to(device=device, dtype=hidden.dtype)

        # Q input: [hidden, regime (or zeros), dd_state]
        q_parts = [hidden]
        if self.regime_dim > 0:
            if regime_probs is not None:
                q_parts.append(regime_probs)
            else:
                q_parts.append(torch.zeros(B, self.regime_dim, device=device,
                                           dtype=hidden.dtype))
        q_parts.append(dd_state_t)
        q_input = torch.cat(q_parts, dim=-1)

        q = self.tanh(self.expert_q_heads[expert_name](q_input)).unsqueeze(-1)

        # P from expert-specific head
        p_diag = torch.sigmoid(self.expert_p_heads[expert_name](hidden))
        p = torch.diag_embed(p_diag)

        # Omega: shared omega logic (base class heads)
        if self.omega_mode == 'learnable':
            omega_diag = self.softplus(self.head_omega_diag(hidden)) + 1e-6
            omega = torch.diag_embed(omega_diag)
        elif self.omega_mode == 'formula':
            omega = self._compute_omega_formula(p_diag, sigma, tau=0.05)
        elif self.omega_mode == 'hybrid':
            omega = self._compute_omega_hybrid(p_diag, sigma, tau=0.05)
        else:
            raise ValueError(f"Unknown omega_mode: {self.omega_mode}")

        return p, q, omega

    def forward(self, x: torch.Tensor,
                pi: torch.Tensor = None, sigma: torch.Tensor = None,
                is_crisis: float = 0.0,
                regime_probs: torch.Tensor = None,
                macro_features: torch.Tensor = None,
                dd_state: torch.Tensor = None) -> torch.Tensor:
        """
        Forward Pass (Dual-Shock Expert).

        Args:
            x: (B, Seq, Feat)
            pi, sigma: optional prior mean / covariance
            is_crisis: scalar float passed to CVaR layer
            regime_probs: (B, R) HMM regime probabilities
            macro_features: (B, macro_dim) for internal RegimeHead
            dd_state: (B, 3) drawdown state

        Returns:
            weights: (B, N)  or  (weights, e2e_regime_probs) if e2e_regime=True
        """
        bs, _, _ = x.size()
        device = x.device

        # 0. sigma / pi estimation
        if sigma is None:
            asset_returns = x[:, :, :self.num_assets]
            sigma = estimate_covariance(asset_returns, shrinkage=0.1)
        if pi is None:
            if 'asset_returns' not in locals():
                asset_returns = x[:, :, :self.num_assets]
            pi = asset_returns.mean(dim=1)

        # 1. Shared encoding
        hidden = self.encode(x)

        # 1.5. Regime processing (mirrors BaseBLModel.forward)
        e2e_regime_probs = None
        if self.e2e_regime and hasattr(self, 'e2e_regime_head'):
            e2e_regime_probs = self.e2e_regime_head(hidden)
            regime_probs_for_model = e2e_regime_probs
        else:
            regime_probs_for_model = regime_probs

        if hasattr(self, 'macro_regime_head') and macro_features is not None:
            regime_probs_for_model = self.macro_regime_head(
                hidden, macro_features=macro_features)

        # 2. Gate probabilities: [hidden, regime] → softmax(3)
        gate_parts = [hidden]
        if self.regime_dim > 0:
            if regime_probs_for_model is not None:
                gate_parts.append(regime_probs_for_model)
            else:
                gate_parts.append(torch.zeros(bs, self.regime_dim,
                                              device=device, dtype=hidden.dtype))
        gate_logits = self.gate(torch.cat(gate_parts, dim=-1))  # (B, 3)
        gate_probs = torch.softmax(gate_logits, dim=-1)          # (B, 3)

        # 3. Per-expert BL → CVaR → weights
        expert_weights = []
        for i, name in enumerate(self.EXPERT_NAMES):
            p, q, omega = self._get_expert_bl_params(
                name, hidden, sigma, regime_probs_for_model, dd_state)

            mu_bl, sigma_out = self.black_litterman_formula(
                p, q, omega, pi, sigma)

            lam = self.expert_lambda_risk[i].item()
            w_i = self.opt_layer(mu_bl, sigma_out,
                                 is_crisis=is_crisis, lambda_risk=lam)
            expert_weights.append(w_i)

        # 4. Blend experts
        expert_stack = torch.stack(expert_weights, dim=1)   # (B, 3, N)
        gate_expanded = gate_probs.unsqueeze(-1)             # (B, 3, 1)
        weights = (gate_expanded * expert_stack).sum(dim=1)  # (B, N)

        # 5. Crisis Overlay
        if (self.regime_dim > 0
                and regime_probs_for_model is not None
                and hasattr(self, 'crisis_overlay')):
            weights = self.crisis_overlay(weights, regime_probs_for_model)

        # 6. E2E regime return
        if self.e2e_regime and e2e_regime_probs is not None:
            return weights, e2e_regime_probs

        return weights


# =============================================================================
# 5. 모델 팩토리 함수
# =============================================================================

def get_model(model_type, input_dim, num_assets, device='cpu',
              omega_mode='learnable', sigma_mode='prior', lambda_risk=0.0,
              hidden_dim=64, regime_dim=0, macro_dim=0, max_bil_floor=0.5,
              dist_type='t', t_df=5.0, e2e_regime=False,
              gate_hidden=16,
              group_masks=None, group_caps=None, group_floors=None,
              learnable_t_df=False, t_df_prior=None):
    """
    모델 팩토리 함수 (v5: dual_shock_expert 추가).
    """
    if model_type == 'dual_shock_expert':
        # lambda_risk → expert lambda scale
        # baseline lambda_risk=2.0 → scale=1.0 (기존 [1,3,5] 유지)
        # lambda_risk=1.0 → scale=0.5 → [0.5, 1.5, 2.5] (공격적)
        expert_scale = lambda_risk / 2.0 if lambda_risk > 0 else 1.0
        return DualShockExpertModel(
            input_dim=input_dim, num_assets=num_assets,
            hidden_dim=hidden_dim,
            omega_mode=omega_mode, sigma_mode=sigma_mode,
            regime_dim=regime_dim if regime_dim > 0 else 4,
            macro_dim=macro_dim, max_bil_floor=max_bil_floor,
            dist_type=dist_type, t_df=t_df,
            e2e_regime=e2e_regime,
            gate_hidden=gate_hidden,
            lambda_risk_scale=expert_scale,
        ).to(device)

    model_map = {
        'lstm': LSTMModel, 'gru': GRUModel, 'tcn': TCNModel,
        'transformer': TransformerModel, 'tft': TFTModel,
    }
    if model_type not in model_map:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from {list(model_map.keys()) + ['dual_shock_expert']}"
        )

    kwargs = dict(
        input_dim=input_dim, num_assets=num_assets, hidden_dim=hidden_dim,
        omega_mode=omega_mode, sigma_mode=sigma_mode, lambda_risk=lambda_risk,
    )
    if model_type in ['gru', 'lstm']:
        kwargs['dist_type'] = dist_type
        kwargs['t_df'] = t_df

    if model_type == 'gru' and regime_dim > 0:
        kwargs['regime_dim'] = regime_dim
        kwargs['macro_dim'] = macro_dim
        kwargs['max_bil_floor'] = max_bil_floor
        kwargs['e2e_regime'] = e2e_regime

    # Forward the new P2.1 / P2.2 kwargs — every encoder accepts **kwargs.
    for k, v in (
        ('group_masks', group_masks),
        ('group_caps', group_caps),
        ('group_floors', group_floors),
        ('learnable_t_df', learnable_t_df),
        ('t_df_prior', t_df_prior),
    ):
        if v is not None or k == 'learnable_t_df':
            kwargs[k] = v

    return model_map[model_type](**kwargs).to(device)





# =============================================================================
# 7. Phase 2 — Standalone BL Formula (모델 독립)
# =============================================================================

def black_litterman_formula(p, q, omega, pi, sigma, tau=0.05, sigma_mode='prior'):
    """
    Black-Litterman 공식 (모델 독립 함수).
    
    BaseBLModel.black_litterman_formula에서 추출하여 재사용 가능하게 분리.
    
    P=I일 때: 시장 균형(π)과 모델 전망(Q)의 Bayesian 결합.
    Ω 크면 → Ω⁻¹ 작음 → Q 영향↓ → π에 가까운 보수적 결과
    Ω 작으면 → Ω⁻¹ 큼 → Q 영향↑ → 공격적 결과
    
    Args:
        p: (B, N, N) Pick 행렬
        q: (B, N, 1) 전망 벡터
        omega: (B, N, N) 불확실성 행렬
        pi: (B, N) 사전 기대수익률
        sigma: (B, N, N) 사전 공분산
        tau: BL scaling factor
        sigma_mode: 'prior' (sigma 그대로) 또는 'posterior' (sigma_bl 반환)
        
    Returns:
        mu_bl: (B, N) 사후 기대수익률
        sigma_out: (B, N, N) 공분산
    """
    num_assets = p.size(1)
    device = p.device
    
    tau_sigma = tau * sigma
    eye = torch.eye(num_assets, device=device)
    eye_batch = eye.unsqueeze(0).expand_as(tau_sigma)
    jitter = BL_SOLVER_JITTER * eye
    jitter_batch = BL_SOLVER_JITTER * eye_batch
    
    try:
        inv_tau_sigma = torch.linalg.solve(tau_sigma + jitter, eye_batch)
        inv_omega = torch.linalg.solve(omega + jitter_batch, eye_batch)
    except RuntimeError:
        inv_tau_sigma = torch.linalg.pinv(tau_sigma + jitter_batch)
        inv_omega = torch.linalg.pinv(omega + jitter_batch)
    
    # 사후 정밀도 = (τΣ)^-1 + P^T Ω^-1 P
    term_b = torch.bmm(p.transpose(1, 2), torch.bmm(inv_omega, p))
    post_precision = inv_tau_sigma + term_b
    
    try:
        sigma_bl = torch.linalg.solve(post_precision + jitter_batch, eye_batch)
    except RuntimeError:
        sigma_bl = torch.linalg.pinv(post_precision + jitter_batch)
    
    # 사후 기대수익률
    if pi.dim() == 2:
        pi = pi.unsqueeze(-1)
    
    term_c = torch.bmm(inv_tau_sigma, pi)
    term_d = torch.bmm(p.transpose(1, 2), torch.bmm(inv_omega, q))
    mu_bl = torch.bmm(sigma_bl, term_c + term_d)
    
    sigma_out = sigma if sigma_mode == 'prior' else sigma_bl
    
    return mu_bl.squeeze(-1), sigma_out


# =============================================================================
# 8. Phase 2 — SharedBLHead + FiLM
# =============================================================================

class SharedBLHead(nn.Module):
    """
    FiLM-conditioned BL Parameter Head (Phase 2).
    
    핵심 설계:
    - P = I (고정, 절대 전망) → 구조 안정성
    - Q: tanh base + bounded FiLM → regime이 전망 크기 조절
    - Ω: softplus base + exp(γ) FiLM → regime이 불확실성 조절 (항상 양수)
    - FiLM zero-init → 초기 동작 = regime 무관
    
    Args:
        hidden_dim: GRU hidden dimension
        num_assets: 자산 수
        n_regimes: regime 수
        embed_dim: regime embedding 차원
    """
    def __init__(
        self,
        hidden_dim: int,
        num_assets: int,
        n_regimes: int = 3,
        embed_dim: int = 4,
    ):
        super().__init__()
        self.num_assets = num_assets
        
        # P = I (고정, 비학습)
        self.register_buffer('P', torch.eye(num_assets))
        
        # Q, Omega base heads
        self.head_q = nn.Linear(hidden_dim, num_assets)
        self.head_omega = nn.Linear(hidden_dim, num_assets)
        self._softplus = nn.Softplus()
        
        # Regime Embedding
        self.regime_embed = nn.Embedding(n_regimes, embed_dim)
        
        # FiLM layers (Q: gamma + beta, Omega: gamma만)
        self.film_q_gamma = nn.Linear(embed_dim, num_assets)
        self.film_q_beta = nn.Linear(embed_dim, num_assets)
        self.film_omega_gamma = nn.Linear(embed_dim, num_assets)
        
        # FiLM zero-init (초기 동작 = regime 무관)
        for layer in [self.film_q_gamma, self.film_q_beta, 
                      self.film_omega_gamma]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, hidden: torch.Tensor, regime_probs: torch.Tensor):
        """
        Args:
            hidden: (B, hidden_dim) GRU 출력
            regime_probs: (B, n_regimes) regime 확률
            
        Returns:
            P: (B, N, N) = Identity
            Q: (B, N, 1) regime-conditioned 전망
            Omega: (B, N, N) regime-conditioned 불확실성 (대각, 항상 양수)
        """
        B = hidden.size(0)
        
        # Soft regime embedding (확률 가중 평균)
        r = regime_probs @ self.regime_embed.weight  # (B, embed_dim)
        
        # Q: tanh base + bounded FiLM
        q_base = torch.tanh(self.head_q(hidden))               # [-1, 1]
        gamma_q = torch.tanh(self.film_q_gamma(r))              # [-1, 1]
        beta_q = 0.1 * torch.tanh(self.film_q_beta(r))          # [-0.1, 0.1]
        q = torch.clamp((1 + gamma_q) * q_base + beta_q, -1.5, 1.5)
        
        # Omega: softplus base + exp(γ) FiLM (항상 양수)
        omega_base = self._softplus(self.head_omega(hidden)) + 1e-6
        gamma_o = self.film_omega_gamma(r)
        omega_diag = torch.exp(gamma_o) * omega_base  # > 0 guaranteed
        
        P = self.P.unsqueeze(0).expand(B, -1, -1)
        return P, q.unsqueeze(-1), torch.diag_embed(omega_diag)


# =============================================================================
# 9. Phase 2 — RegimeConditionedBLModel (최종 모델)
# =============================================================================

class RegimeConditionedBLModel(nn.Module):
    """
    Phase 2 최종 모델: Regime-Conditioned MIDAS-BL with Mean-CVaR.
    
    파이프라인:
      Daily Data → LearnableMIDAS → Monthly Features
      concat Monthly → GRU → hidden
      hidden → RegimeHead → regime_probs → λ, is_crisis
      hidden → SharedBLHead(FiLM) → P=I, Q, Ω
      BL Formula → μ_BL, Σ_BL
      Mean-CVaR Solver(μ_BL, λ, is_crisis) → Weights
    """
    def __init__(
        self,
        monthly_input_dim: int,
        n_daily_vars: int = 3,
        K_midas: int = 66,
        num_assets: int = 10,
        hidden_dim: int = 32,
        n_regimes: int = 3,
        dropout: float = 0.3,
        hmm_prior_mean: torch.Tensor = None,
        num_scenarios: int = 200,
        confidence_level: float = 0.95,
        bil_index: int = 4,
    ):
        super().__init__()
        self.num_assets = num_assets
        self.n_regimes = n_regimes
        
        # Import Phase 2 components
        from src.midas_layer import LearnableMIDASLayer
        from src.regime import RegimeHead
        from src.optimization import MeanCVaROptimizationLayer
        
        # 1. Learnable MIDAS
        self.midas_layer = LearnableMIDASLayer(n_daily_vars, K_midas)
        
        # 2. GRU (1-layer, 용량 축소)
        gru_input_dim = monthly_input_dim + n_daily_vars
        self.gru = nn.GRU(
            gru_input_dim, hidden_dim, 
            num_layers=1, batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        
        # 3. Regime Head
        self.regime_head = RegimeHead(
            hidden_dim, n_regimes, 
            hmm_prior_mean=hmm_prior_mean,
        )
        
        # 4. Shared BL Head + FiLM
        self.bl_head = SharedBLHead(
            hidden_dim, num_assets, n_regimes, embed_dim=4,
        )
        
        # 5. Mean-CVaR Solver
        self.opt_layer = MeanCVaROptimizationLayer(
            num_assets=num_assets,
            num_scenarios=num_scenarios,
            confidence_level=confidence_level,
            bil_index=bil_index,
        )
    
    def forward(
        self, 
        x_monthly: torch.Tensor, 
        x_daily: torch.Tensor,
        pi: torch.Tensor = None, 
        sigma: torch.Tensor = None,
    ):
        """
        Args:
            x_monthly: (B, Seq, monthly_dim) 월간 features
            x_daily: (B, Seq, K, n_daily_vars) 일간 데이터
            pi: (B, N) 사전 기대수익률 (None이면 자동 추정)
            sigma: (B, N, N) 사전 공분산 (None이면 자동 추정)
            
        Returns:
            weights: (B, N) 포트폴리오 비중
            regime_probs: (B, n_regimes) regime 확률 (Loss 계산용)
        """
        B, Seq, _ = x_monthly.shape
        
        # 1. MIDAS: 각 월의 일간 데이터 → 월간 압축
        midas_seq = []
        for t in range(Seq):
            midas_t = self.midas_layer(x_daily[:, t])  # (B, n_daily_vars)
            midas_seq.append(midas_t)
        midas_features = torch.stack(midas_seq, dim=1)  # (B, Seq, n_daily_vars)
        
        # 2. concat → GRU
        combined = torch.cat([x_monthly, midas_features], dim=-1)
        _, h_n = self.gru(combined)
        hidden = self.drop(h_n[-1])  # (B, hidden_dim)
        
        # 3. Regime (HMM prob은 Input에 안 들어감)
        regime_probs = self.regime_head(hidden)
        
        # 4. BL parameters (FiLM conditioned)
        p, q, omega = self.bl_head(hidden, regime_probs)
        
        # 5. BL formula → μ_BL, Σ_BL
        if pi is None:
            pi = x_monthly[:, :, :self.num_assets].mean(dim=1)
        if sigma is None:
            asset_ret = x_monthly[:, :, :self.num_assets]
            sigma = estimate_covariance(asset_ret, shrinkage=0.1)
        
        mu_bl, sigma_out = black_litterman_formula(
            p, q, omega, pi, sigma, tau=0.05)
        
        # 6. Regime → λ + is_crisis
        lambda_risk = self.regime_head.get_lambda_risk(regime_probs)
        is_crisis = self.regime_head.get_is_crisis(regime_probs)
        
        # 7. Mean-CVaR
        weights = self.opt_layer(
            mu_bl, sigma_out,
            lambda_risk=lambda_risk,
            is_crisis=is_crisis,
        )
        
        return weights, regime_probs
    
    def param_count(self) -> int:
        """모델 파라미터 수 보고."""
        groups = {
            'MIDAS': self.midas_layer,
            'GRU': self.gru,
            'RegimeHead': self.regime_head,
            'SharedBLHead': self.bl_head,
        }
        total = 0
        for name, module in groups.items():
            n = sum(p.numel() for p in module.parameters())
            print(f"  {name:20s}: {n:>6,}")
            total += n
        print(f"  {'TOTAL':20s}: {total:>6,}")
        return total


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Models Self-Test: 5 Models × 3 Omega × 2 Sigma")
    print("=" * 60)
    
    B, S, F_in = 4, 12, 10
    num_assets = 10
    x = torch.randn(B, S, F_in)
    
    omega_modes = ['learnable', 'formula', 'hybrid']
    sigma_modes = ['prior', 'residual']
    
    # 빠른 테스트: TFT만 전체 모드 테스트, 나머지는 기본 모드
    test_configs = []
    
    # TFT: 모든 모드 조합 테스트
    for om in omega_modes:
        for sm in sigma_modes:
            test_configs.append(('tft', om, sm))
    
    # 나머지 모델: 기본 모드만 테스트
    for m_name in ['lstm', 'gru', 'tcn', 'transformer']:
        test_configs.append((m_name, 'learnable', 'prior'))
    
    passed = 0
    failed = 0
    
    for m_name, om, sm in test_configs:
        label = f"{m_name.upper()} (omega={om}, sigma={sm})"
        try:
            model = get_model(m_name, F_in, num_assets, omega_mode=om, sigma_mode=sm)
            w = model(x)
            
            # 검증: 출력 형태 + 합계 ≈ 1.0
            assert w.shape == (B, num_assets), f"Shape mismatch: {w.shape}"
            weight_sum = w[0].sum().item()
            assert 0.95 <= weight_sum <= 1.05, f"Weight sum out of range: {weight_sum}"
            
            # 파라미터 수 계산
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  [OK] {label:50s} | params: {n_params:>6,} | w_sum: {weight_sum:.4f}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {label:50s} | ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'=' * 60}")
