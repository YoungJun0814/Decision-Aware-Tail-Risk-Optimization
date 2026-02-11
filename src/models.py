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
                 omega_mode: str = 'learnable', sigma_mode: str = 'prior'):
        super(BaseBLModel, self).__init__()
        self.num_assets = num_assets
        self.dropout = dropout
        self.omega_mode = omega_mode
        self.sigma_mode = sigma_mode
        
        # --- 전망(View) 생성 헤드 ---
        # Q: 기대 수익률 전망 (-1 ~ +1, tanh)
        self.head_q = nn.Linear(hidden_dim, num_assets)
        # P: 자산 선택 행렬의 대각 원소 (0 ~ 1, sigmoid)
        self.head_p_diag = nn.Linear(hidden_dim, num_assets)
        
        # Omega 헤드: omega_mode에 따라 다르게 초기화
        if omega_mode == 'learnable':
            # 기존 방식: 신경망이 직접 Omega 출력
            self.head_omega_diag = nn.Linear(hidden_dim, num_assets)
        elif omega_mode == 'hybrid':
            # 하이브리드: 수식 기반 + 학습 가능 스케일링
            # α_i = exp(log_alpha_i), 초기값 0 → α = 1.0 (표준 BL과 동일)
            self.log_alpha = nn.Parameter(torch.zeros(num_assets))
        # formula 모드: 추가 파라미터 없음

        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        
        # Residual sigma 모드의 블렌딩 강도
        if sigma_mode == 'residual':
            self.lambda_blend = nn.Parameter(torch.tensor(0.1))
        
        # --- 최적화 레이어 ---
        # 안전망이 포함된 CVaR 최적화 (BIL=4)
        self.opt_layer = CVaROptimizationLayer(
            num_assets=num_assets,
            num_scenarios=200,
            confidence_level=0.95,
            bil_index=9,      # BIL 인덱스 (10자산 유니버스에서 마지막)
            safety_threshold=0.5
        )

    def get_bl_parameters(self, hidden_features: torch.Tensor, sigma: torch.Tensor = None):
        """
        BL 파라미터(P, Q, Omega) 생성.
        
        Args:
            hidden_features: (B, hidden_dim) 인코더 출력
            sigma: (B, N, N) 공분산 행렬 (formula/hybrid 모드에서 필요)
        
        Returns:
            p: (B, N, N) Pick 행렬
            q: (B, N, 1) 전망 벡터
            omega: (B, N, N) 불확실성 행렬
        """
        # Q: 기대 수익률 전망
        q = self.tanh(self.head_q(hidden_features)).unsqueeze(-1)
        
        # P: 대각 행렬
        p_diag = torch.sigmoid(self.head_p_diag(hidden_features))
        p = torch.diag_embed(p_diag)
        
        # Omega: 모드에 따라 다르게 계산
        if self.omega_mode == 'learnable':
            # 기존 방식: 신경망 직접 출력
            omega_diag = self.softplus(self.head_omega_diag(hidden_features)) + 1e-6
            omega = torch.diag_embed(omega_diag)
            
        elif self.omega_mode == 'formula':
            # 수식 기반: Ω_ii = τ · p_i² · Σ_ii
            omega = self._compute_omega_formula(p_diag, sigma, tau=0.05)
            
        elif self.omega_mode == 'hybrid':
            # 하이브리드: Ω_ii = α_i · τ · p_i² · Σ_ii
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
        
        try:
            inv_tau_sigma = torch.inverse(tau_sigma + 1e-6 * torch.eye(self.num_assets, device=p.device))
            inv_omega = torch.inverse(omega)
        except RuntimeError:
            inv_tau_sigma = torch.linalg.pinv(tau_sigma)
            inv_omega = torch.linalg.pinv(omega)

        term_b = torch.bmm(p.transpose(1, 2), torch.bmm(inv_omega, p))
        post_precision = inv_tau_sigma + term_b
        
        try:
            sigma_bl = torch.inverse(post_precision + 1e-6 * torch.eye(self.num_assets, device=p.device))
        except RuntimeError:
            sigma_bl = torch.linalg.pinv(post_precision)

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

    def forward(self, x: torch.Tensor, pi: torch.Tensor = None, sigma: torch.Tensor = None, is_crisis: float = 0.0) -> torch.Tensor:
        """
        Forward Pass (순전파)
        Args:
            x: (Batch, Seq, Feat) - 자산 수익률 또는 특징 데이터
            is_crisis: 1.0이면 위기 상황(VIX > 30)으로 간주하여 안전망 가동.
        """
        bs, seq_len, _ = x.size()
        device = x.device
        
        # 0. sigma/pi 추정 (미리 계산 — formula/hybrid 모드에서 BL 파라미터 생성 시 필요)
        if sigma is None:
            asset_returns = x[:, :, :self.num_assets]
            sigma = estimate_covariance(asset_returns, shrinkage=0.1)
            
        if pi is None:
            if 'asset_returns' not in locals():
                asset_returns = x[:, :, :self.num_assets]
            pi = asset_returns.mean(dim=1)

        # 1. 인코딩 (딥러닝 모델이 시장 상황 압축)
        hidden = self.encode(x) 
        
        # 2. 뷰(View) 생성 (sigma를 전달하여 formula/hybrid omega 계산에 활용)
        p, q, omega = self.get_bl_parameters(hidden, sigma=sigma)
        
        # 3. Black-Litterman 공식 적용 (시장 데이터 + AI 전망 결합)
        mu_bl, sigma_out = self.black_litterman_formula(p, q, omega, pi, sigma)
        
        # 4. CVaR 최적화 수행 (위험을 최소화하는 비중 계산)
        weights = self.opt_layer(mu_bl, sigma_out, is_crisis=is_crisis)
        
        return weights
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("각 모델은 encode() 메서드를 구현해야 합니다.")


# =============================================================================
# 2. 5가지 벤치마크 모델 (Encoders)
# =============================================================================

class LSTMModel(BaseBLModel):
    def __init__(self, input_dim, num_assets, hidden_dim=64, num_layers=2, dropout=0.2,
                 omega_mode='learnable', sigma_mode='prior'):
        super().__init__(input_dim, num_assets, hidden_dim, dropout, omega_mode, sigma_mode)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
    
    def encode(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]

class GRUModel(BaseBLModel):
    def __init__(self, input_dim, num_assets, hidden_dim=64, num_layers=2, dropout=0.2,
                 omega_mode='learnable', sigma_mode='prior'):
        super().__init__(input_dim, num_assets, hidden_dim, dropout, omega_mode, sigma_mode)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
    
    def encode(self, x):
        _, h_n = self.gru(x)
        return h_n[-1]

class TCNModel(BaseBLModel):
    def __init__(self, input_dim, num_assets, hidden_dim=64, start_kernel_size=3, dropout=0.2,
                 omega_mode='learnable', sigma_mode='prior'):
        super().__init__(input_dim, num_assets, hidden_dim, dropout, omega_mode, sigma_mode)
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
                 omega_mode='learnable', sigma_mode='prior'):
        super().__init__(input_dim, num_assets, hidden_dim, dropout, omega_mode, sigma_mode)
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
                 omega_mode='learnable', sigma_mode='prior'):
        super().__init__(input_dim, num_assets, hidden_dim, dropout, omega_mode, sigma_mode)
        
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
# 3. 모델 팩토리 함수
# =============================================================================

def get_model(model_type, input_dim, num_assets, device='cpu',
              omega_mode='learnable', sigma_mode='prior'):
    """
    모델 팩토리 함수.
    
    Args:
        model_type: 모델 아키텍처 ('lstm', 'gru', 'tcn', 'transformer', 'tft')
        input_dim: 입력 피처 차원
        num_assets: 자산 수
        device: 학습 디바이스
        omega_mode: Omega 산출 방식 ('learnable', 'formula', 'hybrid')
        sigma_mode: Sigma 반환 방식 ('prior', 'residual')
    
    Returns:
        model: 초기화된 BaseBLModel 인스턴스
    """
    model_map = {
        'lstm': LSTMModel, 'gru': GRUModel, 'tcn': TCNModel,
        'transformer': TransformerModel, 'tft': TFTModel
    }
    if model_type not in model_map:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(model_map.keys())}")
    
    return model_map[model_type](
        input_dim, num_assets, hidden_dim=64,
        omega_mode=omega_mode, sigma_mode=sigma_mode
    ).to(device)


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
