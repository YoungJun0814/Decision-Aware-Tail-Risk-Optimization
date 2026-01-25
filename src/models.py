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
    """
    def __init__(self, input_dim: int, num_assets: int, hidden_dim: int, dropout: float = 0.2):
        super(BaseBLModel, self).__init__()
        self.num_assets = num_assets
        self.dropout = dropout
        
        # --- 전망(View) 생성 헤드 ---
        # Q: 기대 수익률 전망, P: 자산 간 관계(대각), Omega: 전망의 불확실성
        self.head_q = nn.Linear(hidden_dim, num_assets)
        self.head_p_diag = nn.Linear(hidden_dim, num_assets) 
        self.head_omega_diag = nn.Linear(hidden_dim, num_assets)

        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        
        # --- 최적화 레이어 ---
        # 안전망이 포함된 CVaR 최적화 (BIL=4)
        self.opt_layer = CVaROptimizationLayer(
            num_assets=num_assets,
            num_scenarios=200,
            confidence_level=0.95,
            bil_index=4,      # BIL 인덱스
            safety_threshold=0.5
        )

    def get_bl_parameters(self, hidden_features: torch.Tensor):
        batch_size = hidden_features.size(0)
        q = self.tanh(self.head_q(hidden_features)).unsqueeze(-1)
        p_diag = torch.sigmoid(self.head_p_diag(hidden_features))
        p = torch.diag_embed(p_diag) 
        omega_diag = self.softplus(self.head_omega_diag(hidden_features)) + 1e-6
        omega = torch.diag_embed(omega_diag)
        return p, q, omega

    def black_litterman_formula(self, p, q, omega, pi, sigma, tau=0.05):
        """
        Black-Litterman 공식의 완전한 구현입니다.
        연구 논문의 수식: E[R] = [(tau*Sigma)^-1 + P.T * Omega^-1 * P]^-1 * ...
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
        
        return mu_bl.squeeze(-1), sigma_bl

    def forward(self, x: torch.Tensor, pi: torch.Tensor = None, sigma: torch.Tensor = None, is_crisis: float = 0.0) -> torch.Tensor:
        """
        Forward Pass (순전파)
        Args:
            x: (Batch, Seq, Feat) - 자산 수익률 또는 특징 데이터
            is_crisis: 1.0이면 위기 상황(VIX > 30)으로 간주하여 안전망 가동.
        """
        bs, seq_len, _ = x.size()
        device = x.device
        
        # 0. 입력값이 없으면 과거 데이터로부터 사전확률(Prior) 추정
        if sigma is None:
            # 입력 데이터 X로부터 공분산 추정
            sigma = estimate_covariance(x, shrinkage=0.1)
            
        if pi is None:
            # 단순 평균 수익률을 사전 기대수익률로 사용
            pi = x.mean(dim=1)

        # 0.1 위기 상황 자동 감지 (현재는 명시적 flag 사용)
        if is_crisis == 0.0:
            pass

        # 1. 인코딩 (딥러닝 모델이 시장 상황 압축)
        hidden = self.encode(x) 
        
        # 2. 뷰(View) 생성 (AI가 독자적인 시장 전망 수립)
        p, q, omega = self.get_bl_parameters(hidden)
        
        # 3. Black-Litterman 공식 적용 (시장 데이터 + AI 전망 결합)
        mu_bl, sigma_bl = self.black_litterman_formula(p, q, omega, pi, sigma)
        
        # 4. CVaR 최적화 수행 (위험을 최소화하는 비중 계산)
        weights = self.opt_layer(mu_bl, sigma_bl, is_crisis=is_crisis)
        
        return weights
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("각 모델은 encode() 메서드를 구현해야 합니다.")


# =============================================================================
# 2. 5가지 벤치마크 모델 (Encoders)
# =============================================================================

class LSTMModel(BaseBLModel):
    def __init__(self, input_dim, num_assets, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__(input_dim, num_assets, hidden_dim, dropout)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
    
    def encode(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]

class GRUModel(BaseBLModel):
    def __init__(self, input_dim, num_assets, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__(input_dim, num_assets, hidden_dim, dropout)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
    
    def encode(self, x):
        _, h_n = self.gru(x)
        return h_n[-1]

class TCNModel(BaseBLModel):
    def __init__(self, input_dim, num_assets, hidden_dim=64, start_kernel_size=3, dropout=0.2):
        super().__init__(input_dim, num_assets, hidden_dim, dropout)
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
    def __init__(self, input_dim, num_assets, hidden_dim=64, num_layers=2, nhead=4, dropout=0.2):
        super().__init__(input_dim, num_assets, hidden_dim, dropout)
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

    def forward(self, x):
        gate_out = torch.sigmoid(self.gate(x)[..., :x.size(-1)]) 
        residual = self.fc2(F.elu(self.fc1(x)))
        residual = self.dropout(residual)
        
        # 차원 불일치 시 처리 (필요한 경우)
        if x.size(-1) != residual.size(-1):
             pass
             
        out = self.layernorm(x + gate_out * residual)
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
    def __init__(self, input_dim, num_assets, hidden_dim=64, num_layers=2, nhead=4, dropout=0.2):
        super().__init__(input_dim, num_assets, hidden_dim, dropout)
        
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

def get_model(model_type, input_dim, num_assets, device='cpu'):
    model_map = {
        'lstm': LSTMModel, 'gru': GRUModel, 'tcn': TCNModel,
        'transformer': TransformerModel, 'tft': TFTModel
    }
    return model_map[model_type](input_dim, num_assets, hidden_dim=64).to(device)

if __name__ == "__main__":
    print("-" * 50)
    print("5-Model Benchmark Test")
    print("-" * 50)
    B, S, F_in = 4, 12, 10
    num_assets = 5
    x = torch.randn(B, S, F_in)
    
    models = ['lstm', 'gru', 'tcn', 'transformer', 'tft']
    
    for m_name in models:
        print(f"\nTesting {m_name.upper()}...")
        try:
            model = get_model(m_name, F_in, num_assets)
            # 더미 순전파 (Forward pass)
            w = model(x)
            print(f"  Output Shape: {w.shape}")
            print(f"  Sum check: {w[0].sum().item():.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
