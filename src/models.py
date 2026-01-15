"""
Models Module
=============
[Step 2] 딥러닝 모델

역할: 과거 데이터를 보고 "최적화에 필요한 파라미터(θ)"를 예측합니다.
- Encoder: LSTM/GRU를 사용해 시계열 패턴을 읽습니다.
- Head: Linear 레이어를 통해 최적화기에게 넘겨줄 값을 출력합니다.

주의: 여기서 바로 비중(w)이 나오는 게 아닙니다! 비중을 계산하기 위한 재료가 나옵니다.
"""

import torch
import torch.nn as nn


class DecisionAwareNet(nn.Module):
    """
    Decision-Aware Neural Network
    
    시장 데이터를 읽고 최적화에 필요한 파라미터를 예측하는 네트워크입니다.
    
    Args:
        input_dim: 입력 특성 개수 (자산 수 + 거시변수 수)
        hidden_dim: LSTM 히든 레이어 차원
        num_layers: LSTM 레이어 수
        num_assets: 자산 개수 (출력 차원)
        dropout: 드롭아웃 비율
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_assets: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(DecisionAwareNet, self).__init__()
        
        self.input_dim = input_dim
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        
        # 1. Feature Extractor (시장 상황 판단)
        # LSTM: 시계열 데이터(과거 12개월 등)를 읽어서 특징 추출
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 2. Parameter Predictor (최적화에 필요한 파라미터 예측)
        # 예: 다음 달 기대수익률이나 변동성을 예측
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_assets)
        )
        
        # 3. Optimization Layer (cvxpylayers)
        # TODO: Researcher가 CVaR 수식 확정하면 여기에 구현할 예정
        self.optimization_layer = None
        
        # TODO: Researcher가 출력 활성화 함수를 확정하면 교체
        # 지금은 기본값으로 Tanh 사용 (출력 범위 제한)
        # 옵션: nn.Softplus(), nn.Sigmoid(), nn.Identity()
        self.output_activation = nn.Tanh()

    def forward(self, x: torch.Tensor, use_optimization: bool = False) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (Batch, Sequence, Features) 형태의 입력 텐서
            use_optimization: cvxpylayers 최적화 레이어 사용 여부
        
        Returns:
            weights: (Batch, Num_assets) 형태의 포트폴리오 비중
        """
        # Step 1: 시장 데이터 읽기
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # 마지막 시점의 정보
        
        # Step 2: 예측 파라미터 생성
        pred_params = self.fc_layers(last_hidden)
        
        # Step 3: 출력 활성화 (값의 범위 제한)
        # TODO: Researcher가 활성화 함수 교체 가능
        pred_params = self.output_activation(pred_params)
        
        # Step 4: 최적화 레이어 (지금은 Softmax로 비중 합 1만 맞춤)
        # TODO: Researcher가 CVaR 수식 확정하면 cvxpylayers로 교체
        if use_optimization and self.optimization_layer is not None:
            weights = self.optimization_layer(pred_params)
        else:
            # 임시: Softmax로 비중 합 1 맞춤
            weights = torch.softmax(pred_params, dim=1)
        
        return weights
    
    def get_predicted_params(self, x: torch.Tensor) -> torch.Tensor:
        """
        최적화 레이어에 들어가기 전 예측 파라미터만 반환
        (디버깅/분석용)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        pred_params = self.fc_layers(last_hidden)
        pred_params = self.output_activation(pred_params)
        return pred_params


class TransformerNet(nn.Module):
    """
    Transformer 기반 네트워크 (대안)
    
    TODO: Researcher가 Transformer 아키텍처를 선택하면 완성
    
    Args:
        input_dim: 입력 특성 개수
        num_assets: 자산 개수
        d_model: Transformer 모델 차원
        nhead: 어텐션 헤드 수
        num_layers: Transformer 레이어 수
    """
    
    def __init__(
        self,
        input_dim: int,
        num_assets: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(TransformerNet, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, num_assets)
        
        # TODO: Researcher가 출력 활성화 함수를 확정하면 교체
        self.output_activation = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # 입력 프로젝션
        x = self.input_projection(x)
        
        # Transformer 인코더
        transformer_out = self.transformer(x)
        
        # 마지막 시점 사용
        last_hidden = transformer_out[:, -1, :]
        
        # 출력
        pred_params = self.fc_out(last_hidden)
        pred_params = self.output_activation(pred_params)
        
        # 임시: Softmax로 비중 합 맞춤
        weights = torch.softmax(pred_params, dim=1)
        
        return weights


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Model Test")
    print("=" * 60)
    
    # 테스트 입력: (배치크기 10, 과거 12개월, 변수 4개)
    batch_size = 10
    seq_length = 12
    input_dim = 4  # 자산 수 (Macro 없는 경우)
    num_assets = 4
    
    dummy_input = torch.randn(batch_size, seq_length, input_dim)
    
    # LSTM 모델 테스트
    print("\n--- LSTM Model Test ---")
    lstm_model = DecisionAwareNet(input_dim=input_dim, num_assets=num_assets)
    lstm_output = lstm_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {lstm_output.shape}")  # (10, 4) 예상
    print(f"Weights sum (should be ~1): {lstm_output.sum(dim=1)[:3]}")
    
    # Transformer 모델 테스트
    print("\n--- Transformer Model Test ---")
    transformer_model = TransformerNet(input_dim=input_dim, num_assets=num_assets)
    transformer_output = transformer_model(dummy_input)
    print(f"Output shape: {transformer_output.shape}")
    print(f"Weights sum (should be ~1): {transformer_output.sum(dim=1)[:3]}")
    
    print("\n[Success] Model tests passed!")
