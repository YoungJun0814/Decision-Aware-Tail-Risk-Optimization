"""
Decision-Aware Tail Risk Optimization (v2)
===========================================
Main entry point for the project.

이 파일은 전체 파이프라인을 연결하고 실행합니다:
1. 데이터 로딩 (data_loader.py) - VIX 포함
2. 모델 생성 (models.py) - TFT 및 벤치마크 모델
3. 손실 함수 설정 (loss.py) - DecisionAwareLoss
4. 학습 실행 (trainer.py)
5. 결과 확인 및 저장

사용법:
    python main.py
"""

import torch
import torch.optim as optim
import numpy as np
import warnings

# 경고 무시 (yfinance 관련)
warnings.filterwarnings('ignore')

# 프로젝트 모듈 임포트
from src.data_loader import (
    prepare_training_data, 
    get_monthly_asset_data,
    ASSET_TICKERS,
    INVERSE_ETF_INDEX
)
from src.models import get_model  # v2.1: 모델 팩토리
from src.loss import DecisionAwareLoss  # v2: DecisionAwareLoss 사용
from src.trainer import Trainer, create_dataloaders


def main():
    """
    메인 실행 함수
    """
    print("=" * 70)
    print("  Decision-Aware Tail Risk Optimization (v2)")
    print("  End-to-End Learning Pipeline")
    print("=" * 70)
    
    # =========================================================================
    # 설정 (Configuration)
    # =========================================================================
    config = {
        # 데이터 설정
        'start_date': '2005-01-01',
        'end_date': '2024-01-01',
        'seq_length': 12,  # 12개월 lookback
        
        # 모델 설정 (lstm, gru, tcn, transformer, tft 중 선택)
        'model_type': 'tft', 
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.2,
        
        # 학습 설정
        'batch_size': 32,
        'learning_rate': 1e-3,
        'epochs': 50,
        'train_ratio': 0.8,
        
        # Loss 함수 설정 (DecisionAwareLoss v2)
        'eta': 1.0,              # 리스크 페널티 가중치
        'kappa_base': 0.001,     # 기본 거래비용 계수
        'kappa_vix_scale': 0.0001,  # VIX 연동 거래비용 스케일
        
        # 기타
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42
    }
    
    print(f"\n[Config]")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 재현성을 위한 시드 설정
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # =========================================================================
    # Step 1: 데이터 로딩 (v2: VIX 포함)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 1] Loading and Preprocessing Data (v2: with VIX)")
    print("=" * 70)
    
    # v2: prepare_training_data는 이제 vix_tensor와 y_dates도 반환합니다
    X, y, vix, scaler, asset_names, _ = prepare_training_data(
        start_date=config['start_date'],
        end_date=config['end_date'],
        seq_length=config['seq_length'],
        normalize=True
    )
    
    print(f"\nAssets: {asset_names}")
    print(f"X shape: {X.shape}  (Batch, Seq, Features)")
    print(f"y shape: {y.shape}  (Batch, Num_assets)")
    print(f"VIX shape: {vix.shape}  (Batch,)")
    
    # v2: 데이터로더에 VIX도 포함
    train_loader, val_loader = create_dataloaders(
        X, y, vix,  # VIX 추가
        batch_size=config['batch_size'],
        train_ratio=config['train_ratio']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # =========================================================================
    # Step 2: 모델 생성
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 2] Creating Model")
    print("=" * 70)
    
    input_dim = X.shape[-1]  # Features 차원
    num_assets = len(asset_names)
    
    # v2.1: Model Factory 사용 (5 Benchmark Models)
    model = get_model(
        model_type=config['model_type'],
        input_dim=input_dim,
        num_assets=num_assets,
        device=config['device']
    )
    
    print(f"\nModel Architecture: {config['model_type'].upper()}")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Output (assets): {num_assets}")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # =========================================================================
    # Step 3: 손실 함수 및 옵티마이저 (v2: DecisionAwareLoss)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 3] Setting up Loss Function and Optimizer (v2)")
    print("=" * 70)
    
    # v2: DecisionAwareLoss 사용 (VIX 연동 거래비용)
    loss_fn = DecisionAwareLoss(
        eta=config['eta'],
        kappa_base=config['kappa_base'],
        kappa_vix_scale=config['kappa_vix_scale']
    )
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print(f"\nLoss Function: DecisionAwareLoss")
    print(f"  - eta (Risk): {config['eta']}")
    print(f"  - kappa_base (Transaction Cost): {config['kappa_base']}")
    print(f"  - kappa_vix_scale (VIX Scale): {config['kappa_vix_scale']}")
    print(f"Optimizer: Adam (lr={config['learning_rate']})")
    
    # =========================================================================
    # Step 4: 학습
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 4] Training Model")
    print("=" * 70)
    
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=config['device']
    )
    
    print(f"\nDevice: {config['device']}")
    print(f"Epochs: {config['epochs']}")
    print("-" * 50)
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        verbose=True,
        early_stopping_patience=10
    )
    
    # =========================================================================
    # Step 5: 결과 확인
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 5] Results")
    print("=" * 70)
    
    # 최종 손실
    print(f"\nFinal Train Loss: {history['train_losses'][-1]:.6f}")
    if history['val_losses']:
        print(f"Final Val Loss: {history['val_losses'][-1]:.6f}")
        print(f"Best Val Loss: {history['best_val_loss']:.6f}")
    
    # 샘플 예측
    print(f"\n--- Sample Predictions ---")
    sample_X = X[:5].to(config['device'])
    sample_weights = trainer.predict(sample_X)
    
    print(f"Input shape: {sample_X.shape}")
    print(f"Output weights shape: {sample_weights.shape}")
    print(f"\nSample portfolio weights:")
    for i, w in enumerate(sample_weights.cpu().numpy()):
        weights_str = ", ".join([f"{asset}: {wt:.2%}" for asset, wt in zip(asset_names, w)])
        print(f"  Sample {i+1}: {weights_str}")
    
    # =========================================================================
    # 완료
    # =========================================================================
    print("\n" + "=" * 70)
    print("[SUCCESS] Training Complete!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Run benchmark: python src/benchmark.py")
    print("  2. Run XAI Analysis: python run_xai.py")


if __name__ == "__main__":
    main()
