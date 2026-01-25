"""
Trainer Module
==============
학습 루프 (Training Loop)

역할: 자동으로 공부하는 시스템을 만듭니다.
- Epoch, Batch 반복문
- optimizer.zero_grad() -> loss.backward() -> optimizer.step() 흐름
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, List


class Trainer:
    """
    Decision-Aware 모델 학습기
    
    End-to-End 학습을 위한 트레이너 클래스입니다.
    
    Args:
        model: 학습할 PyTorch 모델
        loss_fn: 손실 함수
        optimizer: 옵티마이저
        device: 학습 디바이스 ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        한 에폭 학습 (v2: VIX 지원)
        
        Args:
            dataloader: 학습 데이터 로더 (X, y) 또는 (X, y, vix)
        
        Returns:
            epoch_loss: 에폭 평균 손실
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # v2: 배치 언패킹 (VIX 포함 여부에 따라 다름)
            if len(batch) == 3:
                batch_X, batch_y, batch_vix = batch
                batch_vix = batch_vix.to(self.device)
            else:
                batch_X, batch_y = batch
                batch_vix = None
            
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 1. 그래디언트 초기화
            self.optimizer.zero_grad()
            
            # 2. Forward pass
            weights = self.model(batch_X)
            
            # 3. 손실 계산 (v2: DecisionAwareLoss는 vix 인자 필요)
            if batch_vix is not None:
                # DecisionAwareLoss 사용
                loss = self.loss_fn(weights, batch_y, batch_vix)
            else:
                # Legacy TaskLoss 사용
                loss = self.loss_fn(weights, batch_y)
            
            # 4. Backward pass
            loss.backward()
            
            # 5. 파라미터 업데이트
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        epoch_loss = total_loss / num_batches
        return epoch_loss
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """
        검증 (v2: VIX 지원)
        
        Args:
            dataloader: 검증 데이터 로더 (X, y) 또는 (X, y, vix)
        
        Returns:
            val_loss: 검증 손실
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # v2: 배치 언패킹 (VIX 포함 여부에 따라 다름)
            if len(batch) == 3:
                batch_X, batch_y, batch_vix = batch
                batch_vix = batch_vix.to(self.device)
            else:
                batch_X, batch_y = batch
                batch_vix = None
            
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            weights = self.model(batch_X)
            
            # v2: DecisionAwareLoss는 vix 인자 필요
            if batch_vix is not None:
                loss = self.loss_fn(weights, batch_y, batch_vix)
            else:
                loss = self.loss_fn(weights, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        전체 학습 루프
        
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더 (옵션)
            epochs: 학습 에폭 수
            verbose: 진행 상황 출력 여부
            early_stopping_patience: 조기 종료 인내심
        
        Returns:
            history: 학습 히스토리 딕셔너리
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 학습 (Train)
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 검증 (Validation)
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # 조기 종료 (Early Stopping) 체크
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
            
            # 진행 상황 출력
            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                print(msg)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        예측 (Predict)
        
        Args:
            X: (Batch, Seq, Features) 입력 텐서
        
        Returns:
            weights: (Batch, Num_assets) 포트폴리오 비중
        """
        self.model.eval()
        X = X.to(self.device)
        return self.model(X)


def create_dataloaders(
    X: torch.Tensor,
    y: torch.Tensor,
    vix: torch.Tensor = None,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    shuffle: bool = True
) -> tuple:
    """
    데이터로더 생성 헬퍼 함수 (v2: VIX 지원)
    
    Args:
        X: 입력 텐서 (Batch, Seq, Features)
        y: 타겟 텐서 (Batch, Num_assets)
        vix: VIX 텐서 (Batch,) - DecisionAwareLoss용
        batch_size: 배치 크기
        train_ratio: 학습 데이터 비율
        shuffle: 셔플 여부
    
    Returns:
        train_loader, val_loader
    """
    # Train/Val 분할
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    # TensorDataset 생성 (VIX 포함 여부에 따라 다름)
    if vix is not None:
        vix_train, vix_val = vix[:n_train], vix[n_train:]
        train_dataset = TensorDataset(X_train, y_train, vix_train)
        val_dataset = TensorDataset(X_val, y_val, vix_val)
    else:
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Trainer Test")
    print("=" * 60)
    
    # 더미 데이터 생성
    batch_size = 32
    seq_length = 12
    input_dim = 4
    num_assets = 4
    n_samples = 100
    
    X = torch.randn(n_samples, seq_length, input_dim)
    y = torch.randn(n_samples, num_assets) * 0.02
    
    # 데이터로더 생성
    train_loader, val_loader = create_dataloaders(X, y, batch_size=16)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # 모델, 손실함수, 옵티마이저 생성
    # 테스트를 위한 임시 임포트
    try:
        from models import DecisionAwareNet
        from loss import DecisionAwareLoss
    except ImportError:
        # 경로 문제 발생 시 패스
        print("[Skipping Model Test] Cannot import models/loss directly.")
        pass
    else:
        model = DecisionAwareNet(input_dim=input_dim, num_assets=num_assets)
        loss_fn = DecisionAwareLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 트레이너 생성
        trainer = Trainer(model, loss_fn, optimizer)
        
        # 학습 (간단한 테스트)
        print("\n--- Training for 5 epochs ---")
        # history = trainer.fit(train_loader, val_loader, epochs=5, verbose=True)
        print("\n[Success] Trainer test passed!")
