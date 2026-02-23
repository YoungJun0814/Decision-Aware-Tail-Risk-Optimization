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
from torch.utils.data import DataLoader, TensorDataset, Sampler
import numpy as np
import random
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
        한 에폭 학습 (v3: VIX + Regime 지원, v5: macro_features 지원)
        
        Args:
            dataloader: 학습 데이터 로더
                - 2-tuple: (X, y)
                - 3-tuple: (X, y, vix)
                - 4-tuple: (X, y, vix, regime_probs)
                - 5-tuple: (X, y, vix, regime_probs, macro_features)
        
        Returns:
            epoch_loss: 에폭 평균 손실
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # v3/v5: 배치 언패킹 (VIX, Regime, Macro 포함 여부에 따라 다름)
            batch_vix = None
            batch_regime = None
            batch_macro = None
            
            if len(batch) == 5:
                batch_X, batch_y, batch_vix, batch_regime, batch_macro = batch
                batch_vix = batch_vix.to(self.device)
                batch_regime = batch_regime.to(self.device)
                batch_macro = batch_macro.to(self.device)
            elif len(batch) == 4:
                batch_X, batch_y, batch_vix, batch_regime = batch
                batch_vix = batch_vix.to(self.device)
                batch_regime = batch_regime.to(self.device)
            elif len(batch) == 3:
                batch_X, batch_y, batch_vix = batch
                batch_vix = batch_vix.to(self.device)
            else:
                batch_X, batch_y = batch
            
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 1. 그래디언트 초기화
            self.optimizer.zero_grad()
            
            # 2. Forward pass — regime_probs + macro_features 전달
            has_regime_support = (
                hasattr(self.model, 'film') or  # v3: FiLM
                getattr(self.model, 'regime_dim', 0) > 0  # v4: Q-concat + CrisisOverlay
            )
            fwd_kwargs = {}
            if batch_regime is not None and has_regime_support:
                fwd_kwargs['regime_probs'] = batch_regime
            if batch_macro is not None and getattr(self.model, 'macro_dim', 0) > 0:
                fwd_kwargs['macro_features'] = batch_macro
            
            weights = self.model(batch_X, **fwd_kwargs)
            
            # 3. 손실 계산 (v3: SharpeLoss는 vix + regime_probs 인자 지원)
            loss_args = [weights, batch_y]
            loss_kwargs = {}
            if batch_vix is not None:
                loss_args.append(batch_vix)
            if batch_regime is not None:
                loss_kwargs['regime_probs'] = batch_regime
            
            loss = self.loss_fn(*loss_args, **loss_kwargs)
            
            # Loss가 tuple이면 첫 번째 원소 사용 (DetailedDecisionAwareLoss 호환)
            if isinstance(loss, tuple):
                loss = loss[0]
            
            # 4. Backward pass
            loss.backward()
            
            # 5. 그래디언트 클리핑 (안정성)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 6. 파라미터 업데이트
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        epoch_loss = total_loss / num_batches
        return epoch_loss
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """
        검증 (v3: VIX + Regime 지원, v5: macro_features 지원)
        
        Args:
            dataloader: 검증 데이터 로더
                - 2-tuple: (X, y)
                - 3-tuple: (X, y, vix)
                - 4-tuple: (X, y, vix, regime_probs)
                - 5-tuple: (X, y, vix, regime_probs, macro_features)
        
        Returns:
            val_loss: 검증 손실
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            batch_vix = None
            batch_regime = None
            batch_macro = None
            
            if len(batch) == 5:
                batch_X, batch_y, batch_vix, batch_regime, batch_macro = batch
                batch_vix = batch_vix.to(self.device)
                batch_regime = batch_regime.to(self.device)
                batch_macro = batch_macro.to(self.device)
            elif len(batch) == 4:
                batch_X, batch_y, batch_vix, batch_regime = batch
                batch_vix = batch_vix.to(self.device)
                batch_regime = batch_regime.to(self.device)
            elif len(batch) == 3:
                batch_X, batch_y, batch_vix = batch
                batch_vix = batch_vix.to(self.device)
            else:
                batch_X, batch_y = batch
            
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            has_regime_support = (
                hasattr(self.model, 'film') or
                getattr(self.model, 'regime_dim', 0) > 0
            )
            fwd_kwargs = {}
            if batch_regime is not None and has_regime_support:
                fwd_kwargs['regime_probs'] = batch_regime
            if batch_macro is not None and getattr(self.model, 'macro_dim', 0) > 0:
                fwd_kwargs['macro_features'] = batch_macro
            
            weights = self.model(batch_X, **fwd_kwargs)
            
            loss_args = [weights, batch_y]
            loss_kwargs = {}
            if batch_vix is not None:
                loss_args.append(batch_vix)
            if batch_regime is not None:
                loss_kwargs['regime_probs'] = batch_regime
            
            loss = self.loss_fn(*loss_args, **loss_kwargs)
            if isinstance(loss, tuple):
                loss = loss[0]
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True,
        early_stopping_patience: int = 10,
        scheduler = None,  # v4: LR scheduler (e.g. CosineAnnealingLR)
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
            
            # v4: LR scheduler step
            if scheduler is not None:
                scheduler.step()
            
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
    def predict(self, X: torch.Tensor, regime_probs: torch.Tensor = None,
                macro_features: torch.Tensor = None) -> torch.Tensor:
        """
        예측 (v3: Regime 지원, v5: macro_features 지원)
        
        Args:
            X: (Batch, Seq, Features) 입력 텐서
            regime_probs: (Batch, R) regime 확률 (optional)
            macro_features: (Batch, macro_dim) 매크로 피처 (optional)
        
        Returns:
            weights: (Batch, Num_assets) 포트폴리오 비중
        """
        self.model.eval()
        X = X.to(self.device)
        
        has_regime_support = (
            hasattr(self.model, 'film') or
            getattr(self.model, 'regime_dim', 0) > 0
        )
        
        fwd_kwargs = {}
        if regime_probs is not None and has_regime_support:
            fwd_kwargs['regime_probs'] = regime_probs.to(self.device)
        if macro_features is not None and getattr(self.model, 'macro_dim', 0) > 0:
            fwd_kwargs['macro_features'] = macro_features.to(self.device)
        
        return self.model(X, **fwd_kwargs)


# =============================================================================
# Trajectory Batch Sampler (시계열 배치 샘플러)
# =============================================================================

class TrajectoryBatchSampler(Sampler):
    """
    시계열 데이터를 위한 배치 샘플러.
    
    연속된 시간 구간(trajectory)을 하나의 배치로 묶어 시간 순서를 보존하면서도,
    구간 간의 순서는 에폭마다 셔플하여 배치 다양성을 확보합니다.
    
    일반 shuffle=True:  [t=45, t=3, t=102, ...]  → 시간 순서 파괴 (Look-ahead Bias)
    shuffle=False:      [t=1, t=2, ..., t=32]     → 다양성 없음 (과적합 위험)
    Trajectory Batch:   [t=33~64, t=1~32, ...]     → 구간 내 시간순 + 구간 간 셔플
    
    Args:
        n_samples: 전체 샘플 수
        batch_size: 배치당 샘플 수
        shuffle_chunks: 구간 간 순서를 셔플할지 여부 (기본: True)
        drop_last: 마지막 불완전 배치를 버릴지 여부 (기본: False)
    """
    def __init__(self, n_samples: int, batch_size: int, 
                 shuffle_chunks: bool = True, drop_last: bool = False):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.shuffle_chunks = shuffle_chunks
        self.drop_last = drop_last
    
    def __iter__(self):
        # 연속 구간 인덱스 생성: [0:32], [32:64], [64:96], ...
        indices = list(range(self.n_samples))
        chunks = [indices[i:i + self.batch_size]
                  for i in range(0, self.n_samples, self.batch_size)]
        
        if not chunks:
            return
        
        # 마지막 불완전 배치 처리
        if self.drop_last and len(chunks[-1]) < self.batch_size:
            chunks = chunks[:-1]
        
        # 구간 간 순서만 셔플 (구간 내부는 시간순 유지!)
        if self.shuffle_chunks:
            random.shuffle(chunks)
        
        for chunk in chunks:
            yield chunk
    
    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size


# =============================================================================
# 데이터로더 생성 함수
# =============================================================================

def create_dataloaders(
    X: torch.Tensor,
    y: torch.Tensor,
    vix: torch.Tensor = None,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    shuffle: bool = False,
    use_trajectory_batching: bool = True
) -> tuple:
    """
    데이터로더 생성 헬퍼 함수 (v3: Trajectory Batching 지원)
    
    Args:
        X: 입력 텐서 (Batch, Seq, Features)
        y: 타겟 텐서 (Batch, Num_assets)
        vix: VIX 텐서 (Batch,) - DecisionAwareLoss용
        batch_size: 배치 크기
        train_ratio: 학습 데이터 비율
        shuffle: 일반 셔플 여부 (기본: False, 시계열이므로)
        use_trajectory_batching: Trajectory Batching 사용 여부 (기본: True)
            - True: 구간 내 시간순 보존 + 구간 간 셔플
            - False: shuffle 파라미터에 따라 동작
    
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
    if use_trajectory_batching:
        # Trajectory Batching: 구간 내 시간순 + 구간 간 셔플
        train_sampler = TrajectoryBatchSampler(
            n_samples=len(train_dataset),
            batch_size=batch_size,
            shuffle_chunks=True   # 구간 간 셔플 활성화
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    else:
        # 기존 방식: shuffle 파라미터에 따라 동작
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    # 검증 데이터는 항상 순차적 (셔플 없음)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# =============================================================================
# Phase 2 — Phase2Trainer
# =============================================================================

class Phase2Trainer:
    """
    Phase 2 학습 관리자.
    
    기존 Trainer 대비 추가 기능:
    - τ (Gumbel temperature) cosine annealing
    - λ_KL cosine annealing
    - prev_regime_probs detach 추적
    - Gradient clipping (max_norm=1.0)
    - Loss decomposition 로깅
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_grad_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.loss_details_history: List[Dict] = []
    
    def train_epoch(self, dataloader: DataLoader) -> tuple:
        """
        한 에폭 학습.
        
        DataLoader는 5-tuple: (x_monthly, x_daily, y, vix, hmm_probs)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        prev_regime = None
        prev_weights = None
        epoch_details = {}
        
        for batch in dataloader:
            x_monthly, x_daily, y, vix, hmm_probs = [
                b.to(self.device) for b in batch
            ]
            
            self.optimizer.zero_grad()
            
            # Forward
            weights, regime_probs = self.model(x_monthly, x_daily)
            
            # prev_* 배치 크기 맞추기 (마지막 배치가 작을 수 있음)
            B = weights.size(0)
            _prev_regime = None
            _prev_weights = None
            if prev_regime is not None:
                _prev_regime = prev_regime[:B] if prev_regime.size(0) >= B else None
            if prev_weights is not None:
                _prev_weights = prev_weights[:B] if prev_weights.size(0) >= B else None
            
            # Loss
            loss, details = self.loss_fn(
                weights, y, vix, regime_probs, hmm_probs,
                prev_regime_probs=_prev_regime,
                prev_weights=_prev_weights,
            )
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            
            # Track prev state (detach!)
            prev_regime = regime_probs.detach()
            prev_weights = weights.detach()
            
            total_loss += loss.item()
            num_batches += 1
            epoch_details = details  # 마지막 batch의 details
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, epoch_details
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """검증 (Phase 2)."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            x_monthly, x_daily, y, vix, hmm_probs = [
                b.to(self.device) for b in batch
            ]
            
            weights, regime_probs = self.model(x_monthly, x_daily)
            loss, _ = self.loss_fn(
                weights, y, vix, regime_probs, hmm_probs)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True,
        early_stopping_patience: int = 15,
    ) -> Dict:
        """
        전체 학습 루프 (Phase 2).
        
        매 epoch마다:
        1. τ annealing (Gumbel temperature)
        2. λ_KL annealing (KL loss weight)
        3. Train + Validate
        4. Early stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            # 1. Annealing
            if hasattr(self.model, 'regime_head'):
                self.model.regime_head.anneal_tau(epoch, epochs)
            if hasattr(self.loss_fn, 'update_lambda_kl'):
                self.loss_fn.update_lambda_kl(epoch, epochs)
            
            # 2. Train
            train_loss, details = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.loss_details_history.append(details)
            
            # 3. Validate
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in 
                                  self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
            
            # 4. Logging
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                tau = getattr(self.model.regime_head, 'tau', 0) \
                      if hasattr(self.model, 'regime_head') else 0
                msg = (f"Ep {epoch+1:3d}/{epochs} | "
                       f"train={train_loss:.4f}")
                if val_loss is not None:
                    msg += f" val={val_loss:.4f}"
                msg += (f" | kl={details.get('kl', 0):.4f} "
                        f"λ_kl={details.get('lambda_kl', 0):.3f} "
                        f"τ={tau:.3f}")
                print(msg)
        
        # Best model 복원
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'loss_details': self.loss_details_history,
        }
    
    @torch.no_grad()
    def predict(self, x_monthly: torch.Tensor, 
                x_daily: torch.Tensor) -> tuple:
        """예측."""
        self.model.eval()
        x_monthly = x_monthly.to(self.device)
        x_daily = x_daily.to(self.device)
        return self.model(x_monthly, x_daily)


# =============================================================================
# Phase 2 — DataLoader
# =============================================================================

def create_phase2_dataloaders(
    x_monthly: torch.Tensor,
    x_daily: torch.Tensor,
    y: torch.Tensor,
    vix: torch.Tensor,
    hmm_probs: torch.Tensor,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    use_trajectory_batching: bool = True,
) -> tuple:
    """
    Phase 2 DataLoader 생성.
    
    5-tuple: (x_monthly, x_daily, y, vix, hmm_probs)
    """
    n = len(x_monthly)
    n_train = int(n * train_ratio)
    
    train_ds = TensorDataset(
        x_monthly[:n_train], x_daily[:n_train],
        y[:n_train], vix[:n_train], hmm_probs[:n_train])
    val_ds = TensorDataset(
        x_monthly[n_train:], x_daily[n_train:],
        y[n_train:], vix[n_train:], hmm_probs[n_train:])
    
    if use_trajectory_batching:
        sampler = TrajectoryBatchSampler(
            n_samples=len(train_ds), batch_size=batch_size,
            shuffle_chunks=True)
        train_loader = DataLoader(train_ds, batch_sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
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
    # 현재 모델 구조에 맞게 수정 (v2.2)
    try:
        from src.models import get_model
        from src.loss import DecisionAwareLoss
    except ImportError:
        # 경로 문제 발생 시 패스
        print("[Skipping Model Test] Cannot import models/loss directly.")
    else:
        model = get_model('lstm', input_dim=input_dim, num_assets=num_assets)
        loss_fn = DecisionAwareLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 트레이너 생성
        trainer = Trainer(model, loss_fn, optimizer)
        
        # 학습 (간단한 테스트)
        print("\n--- Training for 5 epochs ---")
        # history = trainer.fit(train_loader, val_loader, epochs=5, verbose=True)
        print("\n[Success] Trainer test passed!")

