"""
Benchmark Module (5 Deep Learning Models)
=========================================
[Step 5] 5개 모델 벤치마킹
- LSTM, GRU, TCN, Transformer, TFT
- 각 모델을 순차적으로 학습하고 성능(검증 손실, Sharpe 등)을 비교합니다.

사용법:
    python -m src.benchmark
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from src.data_loader import prepare_training_data, ASSET_TICKERS
from src.models import get_model
from src.loss import DecisionAwareLoss
from src.utils import set_seed, get_device

def train_and_evaluate(model_type, config, X_tensor, y_tensor, vix_tensor, scaler=None):
    """단일 모델 학습 및 평가"""
    device = config['device']
    input_dim = X_tensor.shape[-1]
    num_assets = y_tensor.shape[-1]
    
    print(f"\n[{model_type.upper()}] 모델 학습 시작...")
    
    # 1. 모델 생성
    model = get_model(model_type, input_dim, num_assets, device=device)
    
    # 2. 손실 함수 (DecisionAwareLoss) 및 옵티마이저
    loss_fn = DecisionAwareLoss(
        eta=config['eta'],
        kappa_base=config['kappa_base'],
        kappa_vix_scale=config['kappa_vix_scale']
    )
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 3. 데이터 분할 (Train/Val)
    train_size = int(len(X_tensor) * config['train_ratio'])
    X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
    vix_train, vix_val = vix_tensor[:train_size], vix_tensor[train_size:]
    
    # 4. 학습 루프
    best_val_loss = float('inf')
    patience = 20 # [MODIFIED] Early Stopping 조건을 10 -> 20으로 완화
    no_improve = 0
    val_weights = None  # 검증 결과 저장을 위한 변수 초기화
    
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        
        weights = model(X_train)
        loss = loss_fn(weights, y_train, vix_train)
        
        loss.backward()
        optimizer.step()

        # [LOGGING] 10 에포크마다 로그 출력
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{config['epochs']}], Loss: {loss.item():.4f}")
        
        # 검증 (Validation)
        model.eval()
        with torch.no_grad():
            current_val_weights = model(X_val)
            val_loss = loss_fn(current_val_weights, y_val, vix_val)
            
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            no_improve = 0
            val_weights = current_val_weights  # Best 가중치 저장
        else:
            no_improve += 1
            if no_improve >= patience:
                # Early Stopping
                break
                
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    
    # 5. 최종 지표 계산 (Validation Set 기준 Sharpe Ratio 등)
    # [FIX] 스케일링된 데이터를 원복(De-normalization)하여 실제 수익률 계산
    if val_weights is not None:
        # y_val (Target)은 Scaled 상태이므로 원복 필요
        if scaler is not None:
            # scaler.mean_, scaler.scale_ 사용 (앞쪽 num_assets 개수만큼)
            # y_val: (Batch, Assets)
            mean = torch.tensor(scaler.mean_[:num_assets], device=device).float()
            scale = torch.tensor(scaler.scale_[:num_assets], device=device).float()
            y_val_real = y_val * scale + mean
        else:
            y_val_real = y_val

        portfolio_returns = (val_weights * y_val_real).sum(dim=1).cpu().numpy()
        mean_ret = np.mean(portfolio_returns) * 12
        std_ret = np.std(portfolio_returns) * np.sqrt(12)
        sharpe = mean_ret / (std_ret + 1e-6)
        
        # MDD 계산
        from src.utils import calculate_mdd
        mdd = calculate_mdd(portfolio_returns)
    else:
        # 학습 실패 시
        mean_ret = 0.0
        sharpe = 0.0
        mdd = 0.0
    
    return {
        'model': model_type,
        'val_loss': best_val_loss,
        'sharpe': sharpe,
        'annual_return': mean_ret,
        'mdd': mdd,
        'val_returns': portfolio_returns if val_weights is not None else np.zeros(len(y_val))
    }

def run_5_model_benchmark():
    # 설정 (Config)
    config = {
        'start_date': '2005-01-01',
        'end_date': '2024-01-01',
        'seq_length': 12,
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100, 
        'train_ratio': 0.8,
        'eta': 1.0,
        'kappa_base': 0.001,
        'kappa_vix_scale': 0.0001,
        'device': get_device(),
        'seed': 42
    }
    
    set_seed(config['seed'])
    
    # 데이터 로드
    print("데이터 로딩 중...")
    X_tensor, y_tensor, vix_tensor, scaler, asset_names, y_dates = prepare_training_data(
        start_date=config['start_date'],
        end_date=config['end_date'],
        seq_length=config['seq_length']
    )
    
    X_tensor = X_tensor.to(config['device']).float()
    y_tensor = y_tensor.to(config['device']).float()
    vix_tensor = vix_tensor.to(config['device']).float()
    
    # 비교할 모델 리스트
    models = ['lstm', 'gru', 'tcn', 'transformer', 'tft']
    results = []
    
    print("\n" + "="*50)
    print("5개 모델 벤치마킹 시작")
    print("="*50)
    
    for m in models:
        try:
            res = train_and_evaluate(m, config, X_tensor, y_tensor, vix_tensor, scaler) # scaler 전달
            results.append(res)
        except Exception as e:
            print(f"[{m.upper()}] 학습 실패: {e}")
            import traceback
            traceback.print_exc()
            
    # 결과 요약
    print("\n" + "="*50)
    print("벤치마크 결과")
    print("="*50)
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        # Loss가 낮은 순서대로 정렬 (낮을수록 좋음)
        df_res = df_res.sort_values('val_loss') 
        print(df_res)
        
        # CSV 저장
        df_res.to_csv("benchmark_results.csv", index=False)
        print("\n결과가 benchmark_results.csv 파일로 저장되었습니다.")
    else:
        print("결과가 생성되지 않았습니다.")
    
    # [NEW] 시계열 수익률 저장 (Visualization용)
    if results:
        # Validation Dates 구하기
        train_size = int(len(X_tensor) * config['train_ratio'])
        val_dates = y_dates[train_size:]
        
        # DataFrame 생성
        df_returns = pd.DataFrame(index=val_dates)
        for res in results:
            df_returns[res['model']] = res['val_returns']
            
        df_returns.to_csv("benchmark_returns.csv")
        print("시계열 수익률 결과가 benchmark_returns.csv 파일로 저장되었습니다.")

if __name__ == "__main__":
    run_5_model_benchmark()