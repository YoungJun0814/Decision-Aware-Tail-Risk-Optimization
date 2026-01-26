"""
Run XAI Analysis (TFT Focus)
============================
[Step 6] "Decision-Aware" 설명력 분석 실행

이 스크립트는 모델(TFT 등)이 왜 특정 시점, 특정 자산에 투자했는지를 분석합니다.
주요 분석 내용:
1. Variable Selection: TFT가 어떤 거시변수/자산수익률을 중요하게 보았는가?
2. Saliency Map: 특정 자산 비중 결정에 과거 데이터가 어떤 영향을 주었는가?
3. Stress Test: 시장 충격(Crash) 시 모델은 어떻게 반응하는가?
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

# Add project root to path for imports when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import prepare_training_data, ASSET_TICKERS
from src.models import get_model
from src.explainability import GradientSaliency, TFTAnalyzer, CounterfactualAnalyzer
from src.utils import get_device, set_seed

def run_analysis():
    # GPU 사용 가능 시 GPU 사용, XAI 분석 시에만 CuDNN 비활성화
    device = torch.device(get_device())
    
    # XAI에서 Input Gradient 계산 시 CuDNN의 RNN backward 호환성 문제 방지
    # CuDNN을 끄면 RNN 계열 모델에서도 Input Saliency 계산 가능
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False
        print("[INFO] CuDNN 비활성화됨 (XAI Input Gradient 호환성 확보)")
    
    set_seed(42)
    
    print("\n" + "="*50)
    print("XAI 분석 시작: TFT 모델 해석")
    print("="*50)
    
    # 1. 데이터 로드
    print("[1] 데이터 로드 중...")
    start_date = '2005-01-01'
    end_date = '2024-01-01'
    seq_length = 12
    
    X_tensor, y_tensor, vix_tensor, scaler, asset_names, _ = prepare_training_data(
        start_date=start_date,
        end_date=end_date,
        seq_length=seq_length
    )
    X_tensor = X_tensor.to(device).float()
    
    # 2. TFT 모델 초기화
    print("[2] TFT 모델 초기화 중...")
    input_dim = X_tensor.shape[-1]
    num_assets = len(ASSET_TICKERS)
    
    model = get_model('tft', input_dim, num_assets, device=device)
    
    # 주의: 원래는 저장된 가중치(best_model.pth)를 로드해야 합니다.
    # 여기서는 데모를 위해 약식으로 5 Epoch 동안 빠르게 재학습하여 의미 있는 가중치를 생성합니다.
    # (랜덤 가중치로는 랜덤한 해석만 나옵니다)
    print("    (의미 있는 설명을 위해 5 Epoch 약식 재학습 수행)...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss() # 속도를 위해 MSE 사용 (약식)
    
    for _ in range(5):
        w = model(X_tensor)
        loss = loss_fn(w, y_tensor.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    model.eval()
    
    # 3. Variable Selection (VSN) 분석
    print("\n[3] 변수 선택 네트워크(VSN) 중요도 분석...")
    tft_xai = TFTAnalyzer(model)
    
    try:
        # 전역 중요도 추출
        vsn_weights = tft_xai.compute_variable_selection_weights(X_tensor)
        
        # 특성 이름 매핑 (현재는 자산 수익률이 곧 Feature라고 가정)
        # 실제로는 [SPY, XLV, TLT, GLD, BIL] + [Macro...] 순서
        features = ASSET_TICKERS
        if len(features) < len(vsn_weights):
            features += [f"Feat_{i}" for i in range(len(features), len(vsn_weights))]
        
        print(f"    Raw Weights: {vsn_weights}")
        
        # 차트 저장
        save_path = "results/plots/xai_tft_variable_importance.png"
        tft_xai.plot_selection_weights(vsn_weights, features[:len(vsn_weights)], save_path=save_path)
        print(f"    [성공] 변수 중요도 차트 저장됨: {save_path}")
        
    except Exception as e:
        print(f"    [오류] VSN 분석 실패 (모델이 TFT가 아닐 수 있음): {e}")
        # import traceback
        # traceback.print_exc()
    
    # 4. Input Saliency (민감도) 분석
    print("\n[4] 입력 민감도(Saliency) 분석...")
    saliency = GradientSaliency(model)
    
    # 마지막 시점(가장 최근 데이터) 분석
    sample_idx = -1 
    target_idx = 0 # SPY 비중에 대한 설명
    
    sal_map = saliency.compute(X_tensor, target_asset_idx=target_idx)
    
    # 히트맵 저장
    heatmap_path = "results/plots/xai_saliency_heatmap.png"
    saliency.plot_heatmap(
        sal_map, 
        features, 
        sample_idx=sample_idx, 
        title="Input Saliency Heatmap (Last Time Step)", 
        save_path=heatmap_path
    )
    print(f"    [성공] Saliency 히트맵 저장됨: {heatmap_path}")
    
    # 5. Counterfactual Analysis (스트레스 테스트)
    print("\n[5] 스트레스 테스트 (시장 폭락 시나리오)...")
    cf = CounterfactualAnalyzer(model)
    
    # 최근 데이터를 기준으로 SPY(인덱스 0) 변동성 충격 시뮬레이션
    # scales: 표준편차의 몇 배만큼 충격을 줄 것인가 (예: -3 sigma)
    # mode='add' 사용
    crash_results = cf.perturb_feature(
        X_tensor[-10:], # 최근 10개 샘플 사용
        feature_idx=0,  # SPY
        scales=[-3.0, -2.0, -1.0, 0.0, 1.0], 
        mode='add' 
    )
    
    # 민감도 곡선 저장
    cf_path = "results/plots/xai_stress_test.png"
    cf.plot_sensitivity_curve(
        crash_results, 
        asset_names=ASSET_TICKERS, 
        feature_name="SPY Shock (Sigma)", 
        title="Portfolio Weight Change during Market Crash",
        save_path=cf_path
    )
    # print(f"    [성공] 스트레스 테스트 차트 저장됨: {cf_path}")
    
    print("\n" + "="*50)
    print("XAI 분석 완료.")
    print("="*50)

if __name__ == "__main__":
    run_analysis()
