# Experiment Log & Analysis Report

## Phase 1: 5-Asset Universe Experiment (Initial Test)

### 1. 실험 설정 (Experimental Setup)

- **자산 유니버스 (5개)**: `['SPY', 'XLV', 'TLT', 'GLD', 'BIL']`
    - 보수적인 자산군 위주, 공격 자산 부족 (SPY 유일)
- **기간**: 2007-07-01 ~ 2024-01-01
- **목적**: Black-Litterman + CVaR 프레임워크의 기초 성능 검증

### 2. 5-Model Benchmark 결과 (Leader: TFT)

| Rank | Model | Sharpe | Ann. Return | MDD | 비고 |
|------|-------|--------|-------------|-----|------|
| **1** | **TFT** | **0.337** | **3.03%** | **-16.58%** | **Best** |
| 2 | GRU | 0.335 | 3.03% | -16.65% | 미세한 차이 |
| 3 | TCN | 0.334 | 3.02% | -16.66% | |
| 4 | Transformer | 0.333 | 3.01% | -16.64% | |
| 5 | LSTM | 0.331 | 2.98% | -16.65% | |

> **Finding**: 모든 모델의 성능이 거의 동일 (동조화 현상). 그러나 미세하게 TFT가 가장 우수하여 TFT를 기준으로 후속 실험 진행.

### 3. Deep Dive Benchmarks (TFT 고정)

**A. Omega Mode 비교 (`Ω`: Views의 불확실성 행렬)**
- **Formula (`Ω = τ·p²·Σ`)**: **Sharpe 0.451** (압도적 1위), Ann. Return 4.72%
  > *설명*: Black-Litterman 공식에서 제안하는 이론적 방식을 그대로 사용. 시장 공분산(Σ)에 비례하여 불확실성을 설정하므로 가장 안정적이고 과적합 위험이 적음.
- **Hybrid (`Ω = α · Formula`)**: Sharpe 0.421 (2위)
  > *설명*: Formula 방식에 학습 가능한 스케일링 파라미터(α)를 추가. 이론적 구조를 유지하면서 AI가 불확실성의 크기만 조절하도록 함.
- **Learnable (`Ω = NN(x)`)**: Sharpe 0.340
  > *설명*: AI(신경망)가 데이터로부터 불확실성 행렬 전체를 직접 학습. 자유도가 가장 높아 데이터가 충분하지 않으면 과적합되기 쉬움.

**B. Sigma Mode 비교 (`Σ_out`: 최적화에 사용될 최종 공분산)**
- **Prior (`Σ_out = Σ_estimate`)**: **Sharpe 0.371** (1위)
  > *설명*: 데이터로부터 추정한 사전 공분산(Covariance Shrinkage)을 그대로 사용. BL 모델의 사후 공분산(Posterior Sigma) 계산 과정을 최적화 파이프라인에서 제외하여 학습 안정성을 높임.
- **Residual (`Σ_out = Σ + λ(Σ_bl - Σ)`)**: Sharpe 0.300
  > *설명*: 사전 공분산(Σ)과 사후 공분산(Σ_bl)의 차이(잔차)만큼을 학습 가능한 가중치(λ)로 반영. 이론적으로는 더 정교해야 하나, 초기 실험(5자산)에서는 노이즈로 작용.

---

## Phase 2: 10-Asset Universe Expansion (Scale-up)

### 1. 실험 설정 변경 (Setup Change)

- **자산 유니버스 확장 (10개)**:
    - 공격 추가: `QQQ` (나스닥), `XLE` (에너지, 인플레 헷지)
    - 방어 보강: `XLP` (필수소비재), `IEF` (중기채), `VNQ` (리츠)
    - 기존 유지: `SPY`, `XLV`, `TLT`, `GLD`, `BIL`
- **목적**: 다양한 위기 상황(금융위기, 코로나, 인플레)에 대응 가능한 포트폴리오 구성 및 수익률 개선

### 2. 5-Model Benchmark 결과 (Leader Change: GRU)

| Rank | Model | Sharpe | Ann. Return | MDD | 변화 (vs 5자산) |
|------|-------|--------|-------------|-----|---------------|
| **1** | **GRU** | **0.730** | **8.14%** | **-14.12%** | **Sharpe 2.2배 ↑** |
| 2 | LSTM | 0.725 | 8.08% | -14.11% | |
| 3 | TCN | 0.721 | 8.00% | -14.08% | |
| 4 | Transformer | 0.707 | 7.86% | -13.89% | |
| **5** | **TFT** | **0.534** | **5.48%** | **-15.61%** | **1위 → 꼴찌 (역전)** |

> **Critical Finding**:
> 1. **성능 폭발**: 자산 확장만으로 Sharpe 0.33 → 0.73, 수익률 3% → 8%로 급상승.
> 2. **순위 역전 (Bias-Variance Tradeoff)**: 복잡한 TFT(파라미터 712K)는 제한된 데이터에서 과적합 발생하여 성능 저하. 반면 단순한 GRU(40K), LSTM(54K)이 일반화 성능 압도적 우위.

### 3. Deep Dive Benchmarks (GRU 고정)

**A. Omega Mode 비교 (Leader Change)**
- **Learnable (`Ω = NN(x)`)**: **Sharpe 0.723** (New Leader)
  > *변화 원인*: 5자산 때와 달리, 10자산에서는 모델(GRU)이 단순하여 과적합 위험이 줄어듦. 이에 따라 Learnable 방식의 높은 자유도가 오히려 정교한 불확실성 추정에 도움이 됨.
- Hybrid: Sharpe 0.684 (2위)
- Formula: Sharpe 0.626 (하락)
  > *변화 원인*: 자산이 늘어나면서 공분산 행렬(Σ)이 커졌는데, 수식 기반 제약이 불확실성을 지나치게 크게 추정하여(Risk Averse) 수익 기회를 놓침.

**B. Sigma Mode 비교**
- **Prior ≈ Residual**: Sharpe 0.730 vs 0.726 (무차별)
  > *변화 원인*: 10자산으로 늘어나면서 공분산 행렬의 정보량이 풍부해짐. 따라서 Residual 방식(학습)이 노이즈가 아닌 유의미한 정보를 반영하기 시작했으나, Prior 방식(고정)도 충분히 안정적이어서 성능 차이가 희석됨. 안정성 측면에서 Prior 권장.

---

## Phase 3: 종합 비교 분석 (Comparative Analysis)

### 1. 유니버스 확장에 따른 최적 설정 변화 (Paradigm Shift)

| 구분 | 5-Asset Phase | 10-Asset Phase | 원인 분석 |
|------|---------------|----------------|-----------|
| **Best Model** | **TFT** (Complex) | **GRU** (Simple) | 데이터/복잡도 비율 변화. 10자산에선 Simple 모델이 강세. |
| **Best Omega** | **Formula** (Rule-based) | **Learnable** (AI-based) | 단순 모델(GRU)은 AI의 자유도(Learnable)를 활용 가능. |
| **Best Sigma** | **Prior** (Fixed) | **Prior ≈ Residual** | 충분한 정보량 확보로 학습(Residual)의 부작용 감소. |
| **Sharpe** | 0.337 | **0.730** | 자산 다양성(Diversification) 효과 극대화. |

### 2. MVO(Markowitz) vs AI(Decision-Aware) 성과 비교

| Metric | AI Model (GRU) | Classic MVO | Improvement |
|--------|----------------|-------------|-------------|
| **Sharpe** | **0.730** | 0.16 | **4.6배** |
| **Ann. Return** | **8.14%** | 3.05% | **2.7배** |
| **MDD** | **-14.12%** | -22.15% | **+36% 방어력** |
| **특이사항** | 동적 자산 배분 | BIL(현금) 93% 몰빵 (Corner Solution) | |

### 3. 최종 결론 (Final Conclusion)

본 연구의 데이터 환경(월간 데이터, ~200샘플)에서는 **"복잡한 모델(TFT) + 제약된 파라미터(Formula)"** 조합보다, **"단순한 모델(GRU) + 유연한 파라미터(Learnable)"** 조합이 월등히 우수한 성과를 보임.

특히 자산 유니버스의 확장이 모델 아키텍처 개선보다 훨씬 큰 성능 향상(+117%)을 가져왔으며, 이는 금융 시계열 예측에서 **Data-Centric AI** 접근의 중요성을 시사함.

### 4. 최종 확정 모델 설정 (Optimal Configuration)

```python
FINAL_CONFIG = {
    'universe': 10_ASSETS,      # ['SPY','QQQ','XLV','XLP','XLE','TLT','IEF','GLD','VNQ','BIL']
    'model_type': 'gru',        # Bias-Variance Tradeoff 최적점
    'omega_mode': 'learnable',  # GRU의 단순성을 보완하는 유연성
    'sigma_mode': 'prior',      # 안정성 확보
}
```
