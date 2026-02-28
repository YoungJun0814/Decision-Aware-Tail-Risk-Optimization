# Model Architecture Evolution (v1 ~ v5)

줌 미팅 발표용 아키텍처 진화 과정 요약본입니다.

---

## 1. 실험 흐름도 (Experiment Process)
단순한 모델 변경이 아닌, **"자산 확장"**과 **"OOS 검증 및 리스크 통제"**가 성능 향상의 핵심 열쇠였음을 확인했습니다.

```text
[5자산] 모델비교 → TFT 1위 → TFT로 Omega/Sigma 실험 → MVO 비교
                                    ↓
                         자산 확장 (5→10)  (Sharpe 2.2배 점프)
                                    ↓
[10자산] 모델비교 → GRU 1위 → GRU로 Omega/Sigma 실험 → MVO 비교
                                    ↓
                       OOS 방어 메커니즘 도입 (v4)
```

---

## 2. 버전별 아키텍처 및 성능 진화 (v1 ~ v5)

### v1 & v2: Baseline Framework & Universe Expansion
**"엔드투엔드(End-to-End) 학습 프레임워크 검증 및 10-Asset 스케일업"**

```text
[수익률 피처 (v1: 5개 자산 → v2: 10개 자산)]
        ↓ 
GRU (은닉층, hidden=32)
        ↓
P, Q, Ω 행렬/벡터 (블랙-리터만 뷰 예측)
        ↓
Black-Litterman 결합 → μ_BL
        ↓
CVaR 최적화
        ↓
Decision-Aware Loss: -Return + η*Risk(CVaR) + κ*Turnover
▶ 역전파 (Backprop)
```

**[기술적 딥다이브: BL 수식 고도화 결과]**
*   **Omega (불확실성 Ω)**: Learnable 모드(AI가 직접 추론)가 압도적 1위. Formula(수식 적용)는 너무 보수적이라 거의 현금만 보유하는 문제 발생.
*   **Sigma (공분산 Σ)**: 사전 공분산(Prior) 방식이 안정성과 성능 측면에서 최종 권장됨. Residual(학습) 방식도 성과가 좋았으나 안정성 우려.

**[성과 비교]**
*   **v1 성과 (5 Assets)**: Sharpe 0.335, Return 3.03%, MDD -16.65%
*   **v2 성과 (10 Assets)**: Sharpe 0.730, Return 8.14%, MDD -14.12%
*   **핵심 Takeaway**: 아키텍처는 동일하나, 보수적 5자산에서 10자산 세분화(섹터/채권/대안)로 유니버스를 넓히자마자 **수익률 3.2배, Sharpe 2.2배 폭발**. (가장 큰 첫 번째 점프)

---

### v3: Regime Detection & MIDAS 실험
**"거시 경제 상태(Regime) 인식과 하방 위험 관리에 집중"**

```text
[기존 피처] + [HMM Regime 확률 (3-state)]
        ↓ 
GRU → BL → CVaR
        +
[MIDAS VIX Feature (Almon Polynomial OLS)]
```

*   **성과**: (In-sample 참고치) Sharpe ~0.748, Return ~8.3%, MDD ~-13.5%
*   **MIDAS 한계**: MIDAS VIX와 월말 VIX의 상관계수가 0.9+로 거의 동일 정보 지님. 즉, 일간→월간 압축 과정에서 극값이 희석되어 무의미함을 확인. Sharpe 개선 미미(+0.018).

---

### 🏆 v4: Walk-Forward + Full Architecture (현재 최고성능)
**"논문 채택용 OOS 검증(4 Folds, 3 Seeds, 90개월) 및 사후 Risk Control 완성"**

*   **Walk-Forward OOS 성과 (Ensemble, After VT 기준)**: **Sharpe 0.8153, Return 8.41%, MDD -11.08%**

```text
아키텍처 구조도 (v4)

[수익률(10) + Regime확률(3) + Momentum(10) = 23 Feature]
        ↓ 
GRU (hidden=32, seq=12)
hidden state (32dim)
        ↓
┌── Q Head: cat(hidden, 4-state HMM probs) → 전망벡터
├── P Head: hidden → Pick행렬
└── Ω Head: hidden → 뷰 불확실성 (Learnable)
        ↓
Black-Litterman 결합 → μ_BL 계산
        ↓
CVaR (cvxpylayers, β=0.95, 200 시나리오 시뮬레이션)
 ├── λ_risk = f(regime): Crisis→2.0×, Bull→0.5× 강도 조절
 └── BIL(현금) floor: p_crisis × 70% 보장
        ↓
Drawdown Control + Vol Targeting (사후 통제 오버레이)
 ├── target_vol=10%, lookback=3개월
 ├── DD>3%: 방어 모드 진입
 └── DD>5%: 위기 모드 진입 (레버리지 억제)
        ↓
Regime Leverage Control: Bull→2.0×, Crisis→1.0×
```

**[Ablation Study (컴포넌트 기여도 분석)]**
| 실험 (제외/변경 모듈) | Sharpe | Return | MDD | ΔMDD |
| :--- | :---: | :---: | :---: | :---: |
| **Full Model (v4 기준)** | **0.8153** | **8.41%** | **-11.08%** | — |
| No Drawdown Control | 0.8195 | 7.96% | **-11.76%** | +0.68%p |
| No Regime (no_crisis) | 0.8165 | 8.14% | **-11.55%** | +0.47%p |
| No Momentum | 0.8128 | 8.38% | -11.09% | +0.01%p |
| No Regime Leverage | 0.8153 | 8.41% | -11.08% | 0.00%p |

**[전통 전략(벤치마크) 비교 성과표]**
| 전략 | Sharpe | Return | MDD |
| :--- | :---: | :---: | :---: |
| **Our Model (v4)** | **0.8153** | **8.41%** | **-11.08%** |
| 1/N | 0.7589 | 7.79% | -14.04% |
| HRP | 0.7589 | 7.79% | -14.04% |
| Momentum | 0.7137 | 7.93% | -14.25% |
| Risk Parity | 0.6864 | 5.30% | -14.26% |
| Min Variance | 0.6094 | 3.33% | -7.97% |
| 60/40 | 0.0720 | 0.41% | -19.58% |
*   **핵심 Takeaway**: 두 번째 강력한 점프. Walk-Forward 검증 체계와 MDD 방어 컴포넌트(Vol Targeting & DD Control)가 리스크를 확실히 통제하면서도 최상위 수익률을 견인.

---

### ⚠️ v5: Micro-tuning (매크로 통합 및 구조 고도화 한계 확인)
**"v4 + 매크로 (T10Y3M, BAA10Y) → RegimeHead 결합"**

*   **결과 (Walk-Forward)**: Sharpe 0.8122, Return 8.38%, MDD -11.10% (v4 대비 모두 소폭 하락)
*   **결론**: v4 이후 매크로 직접 투입, hidden_dim 64 변환 등 모든 시도가 노이즈 범위 내에서만 변동. 373개월이라는 월간 데이터의 구조적 한계성 입증. **v4 Full이 실질적 최고 성능(Global Optima)이다.**

---

## 3. 향후 논의 사항: End-to-End 구조의 재정립 고민 (Daily 빈도 확장)

**[현재 구조의 한계]**
현재 파이프라인은 `[데이터] → [HMM 레짐 분류] → [AI (GRU/BL/CVaR)]` 로 **분리**되어 있습니다. 이는 학습 데이터(월간 373개월)가 극도로 제한된 환경에서 나온 선택이었습니다.

**[이상적인 목표 아키텍처 (일간 데이터 패러다임 도입)]**
레짐 분류가 독립적인 것이 아니라 손실(Loss)을 최소화하는 방향으로 **포트폴리오 성과 최적화와 함께 학습**되어야 합니다. 또한, 부족한 월간 데이터셋(373개월)을 벗어나 **일간 데이터(~7,500일)** 정보를 손실 없이 융합(MIDAS)시키는 구조가 필요합니다.

```text
[일간 VIX, SPY, Credit Spread] 
         │
    MIDAS Layer (Learnable Almon Polynomial)
         │ ← 각 월의 일간 데이터를 최적 가중 합산
         ▼
    [최적 가중 월간 Feature 벡터]
         │
    + [월간 Macro 변수들]
         │
    GRU Encoder (Sequence of monthly embeddings)
         │
    ├── Regime (Gumbel-Softmax, K=4) → 미분 가능! Gradient가 레짐 분류까지 역전파
    ├── BL Parameters (Q, P, Ω, Σ)
         │
    Regime-Conditioned CVaR Optimization
         │
    Loss = CVaR(성과) + λ₁·RegimeStability(안정성) + λ₂·Entropy(다양성)
```

**[핵심 논점]**
*   **Gumbel-Softmax**: 이산 변수(Regime)를 미분 가능하게 만들어 포트폴리오 성과(Gradient)로 레짐을 자동 학습시킴.
*   **Learnable MIDAS Layer**: 각 달에서 **어느 시점의(며칠 전) 데이터가 레짐 판별에 중요한가**를 AI가 스스로 학습(최적 가중). 일간 데이터의 극값이 희석되는 기존 한계(v3) 완전 돌파!
