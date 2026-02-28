# 📑 Decision-Aware Tail Risk Optimization — 프로젝트 중간 보고
**작성일**: 2026년 2월 23일  
**작성자**: Jun  
**목적**: 2025년 초~2026년 2월까지 전체 실험 진행 사항 통합 보고 (팀원 공유용)

---

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [Phase 1: 5자산 초기 실험](#2-phase-1-5자산-초기-실험)
3. [Phase 2: 10자산 확장](#3-phase-2-10자산-확장)
4. [Phase 3: 종합 비교 분석](#4-phase-3-종합-비교-분석)
5. [Phase 4: v4 아키텍처 고도화 + Walk-Forward + Ablation](#5-phase-4-v4-아키텍처-고도화--walk-forward--ablation)
6. [Phase 5: 성능 개선 실험](#6-phase-5-성능-개선-실험)
7. [모델 구조 상세 설명](#7-모델-구조-상세-설명)
8. [현재 최선 성능 요약](#8-현재-최선-성능-요약)
9. [미해결 과제 및 향후 방향](#9-미해결-과제-및-향후-방향)

---

## 1. 프로젝트 개요

### 핵심 아이디어

> **"예측 정확도가 아닌 포트폴리오 의사결정 품질을 직접 최적화한다"**

기존 딥러닝 포트폴리오 연구: 수익률 예측 → 포트폴리오 구성 (2단계)  
본 연구: 포트폴리오 성과(Sharpe, CVaR, Drawdown)를 Loss로 삼아 **End-to-End 직접 최적화**

### 시스템 구성 요약

| 컴포넌트 | 기술 | 역할 |
|---------|------|------|
| **Encoder** | GRU (hidden=32) | 12개월 시계열 패턴 학습 |
| **Bayesian Update** | Black-Litterman (P, Q, Ω) | AI 전망을 시장 균형에 베이지안 통합 |
| **최적화** | CVaR (cvxpylayers, β=0.95) | 하위 5% 최악 손실 최소화 |
| **Regime** | 4-State HMM | Bull/Sideways/Correction/Crisis 국면 판단 |
| **Risk Control** | Vol Targeting + CrisisOverlay | 사후적 리스크 관리 |
| **Loss** | Decision-Aware | 수익률·CVaR·Turnover·Drawdown 종합 패널티 |

### 자산 유니버스 (10개)

| 카테고리 | 티커 | 설명 |
|---------|------|------|
| 공격 | SPY, QQQ | 미국 대형주, 나스닥 |
| 방어 | XLV, XLP | 헬스케어, 필수소비재 |
| 에너지/인플레 헷지 | XLE | 에너지 섹터 |
| 채권 | TLT, IEF | 장기채, 중기채 |
| 안전자산 | GLD, BIL | 금, 단기채(현금) |
| 대안 | VNQ | 리츠 |

---

## 2. Phase 1: 5자산 초기 실험

### 실험 설정
- **자산**: SPY, XLV, TLT, GLD, BIL (보수적 5개)
- **기간**: 2007-07-01 ~ 2024-01-01
- **목적**: BL+CVaR 프레임워크 기초 성능 검증

### 5-Model Benchmark 결과

| 순위 | 모델 | Sharpe | Return | MDD |
|------|------|--------|--------|-----|
| **1** | **TFT** | **0.337** | **3.03%** | **-16.58%** |
| 2 | GRU | 0.335 | 3.03% | -16.65% |
| 3 | TCN | 0.334 | 3.02% | -16.66% |
| 4 | Transformer | 0.333 | 3.01% | -16.64% |
| 5 | LSTM | 0.331 | 2.98% | -16.65% |

> **관찰**: 모든 모델이 거의 동일한 성능 (동조화). SPY 단독 자산이 너무 적어 다양성 부족.

### BL 파라미터 최적 설정 탐색

**Omega Mode (불확실성 행렬 Ω 설정방식)**
- Formula 방식 (`Ω = τ·p²·Σ`): **Sharpe 0.451** 1위 — 이론적으로 가장 안정적
- Hybrid: Sharpe 0.421, Learnable: Sharpe 0.340

**Sigma Mode (공분산 설정방식)**
- Prior (사전 공분산 유지): **Sharpe 0.371** 1위

---

## 3. Phase 2: 10자산 확장

### 핵심 변경: SPY 외 9개 자산 추가

| | 5자산 | 10자산 | 변화 |
|--|-------|--------|------|
| **Sharpe (GRU)** | 0.335 | **0.730** | **+117%** |
| **Return** | 3.03% | **8.14%** | **+5.11%p** |
| **MDD** | -16.65% | -14.12% | 개선 |

### 순위 역전 현상 (Bias-Variance Tradeoff)

| 순위 | 5자산 | 10자산 |
|------|------|--------|
| 1위 | TFT (복잡, 712K param) | **GRU (단순, 40K param)** |
| 꼴찌 | LSTM | **TFT** (과적합으로 역전) |

> **핵심 교훈**: 월간 데이터 ~200개 샘플에서는 **단순 모델이 압도적으로 유리**.  
> TFT(712K 파라미터)는 데이터에 비해 지나치게 복잡해 OOS에서 성능이 무너짐.

### 최적 BL 설정 변화

| | 5자산 | 10자산 | 이유 |
|--|-------|--------|------|
| Omega | Formula | **Learnable** | GRU가 단순해 Learnable 자유도를 흡수 가능 |
| Sigma | Prior | Prior≈Residual | 자산 증가로 정보량 충분해 Residual도 유효 |

---

## 4. Phase 3: 종합 비교 분석

### MVO(Markowitz) vs AI 성과 비교

| 지표 | AI (GRU) | Classic MVO | 개선 |
|------|---------|-------------|------|
| Sharpe | **0.730** | 0.16 | **4.6배** |
| Return | **8.14%** | 3.05% | **2.7배** |
| MDD | **-14.12%** | -22.15% | **+36% 방어** |

> MVO는 BIL(현금)에 93% 몰빵하는 Corner Solution 발생 — 이론과 실무의 괴리.

---

## 5. Phase 4: v4 아키텍처 고도화 + Walk-Forward + Ablation

### 5-1. v4 최종 아키텍처

```
입력 피처 (23개):
  ├── 10자산 월간 수익률
  ├── 3개 Regime 확률 (HMM 입력 피처)
  └── 10개 12M Momentum
          ↓
  GRU Encoder (hidden=32, seq=12)
          ↓
  ┌── Q Head: cat(hidden, 4-state regime_probs) → 전망 벡터
  ├── P Head: hidden → Pick 행렬
  └── Ω Head: hidden → 불확실성 (Learnable)
          ↓
  Black-Litterman Formula → μ_BL (보정 기대수익률)
          ↓
  CVaR Optimization (cvxpylayers, 200 시나리오, β=0.95)
     ├── λ_risk = f(regime) — Crisis 시 2배 강화
     └── BIL floor: p_crisis에 비례 (max 70%)
          ↓
  Drawdown Control + Vol Targeting (post-hoc)
     ├── p_bull > 1.5× → 레버리지 최대 2.0×
     ├── p_crisis → 레버리지 최대 1.0×
     └── target_vol=10%, DD trigger: 3%/5%
```

### 5-2. 학습 설정 (v4 CONFIG)

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| hidden_dim | 32 | GRU 은닉층 크기 |
| regime_dim | 4 | 4-State HMM 확률 사용 |
| λ_risk (CVaR) | 2.0 | Mean-CVaR 패널티 강도 |
| λ_dd | 2.0 | Drawdown 패널티 (Crisis 시 ×3.0) |
| Target vol | 10% | 포트폴리오 목표 변동성 |
| DD threshold 1 | 3% | 초기 방어 발동 |
| DD threshold 2 | 5% | 위기 모드 진입 |
| Bull leverage | 2.0× | 강세장 공격 배분 |
| Crisis leverage | 1.0× | 위기 시 레버리지 제한 |
| n_seeds | 3 | 앙상블 평균 |
| epochs | 100 (early stop) | |
| lr | 1e-3 (cosine decay) | |

### 5-3. Walk-Forward 결과

- **방식**: Expanding Window (4 Folds)
- **Folds**: 2016~2018 / 2018~2020 / 2020~2022 / 2022~2025
- **OOS**: 90개월 (7.5년)

| 구분 | Sharpe | Return | MDD |
|------|--------|--------|-----|
| **Our Model (Ensemble, After VT)** | **0.8153** | **8.41%** | **-11.08%** |
| 1/N Equal Weight | 0.759 | 7.79% | -14.04% |
| Before Vol Targeting | 0.821 | 7.85% | -12.22% |

**Seed 안정성**: Sharpe std = **0.002** (극도로 안정)

### 5-4. Ablation Study — 컴포넌트 기여도 분석

> 각 컴포넌트를 하나씩 제거하여 기여도를 측정 (논문의 핵심 실험)

| 실험 | Sharpe | Return | MDD | ΔMDD |
|------|--------|--------|-----|------|
| **Full Model** | **0.8153** | **8.41%** | **-11.08%** | — |
| No Drawdown Control | 0.8195 | 7.96% | -11.76% | **+0.68% ↑** |
| No Crisis Overlay | 0.8165 | 8.14% | -11.55% | **+0.47% ↑** |
| No Momentum | 0.8128 | 8.38% | -11.09% | +0.01% |
| No Regime Leverage | 0.8153 | 8.41% | -11.08% | 0.00% |

**핵심 발견:**

✅ **효과 있는 컴포넌트**
- **Drawdown Control**: MDD +0.68% 악화 → **가장 중요한 리스크 관리 메커니즘**
- **Crisis Overlay**: MDD +0.47% 악화 → BIL floor 강제의 실질 효과 입증

⚠️ **효과 없는 컴포넌트**
- **Momentum Feature**: ΔSharpe=-0.002, ΔMDD=+0.01% → 사실상 기여 없음
- **Regime-Adaptive Leverage**: 완전히 동일 (ΔMDD=0.00%) → OOS 기간에서 `p_bull>0.5` 또는 `p_crisis>0.5` 시점이 극히 드물어 활성화 안 됨

### 5-5. 전통 전략 비교 (7개 전략)

| 전략 | Sharpe | Return | MDD | 비고 |
|------|--------|--------|-----|------|
| **Our Model** | **0.816** | **8.42%** | **-11.08%** | **1위** |
| 1/N Equal Weight | 0.759 | 7.79% | -14.04% | |
| HRP | 0.759 | 7.79% | -14.04% | |
| Momentum | 0.714 | 7.93% | -14.25% | |
| Risk Parity | 0.686 | 5.30% | -14.26% | |
| Min Variance | 0.609 | 3.33% | **-7.97%** | MDD만 1위 (Return 매우 낮음) |
| 60/40 (SPY+TLT) | 0.072 | 0.41% | -19.58% | 채권 대하락기(2022)에 매우 불리 |

> **결론**: Sharpe 기준으로 모든 전통 전략 대비 1위.  
> MDD도 Min Variance 제외 전부 우위 (Min Variance는 Return 3.3%로 실용성 없음).

---

## 6. Phase 5: 성능 개선 실험

### 6-1. 매크로 피처 → RegimeHead 통합 (v5)

**목적**: GRU가 가격 패턴만으로 Regime을 추론하는 한계를 극복.  
실제 거시경제 선행지표를 RegimeHead에 직접 투입.

| 추가된 피처 | FRED 코드 | 의미 |
|-----------|-----------|------|
| Term Spread | T10Y3M | 10년물-3개월물 금리차 → 경기침체 선행지표 |
| Credit Spread | BAA10Y | 회사채-국채 스프레드 → 위험회피 강도 |

```
기존: GRU hidden → RegimeHead → regime_probs
변경: GRU hidden + [T10Y3M, BAA10Y] → RegimeHead → regime_probs
```

**v5 결과 비교**:

| 지표 | Ablation Full (v4, 기준) | v5 + Macro | 변화 |
|------|--------------------------|------------|------|
| Sharpe | **0.8153** | 0.8121 | -0.003 (노이즈 범위) |
| Return | **8.41%** | 8.38% | -0.03%p |
| MDD | -11.08% | -11.08% | 동일 |

> **결론**: 매크로 피처 2개(T10Y3M, BAA10Y)로는 성능 개선 없음.  
> **현재 최고 성능은 Ablation Full (v4)** 임을 확인.

---

### 6-2. Proposal A: hidden_dim 확장 (32 → 64)

**가설**: hidden=32는 23 feature × 12개월을 압축하기에 너무 작을 수 있음.

| 지표 | hidden=32 (Best) | hidden=64 | 변화 |
|------|-----------------|-----------|------|
| Sharpe | 0.8153 | 0.8170 | +0.0017 |
| Return | 8.41% | 8.39% | -0.02%p |
| MDD | -11.08% | -11.06% | +0.02%p |
| Seed Std | **0.0021** | 0.0028 | **불안정↑** |

> **결론**: 실질 개선 없음 + Seed 43에서 100 epoch 전부 소진 (학습 불안정 신호).  
> 200개 월간 샘플 대비 파라미터 과다 (40K→120K).  
> **hidden=32 유지** 결정.

---

### 6-3. Proposal B: Cross-Sectional 모멘텀 피처 (구현 완료, 실험 예정)

**아이디어**: 기존 절대 모멘텀(각 자산의 12개월 수익률)에 더해,  
**동일 시점 10개 자산 간 상대 강약**을 학습 신호로 추가.

| 추가 피처 | 계산 | 총 수 |
|---------|------|------|
| `{TICKER}_CS_RANK` | 시점별 자산 간 백분위 순위 (0~1) | 10개 |
| `{TICKER}_CS_Z` | 시점별 자산 간 z-score | 10개 |

```python
# 구현 핵심 (look-ahead bias 없음 — 동일 시점 내 자산 간 비교)
cs_rank  = mom_matrix.rank(axis=1, pct=True)   # 0~1 순위
cs_zscore = (mom - mom.mean(axis=1)) / mom.std(axis=1).clip(1e-8)
```

- **input_dim**: 23 → **43** (모델 코드 변경 없음, 자동 반영)
- **Look-ahead bias**: 없음 ✅

> **실험 결과**: 진행 예정 → `python run_walkforward.py`

---

### 6-4. Proposal C: Regime-Aware Dynamic Loss (계획)

**아이디어**: 현재 Loss는 모든 Regime에서 동일한 패널티를 사용.  
Crisis 구간에서는 CVaR 패널티를 강화, Bull에서는 Return 목표를 높이도록 동적 조절.

```python
# src/loss.py 변경 (개념)
p_crisis = regime_probs[:, 3].mean().detach()  # gradient 차단
dynamic_eta = self.eta * (1.0 + 2.0 * p_crisis)
loss = return_loss + dynamic_eta * risk_penalty + ...
```

- **예상 공수**: 2~3일 (loss.py + trainer.py + 6개 caller 수정)
- **진행 조건**: Proposal B 결과 확인 후 결정

---

## 7. 모델 구조 상세 설명

### 데이터 흐름 (Step-by-step)

**Step 1**: 입력 데이터 (매월 리밸런싱)
```
과거 12개월 × 23개 Feature
  - 10개 자산 수익률
  - 3개 HMM Regime 확률 (Bull/Uncertain/Crisis)
  - 10개 12M 모멘텀
```

**Step 2**: GRU Encoder
```
(batch, 12, 23) → GRU → (batch, 32) hidden state
→ "시장이 지금까지 어떤 흐름이었는가"를 32차원 벡터로 표현
```

**Step 3**: BL Views 생성
```
hidden → P (Pick 행렬: 어떤 자산에 대한 견해인가)
hidden → Q (View 수익률: 얼마나 오를 것인가)
hidden → Ω (View 불확실성: 얼마나 확신하는가)
→ Bayesian Update: μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹[(τΣ)⁻¹π + PᵀΩ⁻¹Q]
```

**Step 4**: Regime 조건부 Risk Aversion
```
HMM p_crisis 높음 → λ_risk 크게 (최대 6.0) → 리스크 회피 강화
HMM p_bull 높음  → λ_risk 작게 (최소 0.5) → 공격적 투자
```

**Step 5**: CVaR 최적화 레이어
```
μ_BL + Σ_BL → 200개 수익률 시나리오 샘플링
→ Min CVaR(w) s.t. Σw=1, w≥0, w_BIL ≥ p_crisis × 0.7
→ 하위 5% 최악 손실을 최소화하는 포트폴리오 도출
```

**Step 6**: Vol Targeting + Regime Leverage
```
Raw Weights
  → 실현변동성 > 10%? → 비중 스케일다운
  → DD > 3%?         → 방어 모드
  → DD > 5%?         → 위기 모드
  → p_bull > 0.5?    → 레버리지 최대 2.0×
  → p_crisis > 0.5?  → 레버리지 최대 1.0×
→ Final Weights
```

**손실 함수**:
```
Loss = -Return
     + η × CVaR(w, r)               (테일 리스크 패널티)
     + κ(VIX) × Turnover            (VIX 높을수록 매매 억제)
     + λ_dd × MaxDrawdown           (낙폭 패널티)
     + λ_dd × 3.0 × Crisis_DD       (위기 시 낙폭 패널티 3배)
```

---

## 8. 현재 최선 성능 요약

**기준 모델**: Ablation Full (v4 + Walk-Forward, hidden=32, 매크로 없음)

| 지표 | 값 | 비교 (1/N) |
|------|-----|-----------|
| **Sharpe** | **0.8153** | +7.4% ↑ |
| **연 수익률** | **8.41%** | +0.62%p ↑ |
| **MDD** | **-11.08%** | +2.96%p 개선 ↑ |
| Seed 안정성 (Std) | **0.002** | 극도로 안정 |

**실험한 개선 시도 결과 요약**:

| 시도 | 결과 | 결론 |
|-----|------|------|
| v5: 매크로 피처 (T10Y3M, BAA10Y) | Sharpe -0.003 | ❌ 효과 없음 |
| hidden_dim 64 | Sharpe +0.0017, 불안정↑ | ❌ 채택 안 함 |
| CS 모멘텀 피처 (43 feat) | 실험 예정 | 🔄 진행 중 |

---

## 9. 미해결 과제 및 향후 방향

### 9-1. 단기 실행 계획

| 우선순위 | 과제 | 예상 소요 |
|---------|------|---------|
| 1 | **Proposal B 실험**: CS 모멘텀 Walk-Forward 실행 | 실행 1회 (~5분) |
| 2 | **통계적 유의성**: n_seeds 3→10 + Bootstrap 신뢰구간 | 1일 |
| 3 | **Proposal C**: Regime-Aware Dynamic Loss 구현 | 2~3일 |

### 9-2. 근본적 한계 및 해결 방향

**문제**: 월간 리밸런싱은 일중 급락을 포착하지 못함 (2020년 3월 등)

**이전 시도 — MIDAS (실패)**:
```
표준 MIDAS: feature_t = Σ w(k; θ) × x_{t,k}  (Almon Polynomial)
→ w(k)가 시간 위치에만 의존 (데이터 값 무관)
→ VIX=82인 날과 VIX=20인 날이 같은 가중치 → 월말 스냅샷과 상관 0.9+
→ 결과: Sharpe +0.018에 그침 (사실상 월말 스냅샷과 동일 정보)
```

**제안 — EA-MIDAS (미구현)**:
```
표준 MIDAS:  w(k) = softmax(θ₁k̃ + θ₂k̃²)          ← 시간만 반영
EA-MIDAS:    w(k) = softmax(θ₁k̃ + θ₂k̃² + θ₃·x̃ₖ) ← 시간 + 값 반영

θ₃ > 0 학습 시 → VIX 높은 날(급락 시)에 더 큰 가중치 → 스파이크 포착
θ₃ = 0 시    → 표준 MIDAS로 퇴화 (기존 방법이 Special case)
```

**논문 기여**: "MIDAS의 일반화 (Generalization of MIDAS)"

### 9-3. 장기 비전: Joint End-to-End 아키텍처

```
현재 구조:
  [데이터] → [HMM (독립 사전학습)] → [GRU+BL+CVaR (독립)] → Loss

목표 구조:
  [일간 데이터] → [EA-MIDAS Layer]
                          ↓
  [월간 Macro]  → concat → [GRU Encoder]
                                 ↓
                       ┌── Regime (Gumbel-Softmax, K=4)  ← 포트폴리오 손실로 직접 학습
                       └── BL Parameters (regime-conditioned)
                                 ↓
                       CVaR Optimization
                                 ↓
                Loss = CVaR + λ₁·Regime Stability + λ₂·Entropy
```

**핵심**: Regime이 포트폴리오 성과로부터 **직접 학습** (현재는 고정된 HMM 사용)  
**전제조건**: EA-MIDAS로 데이터를 185개월 → 7,500일로 확장해야 학습 안정성 확보 가능

---

## 부록: 실험 환경

```
OS: Windows 11
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
Python: 3.11
PyTorch: 2.x
주요 라이브러리: cvxpylayers, yfinance, hmmlearn, fredapi
```

**검증 방식**:
```
Walk-Forward Cross-Validation
  - 4 Folds (Expanding Window)
  - 3 Seeds 앙상블 (seed 42/43/44)
  - OOS: 90개월 (2016.07 ~ 2024.01)
```

---

*본 문서는 지속적으로 업데이트됩니다. 마지막 업데이트: 2026-02-23*
