# Model Architecture Diagram
**Best Config: Correlation Guard + Asset Exclusion (Sharpe=1.1449, Return=11.16%, MDD=-9.61%)**

---

## Full Pipeline Flowchart

```mermaid
flowchart TD
    %% ── DATA LAYER ──────────────────────────────────────────
    A1["📈 Market Data\n(yfinance, 13 assets)\n2016-07 ~ 2025-12"]
    A2["📊 Daily Returns\nSPY, QQQ, XLV, XLP, XLE\nTLT, IEF, GLD, TIP, DBC\nVNQ, VEU, BIL"]

    %% ── UPSTREAM MODELS ──────────────────────────────────────
    B1["🤖 R6 Model\nCVaR Optimization\n(Walk-forward)"]
    B2["🤖 R7 Model\nOffline EA\n(Walk-forward)"]
    B3["🔄 PIT-Shift\n(Point-In-Time)\nLook-ahead bias 제거"]
    B4["🌐 Regime Model\nProb(Bull / Sideways\n/ Correction / Crisis)"]

    %% ── GUARD SIGNAL ─────────────────────────────────────────
    G1["📡 Guard Signal\n60일 SPY-TLT Rolling Corr\n(Lagged, No Look-ahead Bias)"]
    G2{"corr ≤ -0.25?"}
    G_ON["🛑 GUARD ON\nTLT, IEF, TIP → 0%\n→ BIL로 이동"]
    G_OFF["✅ GUARD OFF\n정상 overlay 비중 유지"]

    %% ── PHASE 18 OVERLAY STACK ───────────────────────────────
    C1["① blend_sleeves_v2\ncosine_similarity(R6, R7)\n→ R6/R7 블렌딩"]
    C2["② S3 Conviction Scaling\ns3_factor = f(similarity)\n[smooth: low=0.92, high=1.45]"]
    C3["③ S7 Drawdown Control\n★ dd0=1.5, dd3=1.1 (수정)\ndd6=0.8, dd9=0.65"]
    C4["④ SRB Regime Budget\nbull×1.15 / crisis×0.5\n(regime 반응)"]
    C5["⑤ S1 Vol Targeting\n★ target_vol=0.09 (수정)\nbull_scale=1.2, crisis_scale=0.7"]
    C6["⑥ Sleeve Allocation\nGrowth / Defensive / Cash\n비중 배분"]
    C7["⑦ Tilt\nbull sim>0.8 → growth +20%\ncrisis → defensive +20%"]
    C8["⑧ Rebalance Guard\n과도한 turnover 방지"]
    C9["⑨ Stop-Loss\n★ soft=-2.0%, hard=-3.0% (수정)\nreentry=0.8%"]

    %% ── OUTPUT ───────────────────────────────────────────────
    OUT["📦 월별 포트폴리오 수익률\n→ 누적 성과 계산"]
    RESULT["🏆 Triple Target 달성\nSharpe=1.1449\nReturn=11.16%\nMDD=-9.61%"]

    %% ── CONNECTIONS ──────────────────────────────────────────
    A1 --> A2
    A1 --> B1
    A1 --> B2
    A1 --> B4
    A2 --> G1

    B1 --> B3
    B2 --> B3
    B3 --> C1
    B4 --> C4
    B4 --> C5

    G1 --> G2
    G2 -->|YES| G_ON
    G2 -->|NO| G_OFF

    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> C5
    C5 --> C6
    C6 --> C7
    C7 --> G_ON
    C7 --> G_OFF
    G_ON --> C8
    G_OFF --> C8
    C8 --> C9
    C9 --> OUT
    OUT --> RESULT

    %% ── STYLING ──────────────────────────────────────────────
    style G_ON fill:#ff6b6b,color:#fff,stroke:#c0392b
    style G_OFF fill:#2ecc71,color:#fff,stroke:#27ae60
    style G2 fill:#f39c12,color:#fff,stroke:#e67e22
    style RESULT fill:#3498db,color:#fff,stroke:#2980b9
    style C3 fill:#ffeaa7,stroke:#fdcb6e
    style C5 fill:#ffeaa7,stroke:#fdcb6e
    style C9 fill:#ffeaa7,stroke:#fdcb6e
```

> ★ 노란색 박스 = Best Config에서 수정된 파라미터

---

## Correlation Guard 상세 로직

```mermaid
flowchart LR
    subgraph SIGNAL["Guard Signal 계산 (매월)"]
        S1["SPY 일별 수익률\n(t-1 월말 기준)"]
        S2["TLT 일별 수익률\n(t-1 월말 기준)"]
        S3["60일 Rolling\nPearson Correlation"]
        S1 --> S3
        S2 --> S3
    end

    subgraph DECISION["Guard 발동 결정"]
        D1{"spy_tlt_corr\n≤ -0.25?"}
        D2["GUARD ON\n(71/114개월, 62.3%)"]
        D3["GUARD OFF\n(43/114개월, 37.7%)"]
        S3 --> D1
        D1 -->|Yes| D2
        D1 -->|No| D3
    end

    subgraph ACTION["자산 배분 조정"]
        A1["TLT 비중 → 1.05%\nIEF 비중 → 1.05%\nTIP 비중 → 1.05%\n(최소 잔류)\n잉여 → BIL"]
        A2["정상 overlay 비중\nTLT~3.2%, IEF~4.0%\nTIP~3.4%"]
        D2 --> A1
        D3 --> A2
    end

    subgraph EFFECT["효과 (Guard ON 시)"]
        E1["BIL: 0.4% → 13.9%\n(+13.5pp)"]
        E2["채권 합계: 10.5% → 3.2%\n(-7.3pp)"]
        A1 --> E1
        A1 --> E2
    end

    style D2 fill:#ff6b6b,color:#fff
    style D3 fill:#2ecc71,color:#fff
    style E1 fill:#74b9ff,stroke:#0984e3
```

---

## Overlay Stack 파라미터 비교 (Baseline vs Best)

```mermaid
%%{init: {"theme": "base"}}%%
xychart-beta
    title "S7 Drawdown Scale Factor"
    x-axis ["dd0 (no DD)", "dd3 (-3%)", "dd6 (-6%)", "dd9 (-9%)"]
    y-axis "Scale" 0 --> 2
    bar [1.1, 1.0, 0.8, 0.65]
    bar [1.5, 1.1, 0.8, 0.65]
```

| 파라미터 | Baseline | Best Config | 방향 |
|----------|----------|-------------|------|
| **S7 dd0** | 1.1 | **1.5** | ↑ Drawdown 없을 때 더 공격적 |
| **S7 dd3** | 1.0 | **1.1** | ↑ |
| S7 dd6 | 0.8 | 0.8 | = |
| S7 dd9 | 0.65 | 0.65 | = |
| **S1 target_vol** | 10% | **9%** | ↓ 약간 보수적 |
| S1 bull_scale | 1.2 | 1.2 | = |
| **Stop soft** | -2.0% | **-2.0%** | = |
| **Stop hard** | -3.5% | **-3.0%** | ↑ 더 빠른 손절 |
| **Guard corr_thresh** | None | **-0.25** | NEW |
| **Exclude assets** | None | **TLT, IEF, TIP** | NEW |

---

## 성능 기여도 분해

```mermaid
%%{init: {"theme": "base"}}%%
xychart-beta horizontal
    title "MDD 개선 기여 (Baseline → Best)"
    x-axis ["Baseline", "+Overlay Mods", "+Guard Only", "+Guard+Mods (Best)"]
    y-axis "MDD (%)" -15 --> 0
    bar [-13.92, -12.42, -9.86, -9.61]
```

| 단계 | Sharpe | Return | MDD | Triple |
|------|--------|--------|-----|--------|
| Baseline | 0.886 | 9.10% | -13.92% | ❌ |
| + Overlay Mods만 | 1.017 | 10.59% | -12.42% | ❌ |
| + Guard만 | 1.027 | 9.84% | -9.86% | ❌ |
| **+ Guard + Mods** | **1.145** | **11.16%** | **-9.61%** | **✅** |

---

## 데이터 흐름 요약 (ASCII)

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│  Market Data (13 assets, daily, 2016-07~2025-12, yfinance)     │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
    ┌──────────▼──────────┐       ┌──────────▼──────────┐
    │  Walk-Forward       │       │  Guard Signal        │
    │  R6 (CVaR) weights  │       │  60d SPY-TLT corr   │
    │  R7 (EA)   weights  │       │  (lagged, PIT)       │
    │  → PIT-shift        │       └──────────┬──────────┘
    └──────────┬──────────┘                  │
               │                    corr ≤ -0.25?
    ┌──────────▼──────────────────────────────┐
    │         PHASE 18 OVERLAY STACK          │
    │                                         │
    │  ① blend_sleeves_v2 (cosine sim)        │
    │  ② S3 Conviction (sim → scale)          │
    │  ③ S7 Drawdown ★ dd0=1.5               │
    │  ④ SRB Regime Budget                    │
    │  ⑤ S1 Vol Target ★ vol=0.09            │
    │  ⑥ Sleeve Allocation                    │
    │  ⑦ Tilt                                 │
    │                     ┌───────────────────┤
    │              YES ◄──┤ Guard Active?     ├──► NO
    │              │       └───────────────────┘    │
    │    TLT/IEF/TIP→BIL                    Normal weights
    │              └──────────────┬──────────────────┘
    │  ⑧ Rebalance Guard          │
    │  ⑨ Stop-Loss ★ hard=-3.0%  │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │       MONTHLY RETURN        │
    │  Σ(weight × asset_return)   │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────────────────┐
    │         TRIPLE TARGET ACHIEVED           │
    │  Sharpe = 1.1449  ✅ (target: ≥ 1.0)   │
    │  Return = 11.16%  ✅ (target: ≥ 10%)   │
    │  MDD    = -9.61%  ✅ (target: ≥ -10%)  │
    └──────────────────────────────────────────┘
```
