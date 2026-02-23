# Decision-Aware Tail Risk Optimization

> **Status:** Active Development — Walk-Forward Validation Complete (v4), Performance Improvement Experiments In Progress

## 📖 Project Overview

This project implements a **Decision-Aware** portfolio optimization strategy that directly optimizes for tail risk protection and superior risk-adjusted returns. Unlike traditional methods that forecast returns (MSE loss) and then optimize separately, our model **backpropagates gradients through the optimization problem** to learn portfolio weights end-to-end.

**Core Philosophy**: *"Optimize decision quality, not prediction accuracy."*

---

## 🏆 Latest Results (v4, Walk-Forward OOS)

> Out-of-sample: 90 months (2016.07 ~ 2024.01), 4-fold expanding window, 3-seed ensemble

| Strategy | Sharpe | Ann. Return | MDD |
|---------|--------|-------------|-----|
| **Our Model (v4)** | **0.815** | **8.41%** | **-11.08%** |
| 1/N Equal Weight | 0.759 | 7.79% | -14.04% |
| HRP | 0.759 | 7.79% | -14.04% |
| Cross-Sectional Momentum | 0.714 | 7.93% | -14.25% |
| Risk Parity | 0.686 | 5.30% | -14.26% |
| Min Variance | 0.609 | 3.33% | -7.97% |
| 60/40 (SPY+TLT) | 0.072 | 0.41% | -19.58% |

---

## 🚀 Key Features (v4)

### 1. Decision-Aware Learning
Loss function trains the AI based on **final portfolio performance**, not prediction accuracy:
```
Loss = -Return + η·CVaR + κ(VIX)·Turnover + λ_dd·MaxDD + λ_dd·3×Crisis_DD
```

### 2. Black-Litterman + Differentiable CVaR
- GRU Encoder → BL Views (P, Q, Ω) → Bayesian update → μ_BL
- **cvxpylayers**: End-to-end differentiable CVaR optimization (β=0.95, 200 scenarios)

### 3. 4-State Regime Conditioning
HMM-based regime model (Bull / Sideways / Correction / Crisis):
- Dynamic risk-aversion λ: Crisis→2.0×, Bull→0.5×
- BIL floor: proportional to crisis probability (max 70%)

### 4. Drawdown Control + Vol Targeting
- Target volatility: 10% (3-month rolling)
- DD threshold 1 (3%): early defensive mode
- DD threshold 2 (5%): crisis mode entry
- Regime Leverage: Bull→2.0×, Crisis→1.0×

### 5. Robust 10-Asset Universe
| Category | Tickers |
|---------|---------|
| Equity | SPY, QQQ, XLV, XLP, XLE |
| Bonds | TLT, IEF, BIL |
| Alternatives | GLD, VNQ |

---

## 📊 Ablation Study (Component Contribution)

| Experiment | Sharpe | Return | MDD | ΔMDD |
|-----------|--------|--------|-----|------|
| **Full Model** (baseline) | **0.815** | **8.41%** | **-11.08%** | — |
| No Drawdown Control | 0.820 | 7.96% | -11.76% | **+0.68%** |
| No Crisis Overlay | 0.817 | 8.14% | -11.55% | +0.47% |
| No Momentum | 0.813 | 8.38% | -11.09% | +0.01% |
| No Regime Leverage | 0.815 | 8.41% | -11.08% | 0.00% |

---

## 🛠 Directory Structure

```
.
├── src/
│   ├── data_loader.py       # Data fetching (yfinance, FRED) & preprocessing
│   ├── models.py            # BaseBLModel + GRU/LSTM/TCN/Transformer/TFT
│   ├── loss.py              # DecisionAwareLoss (Return + CVaR + Turnover + DD)
│   ├── optimization.py      # Differentiable CVaR optimization (cvxpylayers)
│   ├── trainer.py           # Training loop with 5-tuple DataLoader support
│   ├── regime.py            # RegimeHead (Gumbel-Softmax, macro feature support)
│   ├── gen_regime_4state.py # 4-State HMM regime label generator
│   ├── midas.py             # MIDAS feature engineering (Phase 1)
│   ├── midas_layer.py       # Learnable MIDAS layer (Phase 2)
│   ├── benchmark.py         # 5-model benchmark
│   ├── explainability.py    # XAI (Gradient Saliency, Counterfactual)
│   └── utils.py             # Utilities (seed, device, MDD)
├── run_walkforward.py       # Walk-Forward CV + Ensemble (main entry)
├── run_ablation.py          # Ablation Study
├── run_baselines.py         # Traditional strategy baselines
├── run_vol_sweep.py         # Vol targeting parameter sweep
├── run_crisis_deepdive.py   # Crisis period deep dive analysis
├── run_midas_benchmark.py   # MIDAS vs baseline benchmark
├── run_phase1_5.py          # Hyperparameter grid search
├── main.py                  # Legacy single-run entry point
├── data/
│   └── processed/
│       ├── prob_data.csv       # HMM 3-state regime probabilities
│       └── regime_4state.csv   # HMM 4-state regime probabilities
├── results/
│   ├── walkforward/        # Walk-forward summary.json
│   ├── ablation/           # Ablation summary.json
│   ├── baselines/          # Baseline summary.json
│   └── metrics/            # Benchmark CSVs
└── docs/
    ├── experiment_log.md   # Full experiment history (v1~v5)
    └── team_report.md      # Team progress report
```

---

## 💻 Installation

### 1. Clone & Create Environment
```bash
git clone https://github.com/YoungJun0814/Decision-Aware-Tail-Risk-Optimization.git
cd Decision-Aware-Tail-Risk-Optimization

conda create -n tail_risk_opt python=3.10 -y
conda activate tail_risk_opt
```

### 2. Install Solvers (Conda — required for Windows)
```bash
conda install -c conda-forge cvxpy pyportfolioopt -y
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set FRED API Key (optional — for macro features)
```powershell
# Windows PowerShell
$env:FRED_API_KEY = 'your_fred_api_key_here'

# Linux / Mac
export FRED_API_KEY='your_fred_api_key_here'
```
> Get a free key at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

---

## 🏃 Usage

### Walk-Forward Validation (Main Experiment)
```bash
python run_walkforward.py
# Results saved to results/walkforward/summary.json
```

### Ablation Study
```bash
python run_ablation.py
# Results saved to results/ablation/summary.json
```

### Traditional Strategy Baselines
```bash
python run_baselines.py
# Results saved to results/baselines/summary.json
```

### Volatility Sweep
```bash
python run_vol_sweep.py
# Efficient frontier across target_vol=[4%~20%]
```

---

## ⚙️ Configuration (`run_walkforward.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_type` | `'gru'` | Best bias-variance tradeoff |
| `omega_mode` | `'learnable'` | AI learns BL view uncertainty |
| `sigma_mode` | `'prior'` | Uses market covariance for stability |
| `hidden_dim` | `32` | Optimal for ~200 monthly samples |
| `regime_dim` | `4` | 4-State HMM regime conditioning |
| `lambda_risk` | `2.0` | CVaR penalty weight |
| `lambda_dd` | `2.0` | Drawdown penalty (×3 in Crisis) |
| `target_vol` | `10%` | Portfolio volatility target |
| `n_seeds` | `3` | Ensemble averaging |

---

## 📈 Version History

| Version | Key Change | Sharpe | Return | MDD |
|---------|-----------|--------|--------|-----|
| v1 (5-asset) | Baseline BL+CVaR | 0.335 | 3.0% | -16.7% |
| v2 (10-asset) | Universe expansion | 0.730 | 8.1% | -14.1% |
| v3 (Regime) | 4-State HMM + MIDAS | ~0.748 | ~8.3% | ~-13.5% |
| **v4 (Walk-Forward)** | Full architecture + OOS | **0.815** | **8.41%** | **-11.08%** |
| v5 (Macro) | RegimeHead + T10Y3M/BAA10Y | 0.812 | 8.38% | -11.10% |
