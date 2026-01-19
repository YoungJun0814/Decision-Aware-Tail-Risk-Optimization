# Decision-Aware Tail Risk Optimization

> **Status:** Active Development (v2 Implementation Complete)  
> **Roles:**
> - **Tech Lead (Role A):** System Architecture & Pipeline (Complete)
> - **Quant Researcher (Role B):** CVaR Optimization & Risk Modeling (Pending)
> - **Strategist (Role C):** Macro Indicator Selection (Pending)

## 📖 Project Overview

This project implements a **Decision-Aware** investment strategy that optimizes for tail risk protection and superior risk-adjusted returns. Unlike traditional methods that forecast future prices (MSE loss) and then optimize, our model learns to **optimize portfolio weights directly** by backpropagating gradients through the optimization problem.

The core philosophy is **"Asymmetric Payoff"**:
- **Upside:** Capture market growth with Long assets (SPY, XLV).
- **Downside:** Defend against crashes with Safe assets (TLT, GLD) and limited Inverse ETF hedging (SH).

## 🚀 Key Features (v2)

### 1. Decision-Aware Learning
We use a specialized loss function that trains the AI based on the **final portfolio performance** (Return - Risk) rather than prediction accuracy.
- **Goal:** Maximize Sharpe Ratio / Minimize CVaR directly.

### 2. Multi-Objective Loss Function (`src/loss.py`)
Our `DecisionAwareLoss` incorporates four key terms:
- **Return Maximization:** $\max \mu^T w$
- **Risk Minimization:** $\min \sigma_p$ (or CVaR)
- **Dynamic Transaction Costs ($\kappa(VIX)$):** Penalizes excessive trading, especially during high volatility (high VIX) to enforce stability.
- **Inverse Decay Penalty ($\rho$):** Penalizes long-term holding of Inverse ETFs (SH) to prevent value erosion from volatility drag.

### 3. Smart Asset Universe
A robust 5-asset universe covering all weather conditions:
- **Growth:** `SPY` (S&P 500)
- **Defensive:** `XLV` (Healthcare - Low Beta)
- **Safe Haven:** `TLT` (Treasuries), `GLD` (Gold)
- **Hedge:** `SH` (Inverse S&P 500) - *Restricted to Max 30%*

### 4. Safety Constraints (`src/optimization.py`)
- **Hard Cap on Inverse ETF:** `w[SH] <= 0.3`. The AI cannot bet the farm on a market crash.
- **No Short Selling:** Long-only portfolio (except for the implicit short via SH).

---

## 🛠 Directory Structure

```plaintext
.
├── data/                   # Data storage
│   ├── raw/                # Raw downloads (yfinance)
│   └── processed/          # Preprocessed tensors
├── src/
│   ├── data_loader.py      # Data fetching (Prices, VIX) & Preprocessing
│   ├── loss.py             # DecisionAwareLoss implementation
│   ├── models.py           # LSTM/Transformer Encoders
│   ├── optimization.py     # Differentiable Optimization Layers (cvxpylayers)
│   ├── trainer.py          # Training Loop & Validation
│   └── benchmark_mvo.py    # Classical Mean-Variance Benchmark
├── main.py                 # Entry point (End-to-End Pipeline)
├── verify_env.py           # Dependency verification script
└── README.md               # Documentation
```

---

## 💻 Installation

To avoid C++ compilation errors with `cvxpy` and `cvxpylayers` on Windows, we **strongly recommend** using Conda.

### 1. Clone & Create Environment
```bash
git clone https://github.com/YoungJun0814/Decision-Aware-Tail-Risk-Optimization.git
cd Decision-Aware-Tail-Risk-Optimization

conda create -n tail_risk_opt python=3.10 -y
conda activate tail_risk_opt
```

### 2. Install Solvers (Conda)
```bash
# Installing binary wheels prevents build errors
conda install -c conda-forge cvxpy pyportfolioopt -y
```

### 3. Install Dependencies (Pip)
```bash
pip install -r requirements.txt
```

---

## 🏃 Usage

### Train the Model
Run the full pipeline (Data -> Model -> Train -> Evaluate):
```bash
python main.py
```
*Expected Output:*
- Data loading logs (including VIX)
- Training progress (Loss decreasing)
- Sample portfolio weights showing diversification

### Run Benchmark
Compare AI performance against a classical Mean-Variance Optimization (Max Sharpe) strategy:
```bash
python -m src.benchmark_mvo
```

---

## ⚙️ Configuration (`main.py`)

You can tune the hyperparameters in the `config` dictionary in `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `eta` | 1.0 | Risk aversion parameter (higher = safer) |
| `kappa_base` | 0.001 | Base transaction cost penalty |
| `kappa_vix_scale` | 0.0001 | Sensitivity of trading cost to VIX |
| `rho` | 0.01 | Penalty strength for holding Inverse ETF (SH) |
| `inverse_cap` | 0.3 | Hard limit on SH allocation (set in `optimization.py`) |

---

## 🤝 Project Roles & Next Steps

### ✅ Tech Lead (Role A) - **Done**
- Constructed the foundational pipeline.
- Implemented `DecisionAwareLoss` and VIX integration.
- Enforced Safety Constraints (Inverse Cap).

### ⏳ Quant Researcher (Role B) - **Next**
- Define the `CVaR` optimization logic in `src/optimization.py`.
- Refine the Risk term in `DecisionAwareLoss`.

### ⏳ Strategist (Role C) - **Pending**
- Select Macroeconomic indicators in `src/data_loader.py`.
- Analyze regime-based performance behavior.
