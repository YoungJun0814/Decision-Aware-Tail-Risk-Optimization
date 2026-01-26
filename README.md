# Decision-Aware Tail Risk Optimization

> **Status:** Active Development (v2.2 Implementation Complete)  


## 📖 Project Overview

This project implements a **Decision-Aware** investment strategy that optimizes for tail risk protection and superior risk-adjusted returns. Unlike traditional methods that forecast future prices (MSE loss) and then optimize, our model learns to **optimize portfolio weights directly** by backpropagating gradients through the optimization problem.

The core philosophy is **"Asymmetric Payoff"**:
- **Upside:** Capture market growth with Long assets (SPY, XLV).
- **Downside:** Defend against crashes with Safe Haven assets (TLT, GLD, BIL).

## 🚀 Key Features (v2.2)

### 1. Decision-Aware Learning
We use a specialized loss function that trains the AI based on the **final portfolio performance** (Return - Risk) rather than prediction accuracy.
- **Goal:** Maximize Sharpe Ratio / Minimize CVaR directly.

### 2. Multi-Objective Loss Function (`src/loss.py`)
Our `DecisionAwareLoss` incorporates three key terms:
- **Return Maximization:** $\max \mu^T w$
- **Risk Minimization:** Supports `std`, `downside_deviation`, or `cvar` modes.
- **Dynamic Transaction Costs (κ(VIX)):** Penalizes excessive trading, especially during high volatility (high VIX) to enforce stability.

### 3. Black-Litterman Integration (`src/models.py`)
All 5 benchmark models (LSTM, GRU, TCN, Transformer, TFT) output **Black-Litterman parameters** (P, Q, Omega matrices) instead of direct weights:
- **P:** View matrix (asset relationships)
- **Q:** Expected return views
- **Omega:** View uncertainty

The BL formula combines market equilibrium with AI views for robust portfolio allocation.

### 4. Smart Asset Universe
A robust 5-asset universe covering all weather conditions:
- **Growth:** `SPY` (S&P 500)
- **Defensive:** `XLV` (Healthcare - Low Beta)
- **Safe Haven:** `TLT` (Treasuries), `GLD` (Gold), `BIL` (Short-Term Treasury)

### 5. Safety Net Mechanism (`src/optimization.py`)
- **Crisis Detection:** When VIX > 30, automatically increase allocation to safe assets (BIL).
- **CVaR Optimization:** Uses cvxpylayers for differentiable convex optimization.
- **No Short Selling:** Long-only portfolio constraint.

### 6. Explainable AI (XAI) (`src/explainability.py`)
To solve the "Black Box" problem, we provide a full suite of interpretability tools:
- **Gradient Saliency:** Identifies which asset's past returns influenced the decision most.
- **TFT Variable Selection:** Visualizes which input features the model weighted most.
- **Counterfactual Scenarios:** Simulates "What if SPY crashed?" to ensure the model reacts defensively.

---

## 🛠 Directory Structure

```plaintext
.
├── data/                   # Data storage
│   ├── raw/                # Raw downloads (yfinance)
│   └── processed/          # Preprocessed tensors
├── notebooks/              # Jupyter Notebooks (Regime Test, etc.)
├── results/                # Output files
│   ├── metrics/            # CSV Results (benchmark_results.csv, etc.)
│   └── plots/              # Charts (benchmark_comparison.png, xai_*.png)
├── scripts/                # Utility & Analysis Scripts
│   ├── run_xai.py          # XAI Analysis Runner
│   ├── verify_assets.py
│   └── verify_import.py
├── src/                    # Core Source Modules
│   ├── data_loader.py      # Data fetching (Prices, VIX) & Preprocessing
│   ├── loss.py             # DecisionAwareLoss implementation
│   ├── models.py           # 5 Benchmark Models (LSTM, GRU, TCN, Transformer, TFT)
│   ├── optimization.py     # Differentiable CVaR Optimization (cvxpylayers)
│   ├── explainability.py   # XAI Module (Saliency, TFT Analyzer, Counterfactuals)
│   ├── trainer.py          # Training Loop & Validation
│   ├── benchmark.py        # 5-Model Benchmark Comparison
│   └── utils.py            # Utility functions (seed, device, MDD)
├── main.py                 # Entry point (End-to-End Pipeline)
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

### Run 5-Model Benchmark
Compare 5 deep learning models (LSTM, GRU, TCN, Transformer, TFT):
```bash
python -m src.benchmark
```

### Run XAI Analysis
Analyze model decisions with explainability tools:
```bash
python scripts/run_xai.py
```

---

## ⚙️ Configuration (`main.py`)

You can tune the hyperparameters in the `config` dictionary in `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `model_type` | 'tft' | Model architecture (lstm, gru, tcn, transformer, tft) |
| `eta` | 1.0 | Risk aversion parameter (higher = safer) |
| `kappa_base` | 0.001 | Base transaction cost penalty |
| `kappa_vix_scale` | 0.0001 | Sensitivity of trading cost to VIX |
| `hidden_dim` | 64 | Hidden layer dimension |
| `epochs` | 50 | Number of training epochs |

---

## 📊 Benchmark Results

Run `python -m src.benchmark` to generate:
- `results/metrics/benchmark_results.csv`: Performance metrics (Sharpe, MDD, Annual Return)
- `results/metrics/benchmark_returns.csv`: Time series of portfolio returns
- `results/plots/benchmark_comparison.png`: Visualization chart

---

