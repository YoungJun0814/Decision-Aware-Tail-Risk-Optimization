# Decision-Aware Tail Risk Optimization

> **Status:** Active Development (10-Asset Expansion & Regime Model Analysis Complete)

## 📖 Project Overview

This project implements a **Decision-Aware** investment strategy that optimizes for tail risk protection and superior risk-adjusted returns. Unlike traditional methods that forecast future prices (MSE loss) and then optimize, our model learns to **optimize portfolio weights directly** by backpropagating gradients through the optimization problem.

The core philosophy is **"Asymmetric Payoff"**:
- **Upside:** Capture market growth with Long assets (SPY, QQQ, Sectors).
- **Downside:** Defend against crashes with Safe Haven assets (TLT, GLD, BIL).

## 🚀 Key Features (v2.3)

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
- **Omega:** View uncertainty (We recommend `Learnable` mode)

The BL formula combines market equilibrium with AI views for robust portfolio allocation.

### 4. Robust 10-Asset Universe
Expanded from 5 to 10 assets to capture diverse market conditions:
- **Equity:** `SPY` (S&P 500), `QQQ` (Nasdaq 100), `XLV` (Healthcare), `XLP` (Staples), `XLE` (Energy)
- **Bonds:** `TLT` (Long-Term), `IEF` (Mid-Term), `BIL` (T-Bills/Cash)
- **Alternatives:** `GLD` (Gold), `VNQ` (Real Estate)

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

## 📚 Analysis & Reports (New!)

Detailed analysis regarding the project's recent milestones can be found in the `docs/` directory:

- [**Benchmark Analysis (10-Asset)**](docs/benchmark_analysis.md): Comparison of 5 models and Omega/Sigma settings.
    - **Key Result:** GRU + Learnable Omega + Prior Sigma achieved **Sharpe 0.73** (vs 0.33 in 5-asset).
- [**Regime Model Analysis**](docs/regime_model_analysis.md): Code review and findings on the Regime Classifier.
- [**Expert Ideas**](docs/regime_expert_ideas.md): Proposals for fixing M1 model and incorporating MIDAS/NFCI.

---

## 🛠 Directory Structure

```plaintext
.
├── data/                   # Data storage
│   ├── raw/                # Raw downloads (yfinance)
│   └── processed/          # Preprocessed tensors
├── docs/                   # Analysis Reports & Documentation
├── notebooks/              # Jupyter Notebooks (Regime Test, etc.)
├── results/                # Output files
│   ├── metrics/            # CSV Results (benchmark_results.csv, etc.)
│   └── plots/              # Charts (benchmark_comparison.png, xai_*.png)
├── scripts/                # Utility & Analysis Scripts
│   ├── run_xai.py          # XAI Analysis Runner
│   ├── verify_assets.py
│   └── ...
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

You can tune the hyperparameters in the `config` dictionary in `main.py`.  
**Recommended Configuration (v2.3):**

| Parameter | Recommended | Description |
|---|---|---|
| `model_type` | **'gru'** | Robust performance with lower complexity |
| `omega_mode` | **'learnable'** | AI learns uncertainty from data |
| `sigma_mode` | **'prior'** | Uses market covariance for stability |
| `asset_class` | **10 Assets** | SPY, QQQ, XLV, XLP, XLE, TLT, IEF, GLD, VNQ, BIL |

---

## 📊 Benchmark Results (10-Asset)

Run `python -m src.benchmark` to reproduce these results.

| Model | Sharpe | Ann. Return | MDD | Recommendation |
|---|---|---|---|---|
| **GRU** | **0.730** | **8.14%** | -14.12% | **🥇 Best Overall** |
| LSTM | 0.725 | 8.08% | -14.11% | Excellent |
| TCN | 0.721 | 8.00% | -14.08% | Strong |
| Transformer | 0.707 | 7.86% | -13.89% | Good |
| TFT | 0.534 | 5.48% | -15.61% | Overfitting Risk |

*Note: Results based on backtest period 2007-2023.*
