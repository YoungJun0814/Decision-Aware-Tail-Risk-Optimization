# Decision-Aware-Tail-Risk-Optimization

## Project Construction
This project aims to optimize tail risk in a decision-aware manner using deep learning and convex optimization.

## Directory Structure
- `data/`: Data storage
    - `raw/`: Raw data (yfinance downloads)
    - `processed/`: Processed data
- `notebooks/`: Jupyter notebooks for experiments
- `src/`: Source code including data loaders, models, and optimization logic
- `main.py`: Main entry point for the pipeline
- `verify_env.py`: Script to verify installed dependencies
- `README.md`: Project documentation
- `requirements.txt`: Python dependencies

## Installation Guide (Windows/Mac/Linux)

### 1. Clone the Repository
```bash
git clone https://github.com/YoungJun0814/Decision-Aware-Tail-Risk-Optimization.git
cd Decision-Aware-Tail-Risk-Optimization
```

### 2. Environment Setup (Critical!)
This project requires specific solvers (`cvxpy`, `cvxpylayers`). To avoid C++ compilation errors on Windows, **we strongly recommend using Conda**.

**Step 1: Create Conda Environment**
```bash
conda create -n tail_risk_opt python=3.10 -y
conda activate tail_risk_opt
```

**Step 2: Install Solvers via Conda (Prevents Build Errors)**
```bash
# Installs binary wheels for optimization libraries
conda install -c conda-forge cvxpy pyportfolioopt -y
```

**Step 3: Install Remaining Dependencies via Pip**
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
Run the main pipeline to check if everything is working:
```bash
python main.py
```
If you see **"Epoch 1 complete"**, the environment is set up correctly.
