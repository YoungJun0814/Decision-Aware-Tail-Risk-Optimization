
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from src.data_loader import prepare_training_data
from src.models import get_model
from src.loss import DecisionAwareLoss
from src.utils import set_seed, get_device, calculate_mdd

# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    'start_date': '2005-01-01',
    'end_date': '2024-01-01',
    'seq_length': 12,
    'train_ratio': 0.8,
    'batch_size': 32,
    'epochs': 50,          # Reduced epochs
    'learning_rate': 0.001,
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'eta': 1.0,           # Risk aversion
    'kappa_base': 0.001,
    'kappa_vix_scale': 0.0001,
    'device': get_device()
}

RISK_TYPES = ['std', 'downside_deviation', 'cvar']

def train_and_evaluate_loss_variant(risk_type, X_tensor, y_tensor, vix_tensor, scaler, asset_names, y_dates):
    """
    특정 loss type으로 TFT 모델 학습 및 평가
    """
    print(f"\n[{risk_type.upper()}] Training Started...")
    set_seed(42)  # For fair comparison
    
    device = CONFIG['device']
    num_assets = len(asset_names)
    input_dim = X_tensor.shape[-1]
    
    # 1. Model (Always TFT)
    model = get_model('tft', input_dim, num_assets, device=device)
    
    # 2. Loss Function with specific risk_type
    loss_fn = DecisionAwareLoss(
        eta=CONFIG['eta'],
        kappa_base=CONFIG['kappa_base'],
        kappa_vix_scale=CONFIG['kappa_vix_scale'],
        risk_type=risk_type
    )
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # 3. Data Split
    train_size = int(len(X_tensor) * CONFIG['train_ratio'])
    X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
    vix_train, vix_val = vix_tensor[:train_size], vix_tensor[train_size:]
    
    # 4. Training Loop
    best_val_loss = float('inf')
    patience = 20
    no_improve = 0
    val_weights = None
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        optimizer.zero_grad()
        
        weights = model(X_train)
        loss = loss_fn(weights, y_train, vix_train)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            curr_val_weights = model(X_val)
            val_loss = loss_fn(curr_val_weights, y_val, vix_val, prev_weights=None) # Simplify prev_weights for val
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                val_weights = curr_val_weights
                no_improve = 0
            else:
                no_improve += 1
                
        if epoch % 10 == 0:
            print(f"  Epoch [{epoch}/{CONFIG['epochs']}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
            
    # 5. Evaluation
    if val_weights is not None:
        # Denormalize returns
        if scaler is not None:
            mean = torch.tensor(scaler.mean_[:num_assets], device=device).float()
            scale = torch.tensor(scaler.scale_[:num_assets], device=device).float()
            y_val_real = y_val * scale + mean
        else:
            y_val_real = y_val
            
        portfolio_returns = (val_weights * y_val_real).sum(dim=1).cpu().numpy()
        mean_ret = np.mean(portfolio_returns) * 12
        std_ret = np.std(portfolio_returns) * np.sqrt(12)
        sharpe = mean_ret / (std_ret + 1e-6)
        mdd = calculate_mdd(portfolio_returns)
        
        return {
            'risk_type': risk_type,
            'val_loss': best_val_loss,
            'sharpe': sharpe,
            'annual_return': mean_ret * 100, # % for readability
            'mdd': mdd * 100,               # % for readability
            'val_returns': portfolio_returns,
            'val_dates': y_dates[train_size:]
        }
    else:
        return None

def run_comparison():
    print("=" * 60)
    print("Loss Function Comparison Benchmark (TFT Model)")
    print("=" * 60)
    
    # Load Data
    print("Loading Data...")
    X_tensor, y_tensor, vix_tensor, scaler, asset_names, y_dates = prepare_training_data(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        seq_length=CONFIG['seq_length']
    )
    
    X_tensor = X_tensor.to(CONFIG['device'])
    y_tensor = y_tensor.to(CONFIG['device'])
    vix_tensor = vix_tensor.to(CONFIG['device'])
    
    results = []
    
    for risk in RISK_TYPES:
        res = train_and_evaluate_loss_variant(risk, X_tensor, y_tensor, vix_tensor, scaler, asset_names, y_dates)
        if res:
            results.append(res)
            
    # Save Results
    if results:
        # Summary CSV
        summary_data = [{k: v for k, v in r.items() if k not in ['val_returns', 'val_dates']} for r in results]
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv("loss_comparison_results.csv", index=False)
        print("\n[Saved] loss_comparison_results.csv")
        print(df_summary)
        
        # Returns CSV for Visualization
        df_returns = pd.DataFrame(index=results[0]['val_dates'])
        for res in results:
            df_returns[res['risk_type']] = res['val_returns']
        df_returns.to_csv("loss_comparison_returns.csv")
        print("[Saved] loss_comparison_returns.csv")
        
        # Visualization Code (Embedded)
        import matplotlib.pyplot as plt
        
        cumulative_returns = (1 + df_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Return
        for col in cumulative_returns.columns:
            final = cumulative_returns[col].iloc[-1]
            axes[0].plot(cumulative_returns.index, cumulative_returns[col], label=f"{col.upper()} ({final:.2f})")
        axes[0].set_title("Cumulative Returns by Loss Type")
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # MDD
        for col in drawdown.columns:
            mdd_val = drawdown[col].min()
            axes[1].plot(drawdown.index, drawdown[col], label=f"{col.upper()} (MDD: {mdd_val:.1%})")
        axes[1].set_title("Drawdown by Loss Type")
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig("loss_comparison_plot.png")
        print("[Saved] loss_comparison_plot.png")

if __name__ == "__main__":
    run_comparison()
