import os
import sys
import numpy as np
import pandas as pd
# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

import config
from utils.dataset import CryptoHedgingDataset
from agents.lstm_agent import LSTMAgent
from agents.rl_agent import RLAgent
from utils.metrics import FinancialMetrics


def main():
    print("=========================================================")
    print("FEDERATED VS LOCAL MODEL COMPARATIVE BACKTEST")
    print("=========================================================\n")

    # Target client to test on 
    target_client = "client_binance"
    model_type = 'rl'
    # 1. Define paths for saved models
    global_weights_path = config.FEDHEDGE_MODEL
    local_model_path = config.MODEL_SAVE_DIR + f"/local_{target_client}_{model_type}_model.pth"
    train_path = f"{config.PROCESSED_TRAIN_DIR}/{target_client}_train.csv"
    test_path = f"{config.PROCESSED_TEST_DIR}/{target_client}_test.csv"

    # Check if both models exist
    if not os.path.exists(global_weights_path) or not os.path.exists(local_model_path):
        print("Error: Missing trained models! Please run the training pipelines first.")
        print(f"Required: {global_weights_path} AND {local_model_path}")
        return

    # 2. Initialize and load the Global Federated Model
    print("-> Loading Global Federated Model...")

    global_agent = LSTMAgent(input_dim=config.NUM_FEATURE_COLS, hidden_dim=64)
    npz_file = np.load(global_weights_path)
    global_weights = [npz_file[key] for key in npz_file.files]
    global_agent.set_weights(global_weights)
    global_agent.model.eval()

    # 3. Initialize and load the Local Baseline Model
    print("-> Loading Local Baseline Model...")
    local_agent = LSTMAgent(input_dim=config.NUM_FEATURE_COLS, hidden_dim=64)
    local_agent.model.load_state_dict(torch.load(local_model_path, map_location=local_agent.device))
    local_agent.model.eval()

    # 4. Load the TEST partition (unseen future data) to ensure fair comparison
    print(f"-> Loading unseen test data for [{target_client.upper()}]...\n")
    test_dataset = CryptoHedgingDataset(csv_file=test_path, seq_length=10)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 5. Simulation arrays
    unhedged_returns = []
    local_hedged_returns = []
    global_hedged_returns = []
    
    local_deltas = []
    global_deltas = []

    # 6. Execute step-by-step trading simulation
    with torch.no_grad():
        for features, target_returns in test_loader:
            features = features.to(global_agent.device)
            raw_return = target_returns.item()
            
            # Predict hedge ratios (Deltas)
            delta_l = local_agent.model(features).item()
            delta_g = global_agent.model(features).item()
            
            # Calculate Hedged Returns
            return_l = raw_return - (delta_l * raw_return)
            return_g = raw_return - (delta_g * raw_return)
            
            # Save historical records
            unhedged_returns.append(raw_return)
            local_hedged_returns.append(return_l)
            global_hedged_returns.append(return_g)
            
            local_deltas.append(delta_l)
            global_deltas.append(delta_g)

    # 7. Metrics Compilation
    df = pd.DataFrame({
        'Unhedged': unhedged_returns,
        'Local_Hedged': local_hedged_returns,
        'Global_Hedged': global_hedged_returns
    })

    # Calculate Cumulative Performance
    df['Unhedged_Cum'] = (1 + df['Unhedged']).cumprod()
    df['Local_Cum'] = (1 + df['Local_Hedged']).cumprod()
    df['Global_Cum'] = (1 + df['Global_Hedged']).cumprod()

    # Annualized Volatility (hourly logs standard scaling)
    ann_factor = np.sqrt(365 * 24)
    vol_unhedged = df['Unhedged'].std() * ann_factor
    vol_local = df['Local_Hedged'].std() * ann_factor
    vol_global = df['Global_Hedged'].std() * ann_factor

    # Max Drawdowns
    fm = FinancialMetrics()
    mdd_unhedged = fm.calculate_max_drawdown(df['Unhedged_Cum'])
    mdd_local = fm.calculate_max_drawdown(df['Local_Cum'])
    mdd_global = fm.calculate_max_drawdown(df['Global_Cum'])

    # 8. Render Comparative Report
    print("PERFORMANCE COMPARISON REPORT")
    print("-" * 75)
    print(f"Strategy              | Annualized Volatility | Max Drawdown (Tail Risk)")
    print("-" * 75)
    print(f"Unhedged Asset (Base) | {vol_unhedged*100:20.2f}% | {mdd_unhedged*100:24.2f}%")
    print(f"Locally Trained AI    | {vol_local*100:20.2f}% | {mdd_local*100:24.2f}%")
    print(f"Globally Federated AI | {vol_global*100:20.2f}% | {mdd_global*100:24.2f}%")
    print("-" * 75)
    
    # Insights
    local_improvement = (vol_unhedged - vol_local) / vol_unhedged * 100
    global_improvement = (vol_unhedged - vol_global) / vol_unhedged * 100
    print(f"Local Model Risk Reduction  : {local_improvement:.2f}%")
    print(f"Global Model Risk Reduction : {global_improvement:.2f}%")
    
    if vol_global < vol_local:
        superiority = (vol_local - vol_global) / vol_local * 100
        print(f"\nSUCCESS: Federated Learning outperformed the Local Model by an extra {superiority:.2f}% risk reduction!")
    else:
        print("\nNote: Local model matches global performance. This can happen if local data distribution is highly identical to global.")

if __name__ == "__main__":
    main()