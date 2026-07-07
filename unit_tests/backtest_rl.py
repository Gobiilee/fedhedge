import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add root directory to path to prevent import errors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.dataset import CryptoHedgingDataset
from agents.rl_agent import RLAgent
from utils.metrics import FinancialMetrics

def main():
    print("=========================================================")
    print("     REINFORCEMENT LEARNING (RL) MODEL BACKTEST")
    print("=========================================================\n")

    # 1. Configuration parameters
    target_client = "client_binance"
    hidden_dim = config.HIDDEN_DIM  # Matches your current local training configuration
    
    # Define paths for the saved model and test data
    local_model_path = os.path.join(config.MODEL_SAVE_DIR, f"local_{target_client}_rl_model.pth")
    test_path = f"{config.PROCESSED_TEST_DIR}/{target_client}_test.csv"

    # Check if the trained model file exists
    if not os.path.exists(local_model_path):
        print(f"Error: Trained model not found at: {local_model_path}")
        print("Please run the local_training pipeline first!")
        return

    # 2. Initialize Agent and load Actor weights
    print(f"-> Initializing RLAgent with Hidden Dim = {hidden_dim}...")
    agent = RLAgent(input_dim=config.NUM_FEATURE_COLS, hidden_dim=hidden_dim)
    
    print("-> Loading Actor weights into evaluation mode...")
    # Load weights directly into the actor network and set to eval() mode
    agent.actor.load_state_dict(torch.load(local_model_path, map_location=agent.device))
    agent.actor.eval() 

    # 3. Load unseen TEST partition (future market data)
    print(f"-> Loading unseen test data for [{target_client.upper()}]...")
    test_dataset = CryptoHedgingDataset(csv_file=test_path, seq_length=10)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 4. Simulation arrays
    unhedged_returns = []
    hedged_returns = []
    predicted_deltas = []

    # 5. Execute step-by-step trading simulation
    print("-> Running step-by-step hedging simulation...")
    with torch.no_grad():
        for features, target_returns in test_loader:
            features = features.to(agent.device)
            raw_return = target_returns.item()
            
            # Extract the last timestep state from the sequence (Shape: [1, NUM_FEATURE_COLS])
            current_state = features[:, -1, :] 
            
            # Predict the hedge ratio (Delta)
            delta = agent.actor(current_state).item()
            
            # Calculate Hedged Return: Hedged_Return = Raw_Return - (Delta * Raw_Return)
            hedged_return = raw_return - (delta * raw_return)
            
            # Save historical records
            unhedged_returns.append(raw_return)
            hedged_returns.append(hedged_return)
            predicted_deltas.append(delta)

    # 6. Metrics Compilation using pandas DataFrame
    df = pd.DataFrame({
        'Unhedged': unhedged_returns,
        'Hedged': hedged_returns,
        'Delta': predicted_deltas
    })

    # Calculate Cumulative Performance
    df['Unhedged_Cum'] = (1 + df['Unhedged']).cumprod()
    df['Hedged_Cum'] = (1 + df['Hedged']).cumprod()

    # Annualized Volatility (assuming hourly data logs scaling)
    ann_factor = np.sqrt(365 * 24)
    vol_unhedged = df['Unhedged'].std() * ann_factor
    vol_hedged = df['Hedged'].std() * ann_factor

    # Max Drawdowns (Tail Risk Evaluation)
    fm = FinancialMetrics()
    mdd_unhedged = fm.calculate_max_drawdown(df['Unhedged_Cum'])
    mdd_hedged = fm.calculate_max_drawdown(df['Hedged_Cum'])

    # 7. Render Comparative Performance Report
    print("\n" + "="*75)
    print("                      RL BACKTEST PERFORMANCE REPORT")
    print("=========================================================")
    print(f"Strategy              | Annualized Volatility | Max Drawdown (Tail Risk)")
    print("-"*75)
    print(f"Unhedged Asset (Base) | {vol_unhedged*100:20.2f}% | {mdd_unhedged*100:24.2f}%")
    print(f"RL AI Hedged Strategy | {vol_hedged*100:20.2f}% | {mdd_hedged*100:24.2f}%")
    print("-"*75)
    
    # Performance insights
    risk_reduction = (vol_unhedged - vol_hedged) / vol_unhedged * 100
    print(f"-> AI Volatility Risk Reduction : {risk_reduction:.2f}%")
    print(f"-> Mean Predicted Delta Ratio   : {df['Delta'].mean():.4f}")
    
    if vol_hedged < vol_unhedged:
        print("\n[SUCCESS]: The Reinforcement Learning model successfully reduced market risk!")
    else:
        print("\n[WARNING]: Underperformance detected. Consider tuning the reward function or training for more epochs.")

if __name__ == "__main__":
    main()