import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

import config 
from utils.dataset import CryptoHedgingDataset
from agents.lstm_agent import LSTMAgent
# from agents.rl_agent import RLAgent
from utils.metrics import FinancialMetrics

def main():
    print("==========================================")
    print("FEDHEDGE BACKTEST & PERFORMANCE REPORT")
    print("==========================================\n")

    # 1. Locate the saved Global Model weights
    weights_path = config.FEDHEDGE_MODEL + ".npz"
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found. Run Server and Clients first.")
        return

    print("Loading Global AI Weights...")
    # Load the .npz file and convert it back to a list of NumPy arrays
    npz_file = np.load(weights_path)
    global_weights = [npz_file[key] for key in npz_file.files]

    # 2. Initialize the Base Agent and Inject the Weights
    # If using RL in the future, change this to RLAgent
    agent = LSTMAgent(input_dim=config.NUM_FEATURE_COLS, hidden_dim=32)
    agent.set_weights(global_weights)
    agent.model.eval() # Set model to evaluation (testing) mode

    # 3. Load Test Data (Simulating a real-world unseen dataset)
    client_name = "client_binance"
    train_path = f"{config.PROCESSED_TRAIN_DIR}/{client_name}_train.csv"
    test_path = f"{config.PROCESSED_TEST_DIR}/{client_name}_test.csv"
    print(f"Loading backtest data from {test_path}...\n")
    dataset = CryptoHedgingDataset(csv_file=test_path, seq_length=10)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # Sequential order

    unhedged_returns = []
    hedged_returns = []
    deltas_history = []

    # 4. Run Inference (Step-by-step Trading Simulation)
    with torch.no_grad():
        for features, target_returns in dataloader:
            features = features.to(agent.device)
            raw_return = target_returns.item()
            # current_state = features[:, -1, :]
            # The AI Agent predicts the Hedge Ratio (Delta)
            delta = agent.model(features).item()
            
            # Hedged Return = Raw Asset Return - (Delta * Raw Asset Return)
            hedged_return = raw_return - (delta * raw_return)
            
            unhedged_returns.append(raw_return)
            hedged_returns.append(hedged_return)
            deltas_history.append(delta)

    # 5. Calculate Financial Metrics
    df = pd.DataFrame({
        'Unhedged_Return': unhedged_returns,
        'Hedged_Return': hedged_returns,
        'Delta': deltas_history
    })

    # Convert to Cumulative Returns
    df['Unhedged_Cum'] = (1 + df['Unhedged_Return']).cumprod()
    df['Hedged_Cum'] = (1 + df['Hedged_Return']).cumprod()

    # Annualized Volatility (assuming hourly data: 365 days * 24 hours)
    unhedged_vol = df['Unhedged_Return'].std() * np.sqrt(365 * 24)
    hedged_vol = df['Hedged_Return'].std() * np.sqrt(365 * 24)

    # Maximum Drawdown (Tail Risk)
    fm = FinancialMetrics()
    unhedged_mdd = fm.calculate_max_drawdown(df['Unhedged_Cum'])
    hedged_mdd = fm.calculate_max_drawdown(df['Hedged_Cum'])

    # 6. Display Output Report
    print("FINAL BACKTEST RESULTS")
    print("-" * 55)
    print(f"Average Hedge Ratio (Delta) : {np.mean(deltas_history):.4f}")
    print("-" * 55)
    print("Metric                  | Unhedged Base | Hedged Portfolio")
    print("-" * 55)
    print(f"Annualized Volatility   | {unhedged_vol*100:12.2f}% | {hedged_vol*100:14.2f}%")
    print(f"Max Drawdown (Risk)     | {unhedged_mdd*100:12.2f}% | {hedged_mdd*100:14.2f}%")
    print("-" * 55)
    
    vol_reduction = (unhedged_vol - hedged_vol) / unhedged_vol * 100
    print(f"\nCONCLUSION: FedHedge AI reduced portfolio volatility by {vol_reduction:.2f}%!\n")

if __name__ == "__main__":
    main()