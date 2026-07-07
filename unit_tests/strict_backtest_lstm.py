import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.dataset import CryptoHedgingDataset
from agents.lstm_agent import LSTMAgent
from utils.metrics import FinancialMetrics


def run_backtest_on_dataset(agent, dataloader):
    """Simulates trading on a given dataloader and returns returns and deltas."""
    unhedged_returns = []
    hedged_returns = []
    deltas = []
    
    with torch.no_grad():
        for features, target_returns in dataloader:
            features = features.to(agent.device)
            raw_return = target_returns.item()
            
            # Predict hedge ratio
            delta = agent.model(features).item()
            # Bound delta between 0 and 1 for realistic hedging
            delta = max(0.0, min(1.0, delta)) 
            
            # Hedged return logic
            hedged_return = raw_return - (delta * raw_return)
            
            unhedged_returns.append(raw_return)
            hedged_returns.append(hedged_return)
            deltas.append(delta)
            
    return np.array(unhedged_returns), np.array(hedged_returns), np.array(deltas)

def evaluate_performance(unhedged, hedged):
    """Computes annualized volatility and max drawdown."""
    df = pd.DataFrame({'Unhedged': unhedged, 'Hedged': hedged})
    df['Unhedged_Cum'] = (1 + df['Unhedged']).cumprod()
    df['Hedged_Cum'] = (1 + df['Hedged']).cumprod()
    
    ann_factor = np.sqrt(365 * 24)  # Hourly data assumption
    vol_unhedged = df['Unhedged'].std() * ann_factor
    vol_hedged = df['Hedged'].std() * ann_factor
    fm = FinancialMetrics()
    mdd_unhedged = fm.calculate_max_drawdown(df['Unhedged_Cum'])
    mdd_hedged = fm.calculate_max_drawdown(df['Hedged_Cum'])
    
    return vol_unhedged, vol_hedged, mdd_unhedged, mdd_hedged

def main():
    print("=========================================================")
    print("STRICT TEST-SPLIT BACKTESTING FRAMEWORK")
    print("=========================================================\n")

    clients = ["client_binance", "client_kraken"]
    global_weights_path = "data/models/global_model.npz"

    if not os.path.exists(global_weights_path):
        print(f"Error: Global model not found at {global_weights_path}. Run simulation first.")
        return

    # 1. Load the Global Federated Model
    global_agent = LSTMAgent(input_dim=config.NUM_FEATURE_COLS, hidden_dim=32)
    npz_file = np.load(global_weights_path)
    global_weights = [npz_file[key] for key in npz_file.files]
    global_agent.set_weights(global_weights)
    global_agent.model.eval()

    # =========================================================
    # PART 1: LOCAL TEST SPLIT BACKTEST (Per Client)
    # =========================================================
    print("--- PART 1: EVALUATING ON LOCAL TEST SPLITS ---")
    
    for client in clients:
        print(f"\nEvaluating Unseen Test Split for [{client.upper()}]")
        train_path = f"{config.PROCESSED_TRAIN_DIR}/{client}_train.csv"
        test_path = f"{config.PROCESSED_TEST_DIR}/{client}_test.csv"
        local_model_path = f"data/models/local_{client}_model.pth"
        
        # Load local strict test data partition (unseen 20%)
        test_dataset = CryptoHedgingDataset(csv_file=test_path, seq_length=10)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # Run Global Model on this local test data
        g_unhedged, g_hedged, g_deltas = run_backtest_on_dataset(global_agent, test_loader)
        g_vol_un, g_vol_he, g_mdd_un, g_mdd_he = evaluate_performance(g_unhedged, g_hedged)
        
        print(f"  [Global AI] Mean Delta: {g_deltas.mean():.4f} | Vol Reduction: {((g_vol_un - g_vol_he)/g_vol_un)*100:.2f}% | MDD: {g_mdd_he*100:.2f}%")

        # Run Local Model if it exists
        if os.path.exists(local_model_path):
            local_agent = LSTMAgent(input_dim=config.NUM_FEATURE_COLS, hidden_dim=32)
            local_agent.model.load_state_dict(torch.load(local_model_path, map_location=local_agent.device))
            local_agent.model.eval()
            
            l_unhedged, l_hedged, l_deltas = run_backtest_on_dataset(local_agent, test_loader)
            l_vol_un, l_vol_he, l_mdd_un, l_mdd_he = evaluate_performance(l_unhedged, l_hedged)
            print(f"  [Local AI ] Mean Delta: {l_deltas.mean():.4f} | Vol Reduction: {((l_vol_un - l_vol_he)/l_vol_un)*100:.2f}% | MDD: {l_mdd_he*100:.2f}%")
        else:
            print(f"  [Local AI ] Missing trained local model file at {local_model_path}")

    # =========================================================
    # PART 2: GLOBAL AGGREGATED TEST BACKTEST (Network Performance)
    # =========================================================
    print("\n" + "="*57)
    print("--- PART 2: EVALUATING ON GLOBAL AGGREGATED TEST SET ---")
    print("="*57)
    print("Combining unseen test splits from all clients to measure macro generalization...\n")
    
    global_unhedged_combined = []
    global_hedged_combined = []
    
    for client in clients:
        train_path = f"{config.PROCESSED_TRAIN_DIR}/{client}_train.csv"
        test_path = f"{config.PROCESSED_TEST_DIR}/{client}_test.csv"
        test_dataset = CryptoHedgingDataset(csv_file=test_path, seq_length=10)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        unhedged, hedged, _ = run_backtest_on_dataset(global_agent, test_loader)
        global_unhedged_combined.extend(unhedged)
        global_hedged_combined.extend(hedged)
        
    global_unhedged_combined = np.array(global_unhedged_combined)
    global_hedged_combined = np.array(global_hedged_combined)
    
    tot_vol_un, tot_vol_he, tot_mdd_un, tot_mdd_he = evaluate_performance(global_unhedged_combined, global_hedged_combined)
    
    print("GLOBAL FEDERATED MODEL MACRO PERFORMANCE REPORT")
    print("-" * 65)
    print("Metric                | Unhedged Network | Hedged Network (Global AI)")
    print("-" * 65)
    print(f"Annualized Volatility | {tot_vol_un*100:15.2f}% | {tot_vol_he*100:24.2f}%")
    print(f"Max Drawdown (Risk)   | {tot_mdd_un*100:15.2f}% | {tot_mdd_he*100:24.2f}%")
    print("-" * 65)
    
    macro_reduction = (tot_vol_un - tot_vol_he) / tot_vol_un * 100
    print(f"System-wide Volatility Reduction via Federated Deep Hedging: {macro_reduction:.2f}%\n")

if __name__ == "__main__":
    main()