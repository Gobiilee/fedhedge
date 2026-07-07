import os
import sys
import time
import schedule
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Add root directory to path
# sys.path.append(os.path.abspath(__file__))

import config
from utils.dataset import CryptoHedgingDataset
from agents.rl_agent import RLAgent
from utils.data_processor import LocalDataProcessor

# --- Online Learning Configuration ---
TARGET_CLIENT = "client_binance"
# The number of recent candles to keep as the Experience Replay Buffer
MEMORY_WINDOW = 500 

def fetch_latest_market_data():
    """
    Simulate or execute CCXT to fetch the latest OHLCV candles and append to raw.csv.
    In a production streaming setup, a separate WebSocket script continuously writes to this file.
    """
    print(f"\n[1] Fetching the latest market data...")
    # TODO: Implement CCXT fetch logic here or rely on external WebSocket writer
    pass 

def process_and_update_memory():
    """
    Preprocess (Calculate RSI, MACD, etc.) new data and update the Processed training file.
    """
    print(f"[2] Recalculating indicator matrix (RSI, MACD...)...")
    raw_path = f"{config.RAW_DATA_DIR}/{TARGET_CLIENT}_raw.csv"
    processed_path = f"{config.PROCESSED_TRAIN_DIR}/{TARGET_CLIENT}_train.csv"
    
    if not os.path.exists(raw_path):
        print("    -> Raw data file not found. Skipping this cycle.")
        return False
        
    df_raw = pd.read_csv(raw_path)
    
    # Maintain exactly the last MEMORY_WINDOW candles (Experience Replay Buffer mechanism)
    if len(df_raw) > MEMORY_WINDOW:
        df_raw = df_raw.tail(MEMORY_WINDOW).reset_index(drop=True)
        
    processor = LocalDataProcessor(window_size=5)
    df_features = processor.engineer_features(df_raw)
    
    # Fit and transform dynamically based only on the current rolling window
    df_processed = processor.fit_transform_local(df_features)
    
    # Overwrite the train file to prepare for DataLoader ingestion
    df_processed.to_csv(processed_path, index=False)
    return True

def learn_and_predict():
    """
    Core execution loop: Read Replay Buffer -> Train AI (Micro-batch) -> Output Hedging Prediction
    """
    print("\n" + "="*60)
    print(f"🔄 STARTING ONLINE LEARNING CYCLE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Update historical market data
    fetch_latest_market_data()
    success = process_and_update_memory()
    if not success: 
        return
    
    # 2. Initialize Agent and load the most recent model memory
    local_model_path = os.path.join(config.MODEL_SAVE_DIR, f"local_{TARGET_CLIENT}_rl_model.pth")
    agent = RLAgent(state_dim=config.NUM_FEATURE_COLS, action_dim=1, hidden_dim=config.HIDDEN_DIM)
    
    if os.path.exists(local_model_path):
        agent.actor.load_state_dict(torch.load(local_model_path, map_location=agent.device))
        print("[3] Loaded previous AI memory (Weights).")
    else:
        print("[3] No previous AI memory found. Initializing from scratch.")

    # 3. Create DataLoader from the latest memory window
    processed_path = f"{config.PROCESSED_TRAIN_DIR}/{TARGET_CLIENT}_train.csv"
    dataset = CryptoHedgingDataset(csv_file=processed_path, seq_length=10)
    
    # Use a small batch size for continuous weight updates
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 4. Review past data and adapt to LATEST market volatility (Train 1-3 Epochs)
    print("[4] Absorbing new market conditions...")
    for epoch in range(3): 
        loss = agent.train_epoch(dataloader)
        
    print(f"    -> Actor network updated successfully! Current Loss: {loss:.6f}")
    
    # 5. Save the updated "Brain" for the next scheduled cycle
    torch.save(agent.actor.state_dict(), local_model_path)
    
    # 6. MAKE ACTUAL PREDICTION (Determine the hedge ratio for the current moment)
    agent.actor.eval()
    with torch.no_grad():
        # Extract the most recent state (the very last row of the dataset sequence)
        latest_features = torch.tensor(dataset.features[-1:]).to(agent.device)
        current_state = latest_features[:, -1, :] 
        
        # Predict the optimal Delta (Hedge Ratio)
        predicted_delta = agent.actor(current_state).item()
        print(f"🚀 PREDICTION OUTPUT: Target Hedging Ratio (Delta) -> {predicted_delta*100:.2f}%")
        print("="*60)

def main():
    print("Initializing Automated Online Learning & Hedging System...")
    
    # Execute the learning and prediction loop once immediately upon startup
    learn_and_predict()
    
    # Schedule: Automatically wake up, process data, and learn every 15 minutes
    schedule.every(15).minutes.do(learn_and_predict)
    
    print("⏳ System is standing by, listening to market updates...")
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()