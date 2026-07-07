import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import flwr as fl
from torch.utils.data import DataLoader

# Import Core Client và Dataset
from clients.base_client import HedgingFlowerClient
from utils.dataset import CryptoHedgingDataset

# Import Agent (Alpha Strategy)
from agents.lstm_agent import LSTMAgent
# from agents.rl_agent import RLAgent

def main():
    print("--- Starting Binance Local Node ---")
    client_name = "client_binance"
    train_path = f"{config.PROCESSED_TRAIN_DIR}/{client_name}_train.csv"
    test_path = f"{config.PROCESSED_TEST_DIR}/{client_name}_test.csv"
    
    print(f"[{client_name}] Loading local train/test data...")
    
    train_dataset = CryptoHedgingDataset(csv_file=train_path, seq_length=10)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = CryptoHedgingDataset(csv_file=test_path, seq_length=10)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"[{client_name}] Initializing AI Agent...")
    agent = LSTMAgent(input_dim=config.NUM_FEATURE_COLS, hidden_dim=32)
    
    client = HedgingFlowerClient(
        client_id=client_name, 
        agent=agent, 
        train_loader=train_loader,
        val_loader=val_loader
    )

    print(f"[{client_name}] Connecting to the Global Server at 127.0.0.1:8080...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )

if __name__ == "__main__":
    main()