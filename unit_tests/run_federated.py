import os
import sys
# Add root directory to the path for module importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
from torch.utils.data import DataLoader

import config
from agents.lstm_agent import LSTMAgent
from agents.rl_agent import RLAgent
from utils.dataset import CryptoHedgingDataset
from clients.base_client import HedgingFlowerClient
from server.fedprox_strategy import get_fedprox_strategy

def client_fn(cid: str) -> fl.client.Client:
    """
    Factory function for Flower Simulation. 
    Whenever the virtual server needs a client, it calls this function to spawn one.
    """
    client_mapping = {
        "0": "client_binance",
        "1": "client_kraken"
    }
    client_name = client_mapping[cid]
    train_path = f"{config.PROCESSED_TRAIN_DIR}/{client_name}_train.csv"
    test_path = f"{config.PROCESSED_TEST_DIR}/{client_name}_test.csv"
    
    # 1. Initialize Datasets with chronological Train/Test split
    train_dataset = CryptoHedgingDataset(csv_file=train_path, seq_length=10)
    val_dataset = CryptoHedgingDataset(csv_file=test_path, seq_length=10)
    
    # 2. Create PyTorch DataLoaders (Shuffle training data, keep validation sequential)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 3. Initialize the AI Agent (LSTM for deep hedging)
    # agent = LSTMAgent(input_dim=config.NUM_FEATURE_COLS, hidden_dim=32)
    agent = RLAgent(input_dim=config.NUM_FEATURE_COLS, hidden_dim=config.HIDDEN_DIM)
    
    # 4. Return the fully configured and isolated Client
    return HedgingFlowerClient(
        client_id=client_name, 
        agent=agent, 
        train_loader=train_loader,
        val_loader=val_loader
    ).to_client()

def main():
    print("=========================================================")
    print("STARTING FEDERATED LEARNING SIMULATION (VIRTUAL MODE)")
    print("=========================================================\n")
    
    # Fetch the FedProx strategy (Includes automatic model saving at the final round)
    strategy = get_fedprox_strategy()
    
    # Launch the centralized simulation (All virtual clients run in this single terminal)
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=1000),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
"""
get_parameters và set_parameters: Đây là rào chắn bảo mật. Flower chỉ lấy mảng NumPy chứa trọng số của mạng nơ-ron (ví dụ: [0.23, -0.11, ...]) truyền qua mạng, hoàn toàn không chạm vào file CSV chứa lịch sử giao dịch gốc của sàn.

Cơ chế hoạt động của start_simulation: Khi bạn chạy file này, Flower sẽ dựng lên một Server ảo. Ở Round 1, Server gửi trọng số khởi tạo xuống. Hàm client_fn lập tức dựng lên 2 thực thể Client (Binance và Kraken). Cả hai sàn tự train ngầm độc lập, in log ra màn hình, rồi gửi kết quả nén về Server. Server tổng hợp xong sẽ chuyển sang Round 2.

1. Tác dụng của file run_federated.py là gì?
Trong hệ thống Flower (Học liên kết), có 2 cách để chạy mạng lưới:

Cách 1: Chạy phân tán thực tế (True Distributed): Bạn phải mở 3 Terminal khác nhau (1 cái cho Server, 2 cái cho Client) như tôi hướng dẫn lúc nãy. Cách này chuẩn với thực tế triển khai, nhưng lại mất thời gian khi bạn chỉ muốn test code nhanh.

Cách 2: Chạy Giả lập trên 1 máy (Simulation Mode): Chính là file run_federated.py này! Khi dùng hàm fl.simulation.start_simulation, Flower sẽ ảo hóa ra Server và tự động "đẻ" ra các Client ngay trong bộ nhớ RAM của cùng 1 Terminal. Bạn chỉ cần gõ 1 lệnh duy nhất là toàn bộ hệ thống Học liên kết sẽ tự chạy từ A-Z. Rất tuyệt vời cho việc nghiên cứu và gỡ lỗi nhanh.
"""