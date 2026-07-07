import os
import sys
import torch
from torch.utils.data import DataLoader

# Add root directory to path so Python can find custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.dataset import CryptoHedgingDataset
from agents.rl_agent import RLAgent
from agents.lstm_agent import LSTMAgent

def train_local_baseline(client_name="client_binance", model_type = "rl", epochs=15, batch_size=32, seq_length=10):
    print(f"============================================================")
    print(f"STARTING LOCAL TRAINING BASELINE FOR [{client_name.upper()}]")
    print(f"============================================================\n")
    
    train_path = f"{config.PROCESSED_TRAIN_DIR}/{client_name}_train.csv"
    test_path = f"{config.PROCESSED_TEST_DIR}/{client_name}_test.csv"
    
    # 1. Load Datasets with Chronological Split (Preventing Data Leakage)
    print(f"Loading and splitting dataset Ratio Train/Test {config.TRAIN_SPLIT_RATIO}")
    train_dataset = CryptoHedgingDataset(csv_file=train_path, seq_length=seq_length)
    val_dataset = CryptoHedgingDataset(csv_file=test_path, seq_length=seq_length)

    # 2. Create DataLoaders
    # Train loader MUST be shuffled so the model learns general patterns, not just sequential memorization
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation loader must remain sequential to simulate real-world forward testing
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Initialize the AI Agent (Using the unified Agent architecture)
    print(f"Initializing {model_type} Agent...")
    if model_type == 'rl':
        agent = RLAgent(input_dim=config.NUM_FEATURE_COLS, hidden_dim=config.HIDDEN_DIM)
    else:
        agent = LSTMAgent(input_dim=config.NUM_FEATURE_COLS, hidden_dim=config.HIDDEN_DIM)
    
    # 4. Training Loop
    print(f"Starting training for {epochs} epochs...\n")
    for epoch in range(1, epochs + 1):
        # Train on historical data (80%)
        train_loss = agent.train_epoch(train_loader)
        
        # Evaluate on unseen future data (20%)
        val_loss = agent.evaluate(val_loader)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.6e} | Val Loss: {val_loss:.6e}")
        
    # 5. Save the local model for future comparison with the Federated model
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    save_path = config.MODEL_SAVE_DIR + f"/local_{client_name}_{model_type}_model.pth"
    
    # Save PyTorch native state_dict since this is a local training run
    torch.save(agent.model.state_dict(), save_path)
    
    print(f"\nFinished Baseline Training!")
    print(f"Local model safely saved to: {save_path}\n")
    
    return agent

if __name__ == "__main__":
    # Test the pipeline locally using Binance data
    model_type = 'rl'
    train_local_baseline(client_name="client_kraken", model_type=model_type, epochs=1000)
    train_local_baseline(client_name="client_binance", model_type=model_type, epochs=1000)
"""
Cơ chế Cửa sổ trượt (Sliding Window): Biến seq_length=10 có nghĩa là để đưa ra quyết định phòng hộ (Hedge Ratio) cho giờ thứ 11, mô hình AI bắt buộc phải nhìn vào chuỗi hành vi của 10 giờ liên tiếp trước đó trong quá khứ. Cách thiết kế dữ liệu dạng khối (tensor) này là bắt buộc khi làm việc với các mạng hồi quy như LSTM hoặc Transformer.

Tại sao không shuffle dữ liệu? Khác với các bài toán nhận diện ảnh (Computer Vision), dữ liệu tài chính có tính phụ thuộc thời gian cực kỳ nghiêm ngặt. Nếu bạn xáo trộn (shuffle=True), bạn sẽ vô tình phá vỡ cấu trúc chuỗi và làm rò rỉ dữ liệu tương lai vào quá khứ (Data Leakage).

Theo dõi Loss: Khi bạn chạy file local_training.py, bạn sẽ thấy chỉ số Average Hedging Loss giảm dần qua từng epoch. Điều này chứng minh rằng mạng LSTM đang học được cách điều chỉnh vị thế mua/bán (Delta) sao cho triệt tiêu phương sai rủi ro của danh mục một cách hiệu quả.

Code siêu gọn gàng: Không còn các vòng lặp loss.backward() hay optimizer.step() phức tạp nữa. Tất cả logic toán học nặng nề đã được giấu kỹ bên trong hàm agent.train_epoch().

Theo dõi độ hội tụ thực tế: Bạn sẽ thấy màn hình in ra cả Train Loss và Val Loss. Nếu Train Loss giảm rất sâu nhưng Val Loss lại tăng vọt lên, bạn sẽ ngay lập tức nhận ra mô hình đang bị học vẹt (Overfitting).

Sandbox thực thụ: Nếu ngày mai bạn muốn test thử thuật toán Học tăng cường (RL), bạn chỉ cần đổi LSTMAgent thành RLAgent ở file này và nhấn "Run". Mọi thứ sẽ chạy trơn tru mà không cần chỉnh sửa logic tập dữ liệu!
    
"""

"""

File local_train.py (Huấn luyện cục bộ) đóng vai trò như một Đường cơ sở (Baseline) trong nghiên cứu Khoa học Dữ liệu. Dưới đây là 2 giá trị cốt lõi của nó:

1. Thước đo chứng minh sức mạnh của Federated Learning (FL)
Khi bạn mang dự án này đi thuyết trình (hoặc viết báo cáo), người ta sẽ luôn hỏi một câu: "Tại sao phải dùng FL cho phức tạp? Tại sao sàn Binance không tự lấy data của nó mà train một cái AI riêng?"

File local_train.py chính là công cụ để bạn trả lời câu hỏi đó:

Bạn dùng file này để tạo ra một "AI cục bộ" chỉ biết mỗi dữ liệu của Binance.

Bạn dùng hệ thống FL để tạo ra "Siêu AI toàn cầu" (global_model.npz).

Sau đó, bạn lấy cả 2 con AI này đi test trên một đợt sập giá (thanh khoản thấp) của Kraken. Con AI cục bộ của Binance chắc chắn sẽ "cháy rụi" vì nó chưa bao giờ thấy trường hợp đó, trong khi Siêu AI FL sẽ sống sót. Đây chính là giá trị cốt lõi của dự án!

2. Môi trường Sandbox (Thử nghiệm nhanh)
Khi bạn muốn đổi hàm Loss, chỉnh số lượng nơ-ron, hoặc thêm các siêu tham số (hyperparameters) mới, việc bật 1 Server và 2 Clients lên để test là rất cồng kềnh. File local_train.py cho phép bạn chạy thử thuật toán độc lập trên 1 máy tính để xem mô hình có bị lỗi syntax hay lỗi logic không trước khi đưa vào mạng lưới phân tá
"""