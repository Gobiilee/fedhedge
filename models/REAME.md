### Tìm hiểu về Kiến trúc Mạng Deep Hedging

Đối với dữ liệu chuỗi thời gian (time-series) như Crypto, mạng **LSTM (Long Short-Term Memory)** là lựa chọn tối ưu vì nó có khả năng nhớ các trạng thái biến động của thị trường trong quá khứ để đưa ra quyết định phòng hộ (hedging) cho hiện tại.

Mô hình của chúng ta sẽ nhận vào 4 tính năng đã scale ở bước trước và trả ra một giá trị gọi là **Hedge Ratio ($\delta$)** nằm trong khoảng `[-1, 1]` (sử dụng hàm kích hoạt `tanh`).

* $\delta = 1$: Mua full vị thế để hedge.
* $\delta = -1$: Bán khống full vị thế để hedge.
* $\delta = 0$: Đứng ngoài thị trường.

---

### Bước 3: Xây dựng mạng Neural và Hàm Loss bằng PyTorch

Chúng ta sẽ tạo 2 file mới trong thư mục `models/` theo đúng sơ đồ kiến trúc:

1. `models/deep_hedging_net.py`: Định nghĩa cấu trúc mạng LSTM.
2. `models/loss_functions.py`: Định nghĩa hàm mất mát tối ưu hóa rủi ro danh mục (Portfolio Variance Minimization).

Tạo file này để định nghĩa cách thực tế mà một AI học cách quản trị rủi ro. Thay vì dùng hàm MSE hay CrossEntropy thông thường, trong tài chính chúng ta cần giảm thiểu **Phương sai của danh mục sau khi phòng hộ (Hedged Portfolio Variance)** kết hợp với phạt chi phí giao dịch (Transaction Cost Penalty).


### Giải thích Logic (Tiếng Việt)

* **Tại sao dùng `last_time_step` trong LSTM?** Khi đưa một chuỗi dữ liệu (ví dụ 10 giờ liên tiếp) vào LSTM, mạng sẽ sinh ra 10 đầu ra tương ứng. Tuy nhiên, để đưa ra quyết định giao dịch cho giờ tiếp theo, chúng ta chỉ cần thông tin cô đọng nhất ở bước thời gian cuối cùng (`lstm_out[:, -1, :]`).
* **Hàm Loss tùy biến (Custom Loss):** Đây chính là điểm làm nên chất "Q1" cho dự án của bạn. Thay vì bắt mô hình dự đoán giá ngày mai tăng hay giảm (vốn rất khó và nhiễu), ta bắt mô hình học cách **giữ cho tài sản ổn định nhất có thể**. Nếu phương sai (`variance`) của danh mục sau khi hedge tiến về 0, nghĩa là rủi ro đã bị triệt tiêu hoàn toàn. `cost_coefficient` được thêm vào để ngăn AI liên tục đổi vị thế mua/bán liên tục làm tốn phí sàn.

Bạn hãy tạo xong 2 file cấu trúc này chưa? Bước tiếp theo chúng ta sẽ viết một script huấn luyện local thử nghiệm trên dữ liệu Binance xem AI có thực sự tối ưu được rủi ro không, trước khi chúng ta gọi Flower để biến nó thành Federated Learning.