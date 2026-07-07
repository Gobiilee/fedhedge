import torch
import torch.nn as nn

class RLHedgeActor(nn.Module):
    """
    ACTOR NETWORK (The Policy pi):
    Observes the market state and decides the continuous hedging action.
    Outputs a deterministic action (delta) between -1 and 1.
    """
    def __init__(self, input_dim=4, hidden_dim=64):
        super(RLHedgeActor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh() # Constrain action to [-1, 1] for short/long ratio
        )

    def forward(self, state):
        # state shape: (batch_size, input_dim)
        return self.network(state)

class RLHedgeCritic(nn.Module):
    """
    CRITIC NETWORK (The Value Function Q or V):
    Evaluates how good an action is given a specific market state, 
    based on expected future rewards (e.g., negative cVaR).
    """
    def __init__(self, input_dim=4, action_dim=1, hidden_dim=64):
        super(RLHedgeCritic, self).__init__()
        # The critic takes BOTH the state and the actor's action as input
        self.network = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Outputs a single Q-value (Expected Reward)
        )

    def forward(self, state, action):
        # Concatenate state and action along the feature dimension
        x = torch.cat([state, action], dim=-1)
        return self.network(x)
    

"""
Mạng này sẽ bao gồm 2 phần:Actor (Nhà giao dịch): Học chiến lược $\pi$, quyết định tỷ lệ Hedge $\delta$.Critic (Nhà quản trị rủi ro): Đánh giá xem quyết định của Actor mang lại cVaR tồi tệ hay an toàn để điều chỉnh lại.

Việc thêm file này mở ra một hướng rẽ (branch) cực kỳ cao cấp cho dự án:State (Trạng thái - $S_t$): Chính là 4 features mà chúng ta đã làm ở Phase 2 (giá, khối lượng, độ lệch, biến động).Action (Hành động - $A_t$): Đầu ra của RLHedgeActor, tương đương $\delta$.Reward (Phần thưởng - $R_t$): Thay vì dùng HedgingVarianceLoss, bạn sẽ định nghĩa một hàm Reward = -cVaR - Transaction_Cost. Nếu cVaR cao (rủi ro lớn), Reward sẽ bị âm nặng, ép mạng Actor phải thay đổi chiến lược $\pi$.

"""