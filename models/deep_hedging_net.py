import torch
import torch.nn as nn

class DeepHedgingLSTM(nn.Module):
    """
    LSTM-based Neural Network for Deep Hedging.
    Takes sequential financial features and outputs a hedge ratio (delta).
    """
    def __init__(self, input_dim=4, hidden_dim=32, num_layers=2):
        super(DeepHedgingLSTM, self).__init__()
        
        # LSTM layer to capture temporal dependencies in financial data
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Fully connected layers to map hidden states to hedging action
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        
        # Tanh constrains the output (Hedge Ratio) between -1.0 and 1.0
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # We only take the output of the last time step in the sequence
        last_time_step = lstm_out[:, -1, :]
        
        out = self.fc1(last_time_step)
        out = self.relu(out)
        out = self.fc2(out)
        hedge_ratio = self.tanh(out)
        
        return hedge_ratio