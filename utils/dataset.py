import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CryptoHedgingDataset(Dataset):
    """
    Custom PyTorch Dataset for crypto time-series data.
    Now supports chronological splitting for Train/Test sets to prevent look-ahead bias.
    """
    def __init__(self, csv_file, seq_length=10):
        self.df = pd.read_csv(csv_file)
        self.seq_length = seq_length
            
        # Extract features and targets
        # self.feature_cols = ['scaled_log_return', 'scaled_volatility', 'scaled_price_deviation', 'scaled_volume_log']
        self.feature_cols = ['scaled_open',
                             'scaled_high',
                             'scaled_low',
                             'scaled_volume',
                             'scaled_volatility',
                             'scaled_log_return_lag1',
                             'scaled_log_return_lag2',
                             'scaled_rsi_14',
                             'scaled_macd',
                             'scaled_macd_signal',
                             'scaled_bollinger_z',
                             'scaled_volume_log',
                             'scaled_volume_ratio'
                            ]
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.raw_returns = self.df['log_return'].values.astype(np.float32)

    def __len__(self):
        # Total available sequences
        return len(self.df) - self.seq_length

    def __getitem__(self, idx):
        # Slice a window of history: shape (seq_length, num_features)
        x = self.features[idx : idx + self.seq_length]
        
        # Target is the asset return at the immediately following time step (to hedge against)
        y = self.raw_returns[idx + self.seq_length]
        
        return torch.tensor(x), torch.tensor([y])