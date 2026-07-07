# utils/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import config  # Import global configs

class LocalDataProcessor:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.scaler = StandardScaler()

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates advanced financial features (RSI, MACD, etc.)"""
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Base Signals
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=self.window_size).std()
        df['log_return_lag1'] = df['log_return'].shift(1)
        df['log_return_lag2'] = df['log_return'].shift(2)
        
        # RSI 14
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        ema_gain = gain.ewm(com=13, adjust=False).mean()
        ema_loss = loss.ewm(com=13, adjust=False).mean()
        rs = ema_gain / (ema_loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands Z-Score
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bollinger_z'] = (df['close'] - sma_20) / (std_20 + 1e-8)
        
        # Volume Dynamics
        df['volume_log'] = np.log1p(df['volume'])
        rolling_volume_mean = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (rolling_volume_mean + 1e-8)
        
        return df.dropna().reset_index(drop=True)

    def fit_transform_and_split(self, df: pd.DataFrame):
        """
        Fits the scaler ONLY on the train slice, scales the whole timeline,
        and cleanly splits the data into separate Train and Test DataFrames.
        """
        exclude_cols = ['timestamp', 'datetime', 'close', 'log_return']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Calculate chronological split boundary based on global config
        split_idx = int(len(df) * config.TRAIN_SPLIT_RATIO)
        
        # Fit ONLY on the historical train slice to completely block lookahead bias
        train_features = df.loc[:split_idx, feature_cols]
        self.scaler.fit(train_features)
        
        # Transform the whole matrix
        scaled_features = self.scaler.transform(df[feature_cols])
        
        # Reconstruct the processed dataframe
        processed_df = df[['timestamp', 'datetime', 'close', 'log_return']].copy()
        for i, col in enumerate(feature_cols):
            processed_df[f'scaled_{col}'] = scaled_features[:, i]
            
        # Clean physical separation
        train_df = processed_df.iloc[:split_idx].reset_index(drop=True)
        test_df = processed_df.iloc[split_idx:].reset_index(drop=True)
        
        return train_df, test_df