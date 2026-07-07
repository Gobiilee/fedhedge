import os
import sys
import time
import ccxt
import pandas as pd

# Sync system paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.data_processor import LocalDataProcessor

def init_environment():
    """Dynamically creates all required output directories defined in config."""
    print("-> Creating and verifying system paths...")
    dirs = [config.RAW_DATA_DIR, config.PROCESSED_TRAIN_DIR, config.PROCESSED_TEST_DIR, config.MODEL_SAVE_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def fetch_history(exchange_id, client_name):
    """Fetches and updates historical OHLCV data utilizing pagination loops."""
    # Dynamic Factory: Instantiate ccxt exchange object dynamically from string ID
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})
    
    print(f"\nDownloading data for [{client_name.upper()}] using {exchange_id} API...")
    all_ohlcv = []
    since_time = exchange.parse8601('2015-01-01T00:00:00Z')
    
    while len(all_ohlcv) < config.DATA_FETCH_LIMIT:
        try:
            remaining = config.DATA_FETCH_LIMIT - len(all_ohlcv)
            chunk = min(1000, remaining)
            batch = exchange.fetch_ohlcv(config.SYMBOL, config.TIMEFRAME, since=since_time, limit=chunk)
            
            if not batch:
                break
            all_ohlcv.extend(batch)
            print(f"   [+] Loaded {len(batch)} rows | Cumulative: {len(all_ohlcv)} / {config.DATA_FETCH_LIMIT}")
            
            last_time = batch[-1][0]
            if since_time == last_time:
                break
            since_time = last_time + 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Network error: {e}")
            break
            
    if not all_ohlcv:
        return None
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

def process_and_save_splits():
    """Loads raw data, engineers math signals, and saves into isolated train/test folders."""
    print("\n" + "="*60)
    print("EXECUTION PHASE: FEATURE ENGINEERING & TIMELINE SPLITTING")
    print("="*60)
    
    clients = config.get_enabled_clients()
    for cid, info in clients.items():
        client_name = info["client_name"]
        raw_path = f"{config.RAW_DATA_DIR}/{client_name}_raw.csv"
        
        if not os.path.exists(raw_path):
            print(f"Raw data missing for {client_name}, skipping calculation step.")
            continue
            
        print(f"Structuring training slices for [{client_name.upper()}]...")
        df_raw = pd.read_csv(raw_path)
        
        processor = LocalDataProcessor(window_size=5)
        df_features = processor.engineer_features(df_raw)
        
        # Clear separation of train and test dataframes
        train_df, test_df = processor.fit_transform_and_split(df_features)
        
        # Physically save to different dedicated directories
        train_path = f"{config.PROCESSED_TRAIN_DIR}/{client_name}_train.csv"
        test_path = f"{config.PROCESSED_TEST_DIR}/{client_name}_test.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Splits saved -> TRAIN rows: {len(train_df)} | TEST rows: {len(test_df)}")

def main():
    init_environment()
    
    # 1. Loop through all clients registered inside config.py
    active_clients = config.get_enabled_clients()
    for cid, info in active_clients.items():
        df_raw = fetch_history(info["exchange_id"], info["client_name"])
        if df_raw is not None:
            raw_save_path = f"{config.RAW_DATA_DIR}/{info['client_name']}_raw.csv"
            df_raw.to_csv(raw_save_path, index=False)
            print(f"Saved raw archive to: {raw_save_path}")
            
    # 2. Math processing and physically separating training data from test evaluation grounds
    process_and_save_splits()
    print("\n Pipeline Complete. All client environments are perfectly decoupled!")

if __name__ == "__main__":
    main()