# config.py
import os

# --- Trading Configurations ---
SYMBOL = 'ETH/USDT'
TIMEFRAME = '1m'
DATA_FETCH_LIMIT = 200000000
TRAIN_SPLIT_RATIO = 0.8
SEQ_LENGTH = 10

# --- Directory Registries ---
RAW_DATA_DIR = "data/raw"
PROCESSED_TRAIN_DIR = "data/processed/train"
PROCESSED_TEST_DIR = "data/processed/test"
MODEL_SAVE_DIR = "data/models"

FEDHEDGE_MODEL = MODEL_SAVE_DIR + "/fedhedge_model"

NUM_FEATURE_COLS = 13
HIDDEN_DIM = 64

# --- GLOBAL CLIENT REGISTRY (The Registry Pattern) ---
# To add a new client, simply append a new entry here. No pipeline code changes required!
CLIENT_REGISTRY = {
    "0": {
        "client_name": "client_binance",
        "exchange_id": "binance",
        "enabled": True
    },
    "1": {
        "client_name": "client_kraken",
        "exchange_id": "kraken",
        "enabled": True
    }
}

def get_enabled_clients():
    """Returns a filtered dictionary of clients that are marked as active."""
    return {cid: info for cid, info in CLIENT_REGISTRY.items() if info["enabled"]}