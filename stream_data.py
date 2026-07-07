import os
import sys
import json
import asyncio
import websockets
import pandas as pd
from datetime import datetime

# Add root directory to path (assuming you have a config.py)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# --- Configuration ---
TIMEFRAME = "1m"
# List of symbols you want to stream simultaneously
SYMBOLS = ["btcusdt", "ethusdt", "bnbusdt"] 

def get_raw_file_path(symbol: str) -> str:
    """Ensure the directory exists and return the full file path for a specific symbol."""
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    # E.g., data/raw/btcusdt_1m_raw.csv
    return f"{config.RAW_DATA_DIR}/{symbol}_{TIMEFRAME}_raw.csv"

async def stream_multiple_symbols():
    """
    Connects directly to Binance via Native WebSockets.
    Listens to multiple symbol streams simultaneously and saves ONLY closed candles.
    """
    print("=========================================================")
    print(f"📡 NATIVE MULTI-SYMBOL WEBSOCKET STREAMER (No CCXT)")
    print(f"Exchange  : Binance")
    print(f"Timeframe : {TIMEFRAME}")
    print(f"Symbols   : {', '.join([s.upper() for s in SYMBOLS])}")
    print("=========================================================\n")
    print("⏳ Connecting to Binance WebSocket...")

    # Format the streams for the Binance Combined Stream URL
    # Format: <symbol>@kline_<interval>
    streams = [f"{symbol.lower()}@kline_{TIMEFRAME}" for symbol in SYMBOLS]
    stream_params = "/".join(streams)
    
    # Binance Combined Stream Endpoint
    ws_url = f"wss://stream.binance.com:9443/stream?streams={stream_params}"

    while True:
        try:
            # Connect to the WebSocket endpoint
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                print("✅ Connected successfully! Listening to real-time data stream...\n")
                
                # Listen to the incoming stream continuously
                async for message in ws:
                    payload = json.loads(message)
                    
                    # Binance combined stream payload structure:
                    # {"stream": "btcusdt@kline_1m", "data": {"e": "kline", "k": {...}}}
                    if "data" in payload and "k" in payload["data"]:
                        kline_data = payload["data"]["k"]
                        
                        is_candle_closed = kline_data["x"]  # Boolean: True if candle is closed
                        symbol = kline_data["s"].lower()    # e.g., "btcusdt"
                        
                        # We ONLY process and save the data when the candle is finalized/closed
                        if is_candle_closed:
                            timestamp = int(kline_data["t"])
                            open_price = float(kline_data["o"])
                            high_price = float(kline_data["h"])
                            low_price = float(kline_data["l"])
                            close_price = float(kline_data["c"])
                            volume = float(kline_data["v"])
                            
                            # Create a DataFrame row
                            df_new_candle = pd.DataFrame([{
                                'timestamp': timestamp,
                                'open': open_price,
                                'high': high_price,
                                'low': low_price,
                                'close': close_price,
                                'volume': volume
                            }])
                            
                            # Determine file path based on the symbol
                            raw_file = get_raw_file_path(symbol)
                            write_header = not os.path.exists(raw_file)
                            
                            # Append to the specific symbol's CSV file
                            df_new_candle.to_csv(raw_file, mode='a', header=write_header, index=False)
                            
                            # Log to console
                            readable_time = datetime.fromtimestamp(timestamp / 1000.0).strftime('%H:%M:%S')
                            print(f"🔐 [{symbol.upper()}] {readable_time} | Close: ${close_price:,.2f} | Appended to memory.")

        except websockets.exceptions.ConnectionClosedError:
            print("⚠️ Connection closed by the Binance server. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"⚠️ Unexpected Stream Error: {e}. Reconnecting...")
            await asyncio.sleep(5)

async def main():
    await stream_multiple_symbols()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Native Streamer gracefully stopped by user.")