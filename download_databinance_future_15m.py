import requests
import pandas as pd
import time

def fetch_binance_futures_klines(symbol="BTCUSDT", interval="15m", total_limit=30000):
    url = "https://fapi.binance.com/fapi/v1/klines"
    limit = 1000
    end_time = int(time.time() * 1000)
    all_data = []

    while len(all_data) < total_limit:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, total_limit - len(all_data)),
            "endTime": end_time
        }
        res = requests.get(url, params=params)
        data = res.json()
        if not data:
            break
        all_data = data + all_data
        end_time = data[0][0] - 1
        time.sleep(0.2)

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # Chỉ chuyển đổi các cột số sang float
    float_cols = ["open", "high", "low", "close", "volume"]
    df[float_cols] = df[float_cols].astype(float)

    df = df[["timestamp"] + float_cols]  # Giữ lại các cột cần thiết
    df.to_csv("data/backtest/btcusdt_futures_15m.csv", index=False)
    print("✅ Đã lưu btcusdt_futures_15m.csv")

fetch_binance_futures_klines()
