# ============================
# 📦 Cài thư viện cần thiết
# ============================
# !pip install ta tensorflow matplotlib scikit-learn requests python-dotenv
# # ============================
# # 🔧 Prepare Futures Data (BTCUSDT 15m) - Google Drive Version
# # ============================

import os
import time
import requests
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# # ============================
# # 🔗 Mount Google Drive
# # ============================
# from google.colab import drive
# drive.mount('/content/drive')

# ============================
# 🔧 Setup Path
# ============================
RAW_FILE = "/content/drive/MyDrive/data/btc_futures_raw.csv"
OUTPUT_FILE = "/content/drive/MyDrive/data/btc_futures_15m.csv"

# Tạo thư mục nếu chưa có
os.makedirs("/content/drive/MyDrive/data", exist_ok=True)

# ============================
# 📥 Fetch Futures data từ Binance
# ============================
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

    df = pd.DataFrame(data=all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    float_cols = ["open", "high", "low", "close", "volume"]
    df[float_cols] = df[float_cols].astype(float)

    df = df[["timestamp"] + float_cols]
    df.to_csv(RAW_FILE, index=False)
    print(f"✅ Đã lưu raw data vào {RAW_FILE}")
    time.sleep(5)

fetch_binance_futures_klines()

# ============================
# 📂 Load raw data
# ============================
print("👀 Loading raw data...")

df = pd.read_csv(RAW_FILE)

# timestamp đã là datetime → chỉ cần convert utc
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Đảm bảo float
df = df.astype({"open": float, "high": float, "low": float, "close": float})

# ============================
# 📊 Tính các chỉ báo kỹ thuật
# ============================
print("📊 Calculating technical indicators...")

df["sma"] = SMAIndicator(df["close"], window=14).sma_indicator()
df["ema"] = EMAIndicator(df["close"], window=14).ema_indicator()
macd = MACD(df["close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_diff"] = macd.macd_diff()
df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()

df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

bb = BollingerBands(df["close"], window=20)
df["bb_bbm"] = bb.bollinger_mavg()
df["bb_bbh"] = bb.bollinger_hband()
df["bb_bbl"] = bb.bollinger_lband()
df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

# ============================
# 🗑️ Clean & Save
# ============================
print("🗑️ Cleaning data...")
df.dropna(inplace=True)

final_cols = ["timestamp", "close", "sma", "ema", "macd", "macd_signal", "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]
df[final_cols].to_csv(OUTPUT_FILE, index=False)

print(f"🚀 Data prepared and saved to {OUTPUT_FILE}")
