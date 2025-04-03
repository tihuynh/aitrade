# import pandas as pd
# import time
# from pybit.unified_trading import HTTP
#
# # ===== CẤU HÌNH =====
# API_KEY = "Hm5gG0HKbm5MDo5bpo"
# API_SECRET = "D6iP8YwCisA8pUylvh6916rnvWxoyKQnq1jp"
# SYMBOL = "BTCUSDT"
# TIMEFRAME = "60"  # 1h
# LIMIT = 2000
# OUTFILE = "data/backtest/btc_1h.csv"
# USE_TESTNET = False
#
# # ===== KẾT NỐI BYBIT =====
# session = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=USE_TESTNET)
#
# # ===== LẤY DỮ LIỆU & LƯU FILE =====
# def download_data():
#     print("⏳ Đang tải dữ liệu từ Bybit...")
#     response = session.get_kline(
#         category="spot", symbol=SYMBOL, interval=TIMEFRAME, limit=LIMIT
#     )
#     data = response["result"]["list"]
#     df = pd.DataFrame(data, columns=[
#         "timestamp", "open", "high", "low", "close", "volume", "quote_volume"
#     ])
#
#     # Ép kiểu và sắp xếp thời gian
#     df = df.sort_values("timestamp")
#     df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
#     for col in ["open", "high", "low", "close", "volume"]:
#         df[col] = df[col].astype(float)
#
#     df.to_csv(OUTFILE, index=False)
#     print(f"✅ Đã lưu dữ liệu vào: {OUTFILE}")
#     print(f"👉 Thời gian: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
#     print(f"👉 Số nến: {len(df)}")
#
# if __name__ == "__main__":
#     download_data()
from pybit.unified_trading import HTTP
import pandas as pd
import time

session = HTTP()
symbol = "BTCUSDT"
interval = "60"  # 1 giờ
limit_per_request = 1000
total_needed = 2000
data = []

print("⏳ Đang tải dữ liệu...")
start = int(time.time() * 1000)

for _ in range(total_needed // limit_per_request):
    res = session.get_kline(
        category="spot",
        symbol=symbol,
        interval=interval,
        limit=limit_per_request,
        end=start
    )
    batch = res["result"]["list"]
    data = batch + data
    start = int(batch[0][0]) - 1
    time.sleep(0.2)

df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
df = df.sort_values("timestamp")
df.to_csv("data/backtest/btc_60m.csv", index=False)

print("✅ Đã lưu file btc_60m.csv với", len(df), "nến.")
