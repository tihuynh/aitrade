import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Cấu hình
symbol = "BTCUSDT"
interval = "15m"
limit = 1000  # Binance API tối đa 1000 nến mỗi lần
total_candles = 30000

# Tính tổng số lần gọi API
iterations = total_candles // limit

# Khởi tạo list lưu dữ liệu
all_candles = []

# Timestamp hiện tại (Binance dùng ms)
end_time = int(time.time() * 1000)

print(f"Bắt đầu tải {total_candles} nến BTCUSDT khung 15 phút từ Binance...")

for i in range(iterations):
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "endTime": end_time,
        "limit": limit
    }
    response = requests.get(url, params=params)
    data = response.json()

    if not data:
        print("❌ Không có dữ liệu, dừng quá trình.")
        break

    all_candles.extend(data)

    # Cập nhật end_time cho lần tiếp theo
    end_time = data[0][0] - 1  # Lùi về trước để tránh trùng nến
    print(f"Đã tải {(i + 1) * limit} / {total_candles} nến...")

    # Delay nhẹ để tránh rate limit Binance
    time.sleep(0.3)

# Chuyển về DataFrame
df = pd.DataFrame(all_candles, columns=[
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
])

# Xử lý DataFrame
df = df[["timestamp", "open", "high", "low", "close", "volume"]]
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

# Sort tăng dần theo thời gian
df.sort_values("timestamp", inplace=True)

# Lưu file
df.to_csv("data/backtest/binance_btc_15m.csv", index=False)
print("✅ Đã lưu file binance_btc_15m.csv thành công!")
