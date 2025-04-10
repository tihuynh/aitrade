from pybit.unified_trading import HTTP
import pandas as pd
import time

session = HTTP()
symbol = "BTCUSDT"
interval = "15"
limit_per_request = 1000
total_needed = 30000
data = []

print("⏳ Đang tải dữ liệu...")
start = int(time.time() * 1000)  # timestamp hiện tại

for _ in range(total_needed // limit_per_request):
    res = session.get_kline(
        category="spot",
        symbol=symbol,
        interval=interval,
        limit=limit_per_request,
        end=start
    )
    batch = res["result"]["list"]
    data = batch + data  # nối về đầu
    start = int(batch[0][0]) - 1  # lùi lại để lấy batch kế tiếp

    time.sleep(0.2)

df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
df = df.sort_values("timestamp")
df.to_csv("data/backtest/btc_15m.csv", index=False)

print("✅ Đã lưu file btc_15m.csv với", len(df), "nến.")
