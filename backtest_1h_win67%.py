import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import ta


# ====== LOAD CSV + THÊM INDICATORS ======
def load_csv_and_add_indicators(csv_file):
    df = pd.read_csv(csv_file)
    # Tự động xử lý timestamp đúng kiểu
    if df["timestamp"].dtype == "int64" or df["timestamp"].dtype == "float64":
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    print(f"Tổng số nến: {len(df)}")
    print("Thời gian đầu:", df['timestamp'].iloc[0])
    print("Thời gian cuối:", df['timestamp'].iloc[-1])
    df.sort_values("timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df["sma"] = ta.trend.sma_indicator(df["close"], window=14)
    df["ema"] = ta.trend.ema_indicator(df["close"], window=14)
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr"] = atr.average_true_range()

    df.dropna(inplace=True)
    return df

# ====== BACKTEST CHIẾN LƯỢC ======
def backtest_ai_strategy(model, scaler, data, initial_balance=5000):
    LOOKBACK = 100
    balance = initial_balance
    equity_curve = []
    timestamps = []
    position = 0
    buy_price = 0
    win_count = 0
    loss_count = 0
    trade_log = []
    take_profit = 0
    stop_loss = 0
    if len(data) < LOOKBACK + 10:
        print(f"❌ Không đủ dữ liệu để backtest! Cần ít nhất {LOOKBACK + 10} nến.")
        return
    else:
        print(f"✅ Đủ dữ liệu để backtest. Số nến: {len(data)}")
    for i in range(LOOKBACK, len(data)):

        window = data.iloc[i - LOOKBACK:i]
        last_sequence = scaler.transform(window[[
            "close", "sma", "ema", "macd", "macd_signal", "macd_diff",
            "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr"
        ]])
        last_sequence = last_sequence.reshape(1, LOOKBACK, 11)
        # Dự đoán (scaled)
        scaled_pred = model.predict(last_sequence, verbose=0)[0][0]

        # Tạo dummy row để inverse transform
        dummy = np.zeros((1, 11))  # 11 là số feature
        dummy[0][0] = scaled_pred
        prediction = scaler.inverse_transform(dummy)[0][0]

        current_price = data.iloc[i]["close"]
        # BƯỚC 3: In 10 lần đầu tiên
        if i < LOOKBACK + 10:
            print(f"[DEBUG] Dự đoán: {prediction:.2f} | Giá hiện tại: {current_price:.2f}")

        atr = data.iloc[i]["atr"]
        timestamp = data.iloc[i]["timestamp"]
        # ==== ĐIỀU KIỆN VÀO LỆNH THÔNG MINH ====
        macd_val = data.iloc[i]["macd"]
        macd_sig = data.iloc[i]["macd_signal"]
        rsi_val = data.iloc[i]["rsi"]
        ema_val = data.iloc[i]["ema"]

        # Nới điều kiện dự đoán AI chỉ cần > 0.5%
        ai_confidence = prediction > current_price * 1.005

        # MACD cắt lên nhưng cho phép chênh lệch nhỏ
        macd_bullish = (macd_val - macd_sig) > -5

        # RSI không quá thấp (từ 45 trở lên)
        rsi_ok = rsi_val > 45

        # Giá cắt lên EMA hoặc bằng EMA
        price_above_ema = current_price >= ema_val

        buy_condition = ai_confidence and macd_bullish and rsi_ok and price_above_ema
        # print(f"[DEBUG] AI ok: {ai_confidence}, MACD: {macd_bullish}, RSI: {rsi_ok}, EMA: {price_above_ema}")

        if position == 0 and buy_condition:
            diff = prediction - current_price
            print(
                f"[CHECK] {timestamp} | Dự đoán: {prediction:.2f} | Giá hiện tại: {current_price:.2f} | Chênh lệch: {diff:.2f}")
            print("✅ Thỏa điều kiện mua thông minh")

            position = 1
            buy_price = current_price
            take_profit = round(buy_price * 1.012, 2)  # TP: +1.2%
            stop_loss = round(buy_price * 0.992, 2)  # SL: -0.8%
            trade_log.append(f"🟢 Mua tại {buy_price:.2f} ({timestamp}) | TP: {take_profit:.2f}, SL: {stop_loss:.2f}")

        elif position == 1:
            if current_price >= take_profit:
                balance *= 1 + (atr * 3 / buy_price)
                trade_log.append(f"✅ TP tại {current_price:.2f} ({timestamp}) | Số dư: {balance:.2f}")
                win_count += 1
                position = 0
            elif current_price <= stop_loss:
                balance *= 1 - (atr * 2 / buy_price)
                trade_log.append(f"⛔ SL tại {current_price:.2f} ({timestamp}) | Số dư: {balance:.2f}")
                loss_count += 1
                position = 0

        if position == 0:
            equity_curve.append(balance)
            timestamps.append(timestamp)

    # VẼ BIỂU ĐỒ
    import matplotlib.pyplot as plt
    import os
    os.makedirs("backtest_results", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, equity_curve, label="Equity")
    plt.title("📈 Equity Curve")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.xlabel("Time")
    plt.ylabel("Balance (USDT)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig("backtest_results/equity_curve.png")  # Lưu biểu đồ
    print("🖼️ Đã lưu biểu đồ tại: backtest_results/equity_curve.png")

    # KẾT QUẢ
    print("\n===== KẾT QUẢ BACKTEST =====")
    print(f"Số dư cuối: {balance:.2f} USDT")
    print(f"Tổng giao dịch: {win_count + loss_count}")
    print(f"✅ Lệnh thắng: {win_count}")
    print(f"⛔ Lệnh thua: {loss_count}")
    win_rate = (win_count / (win_count + loss_count)) * 100 if (win_count + loss_count) > 0 else 0
    print(f"🎯 Tỷ lệ thắng: {win_rate:.2f}%")

    # XUẤT LOG RA FILE CSV
    import os
    os.makedirs("backtest_results", exist_ok=True)
    log_df = pd.DataFrame(trade_log)
    log_df.to_csv("backtest_results/trade_log.csv", index=False, encoding="utf-8-sig")
    print("🧾 Đã lưu log giao dịch tại: backtest_results/trade_log.csv")

    # IN LOG RA MÀN HÌNH
    print("\n🧾 Chi tiết giao dịch:")
    for log in trade_log:
        print(log)


# ====== CHẠY BACKTEST ======
if __name__ == "__main__":
    csv_file = "data/backtest/btc_60m.csv"
    model_path = "models/real/ai1h_model.keras"

    df = load_csv_and_add_indicators(csv_file)
    scaler = MinMaxScaler()
    scaler.fit(df[[
        "close", "sma", "ema", "macd", "macd_signal", "macd_diff",
        "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr"
    ]])
    model = load_model(model_path)
    backtest_ai_strategy(model, scaler, df)