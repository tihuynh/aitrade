import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import ta
from ta.trend import ADXIndicator


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

    # ADX
    adx = ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx.adx()

    # DMI
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()

    # Candle body và wick (để phát hiện nến rút chân)
    df["candle_body"] = abs(df["close"] - df["open"])
    df["lower_wick"] = np.where(df["close"] > df["open"], df["open"] - df["low"], df["close"] - df["low"])
    df["lower_wick"] = abs(df["lower_wick"])
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
        last_sequence = scaler.transform(window[[  # Thêm "adx"
            "close", "sma", "ema", "macd", "macd_signal", "macd_diff",
            "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx",
            "plus_di", "minus_di", "candle_body", "lower_wick"
        ]])
        last_sequence = last_sequence.reshape(1, LOOKBACK, 16)  # ✅ thêm dòng này
        dummy = np.zeros((1, 16))  # 16 features

        # Dự đoán (scaled)
        scaled_pred = model.predict(last_sequence, verbose=0)[0][0]

        # Tạo dummy row để inverse transform
        dummy = np.zeros((1, 16))  # 16 là số feature
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

        ai_confidence = prediction > current_price * 1.001
        macd_bullish = (macd_val - macd_sig) > -15
        rsi_ok = rsi_val > 40
        volume_ok = data.iloc[i]["volume"] > data.iloc[i - 1]["volume"] * 1.2
        price_near_bottom = current_price <= data.iloc[i - 20:i]["close"].min() * 1.05
        adx_ok = data.iloc[i]["adx"] > 20

        sma_val = data.iloc[i]["sma"]
        ema_val = data.iloc[i]["ema"]
        plus_di = data.iloc[i]["plus_di"]
        minus_di = data.iloc[i]["minus_di"]

        # SMA cắt lên EMA (momentum tăng)
        sma_cross_ema = sma_val > ema_val and data.iloc[i - 1]["sma"] <= data.iloc[i - 1]["ema"]

        # ADX trend mạnh & bullish
        trend_strong = data.iloc[i]["adx"] > 25 and plus_di > minus_di

        # RSI breakout từ vùng quá bán
        rsi_breakout = data.iloc[i - 1]["rsi"] < 40 and data.iloc[i]["rsi"] >= 40

        # Chặn nếu là nến rút chân dài → có thể đảo chiều giảm
        lower_wick = data.iloc[i]["lower_wick"]
        body = data.iloc[i]["candle_body"]
        is_reversal_candle = lower_wick > body * 2


        buy_condition = (
                ai_confidence and macd_bullish and rsi_ok
                and volume_ok and price_near_bottom and adx_ok
                and sma_cross_ema and trend_strong and rsi_breakout
                and not is_reversal_candle  # Loại bỏ nến rút chân dài
        )

        # print(f"[DEBUG] AI ok: {ai_confidence}, MACD: {macd_bullish}, RSI: {rsi_ok}, EMA: {price_above_ema}")

        if position == 0 and buy_condition:
            diff = prediction - current_price
            print(
                f"[CHECK] {timestamp} | Dự đoán: {prediction:.2f} | Giá hiện tại: {current_price:.2f} | Chênh lệch: {diff:.2f}")
            print("✅ Thỏa điều kiện mua thông minh")

            position = 1
            buy_price = current_price
            take_profit = round(buy_price * 1.004, 2)  # TP chỉ 0.4%
            stop_loss = round(buy_price * 0.996, 2)  # SL -0.4%
            trade_log.append(f"🟢 Mua tại {buy_price:.2f} ({timestamp}) | TP: {take_profit:.2f}, SL: {stop_loss:.2f}")

        elif position == 1:
            if current_price >= take_profit:
                # TP hit
                balance *= 1 + (atr * 2 / buy_price)  # giảm từ 3 → 2
                trade_log.append(f"✅ TP tại {current_price:.2f} ({timestamp}) | Số dư: {balance:.2f}")
                win_count += 1
                position = 0
            elif current_price <= stop_loss:
                # SL hit
                balance *= 1 - (atr * 1.5 / buy_price)  # giảm từ 2 → 1.5
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
    csv_file = "data/backtest/btc_15m.csv"
    model_path = "models/real/ai15m_model.keras"

    df = load_csv_and_add_indicators(csv_file)
    scaler = MinMaxScaler()
    scaler.fit(df[[
        "close", "sma", "ema", "macd", "macd_signal", "macd_diff",
        "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx",
        "plus_di", "minus_di", "candle_body", "lower_wick"
    ]])

    model = load_model(model_path)
    backtest_ai_strategy(model, scaler, df)