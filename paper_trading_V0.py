# ============================
# Paper Trading BTCUSDT - 15 phút, Full Auto - VPS Version (Không reset balance)
# ============================

import os
import datetime
import numpy as np
import pandas as pd
import ta
import time
import requests
import joblib  # Thêm thư viện để load scaler
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from pytz import timezone

# ============================
# Cấu hình Telegram
# ============================
TELEGRAM_TOKEN = 'TELEGRAM_TOKEN'
TELEGRAM_CHAT_ID = 'TELEGRAM_CHAT_ID'

# ============================
# Thư mục lưu trữ log và model
# ============================
os.makedirs("logs", exist_ok=True)
model_path = "models_backup/model.keras"  # Model tốt nhất
log_file = "logs/paper_log.csv"

# ============================
# Load mô hình và scaler
# ============================
model = load_model(model_path)
scaler = MinMaxScaler()  # Sẽ fit sau khi lấy dữ liệu mới

# ============================
# Kết nối Bybit API Public để lấy giá
# ============================
def get_latest_candle():
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": "BTCUSDT",
        "interval": "15",
        "limit": 500
    }
    response = requests.get(url, params=params, timeout=10)
    data = response.json()["result"]["list"]
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df
def get_current_price():
    url = "https://api.bybit.com/v5/market/tickers"
    params = {
        "category": "spot",
        "symbol": "BTCUSDT"
    }
    response = requests.get(url, params=params, timeout=10)
    data = response.json()["result"]["list"][0]
    return float(data["lastPrice"])

# ============================
# Thêm indicators vào dataframe
# ============================
def add_indicators(df):
    df["sma"] = SMAIndicator(df["close"], window=14).sma_indicator()
    df["ema"] = EMAIndicator(df["close"], window=14).ema_indicator()
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    bb = BollingerBands(df["close"], window=20)
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    df.dropna(inplace=True)
    return df

# ============================
# Khởi tạo thông số giao dịch ảo
# ============================
initial_balance = 5000
balance = initial_balance
position = 0
buy_price = 0
take_profit = 0
stop_loss = 0

# Nếu log cũ đã có balance thì load balance cuối cùng
if os.path.exists(log_file):
    try:
        df_log = pd.read_csv(log_file)
        if not df_log.empty:
            balance = df_log["balance"].iloc[-1]
            print(f"🔄 Khôi phục balance từ log cũ: {balance:.2f} USDT")
    except:
        print("⚠️ Không thể đọc log cũ, dùng balance mặc định.")

# ============================
# Hàm gửi Telegram
# ============================
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=data, timeout=10)
    except:
        print("⚠️ Lỗi gửi Telegram!")

# Gửi file log về Telegram
def send_log_file():
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    with open(log_file, "rb") as f:
        files = {"document": f}
        data = {"chat_id": TELEGRAM_CHAT_ID}
        try:
            requests.post(url, files=files, data=data, timeout=30)
        except:
            print("⚠️ Lỗi gửi file log Telegram!")

# ============================
# Hàm lưu log
# ============================
def save_log(action, price, balance):
    timestamp = datetime.datetime.now(timezone('UTC')).strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"{timestamp},{action},{price},{balance}\n")

# ============================
# Giả lập Paper Trading
# ============================
def paper_trading():
    global balance, position, buy_price, take_profit, stop_loss

    df = get_latest_candle()
    df = add_indicators(df)

    feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]
    # scaler.fit(df[feature_cols])
    # Load scaler đã fit từ lúc train model
    print(f"[DEBUG] Feature DataFrame shape: {df[feature_cols].shape}")
    print(f"[DEBUG] Feature Columns: {df[feature_cols].columns.tolist()}")
    print(f"[DEBUG] Last sequence data: {df[feature_cols].iloc[-5:]}")

    scaler = joblib.load("models_backup/scaler.pkl")
    print(f"🔍 Scaler loaded: data min {scaler.data_min_} / data max {scaler.data_max_}")
    # if df.shape[0] < 100:
    #     print(f"[WARNING] Data quá ít: {df.shape[0]} rows, cần ít nhất 100.")
    # else:
    #     last_sequence = scaler.transform(df[feature_cols].iloc[-100:]).reshape(1, 100, len(feature_cols))
    #
    # predicted_close = model.predict(last_sequence, verbose=0)[0][0]
    # print(f"[DEBUG] Giá dự đoán: {predicted_close}")
    # dummy = np.zeros((1, len(feature_cols)))
    # dummy[0][0] = predicted_close
    # predicted_close = scaler.inverse_transform(dummy)[0][0]
    # Chuẩn bị last_sequence
    last_sequence = scaler.transform(df[feature_cols].iloc[-100:])
    last_sequence = last_sequence.reshape(1, 100, len(feature_cols))

    # Dự đoán
    predicted_scaled = model.predict(last_sequence, verbose=0)[0][0]

    # Để inverse transform đúng, bạn cần copy hàng cuối cùng từ last_sequence flatten ra
    dummy = last_sequence.copy().reshape(100, len(feature_cols))[-1]  # Lấy step cuối

    # Thay giá trị cột close bằng giá trị dự đoán
    dummy[0] = predicted_scaled  # Cột close là index 0

    # Inverse transform
    predicted_close = scaler.inverse_transform([dummy])[0][0]
    print(f"[DEBUG] Giá dự đoán sau inverse transform: {predicted_close}")

    # current_price = df["close"].iloc[-1]
    current_price = get_current_price()
    print(f"[DEBUG] Real-time current price: {current_price}")
    atr = df["atr"].iloc[-1]

    signal_buy = predicted_close > current_price * 1.001
    signal_sell = position == 1 and (current_price >= take_profit or current_price <= stop_loss)
    print(f"[DEBUG] Current price: {current_price}, Signal buy: {signal_buy}, Signal sell: {signal_sell}")
    print(f"[DEBUG] Position: {position}, Balance: {balance}")

    if position == 0 and signal_buy:
        position = 1
        buy_price = current_price
        take_profit = buy_price * 1.004
        stop_loss = buy_price * 0.996
        save_log("BUY", buy_price, balance)
        send_telegram(f"[Paper Trading] BUY {buy_price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")

    elif signal_sell:
        if current_price >= take_profit:
            balance *= 1 + (atr * 2 / buy_price)
            save_log("TP", current_price, balance)
            send_telegram(f"[Paper Trading] TP {current_price:.2f} | Balance: {balance:.2f}")
        elif current_price <= stop_loss:
            balance *= 1 - (atr * 1.5 / buy_price)
            save_log("SL", current_price, balance)
            send_telegram(f"[Paper Trading] SL {current_price:.2f} | Balance: {balance:.2f}")
        position = 0

# ============================
# Vòng lặp chính Paper Trading
# ============================

print("✅ Paper Trading BTCUSDT - Khung 15 phút đã bắt đầu!")
send_telegram("✅ Paper Trading BTCUSDT - Khung 15 phút đã bắt đầu!")

while True:
    try:
        now_utc = datetime.datetime.now(timezone('UTC'))
        paper_trading()

        # Nếu đến cuối ngày UTC thì gửi báo cáo + file log
        if now_utc.hour == 0 and now_utc.minute == 0:
            send_log_file()
            send_telegram(f"[Daily Report] Balance cuối ngày: {balance:.2f} USDT")
            send_telegram("[Daily] Giữ nguyên balance, tiếp tục giao dịch ngày mới!")

        print(f"✅ Đã xử lý lúc: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        time.sleep(900)  # 15 phút

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        send_telegram(f"❌ Lỗi Paper Trading: {e}")
        time.sleep(60)
