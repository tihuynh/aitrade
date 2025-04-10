# ============================
# Live Trading BTCUSDT - 15 phút, Full Auto - VPS Version (Final - Quantity chuẩn Bybit)
# ============================

import os
import datetime
import numpy as np
import pandas as pd
import ta
import time
import requests
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from pytz import timezone
from pybit.unified_trading import HTTP

# ============================
# Cấu hình API và Telegram
# ============================
API_KEY = "Hm5gG0HKbm5MDo5bpo"
API_SECRET = "D6iP8YwCisA8pUylvh6916rnvWxoyKQnq1jp"
TELEGRAM_TOKEN = '7621293655:AAHaLf_tMtt-vxpb1Qt0K6QEOGmfhmhy0lY'
TELEGRAM_CHAT_ID = '1989267515'

session = HTTP(api_key=API_KEY, api_secret=API_SECRET)

# ============================
# Thư mục lưu trữ log và model
# ============================
os.makedirs("logs", exist_ok=True)
model_path = "models_backup/model.keras"
log_file = "logs/live_log.csv"

# ============================
# Load mô hình và scaler
# ============================
model = load_model(model_path)
scaler = joblib.load("models_backup/scaler.pkl")

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

# ============================
# Gửi log file về Telegram
# ============================
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
# Lấy dữ liệu nến và giá hiện tại
# ============================
def get_latest_candle():
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "spot", "symbol": "BTCUSDT", "interval": "15", "limit": 500}
    response = requests.get(url, params=params, timeout=10)
    data = response.json()["result"]["list"]
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df

def get_current_price():
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": "spot", "symbol": "BTCUSDT"}
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
# Lưu log giao dịch
# ============================
def save_log(action, price, quantity):
    timestamp = datetime.datetime.now(timezone('UTC')).strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"{timestamp},{action},{price},{quantity}\n")

# ============================
# Thực hiện lệnh mua/bán thật
# ============================
def place_order(side, order_value, last_price):
    try:
        if side == "Buy":
            quantity = str(round(order_value, 6))  # Số USDT mua
        else:
            quantity = str(round(order_value / last_price, 6))  # Số BTC bán

        session.place_order(
            category="spot",
            symbol="BTCUSDT",
            side=side,
            orderType="Market",
            qty=quantity,
        )
        send_telegram(f"🚀 Lệnh {side} thành công với số lượng: {quantity}")
    except Exception as e:
        send_telegram(f"❌ Lỗi khi đặt lệnh {side}: {e}")

# ============================
# Live Trading chính
# ============================

position = 0
buy_price = 0
order_value = 0

print("✅ Live Trading BTCUSDT - Khung 15 phút đã bắt đầu!")
send_telegram("✅ Live Trading BTCUSDT - Khung 15 phút đã bắt đầu!")

while True:
    try:
        df = get_latest_candle()
        df = add_indicators(df)

        feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]
        last_sequence = scaler.transform(df[feature_cols].iloc[-100:]).reshape(1, 100, len(feature_cols))
        predicted_scaled = model.predict(last_sequence, verbose=0)[0][0]

        dummy = last_sequence.copy().reshape(100, len(feature_cols))[-1]
        dummy[0] = predicted_scaled
        predicted_close = scaler.inverse_transform([dummy])[0][0]

        current_price = get_current_price()
        atr = df["atr"].iloc[-1]

        signal_buy = predicted_close > current_price * 1.001
        signal_sell = position == 1 and (current_price >= buy_price * 1.004 or current_price <= buy_price * 0.996)

        print(f"[DEBUG] Current: {current_price}, Predicted: {predicted_close}, Buy signal: {signal_buy}, Sell signal: {signal_sell}")

        if position == 0 and signal_buy:
            account_info = session.get_wallet_balance(accountType="SPOT")
            order_value = float(account_info["result"]["list"][0]["coin"][0]["availableToWithdraw"])
            place_order("Buy", order_value, current_price)
            position = 1
            buy_price = current_price
            save_log("BUY", buy_price, order_value)

        elif signal_sell and position == 1:
            place_order("Sell", order_value, current_price)
            position = 0
            save_log("SELL", current_price, order_value)

        now_utc = datetime.datetime.now(timezone('UTC'))
        if now_utc.hour == 0 and now_utc.minute == 0:
            send_log_file()
            send_telegram("[Daily] Đã gửi file log live trading!")

        time.sleep(900)

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        send_telegram(f"❌ Lỗi live trading: {e}")
        time.sleep(60)