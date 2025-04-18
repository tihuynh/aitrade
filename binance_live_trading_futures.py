# ============================
# 🚀 Live Trading Futures Binance
# ============================
import time
import requests
import hmac
import hashlib
import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# ============================
# 🔧 Load biến môi trường
# ============================
load_dotenv()
API_KEY = os.getenv("API_KEY_BINANCE")
API_SECRET = os.getenv("API_SECRET_BINANCE")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN_FUTURES")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID_FUTURES")

# ============================
# 🛠 Hàm gửi Telegram
# ============================
def send_tele(text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        print("Lỗi gửi Telegram:", e)

# ============================
# 🧠 Load model và scaler
# ============================
model = load_model("models/ai_futures_model.keras")
scaler = joblib.load("models/backup/scaler.pkl")
lookback = 100
symbol = "BTCUSDT"
leverage = 2
position = 0
entry_price = 0

# ============================
# 🧮 Hàm tạo chữ ký HMAC
# ============================
def create_signature(query_string):
    return hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()

# ============================
# 📊 Lấy dữ liệu nến mới nhất
# ============================
def get_klines():
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=150"
    r = requests.get(url)
    df = pd.DataFrame(r.json(), columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote", "count", "taker_base", "taker_quote", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.astype({"open": float, "high": float, "low": float, "close": float})
    return df

# ============================
# 🧪 Tính chỉ báo và tạo feature
# ============================
def prepare_features(df):
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands, AverageTrueRange

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
# 🧠 Dự đoán AI và logic vào lệnh
# ============================
def make_decision(df):
    global position, entry_price
    feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]
    latest = df[-lookback:]
    X = scaler.transform(latest[feature_cols])
    X = np.expand_dims(X, axis=0)
    predicted_price = model.predict(X, verbose=0)[0][0]
    current_price = df["close"].iloc[-1]
    atr = df["atr"].iloc[-1]
    macd_bullish = df["macd"].iloc[-1] - df["macd_signal"].iloc[-1] > -15
    rsi_ok = df["rsi"].iloc[-1] > 40
    near_bottom = current_price <= df["close"].iloc[-20:].min() * 1.05
    adx_ok = df["adx"].iloc[-1] > 20

    if position == 0:
        if predicted_price > current_price * 1.001 and macd_bullish and rsi_ok and near_bottom and adx_ok:
            qty = get_quantity()
            order = place_order("BUY", qty)
            if order:
                position = 1
                entry_price = current_price
                send_tele(f"🔰 Mở LONG tại {current_price:.2f}")
        elif predicted_price < current_price * 0.999 and not macd_bullish and not rsi_ok and adx_ok:
            qty = get_quantity()
            order = place_order("SELL", qty)
            if order:
                position = -1
                entry_price = current_price
                send_tele(f"🔻 Mở SHORT tại {current_price:.2f}")

    elif position == 1:
        if current_price >= entry_price * 1.004 or current_price <= entry_price * 0.996:
            qty = get_quantity()
            order = place_order("SELL", qty)
            if order:
                send_tele(f"✅ Đóng LONG tại {current_price:.2f}")
                position = 0

    elif position == -1:
        if current_price <= entry_price * 0.996 or current_price >= entry_price * 1.004:
            qty = get_quantity()
            order = place_order("BUY", qty)
            if order:
                send_tele(f"✅ Đóng SHORT tại {current_price:.2f}")
                position = 0

# ============================
# 📦 Tính khối lượng lệnh từ số dư USDT
# ============================
def get_quantity():
    url = f"https://fapi.binance.com/fapi/v2/balance"
    timestamp = int(time.time() * 1000)
    query = f"timestamp={timestamp}"
    signature = create_signature(query)
    headers = {"X-MBX-APIKEY": API_KEY}
    r = requests.get(f"{url}?{query}&signature={signature}", headers=headers)
    data = r.json()
    usdt_balance = float([x for x in data if x['asset'] == 'USDT'][0]['availableBalance'])
    return round(usdt_balance * leverage / df["close"].iloc[-1], 3)

# ============================
# 📤 Gửi lệnh Futures
# ============================
def place_order(side, quantity):
    try:
        url = "https://fapi.binance.com/fapi/v1/order"
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
            "timestamp": timestamp
        }
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = create_signature(query_string)
        query_string += f"&signature={signature}"
        headers = {"X-MBX-APIKEY": API_KEY}
        response = requests.post(url, headers=headers, data=query_string)
        if response.status_code == 200:
            return response.json()
        else:
            send_tele(f"❌ Lỗi đặt lệnh: {response.text}")
            return None
    except Exception as e:
        send_tele(f"❌ Lỗi kết nối khi gửi lệnh: {e}")
        return None

# ============================
# ♻️ Vòng lặp chạy bot
# ============================
while True:
    try:
        df = get_klines()
        df = prepare_features(df)
        make_decision(df)
    except Exception as e:
        send_tele(f"❌ Lỗi bot: {e}")
    time.sleep(900)  # 15 phút