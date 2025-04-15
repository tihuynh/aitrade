# ============================
# Live Trading BTCUSDT - 15 phút, Full Auto - VPS Version (Không reset balance)
# ============================

import os
import datetime
import numpy as np
import pandas as pd
import ta
import time
import requests
import joblib  # Load scaler
import hmac
import hashlib
import json
from urllib.parse import urlencode
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from pytz import timezone
from dotenv import load_dotenv


# ============================
# Cấu hình API và Telegram
# ============================
load_dotenv()
print(f"[DEBUG] API KEY: {os.getenv('API_KEY_BINANCE')}")
print(f"[DEBUG] TELEGRAM TOKEN: {os.getenv('TELEGRAM_TOKEN')}")
API_KEY = os.getenv("API_KEY_BINANCE")
API_SECRET = os.getenv("API_SECRET_BINANCE")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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
# Gửi file log về Telegram
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
# Hàm lưu log
# ============================
def save_log(action, price, balance):
    timestamp = datetime.datetime.now(timezone('UTC')).strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"{timestamp},{action},{price},{balance}\n")

def get_current_price():
    try:
        url = "https://api.binance.com/api/v3/ticker/price"
        params = {"symbol": "BTCUSDT"}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return float(data["price"])
    except Exception as e:
        print(f"⚠️ Lỗi lấy giá Binance: {e}")
        return None


def get_balance_usdt():
    try:
        print("vao ham get_balance_usdt")
        timestamp = int(time.time() * 1000)
        params = {"timestamp": timestamp}
        query_string = urlencode(params)
        signature = hmac.new(API_SECRET.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        params['signature'] = signature
        headers = {'X-MBX-APIKEY': API_KEY}

        url = "https://api.binance.com/api/v3/account"
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()

        for asset in data.get("balances", []):
            if asset["asset"] == "USDT":
                return float(asset["free"])
    except Exception as e:
        error_msg = f"⚠️ Lỗi lấy balance BTC Binance: {e}"
        print(error_msg)
        send_telegram(error_msg)
    return 0.0



def get_balance_btc():
    try:
        print("vao ham get_balance_btc")
        timestamp = int(time.time() * 1000)
        params = {"timestamp": timestamp}
        query_string = urlencode(params)
        signature = hmac.new(API_SECRET.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        params['signature'] = signature
        headers = {'X-MBX-APIKEY': API_KEY}

        url = "https://api.binance.com/api/v3/account"
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()

        for asset in data.get("balances", []):
            if asset["asset"] == "BTC":
                return float(asset["free"])
    except Exception as e:
        error_msg = f"⚠️ Lỗi lấy balance BTC Binance: {e}"
        print(error_msg)
        send_telegram(error_msg)
    return 0.0



def place_order(side, quantity):
    try:
        print (f"DEBUG - quantity= {quantity}")
        url = "https://api.binance.com/api/v3/order"
        timestamp = int(time.time() * 1000)

        # Khởi tạo params
        params = {
            "symbol": "BTCUSDT",
            "side": "BUY" if side.lower() == "buy" else "SELL",
            "type": "MARKET",
            "timestamp": timestamp
        }
        if side.lower() == "buy":
            params["quoteOrderQty"] = quantity  # số tiền USDT muốn dùng
        else:
            params["quantity"] = quantity       # số BTC muốn bán

        # Tạo query string
        query_string = urlencode(params)

        # Ký signature
        signature = hmac.new(API_SECRET.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        params['signature'] = signature

        # Headers
        headers = {
            'X-MBX-APIKEY': API_KEY
        }

        # Gửi request
        response = requests.post(url, params=params, headers=headers, timeout=10)
        data = response.json()

        print(f"[DEBUG] API response: {data}")

        if 'code' in data and data['code'] != 0:
            send_telegram(f"❌ Lỗi đặt lệnh {side}: {data}")
        return data

    except Exception as e:
        print(f"❌ Exception khi đặt lệnh {side}: {e}")
        send_telegram(f"❌ Exception khi đặt lệnh {side}: {e}")
        return None

# ============================
# Indicator Processing
# ============================
def get_latest_candle():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "15m", "limit": 500}
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


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
# Trading Logic
# ============================
position = 0
buy_price = 0
take_profit = 0
stop_loss = 0

# Khởi tạo file log nếu chưa có
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("timestamp,action,price,balance\n")

def live_trading():
    global balance, position, buy_price, take_profit, stop_loss

    df = get_latest_candle()
    df = add_indicators(df)

    feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl",
                    "atr", "adx"]
    # scaler.fit(df[feature_cols])
    # Load scaler đã fit từ lúc train model
    print(f"[DEBUG] Feature DataFrame shape: {df[feature_cols].shape}")
    print(f"[DEBUG] Feature Columns: {df[feature_cols].columns.tolist()}")
    print(f"[DEBUG] Last sequence data: {df[feature_cols].iloc[-5:]}")

    scaler = joblib.load("models_backup/scaler.pkl")
    print(f"🔍 Scaler loaded: data min {scaler.data_min_} / data max {scaler.data_max_}")
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

    # Logic như backtest
    macd_bullish = df["macd"].iloc[-1] - df["macd_signal"].iloc[-1] > -15
    rsi_ok = df["rsi"].iloc[-1] > 40
    price_near_bottom = current_price <= df["close"].iloc[-20:].rolling(20).min().iloc[-1] * 1.05
    adx_ok = df["adx"].iloc[-1] > 20
    ai_confidence = predicted_close > current_price * 1.001

    signal_buy = ai_confidence and macd_bullish and rsi_ok and price_near_bottom and adx_ok
    signal_sell = position == 1 and (current_price >= take_profit or current_price <= stop_loss)

    print(f"[DEBUG] Signal buy: {signal_buy}, Signal sell: {signal_sell}")
    print(f"[DEBUG] Position: {position}")

    if position == 0 and signal_buy:
        usdt_balance = get_balance_usdt()
        print(f"[DEBUG] usdt_balance: {usdt_balance}")
        if usdt_balance > 5:
            print("===========BUY BTC==========")
            qty = round(usdt_balance, 6)
            place_order("Buy", qty)
            position = 1
            buy_price = current_price
            take_profit = buy_price * 1.004
            stop_loss = buy_price * 0.996
            save_log("BUY", buy_price, usdt_balance)
            send_telegram(f"[Live Trading] BUY {buy_price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")

    elif signal_sell:
        btc_balance = get_balance_btc()
        print(f"[DEBUG] btc_balance: {btc_balance}")
        if btc_balance > 0.00001:
            print("===========SELL BTC==========")
            qty = round(btc_balance, 6)
            place_order("Sell", qty)
            result = "TP" if current_price >= take_profit else "SL"
            save_log(result, current_price, btc_balance * current_price)
            send_telegram(f"[Live Trading] {result} {current_price:.2f} | Balance: {btc_balance * current_price:.2f}")
            position = 0

# ============================
# Main Loop
# ============================
print("✅ Live Trading BTCUSDT - Khung 15 phút đã bắt đầu!")
send_telegram("✅ Live Trading BTCUSDT - Khung 15 phút đã bắt đầu!")

while True:
    try:
        now_utc = datetime.datetime.now(timezone('UTC'))
        live_trading()

        if now_utc.hour == 0 and now_utc.minute == 0:
            send_log_file()
            send_telegram("[Daily Report] Live trading vẫn đang hoạt động!")

        print(f"✅ Đã xử lý lúc: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        time.sleep(900)  # 15 phút

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        send_telegram(f"❌ Lỗi Live Trading: {e}")
        time.sleep(60)
