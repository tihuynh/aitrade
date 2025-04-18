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
from decimal import Decimal, ROUND_DOWN

# ============================
# Cấu hình API và Telegram
# ============================
load_dotenv()
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
state_path = "logs/position_state.json"

# ============================
# Load mô hình và scaler
# ============================
model = load_model(model_path)
scaler = joblib.load("models_backup/scaler.pkl")

# ============================
# Trading State - Biến toàn cục
# ============================
position = 0
buy_price = 0
take_profit = 0
stop_loss = 0



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
    try:
        timestamp = datetime.datetime.now(timezone('UTC')).strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{timestamp},{action},{price},{balance}\n")
    except Exception as e:
        error_msg = f"❌ Lỗi trong hàm save_log(): {e}"
        print(error_msg)
        send_telegram(error_msg)

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
        error_msg = f"⚠️ Lỗi lấy balance USDT Binance: {e}"
        print(error_msg)
        send_telegram(error_msg)
    return 0.0

def get_balance_btc():
    try:
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

def round_step_size(value, step_size):
    try:
        precision = int(round(-np.log10(step_size)))
        return float(Decimal(value).quantize(Decimal(str(step_size)), rounding=ROUND_DOWN))
    except Exception as e:
        error_msg = f"❌ Lỗi trong hàm round_step_size(): {e}"
        print(error_msg)
        send_telegram(error_msg)
def get_lot_step_size(symbol="BTCUSDT"):
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url, timeout=10)
        data = response.json()

        for s in data["symbols"]:
            if s["symbol"] == symbol:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        return float(f["stepSize"])
        return 0.000001  # fallback
    except Exception as e:
        error_msg = f"❌ Lỗi trong hàm get_lot_step_size(): {e}"
        print(error_msg)
        send_telegram(error_msg)
def place_order(side, quantity):
    try:
        url = "https://api.binance.com/api/v3/order"
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": "BTCUSDT",
            "side": "BUY" if side.lower() == "buy" else "SELL",
            "type": "MARKET",
            "timestamp": timestamp
        }
        print(f"DEBUG - Place_order - quantity:{quantity}")
        if side.lower() == "buy":
            params["quoteOrderQty"] = quantity
        else:
            params["quantity"] = quantity

        query_string = urlencode(params)
        signature = hmac.new(API_SECRET.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        params['signature'] = signature
        headers = {'X-MBX-APIKEY': API_KEY}
        response = requests.post(url, params=params, headers=headers, timeout=10)
        data = response.json()
        print(f"DEBUG - Place_order - data: {data}")
        if 'code' in data and data['code'] != 0:
            send_telegram(f"❌ Lỗi đặt lệnh {side.upper()}: {data}")
            return None  # Ngăn không cho tiếp tục xử lý như thành công
        return data
    except Exception as e:
        error_msg = f"❌ Lỗi trong hàm place_order(): {e}"
        print(error_msg)
        send_telegram(error_msg)
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
    try:
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
    except Exception as e:
        error_msg = f"❌ Lỗi trong hàm add_indicators(): {e}"
        print(error_msg)
        send_telegram(error_msg)
def initialize_trading_state():
    global position, buy_price, take_profit, stop_loss

    # Load trạng thái từ file nếu có
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        position = state.get("position", 0)
        buy_price = state.get("buy_price", 0)
        take_profit = state.get("take_profit", 0)
        stop_loss = state.get("stop_loss", 0)
        send_telegram(f"✅ Khôi phục trạng thái: BUY: {buy_price:.2f} TP: {take_profit:.2f} SL: {stop_loss:.2f}")
    else:
        send_telegram("⚙️ Bot khởi động: không có trạng thái cũ, bắt đầu từ 0")

    # Khởi tạo file log nếu chưa có
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("timestamp,action,price,balance\n")

def live_trading():
    global position, buy_price, take_profit, stop_loss
    try:
        df = get_latest_candle()
        df = add_indicators(df)
        feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]

        last_sequence = scaler.transform(df[feature_cols].iloc[-101:-1])  # nến từ -101 đến -2
        last_sequence = last_sequence.reshape(1, 100, len(feature_cols))
        predicted_scaled = model.predict(last_sequence, verbose=0)[0][0]

        dummy = last_sequence.reshape(100, len(feature_cols))[-1]
        dummy[0] = predicted_scaled
        predicted_close = scaler.inverse_transform([dummy])[0][0]

        current_price = get_current_price()

        # macd_bullish = df["macd"].iloc[-2] - df["macd_signal"].iloc[-2] > -15
        # Mới (nới lỏng):
        macd_bullish = df["macd"].iloc[-2] > df["macd_signal"].iloc[-2] * 0.98
        rsi_ok = df["rsi"].iloc[-2] > 40
        recent_rolling_min = df["close"].iloc[-21:-1].min()
        # price_near_bottom = current_price <= recent_rolling_min * 1.05
        # Mới (nới rộng):
        price_near_bottom = current_price <= recent_rolling_min * 1.08
        # adx_ok = df["adx"].iloc[-2] > 20
        # Mới (nới lỏng):
        adx_ok = df["adx"].iloc[-2] > 15
        ai_confidence = predicted_close > current_price * 1.001
        print(f"DEBUG - predicted_close = {predicted_close}")
        print(f"DEBUG - current_price = {current_price}")
        print(f"DEBUG - ai_confidence ={ai_confidence}")
        print(f"DEBUG - macd_bullish ={macd_bullish}")
        print(f"DEBUG - rsi_ok ={rsi_ok}")
        print(f"DEBUG - price_near_bottom ={price_near_bottom}")
        print(f"DEBUG - adx_ok ={adx_ok}")
        print(f"DEBUG - current_price ={current_price}")
        print(f"DEBUG - take_profit ={take_profit}")
        print(f"DEBUG - stop_loss ={stop_loss}")
        signal_buy = ai_confidence and macd_bullish and rsi_ok and price_near_bottom and adx_ok
        signal_sell = position == 1 and (current_price >= take_profit or current_price <= stop_loss)
        print(f"[DEBUG] Signal buy: {signal_buy}, Signal sell: {signal_sell}")
        print(f"[DEBUG] Position: {position}")
        if position == 0 and signal_buy:
            print("===========BUY BTC==========")
            usdt_balance = get_balance_usdt()
            print(f"[DEBUG] usdt_balance: {usdt_balance}")
            if usdt_balance > 5:
                adjusted_qty = round(usdt_balance * 0.998, 6)  # Trừ buffer 0.2%
                print(f"[DEBUG] usdt adjusted: {adjusted_qty}")
                place_order("Buy", adjusted_qty)
                position = 1
                buy_price = current_price
                entry_atr = df["atr"].iloc[-2]
                take_profit = buy_price + entry_atr * 2
                stop_loss = buy_price - entry_atr * 1.5
                with open(state_path, "w") as f:
                    json.dump({"position": position, "buy_price": buy_price, "take_profit": take_profit, "stop_loss": stop_loss}, f)
                save_log("BUY", buy_price, usdt_balance)
                send_telegram(f"[Live Trading] BUY {buy_price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")

        elif signal_sell:
            btc_balance = get_balance_btc()
            print(f"[DEBUG] btc_balance: {btc_balance}")
            if btc_balance > 0.00001:
                print("===========SELL BTC==========")
                step_size = get_lot_step_size()
                qty = round_step_size(btc_balance, step_size)
                print(f"[DEBUG] btc làm tròn: {qty}")
                result = place_order("Sell", qty)
                if result is None:
                    print("❌ SELL thất bại, giữ nguyên trạng thái position = 1")
                    send_telegram("❌ SELL thất bại, giữ nguyên BTC")
                else:
                    result_text = "TP" if current_price >= take_profit else "SL"
                    save_log(result_text, current_price, btc_balance * current_price)
                    send_telegram(
                        f"[Live Trading] {result_text} {current_price:.2f} | Balance: {btc_balance * current_price:.2f}")
                    position = 0
                    if os.path.exists(state_path):
                        os.remove(state_path)

    except Exception as e:
        error_msg = f"❌ Lỗi trong hàm live_trading(): {e}"
        print(error_msg)
        send_telegram(error_msg)

# ============================
# Main Loop
# ============================
print("✅ Live Trading BTCUSDT - Khung 15 phút đã bắt đầu!")
send_telegram("✅ Live Trading BTCUSDT Code mới- Khung 15 phút đã bắt đầu!")
# ✅ Gọi hàm khởi tạo trạng thái
initialize_trading_state()
while True:
    try:
        now_utc = datetime.datetime.now(timezone('UTC'))
        live_trading()
        if now_utc.hour == 0 and now_utc.minute == 0:
            send_log_file()
            send_telegram("[Daily Report] Live trading vẫn đang hoạt động!")
        print(f"✅ Đã xử lý lúc: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("[DEBUG] Chờ 15 phút trước khi chạy tiếp")
        time.sleep(900)

    except Exception as e:
        send_telegram(f"❌ Lỗi Live Trading: {e}")
        time.sleep(60)
