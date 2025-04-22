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
API_KEY = os.getenv("BINANCE_FUTURES_API_KEY")
API_SECRET = os.getenv("BINANCE_FUTURES_API_SECRET")
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
model = load_model("models_backup/ai_futures_model.keras")
scaler = joblib.load("models_backup/scaler_futures.pkl")
lookback = 100
symbol = "BTCUSDT"
leverage = 2
position = 0
entry_price = 0
DEBUG_MODE = True
LOG_FILE = "logs/debug_log.txt"
POSITION_STATE_FILE = "logs/binance_futures_position_state.json"
os.makedirs("logs", exist_ok=True)
def save_position_state(qty):
    state = {
        "position": position,
        "entry_price": entry_price,
        "qty": qty   # lưu chính xác số BTC đã mua
    }
    with open(POSITION_STATE_FILE, "w") as f:
        json.dump(state, f)


def load_position_state():
    if os.path.exists(POSITION_STATE_FILE):
        with open(POSITION_STATE_FILE, "r") as f:
            state = json.load(f)
            return state.get("position", 0), state.get("entry_price", 0), state.get("qty", 0)
    return 0, 0, 0

# ============================
# 🧮 Hàm tạo chữ ký HMAC
# ============================
def create_signature(query_string):
    return hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()

# ============================
# 📊 Lấy dữ liệu nến mới nhất
# ============================
def get_klines():
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=150"
        r = requests.get(url)
        df = pd.DataFrame(r.json(), columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote", "count", "taker_base", "taker_quote", "ignore"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.astype({"open": float, "high": float, "low": float, "close": float})
        return df
    except Exception as e:
        error_msg = f"❌ Lỗi trong get_klines: {e}"
        send_tele(error_msg)
        print(error_msg)
        with open(LOG_FILE, "a") as f:
            f.write(error_msg + "\n")

# ============================
# 🧪 Tính chỉ báo và tạo feature
# ============================
def prepare_features(df):
    try:
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
    except Exception as e:
        error_msg = f"❌ Lỗi trong prepare_features: {e}"
        send_tele(error_msg)
        print(error_msg)
        with open(LOG_FILE, "a") as f:
            f.write(error_msg + "\n")

# ============================
# 📌 Bổ sung sự kiện set đòn bẩy và gởi trước khi mở lệnh
# ============================
def set_leverage(symbol="BTCUSDT", leverage=2):
    url = "https://fapi.binance.com/fapi/v1/leverage"
    timestamp = int(time.time() * 1000)
    params = {
        "symbol": symbol,
        "leverage": leverage,
        "timestamp": timestamp
    }
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = create_signature(query_string)
    headers = {"X-MBX-APIKEY": API_KEY}
    response = requests.post(f"{url}?{query_string}&signature={signature}", headers=headers)
    if response.status_code != 200:
        send_tele(f"⚠️ Lỗi set leverage: {response.text}")
    return response.json()

# ============================
# 🧠 Dự đoán AI và logic vào lệnh
# ============================
def make_decision(df):
    try:
        position, entry_price, qty = load_position_state()
        feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]
        latest = df[-lookback:]
        X = scaler.transform(latest[feature_cols])
        X = np.expand_dims(X, axis=0)
        predicted_price = model.predict(X, verbose=0)[0][0]
        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, 0] = predicted_price
        predicted_price_real = scaler.inverse_transform(dummy)[0, 0]

        current_price = df["close"].iloc[-2]
        atr = df["atr"].iloc[-2]
        macd_bullish = df["macd"].iloc[-2] - df["macd_signal"].iloc[-2] > -15
        rsi_ok = df["rsi"].iloc[-2] > 40
        near_bottom = current_price <= df["close"].iloc[-20:-2].min() * 1.05
        adx_ok = df["adx"].iloc[-2] > 20

        def get_balance():
            url = f"https://fapi.binance.com/fapi/v2/balance"
            timestamp = int(time.time() * 1000)
            query = f"timestamp={timestamp}"
            signature = create_signature(query)
            headers = {"X-MBX-APIKEY": API_KEY}
            r = requests.get(f"{url}?{query}&signature={signature}", headers=headers)
            data = r.json()
            return float([x for x in data if x['asset'] == 'USDT'][0]['availableBalance'])
        if DEBUG_MODE:
            msg = (
                f"[{datetime.utcnow().isoformat()} UTC]\n"
                f"[DEBUG AI]\n"
                f"Giá: {current_price:.2f}\n"
                f"Dự đoán: {predicted_price_real:.2f}\n"
                f"MACD: {macd_bullish}, RSI: {rsi_ok}, ADX: {adx_ok}, NearBottom: {near_bottom}\n"
                f"Position: {position}"
            )
            send_tele(msg)
            print(msg)
            with open(LOG_FILE, "a") as f:
                f.write(msg + "\n")
        if position == 0:
            if predicted_price_real > current_price * 1.001 and macd_bullish and rsi_ok and near_bottom and adx_ok:
                print("Thỏa điều kiện Long, lệnh place order sẽ được thực hiện")
                send_tele("Thỏa điều kiện Long, lệnh place order sẽ được thực hiện")
                set_leverage(symbol, leverage)  # ✅ Gọi API set đòn bẩy trước khi mở lệnh
                balance_before = get_balance()
                qty = get_quantity()
                order = place_order("BUY", qty, reduce_only=False) # mở LONG
                if order:
                    position = 1
                    entry_price = current_price
                    save_position_state(qty)
                    send_tele(f"🔰 Mở LONG tại {current_price:.2f}\n💵 Balance trước lệnh: {balance_before:.2f} USDT")
            elif predicted_price_real < current_price * 0.999 and not macd_bullish and not rsi_ok and adx_ok:
                print("Thỏa điều kiện SHORT, lệnh place order sẽ được thực hiện")
                send_tele("Thỏa điều kiện SHORT, lệnh place order sẽ được thực hiện")
                set_leverage(symbol, leverage)  # ✅ Gọi API set đòn bẩy trước khi mở lệnh
                balance_before = get_balance()
                qty = get_quantity()
                order = place_order("SELL", qty, reduce_only=False)  # mở SHORT
                if order:
                    position = -1
                    entry_price = current_price
                    save_position_state(qty)
                    send_tele(f"🔻 Mở SHORT tại {current_price:.2f}\n💵 Balance trước lệnh: {balance_before:.2f} USDT")

        elif position == 1:
            if current_price >= entry_price * 1.004 or current_price <= entry_price * 0.996:
                print("Thỏa điều kiện đóng lệnh Long, lệnh place order sẽ được thực hiện")
                send_tele("Thỏa điều kiện đóng lệnh Long, lệnh place order sẽ được thực hiện")
                # qty = get_quantity()
                order = place_order("SELL", qty, reduce_only=True)  # đóng LONG
                if order:
                    position = 0
                    balance_after = get_balance()
                    send_tele(f"✅ Đóng LONG tại {current_price:.2f}\n💰 Balance sau đóng lệnh: {balance_after:.2f} USDT")
                    if os.path.exists(POSITION_STATE_FILE):
                        os.remove(POSITION_STATE_FILE)

        elif position == -1:
            if current_price <= entry_price * 0.996 or current_price >= entry_price * 1.004:
                print("Thỏa điều kiện đóng lệnh SHORT, lệnh place order sẽ được thực hiện")
                send_tele("Thỏa điều kiện đóng lệnh SHORT, lệnh place order sẽ được thực hiện")
                # qty = get_quantity()
                order = place_order("BUY", qty, reduce_only=True)   # đóng SHORT
                if order:
                    position = 0
                    balance_after = get_balance()
                    send_tele(f"✅ Đóng SHORT tại {current_price:.2f}\n💰 Balance sau đóng lệnh: {balance_after:.2f} USDT")
                    if os.path.exists(POSITION_STATE_FILE):
                        os.remove(POSITION_STATE_FILE)
    except Exception as e:
        error_msg = f"❌ Lỗi trong make_decision: {e}"
        send_tele(error_msg)
        print(error_msg)
        with open(LOG_FILE, "a") as f:
            f.write(error_msg + "\n")

# ============================
# 📦 Tính khối lượng lệnh từ số dư USDT
# ============================
def get_quantity():
    try:
        url = f"https://fapi.binance.com/fapi/v2/balance"
        timestamp = int(time.time() * 1000)
        query = f"timestamp={timestamp}"
        signature = create_signature(query)
        headers = {"X-MBX-APIKEY": API_KEY}
        r = requests.get(f"{url}?{query}&signature={signature}", headers=headers)
        data = r.json()
        usdt_balance = float([x for x in data if x['asset'] == 'USDT'][0]['availableBalance'])
        return round(usdt_balance * leverage / df["close"].iloc[-1], 3)
    except Exception as e:
        error_msg = f"❌ Lỗi trong get_quantity: {e}"
        send_tele(error_msg)
        print(error_msg)
        with open(LOG_FILE, "a") as f:
            f.write(error_msg + "\n")

# ============================
# 📤 Gửi lệnh Futures
# ============================
def place_order(side, quantity, reduce_only=True):
    try:
        url = "https://fapi.binance.com/fapi/v1/order"
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
            "reduceOnly": reduce_only,  # ✅ dòng này sẽ tránh lỗi notional < 100
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
        error_msg = f"❌ Lỗi trong place_order: {e}"
        send_tele(error_msg)
        print(error_msg)
        with open(LOG_FILE, "a") as f:
            f.write(error_msg + "\n")

# ============================
# ♻️ Vòng lặp chạy bot
# ============================
while True:
    try:
        df = get_klines()
        df = prepare_features(df)
        make_decision(df)
    except Exception as e:
        error_msg = f"❌ Lỗi bot: {e}"
        send_tele(error_msg)
        print(error_msg)
        with open(LOG_FILE, "a") as f:
            f.write(error_msg + "\n")
    time.sleep(900)  # 15 phút