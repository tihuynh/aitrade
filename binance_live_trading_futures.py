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
DEBUG_MODE = False
LOG_FILE = "logs/debug_log.txt"
POSITION_STATE_FILE = "logs/binance_futures_position_state.json"
os.makedirs("logs", exist_ok=True)
def save_position_state(position, entry_price, qty, predicted_price):
    state = {
        "position": position,
        "entry_price": entry_price,
        "qty": qty,
        "predicted_price": predicted_price
    }
    with open(POSITION_STATE_FILE, "w") as f:
        json.dump(state, f)



def load_position_state():
    try:
        if os.path.exists(POSITION_STATE_FILE):
            with open(POSITION_STATE_FILE, "r") as f:
                state = json.load(f)
                return (
                    state.get("position", 0),
                    state.get("entry_price", 0),
                    state.get("qty", 0),
                    state.get("predicted_price", 0)
                )
        return 0, 0, 0, 0  # ✅ THÊM DÒNG NÀY
    except Exception as e:
        send_tele(f"⚠️ Lỗi đọc file trạng thái: {e}, đang reset về 0")
        return 0, 0, 0, 0



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
        "timestamp": timestamp,
        "recvWindow": 5000  # Cho phép lệch 5 giây
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
# ============================
# 🧠 make_decision(df) - Phiên bản nâng cấp dùng ATR
# ============================
# ============================
# 🧠 make_decision(df) - Phiên bản nâng cấp Chống Quét SL
# ============================
def make_decision(df):
    try:
        global position, entry_price, qty
        position, entry_price, qty, _ = load_position_state()

        feature_cols = [
            "close", "sma", "ema", "macd", "macd_signal", "macd_diff",
            "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"
        ]
        latest = df[-lookback:]
        X = scaler.transform(latest[feature_cols])
        X = np.expand_dims(X, axis=0)
        predicted_price = model.predict(X, verbose=0)[0][0]

        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, 0] = predicted_price
        predicted_price_real = scaler.inverse_transform(dummy)[0, 0]

        current_price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-2]
        ema20 = df["ema"].iloc[-2]
        ema50 = df["sma"].iloc[-2]
        macd_bullish = df["macd"].iloc[-2] - df["macd_signal"].iloc[-2] > -15
        rsi_ok = df["rsi"].iloc[-2] > 40
        near_bottom = current_price <= df["close"].iloc[-20:-2].min() * 1.05
        adx_ok = df["adx"].iloc[-2] > 20

        # ✅ Bỏ qua nếu ATR cao >0.8%
        if atr / current_price > 0.008:
            send_tele(f"⚠️ ATR cao ({atr:.2f}), bỏ qua không vào lệnh.")
            return

        if DEBUG_MODE:
            msg = (
                f"[{datetime.utcnow().isoformat()} UTC]\n"
                f"[DEBUG AI]\n"
                f"Giá: {current_price:.2f}\n"
                f"Dự đoán: {predicted_price_real:.2f}\n"
                f"EMA20: {ema20:.2f}, EMA50: {ema50:.2f}\n"
                f"MACD: {macd_bullish}, RSI: {rsi_ok}, ADX: {adx_ok}, NearBottom: {near_bottom}\n"
                f"ATR: {atr:.2f}\n"
                f"Position: {position}"
            )
            send_tele(msg)

        if position == 0:
            if predicted_price_real > current_price * 1.0005 and rsi_ok and adx_ok and ema20 > ema50:
                set_leverage(symbol, leverage)
                balance_before = get_balance()
                qty = get_quantity(current_price)
                notional = qty * current_price

                if notional < 100:
                    send_tele(f"⚠️ Lệnh bị huỷ vì notional < 100: {notional:.2f}")
                    return

                order = place_order("BUY", qty, reduce_only=False)
                if order is None:
                    send_tele("❌ Lỗi khi mở lệnh BUY: Order trả về None.")
                    return
                position = 1
                entry_price = current_price
                save_position_state(position, entry_price, qty, predicted_price_real)

                tp_price = entry_price + atr * 2.5
                sl_price = entry_price - atr * 2.0

                send_tele(
                    f"🔰 Mở LONG tại {entry_price:.2f}\n"
                    f"📈 TP : {tp_price:.2f}\n"
                    f"🛡️ SL : {sl_price:.2f}\n"
                    f"📏 ATR hiện tại: {atr:.2f}\n"
                    f"💵 Balance trước lệnh: {balance_before:.2f} USDT"
                )
                log_trade(position, qty, current_price, notional)

            elif predicted_price_real < current_price * 0.9995 and adx_ok and ema20 < ema50:
                set_leverage(symbol, leverage)
                balance_before = get_balance()
                qty = get_quantity(current_price)
                notional = qty * current_price

                if notional < 100:
                    send_tele(f"⚠️ Lệnh bị huỷ vì notional < 100: {notional:.2f}")
                    return

                order = place_order("SELL", qty, reduce_only=False)
                if order is None:
                    send_tele("❌ Lỗi khi mở lệnh SELL: Order trả về None.")
                    return
                position = -1
                entry_price = current_price
                save_position_state(position, entry_price, qty, predicted_price_real)

                tp_price = entry_price - atr * 2.5
                sl_price = entry_price + atr * 2.0

                send_tele(
                    f"🔻 Mở SHORT tại {entry_price:.2f}\n"
                    f"📈 TP : {tp_price:.2f}\n"
                    f"🛡️ SL : {sl_price:.2f}\n"
                    f"📏 ATR hiện tại: {atr:.2f}\n"
                    f"💵 Balance trước lệnh: {balance_before:.2f} USDT"
                )
                log_trade(position, qty, current_price, notional)

        elif position == 1:
            if current_price >= entry_price + atr * 2.5 or current_price <= entry_price - atr * 2.0:
                notional = qty * current_price
                if notional < 100:
                    send_tele(f"⚠️ Lệnh bị huỷ vì notional < 100: {notional:.2f}")
                    return
                order = place_order("SELL", qty, reduce_only=True)
                if order:
                    position = 0
                    balance_after = get_balance()
                    if balance_after is None:
                        send_tele(f"✅ Đóng LONG tại {current_price:.2f}\n💰 Không lấy được balance sau lệnh.")
                    else:
                        send_tele(
                            f"✅ Đóng LONG tại {current_price:.2f}\n💰 Balance sau đóng lệnh: {balance_after:.2f} USDT")

                    if os.path.exists(POSITION_STATE_FILE):
                        os.remove(POSITION_STATE_FILE)
                    log_trade(position, qty, current_price, notional)

        elif position == -1:
            if current_price <= entry_price - atr * 2.5 or current_price >= entry_price + atr * 2.0:
                notional = qty * current_price
                if notional < 100:
                    send_tele(f"⚠️ Lệnh bị huỷ vì notional < 100: {notional:.2f}")
                    return
                order = place_order("BUY", qty, reduce_only=True)
                if order:
                    position = 0
                    balance_after = get_balance()
                    if balance_after is None:
                        send_tele(f"✅ Đóng SHORT tại {current_price:.2f}\n💰 Không lấy được balance sau lệnh.")
                    else:
                        send_tele(f"✅ Đóng SHORT tại {current_price:.2f}\n💰 Balance sau đóng lệnh: {balance_after:.2f} USDT")

                    if os.path.exists(POSITION_STATE_FILE):
                        os.remove(POSITION_STATE_FILE)
                    log_trade(position, qty, current_price, notional)

    except Exception as e:
        error_msg = f"❌ Lỗi trong make_decision: {e}"
        send_tele(error_msg)
        print(error_msg)
        with open(LOG_FILE, "a") as f:
            f.write(error_msg + "\n")

def log_trade(position, qty, price, notional):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    side = "BUY" if position == 1 else "SELL"
    with open("logs/trade_log.csv", "a") as f:
        f.write(f"{timestamp},{side},{qty},{price},{notional}\n")

def get_balance():
    try:
        url = f"https://fapi.binance.com/fapi/v2/balance"
        timestamp = int(time.time() * 1000)
        query = f"timestamp={timestamp}"
        signature = create_signature(query)
        headers = {"X-MBX-APIKEY": API_KEY}
        r = requests.get(f"{url}?{query}&signature={signature}", headers=headers)
        data = r.json()

        # Nếu data là list, tìm balance USDT
        if isinstance(data, list):
            usdt_info = next((x for x in data if x.get("asset") == "USDT"), None)
            if usdt_info:
                return float(usdt_info["availableBalance"])

        # Nếu đến đây thì không thành công
        send_tele("⚠️ Không thể lấy balance: response không hợp lệ.")
        print("⚠️ Không thể lấy balance: response không hợp lệ.")
        return None

    except Exception as e:
        error_msg = f"❌ Lỗi trong get_balance: {e}"
        send_tele(error_msg)
        print(error_msg)
        with open(LOG_FILE, "a") as f:
            f.write(error_msg + "\n")
        return None

# ============================
# 📦 Tính khối lượng lệnh từ số dư USDT
# ============================
# ============================
# 📦 Tính khối lượng lệnh từ số dư USDT
# ============================
def round_step_size(quantity, step_size=0.001):
    return round(np.floor(quantity / step_size) * step_size, 3)  # round về sàn gần nhất

def get_quantity(current_price):
    try:
        usdt_balance = get_balance()
        max_notional = usdt_balance * leverage * 0.99  # buffer 1%
        raw_qty = max_notional / current_price
        qty = round_step_size(raw_qty)

        notional = qty * current_price
        # send_tele(f"📊 USDT: {usdt_balance:.2f}, Qty BTC: {qty:.6f}, Giá: {current_price:.2f}, Notional: {notional:.2f}")
        return qty
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