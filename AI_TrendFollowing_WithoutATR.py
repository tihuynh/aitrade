import os
import json
import numpy as np
import pandas as pd
import time
from pybit.unified_trading import HTTP
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ===== CẤU HÌNH CHẾ ĐỘ CHẠY BOT =====
TESTNET_MODE = True  # True: Testnet, False: Real Market

if TESTNET_MODE:
    API_KEY = "BIpG5vVt41fsIEUcQe"
    API_SECRET = "pVfG0mafX1ey8clN6quZqTI4EZvkrQgeoziX"
    testnet = True
    model_file = "ai_model_testnet.h5"  # Model riêng cho Testnet
    data_file = "data_testnet.json"  # Dữ liệu riêng cho Testnet
else:
    API_KEY = "your_real_api_key"
    API_SECRET = "your_real_api_secret"
    testnet = False
    model_file = "ai_model_real.h5"  # Model riêng cho tài khoản thật
    data_file = "data_real.json"  # Dữ liệu riêng cho tài khoản thật

# ===== XÓA MODEL CŨ KHI CHUYỂN TỪ TESTNET SANG REAL MARKET =====
if not TESTNET_MODE:
    if os.path.exists("ai_model_testnet.h5"):
        os.remove("ai_model_testnet.h5")  # Xóa model cũ của Testnet
    if os.path.exists("data_testnet.json"):
        os.remove("data_testnet.json")  # Xóa dữ liệu cũ của Testnet
    print("⚠️ Đã chuyển sang thị trường thực - Xóa dữ liệu AI cũ, cần huấn luyện lại!")

# ===== KẾT NỐI API BYBIT =====
session = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=testnet)
# ===== HÀM LẤY DỮ LIỆU GIÁ =====
def fetch_historical_data(symbol="BTCUSDT", timeframe="60", limit=500):
    """Lấy dữ liệu giá từ Bybit (Testnet hoặc Real Market)"""
    response = session.get_kline(category="spot", symbol=symbol, interval=timeframe, limit=limit)
    data = response["result"]["list"]
    print(f"🔍 DEBUG - Dữ liệu API trả về:\n{data[:3]}")  # In 3 dòng đầu để kiểm tra format
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
    df["close"] = df["close"].astype(float)
    return df

# ===== HÀM HUẤN LUYỆN MÔ HÌNH LSTM =====
def train_lstm_model(data):
    """Huấn luyện AI dự đoán giá"""
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data["close"].values.reshape(-1,1))

    X_train, y_train = [], []
    LOOKBACK = 50
    for i in range(len(scaled_data) - LOOKBACK):
        X_train.append(scaled_data[i:i+LOOKBACK])
        y_train.append(scaled_data[i+LOOKBACK])

    X_train, y_train = np.array(X_train), np.array(y_train)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    return model, scaler

# ===== HÀM KIỂM TRA & LOAD MODEL =====
def get_model():
    """Kiểm tra và load model phù hợp"""
    if os.path.exists(model_file):
        print(f"🔹 Đang tải mô hình từ {model_file}")
        model = load_model(model_file)

        # Fit lại scaler với dữ liệu mới trước khi dùng
        data = fetch_historical_data()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data["close"].values.reshape(-1, 1))  # Fit scaler với dữ liệu mới
        return model, scaler
    else:
        print("🚀 Chưa có mô hình, đang huấn luyện mới!")
        data = fetch_historical_data()
        model, scaler = train_lstm_model(data)
        model.save(model_file)
        return model, scaler

# ===== HÀM DỰ ĐOÁN GIÁ =====
def predict_trend(model, scaler, data):
    """Dự đoán giá dựa vào AI"""
    scaled_data = scaler.transform(data["close"].values.reshape(-1,1))
    last_sequence = np.array([scaled_data[-50:]])
    prediction = model.predict(last_sequence)
    return scaler.inverse_transform(prediction)[0][0]


def get_min_order_values():
    """Lấy thông tin về số lượng và giá trị tối thiểu của cặp BTCUSDT"""
    response = session.get_instruments_info(category="spot", symbol="BTCUSDT")

    # In toàn bộ dữ liệu để kiểm tra key nào đúng
    print(f"🔍 DEBUG - API Response: {response}")

    info = response["result"]["list"][0]  # Lấy thông tin của cặp BTCUSDT
    min_qty = float(info.get("minOrderQty", 0))  # Dùng .get() để tránh KeyError
    min_value = float(info.get("minOrderValue", 10))  # Dùng .get() để tránh KeyError

    print(f"🔍 DEBUG - Min BTC Order Qty: {min_qty}, Min Order Value: {min_value}")
    return min_qty, min_value


# ===== HÀM ĐẶT LỆNH =====
open_orders = []  # Danh sách lưu các lệnh mua đang mở
def place_order(side, order_value):
    """Đặt lệnh mua/bán trên Bybit"""
    try:
        take_profit, stop_loss = 0, 0
        min_btc, min_usdt = get_min_order_values()  # Lấy giá trị tối thiểu
        if order_value < min_usdt:
            order_value = min_usdt # Điều chỉnh order_value để >= min_usdt
        last_price = float(fetch_historical_data().iloc[-1]["close"])  # Lấy giá BTC mới nhất
        if side == "Buy":
            qty = str(round(order_value, 6))  # Mua → `qty` là số USDT
        else:  # Sell
            qty = str(round(order_value / last_price, 6))  # Bán → `qty` là số BTC
        order = session.place_order(
            category="spot",
            symbol="BTCUSDT",
            side=side,
            orderType="Market",
            qty=qty
        )
        print(f"✅ Đã đặt lệnh {side} {qty} {'USDT' if side == 'Buy' else 'BTC'} (≈ {order_value:.2f} USDT)")
    except Exception as e:
        print(f"⚠️ Lỗi khi đặt lệnh: {e}")


# ===== HÀM CHẠY BOT =====
def run_bot():
    """Chạy bot AI với môi trường Testnet hoặc Real Market"""
    model, scaler = get_model()

    while True:
        data = fetch_historical_data()
        predicted_price = predict_trend(model, scaler, data)
        last_close = data["close"].iloc[-1]
        if predicted_price > last_close * 1.005:
            place_order("Buy", 10)  # Mua 0.01 BTC nếu giá tăng 0.5%
        elif predicted_price < last_close * 0.995:
            place_order("Sell", 10)  # Bán 0.01 BTC nếu giá giảm 0.5%

        # Lưu dữ liệu giá vào file để phân tích sau
        with open(data_file, "w") as f:
            json.dump(data.to_dict(), f)

        print(f"📊 Giá dự đoán: {predicted_price:.2f}, Giá hiện tại: {last_close:.2f}")
        print(f"🔄 Chờ 15 phút trước khi giao dịch tiếp...")
        time.sleep(900)

# ===== CHẠY BOT =====
run_bot()
