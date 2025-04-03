import os
import json
import numpy as np
import pandas as pd
import time
import ta.volatility
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from pybit.unified_trading import HTTP
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

# ===== CẤU HÌNH CHẾ ĐỘ CHẠY BOT =====
TESTNET_MODE = False  # True: Testnet, False: Real Market
print(f"🚀 Đang chạy ở chế độ: {'TESTNET' if TESTNET_MODE else 'REAL MARKET'}")
if TESTNET_MODE:
    API_KEY = "BIpG5vVt41fsIEUcQe"
    API_SECRET = "pVfG0mafX1ey8clN6quZqTI4EZvkrQgeoziX"
    testnet = True
    model_file = "models/testnet/ai_model.keras"  # Model riêng cho Testnet
    data_file = "data/testnet/data.json"  # Dữ liệu riêng cho Testnet
else:
    API_KEY = "Hm5gG0HKbm5MDo5bpo"
    API_SECRET = "D6iP8YwCisA8pUylvh6916rnvWxoyKQnq1jp"
    testnet = False
    model_file = "models/real/ai_model.keras"  # Model riêng cho tài khoản thật
    data_file = "data/real/data.json"  # Dữ liệu riêng cho tài khoản thật

# ===== TẠO CÁC THƯ MỤC NẾU CHƯA TỒN TẠI =====
os.makedirs(os.path.dirname(model_file), exist_ok=True)
os.makedirs(os.path.dirname(data_file), exist_ok=True)

# ===== KẾT NỐI API BYBIT =====
session = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=testnet)


#
# ===== HÀM LẤY DỮ LIỆU GIÁ VÀ TÍNH INDICATORS =====
def fetch_data_with_indicators(symbol="BTCUSDT", timeframe="60", limit=2000):
    response = session.get_kline(category="spot", symbol=symbol, interval=timeframe, limit=limit)
    data = response["result"]["list"]
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
    df = df.sort_values("timestamp")

    # Chuyển dữ liệu sang float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Tính các chỉ báo kỹ thuật
    # ghi chú : sma_indicator của thư viện ta chuyên dùng tính toán các chỉ số kỹ thuật, window = 14 sẽ chỉ dùng cho trade trong ngày nếu muốn trade dài hơn phải tăng số nến lên
    df["sma"] = ta.trend.sma_indicator(df["close"], window=14)
    df["ema"] = ta.trend.ema_indicator(df["close"], window=14)
    #macd = ta.trend.macd(df["close"])
    #df["macd"] = macd
    #df["macd_signal"] = ta.trend.macd_signal(df["close"])
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()

    # Tính thêm ATR để đưa vào feature
    atr = ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    )
    df["atr"] = atr.average_true_range()

    # Loại bỏ NaN sau khi tính chỉ báo
    df.dropna(inplace=True)
    return df

# ===== HÀM HUẤN LUYỆN MÔ HÌNH LSTM =====
def train_lstm_model(df):
    feature_cols = [
        "close", "sma", "ema", "macd", "macd_signal",
        "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr"
    ]

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    # Chia dữ liệu train/test
    split_index = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:split_index]
    test_data = scaled_data[split_index:]

    X_train, y_train = [], []
    LOOKBACK = 100
    for i in range(LOOKBACK, len(train_data)):
        X_train.append(train_data[i-LOOKBACK:i])
        y_train.append(train_data[i][0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    model = Sequential()
    model.add(Input(shape=(LOOKBACK, len(feature_cols))))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Dùng EarlyStopping để tránh overfitting
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stop])

    return model, scaler

def validate_data(data):
    """Kiểm tra dữ liệu trước khi sử dụng"""
    if data is None or data.empty:
        print("❌ Lỗi: Dữ liệu API trả về rỗng!")
        return False

    """Kiểm tra xem dữ liệu có hợp lệ không trước khi huấn luyện/dự đoán"""
    required_columns = ["timestamp", "open", "high", "low", "close"]
    if not all(col in data.columns for col in required_columns):
        print("❌ Dữ liệu thiếu cột cần thiết.")
        return False

    # Ép kiểu float cho các cột cần thiết
    data["high"] = data["high"].astype(float)
    data["low"] = data["low"].astype(float)
    data["close"] = data["close"].astype(float)

    if not data["timestamp"].is_monotonic_increasing:
        print("⚠️ Cảnh báo: Dữ liệu không theo thứ tự thời gian!")
        return False

    # Kiểm tra sự biến động giá
    data["price_range"] = data["high"] - data["low"]
    if (data["price_range"] == 0).sum() > 3:
        print("⚠️ Cảnh báo: Có nhiều nến không dao động! Dữ liệu có thể sai.")
        return False

    duplicate_rows = data.duplicated(subset=["timestamp"]).sum()
    if duplicate_rows > 0:
        print(f"⚠️ Cảnh báo: Có {duplicate_rows} dòng dữ liệu trùng timestamp!")

    data["price_range"] = data["high"] - data["low"]
    if data["price_range"].max() > 10000:  # Nếu chênh lệch giá lớn hơn 10,000 USDT
        print("⚠️ Cảnh báo: Có sự biến động giá bất thường trong dữ liệu!")

    return True  # Nếu dữ liệu hợp lệ

# ===== HÀM KIỂM TRA & LOAD MODEL =====
def get_model():
    feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr"]
    LOOKBACK = 100

    print(f"🔹 Đang kiểm tra mô hình từ {model_file}")
    data = fetch_data_with_indicators()
    if not validate_data(data):
        raise ValueError("❌ Dữ liệu lịch sử không hợp lệ!")

    # Tạo scaler mới từ dữ liệu mới
    scaler = MinMaxScaler()
    scaler.fit(data[feature_cols])

    if os.path.exists(model_file):
        try:
            model = load_model(model_file)
            input_shape = model.input_shape  # (None, 100, 11)
            expected_shape = (None, LOOKBACK, len(feature_cols))

            if input_shape != expected_shape:
                print("⚠️ Mô hình cũ không khớp shape mới, sẽ huấn luyện lại...")
                raise ValueError("Shape mismatch")

            print("✅ Mô hình cũ hợp lệ, dùng lại.")
            return model, scaler

        except Exception as e:
            print(f"⚠️ Lỗi khi load mô hình cũ: {e}")
            print("🚀 Đang huấn luyện lại mô hình mới...")

    else:
        print("🚀 Chưa có mô hình, huấn luyện mới...")

    # Nếu model không tồn tại hoặc lỗi, train lại
    model, scaler = train_lstm_model(data)
    model.save(model_file)
    print(f"💾 Đã lưu mô hình mới tại {model_file}")
    return model, scaler

# ===== HÀM DỰ ĐOÁN GIÁ =====
def predict_price(model, scaler, df):
    feature_cols = ["close", "sma", "ema", "macd", "macd_signal",
                    "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr"]

    LOOKBACK = 100  # phải khớp với lúc train

    scaled_data = scaler.transform(df[feature_cols])

    # Lấy chuỗi 100 nến cuối
    last_sequence = np.array(scaled_data[-LOOKBACK:])  # shape: (100, 11)
    last_sequence = last_sequence.reshape(1, LOOKBACK, len(feature_cols))  # shape: (1, 100, 11)

    prediction = model.predict(last_sequence)

    # Ghép thêm cột 0s để inverse_transform đúng shape (1, 11)
    dummy = np.zeros((1, len(feature_cols)))
    dummy[0][0] = prediction[0][0]
    predicted_close = scaler.inverse_transform(dummy)[0][0]

    return predicted_close

def get_min_order_values():
    """Lấy thông tin về số lượng và giá trị tối thiểu của cặp BTCUSDT"""
    try:
        response = session.get_instruments_info(category="spot", symbol="BTCUSDT")

        # In toàn bộ dữ liệu để kiểm tra key nào đúng
        print(f"🔍 DEBUG - API Response: {response}")

        if "result" not in response or "list" not in response["result"] or not response["result"]["list"]:
            raise ValueError ("API ko hop le")
        info = response["result"]["list"][0]  # Lấy thông tin của cặp BTCUSDT
        min_qty = float(info.get("minOrderQty", 0))  # Dùng .get() để tránh KeyError
        min_value = float(info.get("minOrderValue", 10))  # Dùng .get() để tránh KeyError

        print(f"🔍 DEBUG - Min BTC Order Qty: {min_qty}, Min Order Value: {min_value}")
        return min_qty, min_value
    except Exception as e:
        print(f"loi khi lay thong tin min order: {e}")
        return 0.00001, 10

# ===== HÀM ĐẶT LỆNH =====
open_orders = []  # Danh sách lưu các lệnh mua đang mở
def place_order(side, order_value, predict_price=None):
    """Đặt lệnh mua/bán trên Bybit"""
    try:
        min_btc, min_usdt = get_min_order_values()  # Lấy giá trị tối thiểu
        if order_value < min_usdt:
            order_value = min_usdt # Điều chỉnh order_value để >= min_usdt
        data = fetch_data_with_indicators()
        if not validate_data(data):
            raise ValueError("Dữ liệu không hợp lệ! Hủy lệnh.")
        last_price = data["close"].iloc[-1]  # Lấy giá BTC mới nhất
        if side == "Buy":
            qty = str(round(order_value, 6))  # Mua → `qty` là số USDT
            tp = round(predict_price, 2) if predict_price else round(last_price * 1.02)
            sl = round(last_price * 0.98, 2)  # Take-Profit = +3 ATR

            # Đặt lệnh Market Buy
            order = session.place_order(
                category="spot",
                symbol="BTCUSDT",
                side=side,
                orderType="Market",
                qty=qty
            )

            print(f"✅ Đã đặt lệnh Mua {order_value:.2f} USDT")
            print(f"🎯 Take-Profit tại: {take_profit}, ⛔ Stop-Loss tại: {stop_loss}")

            # Lưu lệnh vào danh sách để theo dõi TP/SL
            open_orders.append({"entry_price": last_price, "tp": take_profit, "sl": stop_loss})
        else:  # Sell
            qty = str(round(order_value / last_price, 6))  # Bán → `qty` là số BTC
            # Đặt lệnh Market Sell
            order = session.place_order(
                category="spot",
                symbol="BTCUSDT",
                side=side,
                orderType="Market",
                qty=qty
            )
            print(f"✅ Đã bán {qty} BTC")
    except Exception as e:
        print(f"⚠️ Lỗi khi đặt lệnh: {e}")


#===== HÀM CHẠY BOT =====
def run_bot():
    """Chạy bot AI với môi trường Testnet hoặc Real Market"""
    global open_orders
    model, scaler = get_model()

    while True:
        data = fetch_data_with_indicators()
        if not validate_data(data):
            print("⛔ Dữ liệu lỗi, bỏ qua vòng lặp này.")
            time.sleep(900)
            continue
        predicted_price = predict_price(model, scaler, data)
        last_close = data["close"].iloc[-1]

        # Kiểm tra nếu có lệnh mở, xem có đạt TP hoặc SL không
        for order in open_orders[:]:  # Duyệt qua danh sách lệnh mở
            if last_close >= order["tp"]:
                print(f"✅ Chạm Take-Profit! Giá {order['entry_price']} → {order['tp']}")
                place_order("Sell", 10)  # Bán BTC để chốt lời
                open_orders.remove(order)

            elif last_close <= order["sl"]:
                print(f"⛔ Chạm Stop-Loss! Giá {order['entry_price']} → {order['sl']}")
                place_order("Sell", 10)  # Bán BTC để cắt lỗ
                open_orders.remove(order)

        # Nếu không có lệnh mở, đặt lệnh mới theo AI
        if not open_orders:
            if predicted_price > last_close * 1.005:
                place_order("Buy", 10)  # Mua BTC nếu giá dự đoán tăng 0.5%

        # Lưu dữ liệu giá vào file để phân tích sau
        with open(data_file, "w") as f:
            json.dump(data.to_dict(), f)

        print(f"📊 Giá dự đoán: {predicted_price:.2f}, Giá hiện tại: {last_close:.2f}")
        print(f"🔄 Chờ 15 phút trước khi giao dịch tiếp...")
        time.sleep(900)

# ===== CHẠY BOT =====
run_bot()
