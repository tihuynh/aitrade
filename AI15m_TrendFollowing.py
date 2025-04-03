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
from ta.trend import ADXIndicator

# ===== CẤU HÌNH CHẾ ĐỘ CHẠY BOT =====
TESTNET_MODE = False  # True: Testnet, False: Real Market
ALLOW_WEAK_DATA = TESTNET_MODE
print(f"🚀 Đang chạy ở chế độ: {'TESTNET' if TESTNET_MODE else 'REAL MARKET'}")
if TESTNET_MODE:
    API_KEY = "BIpG5vVt41fsIEUcQe"
    API_SECRET = "pVfG0mafX1ey8clN6quZqTI4EZvkrQgeoziX"
    testnet = True
    model_file = "models/testnet/ai15m_model.keras"  # Model riêng cho Testnet
    data_file = "data/testnet/data15m.json"  # Dữ liệu riêng cho Testnet
else:
    API_KEY = "Hm5gG0HKbm5MDo5bpo"
    API_SECRET = "D6iP8YwCisA8pUylvh6916rnvWxoyKQnq1jp"
    testnet = False
    model_file = "models/real/ai15m_model.keras"  # Model riêng cho tài khoản thật
    data_file = "data/real/data15m.json"  # Dữ liệu riêng cho tài khoản thật

# ===== TẠO CÁC THƯ MỤC NẾU CHƯA TỒN TẠI =====
os.makedirs(os.path.dirname(model_file), exist_ok=True)
os.makedirs(os.path.dirname(data_file), exist_ok=True)

# ===== KẾT NỐI API BYBIT =====
session = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=testnet)


#
# ===== HÀM LẤY DỮ LIỆU GIÁ VÀ TÍNH INDICATORS =====
def fetch_data_with_indicators(symbol="BTCUSDT", timeframe="15", limit=12000, retry_attempts=5):
    for attempt in range(retry_attempts):
        try:
            response = session.get_kline(category="spot", symbol=symbol, interval=timeframe, limit=limit)
            if "retCode" in response and response["retCode"] == 10006:
                print("⚠️ Hit API rate limit. Đợi 10 giây rồi thử lại...")
                time.sleep(10)
                continue

            data = response["result"]["list"]
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
            df = df.sort_values("timestamp")

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            # ===== TÍNH INDICATORS =====
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
            time.sleep(1)  # Nghỉ 1s tránh spam API
            return df

        except Exception as e:
            print(f"❌ Lỗi khi gọi API (lần {attempt+1}/{retry_attempts}): {e}")
            time.sleep(5)  # Nghỉ 5s trước khi thử lại

    print("🚫 Gọi API thất bại sau nhiều lần thử.")
    return pd.DataFrame()  # Trả về DataFrame rỗng nếu thất bại


# ===== HÀM HUẤN LUYỆN MÔ HÌNH LSTM =====
def train_lstm_model(df):
    feature_cols = [
        "close", "sma", "ema", "macd", "macd_signal",
        "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx",
        "plus_di", "minus_di", "candle_body", "lower_wick"
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
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
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
    if not ALLOW_WEAK_DATA and (data["price_range"] == 0).sum() > 3:
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
    feature_cols = [
        "close", "sma", "ema", "macd", "macd_signal",
        "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx",
        "plus_di", "minus_di", "candle_body", "lower_wick"
    ]
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
    feature_cols = [
        "close", "sma", "ema", "macd", "macd_signal",
        "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx",
        "plus_di", "minus_di", "candle_body", "lower_wick"
    ]

    LOOKBACK = 100  # phải khớp với lúc train

    scaled_data = scaler.transform(df[feature_cols])

    # Lấy chuỗi 100 nến cuối
    last_sequence = np.array(scaled_data[-LOOKBACK:])  # shape: (100, 11)
    last_sequence = last_sequence.reshape(1, LOOKBACK, len(feature_cols))  # shape: (1, 100, 11)

    prediction = model.predict(last_sequence)

    # # Ghép thêm cột 0s để inverse_transform đúng shape (1, 11)
    # dummy = np.zeros((1, len(feature_cols)))
    # dummy[0][0] = prediction[0][0]
    # predicted_close = scaler.inverse_transform(dummy)[0][0]

    last_row = scaler.transform(df[feature_cols].iloc[-1:].copy())
    last_row[0][0] = prediction[0][0]  # Thay close bằng dự đoán
    predicted_close = scaler.inverse_transform(last_row)[0][0]

    return predicted_close

def get_min_order_values():
    """Lấy thông tin về số lượng và giá trị tối thiểu của cặp BTCUSDT (Spot)"""
    try:
        response = session.get_instruments_info(category="spot", symbol="BTCUSDT")

        if "result" not in response or "list" not in response["result"] or not response["result"]["list"]:
            raise ValueError("❌ API không trả về dữ liệu hợp lệ!")

        info = response["result"]["list"][0]
        min_qty = float(info.get("minOrderQty", 0))
        min_value = float(info.get("minOrderValue", 10))  # fallback = 10 USDT

        print(f"🔍 Min BTC Order Qty: {min_qty}, Min Order Value: {min_value}")
        return min_qty, min_value

    except Exception as e:
        print(f"⚠️ Lỗi khi lấy thông tin min order: {e}")
        # Trả về giá trị mặc định để tránh bot crash
        return 0.00001, 10

# ===== HÀM ĐẶT LỆNH =====
open_orders = []  # Danh sách lưu các lệnh mua đang mở
def place_order(side, order_value, data):
    # print(f"📥 [MOCK] Đặt lệnh {side} với {order_value} USDT")
    # print(f"📈 Giá hiện tại: {data['close'].iloc[-1]}, ATR: {data['atr'].iloc[-1]}")
    # if side == "Buy":
    #     tp = round(data["close"].iloc[-1] + data["atr"].iloc[-1] * 3, 2)
    #     sl = round(data["close"].iloc[-1] - data["atr"].iloc[-1] * 2, 2)
    #     open_orders.append({
    #         "entry_price": data["close"].iloc[-1],
    #         "tp": tp,
    #         "sl": sl
    #     })
    #     print(f"🎯 TP mới: {tp}, SL mới: {sl}")

    """Đặt lệnh mua/bán trên Bybit"""
    try:
        take_profit, stop_loss = 0, 0
        min_btc, min_usdt = get_min_order_values()  # Lấy giá trị tối thiểu
        if order_value < min_usdt:
            order_value = min_usdt # Điều chỉnh order_value để >= min_usdt
        last_price = float(data.iloc[-1]["close"])  # Lấy giá BTC mới nhất
        if side == "Buy":
            qty = str(round(order_value, 6))  # Mua → `qty` là số USDT
            stop_loss = round(last_price - (data["atr"].iloc[-1] * 2), 2)  # Stop-Loss = -2 ATR
            take_profit = round(last_price + (data["atr"].iloc[-1] * 3), 2)  # Take-Profit = +3 ATR

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
    try:
        """Chạy bot AI với môi trường Testnet hoặc Real Market"""
        global open_orders
        model, scaler = get_model()
        just_closed = False
        while True:
            data = fetch_data_with_indicators()
            if not validate_data(data):
                print("⛔ Dữ liệu lỗi, bỏ qua vòng lặp này.")
                time.sleep(900)
                continue
            predicted_price = predict_price(model, scaler, data)
            last_close = data["close"].iloc[-1]
            # # ===== MOCK DATA CHO TEST =====
            # open_orders = [
            #
            # ]
            #
            # # ===== GIÁ GIẢ LẬP =====
            # last_close = 86800  # Giá hiện tại
            # predicted_price = 88000  # Giá AI dự đoán
            # atr_value = 500  # ATR giả định
            #
            # # ===== DỮ LIỆU GIẢ LẬP =====
            # data = pd.DataFrame({
            #     "close": [last_close],
            #     "atr": [atr_value],
            # })

            # # Kiểm tra nếu có lệnh mở, xem có đạt TP hoặc SL không
            # for order in open_orders[:]:  # Duyệt qua danh sách lệnh mở
            #     if last_close >= order["tp"]:
            #         print(f"✅ Chạm Take-Profit! Giá {order['entry_price']} → {order['tp']}")
            #         place_order("Sell", 10, data)  # Bán BTC để chốt lời
            #         open_orders.remove(order)
            #         just_closed = True
            #     elif last_close <= order["sl"]:
            #         print(f"⛔ Chạm Stop-Loss! Giá {order['entry_price']} → {order['sl']}")
            #         place_order("Sell", 10, data)  # Bán BTC để cắt lỗ
            #         open_orders.remove(order)
            #         just_closed = True
            # # Nếu không có lệnh mở, đặt lệnh mới theo AI
            # if not open_orders and not just_closed:
            #     print("Không có lệnh mở, đặt lệnh mới theo AI")
            #     if predicted_price > last_close * 1.005:
            #         place_order("Buy", 10, data)  # Mua BTC nếu giá dự đoán tăng 0.5%
            #
            # # Lưu dữ liệu giá vào file để phân tích sau
            # with open(data_file, "w") as f:
            #     json.dump(data.to_dict(), f)
            #
            # print(f"📊 Giá dự đoán: {predicted_price:.2f}, Giá hiện tại: {last_close:.2f}")
            # print(f"🔄 Chờ 15 phút trước khi giao dịch tiếp...")
            # # time.sleep(900)
            time.sleep(60)
    except Exception as e:
        print(f"Loi trong run_bot {e}")

# ===== CHẠY BOT =====
run_bot()
