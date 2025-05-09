# ============================
# 📦 Cài thư viện cần thiết
# ============================
!pip install ta tensorflow matplotlib scikit-learn requests python-dotenv

import os
import shutil
import datetime
import numpy as np
import pandas as pd
import ta
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import joblib
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Attention
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import requests
from google.colab import drive, files
from dotenv import load_dotenv

# ============================
# 🔧 Load biến môi trường từ .env
# ============================
uploaded_env = files.upload()
for filename in uploaded_env:
    if filename.endswith('.env'):
        os.rename(filename, ".env")

load_dotenv(".env")
telegram_token = os.getenv("TELEGRAM_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

# ============================
# 🗂 Mount Google Drive
# ============================
drive.mount('/content/drive')

# ============================
# 🚀 Cố định seed random để ổn định
# ============================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ============================
# 📂 Tạo thư mục cần thiết
# ============================
os.makedirs("models/backup", exist_ok=True)
os.makedirs("backtest_results", exist_ok=True)

# ============================
# 🗂️ Upload file CSV dữ liệu nến
# ============================
uploaded = files.upload()
data_file = list(uploaded.keys())[0]

# ============================
# 📊 Load và chuẩn bị dữ liệu
# ============================
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df.reset_index(inplace=True)
    if df["timestamp"].dtype in ["int64", "float64"]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df.sort_values("timestamp", inplace=True)

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
# 🔧 Tạo mô hình LSTM + Attention
# ============================
def create_lstm_attention_model(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    attention = Attention()([x, x])
    x = tf.keras.layers.GlobalAveragePooling1D()(attention)
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model

# ============================
# 🤖 Train các mô hình AI, TP, SL
# ============================
def train_model(df, lookback=100):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"ai15m_model_{timestamp}"
    feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff",
                    "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i][0])
    X, y = np.array(X), np.array(y)

    model = create_lstm_attention_model(input_shape=(lookback, len(feature_cols)))
    model.fit(X, y, epochs=10, batch_size=64, verbose=0, callbacks=[EarlyStopping(patience=2)],validation_split=0.1)
    model.save(f"models/{model_name}.keras")
    joblib.dump(scaler, f"models/{model_name}.pkl")
    return model, scaler, model_name

def train_tp_model(df, lookback=100):
    feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff",
                    "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    X, y_tp = [], []
    for i in range(lookback, len(scaled_data)-1):
        X.append(scaled_data[i - lookback:i])
        close_now = df["close"].iloc[i]
        close_next = df["close"].iloc[i+1]
        atr_now = df["atr"].iloc[i]
        y_tp.append(min(close_next, close_now + atr_now * 4))

    X, y_tp = np.array(X), np.array(y_tp)
    model = Sequential([
        Input(shape=(lookback, len(feature_cols))),
        LSTM(32, return_sequences=True),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y_tp, epochs=5, batch_size=32, verbose=0)
    return model, scaler

def train_sl_model(df, lookback=100):
    feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff",
                    "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    X, y_sl = [], []
    for i in range(lookback, len(scaled_data)-1):
        X.append(scaled_data[i - lookback:i])
        close_now = df["close"].iloc[i]
        close_next = df["close"].iloc[i+1]
        atr_now = df["atr"].iloc[i]
        y_sl.append(max(close_next, close_now - atr_now * 1.5))

    X, y_sl = np.array(X), np.array(y_sl)
    model = Sequential([
        Input(shape=(lookback, len(feature_cols))),
        LSTM(32, return_sequences=True),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y_sl, epochs=5, batch_size=32, verbose=0)
    return model, scaler

# ============================
# 📈 Backtest kết hợp AI + TP + SL
# ============================
def backtest(df, ai_model, ai_scaler, tp_model, tp_scaler, sl_model, sl_scaler, lookback=100, atr_tp_factor=4, atr_sl_factor=1.5):
    try:
        feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff",
                        "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]
        fee_rate = 0.001
        risk_per_trade = 0.02  # Rủi ro 2% mỗi giao dịch

        scaled = ai_scaler.transform(df[feature_cols])
        X = np.zeros((len(df) - lookback, lookback, len(feature_cols)))
        for i in range(lookback, len(df)):
            X[i - lookback] = scaled[i - lookback:i]
        pred_scaled = ai_model.predict(X, verbose=0).flatten()
        dummy = np.tile(scaled[lookback:, -1], (len(feature_cols), 1)).T
        dummy[:, 0] = pred_scaled
        preds = ai_scaler.inverse_transform(dummy)[:, 0]

        df = df.iloc[lookback:].copy()
        df["predicted_close"] = preds

        tp_scaled = tp_scaler.transform(df[feature_cols])
        X_tp = np.zeros((len(df) - lookback, lookback, len(feature_cols)))
        for i in range(lookback, len(df)):
            X_tp[i - lookback] = tp_scaled[i - lookback:i]
        df = df.iloc[lookback:].copy()
        df["predicted_tp"] = tp_model.predict(X_tp, verbose=0).flatten() * df["close"]

        sl_scaled = sl_scaler.transform(df[feature_cols])
        X_sl = np.zeros((len(df) - lookback, lookback, len(feature_cols)))
        for i in range(lookback, len(df)):
            X_sl[i - lookback] = sl_scaled[i - lookback:i]
        df["predicted_sl"] = sl_model.predict(X_sl, verbose=0).flatten() * df["close"]

        balance = 5000
        position = 0
        trades = []
        trailing_sl = 0
        position_size = 0

        for i, row in df.iterrows():
            if position == 0 and row["predicted_close"] > row["close"] * 1.0005 and row["rsi"] < 30 and row["macd"] > row["macd_signal"] and row["adx"] > 25:
                buy_price = row["close"]
                tp = row["predicted_tp"]
                sl = row["predicted_sl"]
                position_size = (balance * risk_per_trade) / (buy_price - sl)
                position = 1
                trailing_sl = sl
                balance -= position_size * buy_price * fee_rate
                trades.append([row["timestamp"], "BUY", round(buy_price, 2), "", "", round(balance, 2)])
            elif position == 1:
                trailing_sl = max(trailing_sl, row["close"] - row["atr"] * atr_sl_factor)
                if row["close"] >= tp or row["close"] <= trailing_sl:
                    sell_price = row["close"]
                    pnl = position_size * (sell_price - buy_price)
                    balance += pnl
                    balance -= position_size * sell_price * fee_rate
                    trades.append([row["timestamp"], "SELL", round(buy_price, 2), round(sell_price, 2),
                                   round(pnl, 2), round(balance, 2)])
                    position = 0

        trades_df = pd.DataFrame(trades, columns=["timestamp", "action", "buy_price", "sell_price", "pnl_usdt", "balance"])
        trades_df.to_csv("backtest_results/trade_log.csv", index=False)

        win_rate = len(trades_df[trades_df["pnl_usdt"] > 0]) / len(trades_df[trades_df["pnl_usdt"] != 0]) if len(trades_df) > 0 else 0
        profit_factor = trades_df[trades_df["pnl_usdt"] > 0]["pnl_usdt"].sum() / abs(trades_df[trades_df["pnl_usdt"] < 0]["pnl_usdt"].sum()) if trades_df[trades_df["pnl_usdt"] < 0]["pnl_usdt"].sum() != 0 else float('inf')

        print(f"\nKết quả: Final Balance = {round(balance, 2)} USDT, Win Rate = {win_rate:.2%}, Profit Factor = {profit_factor:.2f}")
        plt.figure(figsize=(10, 4))
        plt.plot(trades_df["balance"].dropna())
        plt.title("Equity Curve")
        plt.tight_layout()
        plt.savefig("backtest_results/equity_curve.png")
        plt.close()

        return balance, win_rate, profit_factor
    except Exception as e:
        print(f"Error in backtest: {e}")
        return 0, 0, 0

# ============================
# 🔁 Train và backtest 30 lần
# ============================
df = load_and_prepare_data(data_file)
best_balance = 0
for i in range(30):
    print(f"\nVòng train-backtest {i+1}/30")
    ai_model, ai_scaler, model_name = train_model(df)
    tp_model, tp_scaler = train_tp_model(df)
    sl_model, sl_scaler = train_sl_model(df)
    balance = backtest(df, ai_model, ai_scaler, tp_model, tp_scaler, sl_model, sl_scaler)

    if balance > best_balance:
        best_balance = balance
        ai_model.save("models/best_model.keras")
        joblib.dump(ai_scaler, "models/best_scaler.pkl")

print(f"\n📊 Tốt nhất sau 30 lần: Balance: {best_balance:.2f} USDT")