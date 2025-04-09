# ============================
# 📦 Cài thư viện cần thiết
# ============================
!pip install ta tensorflow matplotlib scikit-learn requests

import os
import shutil
import datetime
import numpy as np
import pandas as pd
import ta
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import requests
from google.colab import drive, files

# ============================
# 🔧 Cấu hình Telegram & Drive
# ============================
telegram_token = '7621293655:AAHaLf_tMtt-vxpb1Qt0K6QEOGmfhmhy0lY'
telegram_chat_id = '1989267515'
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
# 🤖 Huấn luyện mô hình
# ============================
def train_model(df, lookback=100):
    feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i][0])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        Input(shape=(lookback, len(feature_cols))),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

    model.fit(X, y, epochs=10, batch_size=64, verbose=0, callbacks=[early_stop])

    model.save("models/ai15m_model_colab.keras")
    return model, scaler

# ============================
# 📈 Backtest chiến lược
# ============================
def backtest_strategy(model, scaler, df, initial_balance=5000, lookback=100):
    feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]

    sequences = np.array([scaler.transform(df[feature_cols].iloc[i - lookback:i]) for i in range(lookback, len(df))])
    predictions_scaled = model.predict(sequences, verbose=0).flatten()

    dummy = np.zeros((len(predictions_scaled), len(feature_cols)))
    dummy[:, 0] = predictions_scaled
    predictions = scaler.inverse_transform(dummy)[:, 0]

    current_price = df["close"].iloc[lookback:].values
    atr = df["atr"].iloc[lookback:].values
    macd_bullish = (df["macd"].iloc[lookback:].values - df["macd_signal"].iloc[lookback:].values) > -15
    rsi_ok = df["rsi"].iloc[lookback:].values > 40
    price_near_bottom = current_price <= df["close"].iloc[lookback - 20:-20].rolling(20).min().values * 1.05
    adx_ok = df["adx"].iloc[lookback:].values > 20
    ai_confidence = predictions > current_price * 1.001

    buy_condition = ai_confidence & macd_bullish & rsi_ok & price_near_bottom & adx_ok

    balance = initial_balance
    position = 0
    buy_price, take_profit, stop_loss = 0, 0, 0
    trade_log = []
    equity_curve = []
    timestamps = df["timestamp"].iloc[lookback:].values
    win_count, loss_count = 0, 0

    for i in range(len(current_price)):
        price = current_price[i]
        timestamp = timestamps[i]

        if position == 0 and buy_condition[i]:
            position = 1
            buy_price = price
            take_profit = round(buy_price * 1.004, 2)
            stop_loss = round(buy_price * 0.996, 2)
            trade_log.append(f"Mua tại {buy_price:.2f} | TP: {take_profit} | SL: {stop_loss}")

        elif position == 1:
            if price >= take_profit:
                balance *= 1 + (atr[i] * 2 / buy_price)
                trade_log.append(f"TP tại {price:.2f} | Số dư: {balance:.2f}")
                win_count += 1
                position = 0
            elif price <= stop_loss:
                balance *= 1 - (atr[i] * 1.5 / buy_price)
                trade_log.append(f"SL tại {price:.2f} | Số dư: {balance:.2f}")
                loss_count += 1
                position = 0

        if position == 0:
            equity_curve.append(balance)

    winrate = (win_count / (win_count + loss_count)) * 100 if (win_count + loss_count) > 0 else 0

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps[:len(equity_curve)], equity_curve)
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Balance (USDT)")
    plt.tight_layout()
    plt.savefig("backtest_results/equity_curve.png")
    plt.close()

    pd.DataFrame(trade_log).to_csv("backtest_results/trade_log.csv", index=False, encoding="utf-8-sig")

    if balance >= 5200:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"models/backup/balance{int(balance)}_{timestamp}"
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy("models/ai15m_model_colab.keras", os.path.join(backup_dir, "model.keras"))

    return balance, winrate

# ============================
# 🏁 Chạy train + backtest 30 lần và auto gửi báo cáo
# ============================

df = load_and_prepare_data(data_file)
best_balance = 0

for i in range(30):
    print(f"\nVòng train-backtest {i + 1}/30")
    model, scaler = train_model(df, lookback=100)
    balance, winrate = backtest_strategy(model, scaler, df, initial_balance=5000)

    if balance > best_balance:
        best_balance = balance
        print(f"🔍 Model mới có balance tốt hơn: {best_balance:.2f} USDT")

print(f"\n📊 Tốt nhất sau 30 lần: Balance: {best_balance:.2f} USDT")

os.system('zip -r models_backup.zip models/backup backtest_results')
shutil.copy('models_backup.zip', '/content/drive/MyDrive/models_backup.zip')

with open('models_backup.zip', 'rb') as f:
    response = requests.post(
        f'https://api.telegram.org/bot{telegram_token}/sendDocument?chat_id={telegram_chat_id}',
        files={'document': f}
    )

if response.status_code == 200:
    print("✅ Đã gửi file models_backup.zip về Telegram!")
else:
    print("❌ Lỗi gửi file về Telegram:", response.text)

report = f"Tốt nhất sau 30 lần:\nBalance: {best_balance:.2f} USDT"
requests.post(
    f'https://api.telegram.org/bot{telegram_token}/sendMessage',
    data={'chat_id': telegram_chat_id, 'text': report}
)
