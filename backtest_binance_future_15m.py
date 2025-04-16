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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
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
print("📥 Vui lòng upload file .env trước!")
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
# 🚀 Cố định seed random
# ============================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

os.makedirs("models/backup", exist_ok=True)
os.makedirs("backtest_results", exist_ok=True)

# ============================
# 🗂️ Upload file CSV
# ============================
print("📥 Bây giờ vui lòng upload file dữ liệu CSV!")
uploaded = files.upload()
data_file = list(uploaded.keys())[0]

# ============================
# 📊 Load & feature engineering
# ============================
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
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
    joblib.dump(scaler, 'models/backup/scaler.pkl')
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
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
    model.fit(X, y, epochs=10, batch_size=64, verbose=0, callbacks=[EarlyStopping(patience=2)])

    model.save("models/ai_futures_model.keras")
    return model, scaler

# ============================
# 📈 Backtest Futures Logic (Long/Short)
# ============================
def backtest_strategy(model, scaler, df, initial_balance=5000, lookback=100, leverage=2):
    feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]
    sequences = np.array([scaler.transform(df[feature_cols].iloc[i - lookback:i]) for i in range(lookback, len(df))])
    predictions_scaled = model.predict(sequences, verbose=0).flatten()

    dummy = np.zeros((len(predictions_scaled), len(feature_cols)))
    dummy[:, 0] = predictions_scaled
    predictions = scaler.inverse_transform(dummy)[:, 0]

    close_prices = df["close"].iloc[lookback:].values
    atr = df["atr"].iloc[lookback:].values
    timestamps = df["timestamp"].iloc[lookback:].values

    balance = initial_balance
    position = 0
    entry_price, take_profit, stop_loss = 0, 0, 0
    direction = ""  # "LONG" or "SHORT"
    trade_log = []
    equity_curve = []
    wins, losses = 0, 0

    for i in range(len(close_prices)):
        price = close_prices[i]
        predicted = predictions[i]
        timestamp = timestamps[i]

        if position == 0:
            if predicted > price * 1.001:
                position = 1
                direction = "LONG"
                entry_price = price
                take_profit = price * 1.004
                stop_loss = price * 0.996
                trade_log.append(f"LONG tại {price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")
            elif predicted < price * 0.999:
                position = 1
                direction = "SHORT"
                entry_price = price
                take_profit = price * 0.996
                stop_loss = price * 1.004
                trade_log.append(f"SHORT tại {price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")

        elif position == 1:
            if direction == "LONG":
                if price >= take_profit:
                    profit = leverage * atr[i] * 2 / entry_price
                    balance *= 1 + profit
                    trade_log.append(f"TP LONG tại {price:.2f} | Balance: {balance:.2f}")
                    wins += 1
                    position = 0
                elif price <= stop_loss:
                    loss = leverage * atr[i] * 1.5 / entry_price
                    balance *= 1 - loss
                    trade_log.append(f"SL LONG tại {price:.2f} | Balance: {balance:.2f}")
                    losses += 1
                    position = 0
            elif direction == "SHORT":
                if price <= take_profit:
                    profit = leverage * atr[i] * 2 / entry_price
                    balance *= 1 + profit
                    trade_log.append(f"TP SHORT tại {price:.2f} | Balance: {balance:.2f}")
                    wins += 1
                    position = 0
                elif price >= stop_loss:
                    loss = leverage * atr[i] * 1.5 / entry_price
                    balance *= 1 - loss
                    trade_log.append(f"SL SHORT tại {price:.2f} | Balance: {balance:.2f}")
                    losses += 1
                    position = 0

        if position == 0:
            equity_curve.append(balance)

    winrate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0

    # Lưu biểu đồ và log
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps[:len(equity_curve)], equity_curve)
    plt.title("Futures Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Balance (USDT)")
    plt.tight_layout()
    plt.savefig("backtest_results/equity_curve.png")
    pd.DataFrame(trade_log).to_csv("backtest_results/trade_log.csv", index=False, encoding="utf-8-sig")

    return balance, winrate

# ============================
# 🔁 Train + Backtest 30 lần
# ============================
df = load_and_prepare_data(data_file)
best_balance = 0

for i in range(30):
    print(f"\n🔁 Lần train-backtest {i+1}/30")
    model, scaler = train_model(df)
    balance, winrate = backtest_strategy(model, scaler, df)

    if balance > best_balance:
        best_balance = balance
        print(f"✅ Model mới tốt hơn: {balance:.2f} USDT")

print(f"\n📊 Tốt nhất sau 30 lần: {best_balance:.2f} USDT")

# Gửi qua Telegram
os.system('zip -r models_backup.zip models/backup backtest_results')
shutil.copy('models_backup.zip', '/content/drive/MyDrive/models_futures_backup.zip')

with open('models_backup.zip', 'rb') as f:
    response = requests.post(
        f'https://api.telegram.org/bot{telegram_token}/sendDocument?chat_id={telegram_chat_id}',
        files={'document': f}
    )

if response.status_code == 200:
    print("✅ Đã gửi model backup về Telegram")
else:
    print("❌ Lỗi gửi file về Telegram:", response.text)

report = f"[Backtest Futures]\nBalance tốt nhất: {best_balance:.2f} USDT"
requests.post(
    f'https://api.telegram.org/bot{telegram_token}/sendMessage',
    data={'chat_id': telegram_chat_id, 'text': report}
)
