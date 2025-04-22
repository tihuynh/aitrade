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
# 🔧 Load biến môi trường
# ============================
print("📥 Vui lòng upload file .env trước!")
uploaded_env = files.upload()
for filename in uploaded_env:
    if filename.endswith('.env'):
        os.rename(filename, ".env")

load_dotenv(".env")
telegram_token = os.getenv("TELEGRAM_TOKEN_FUTURES")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID_FUTURES")

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
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    except Exception:
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
def train_model(df, lookback=100, model_index=0):
    feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

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
    model.fit(X, y, epochs=10, batch_size=64, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=2)])

    return model, scaler

# ============================
# 📈 Backtest chiến lược Futures
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

    macd_bullish = (df["macd"].iloc[lookback:].values - df["macd_signal"].iloc[lookback:].values) > -15
    rsi_ok = df["rsi"].iloc[lookback:].values > 40
    price_near_bottom = close_prices <= df["close"].iloc[lookback - 20:-20].rolling(20).min().values * 1.05
    adx_ok = df["adx"].iloc[lookback:].values > 20

    ai_confidence_long = predictions > close_prices * 1.001
    ai_confidence_short = predictions < close_prices * 0.999

    buy_condition = ai_confidence_long & macd_bullish & rsi_ok & price_near_bottom & adx_ok
    sell_condition = ai_confidence_short & (~macd_bullish) & (~rsi_ok) & adx_ok

    balance = initial_balance
    position = 0
    entry_price, take_profit, stop_loss = 0, 0, 0
    direction = ""
    wins, losses = 0, 0

    for i in range(len(close_prices)):
        price = close_prices[i]
        if position == 0:
            if buy_condition[i]:
                position = 1
                direction = "LONG"
                entry_price = price
            elif sell_condition[i]:
                position = 1
                direction = "SHORT"
                entry_price = price

        elif position == 1:
            if direction == "LONG":
                if price >= entry_price * 1.004:
                    profit = leverage * atr[i] * 2 / entry_price
                    balance *= 1 + profit
                    wins += 1
                    position = 0
                elif price <= entry_price * 0.996:
                    loss = leverage * atr[i] * 1.5 / entry_price
                    balance *= 1 - loss
                    losses += 1
                    position = 0
            elif direction == "SHORT":
                if price <= entry_price * 0.996:
                    profit = leverage * atr[i] * 2 / entry_price
                    balance *= 1 + profit
                    wins += 1
                    position = 0
                elif price >= entry_price * 1.004:
                    loss = leverage * atr[i] * 1.5 / entry_price
                    balance *= 1 - loss
                    losses += 1
                    position = 0

    winrate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
    return balance, winrate

# ============================
# 🔁 Train + Backtest 30 lần
# ============================
df = load_and_prepare_data(data_file)
best_balance = 0

for i in range(30):
    print(f"\n🔁 Lần train-backtest {i+1}/30")
    model, scaler = train_model(df, model_index=i)
    balance, winrate = backtest_strategy(model, scaler, df)
    joblib.dump(scaler, f'models/backup/scaler_b{int(balance)}_w{int(winrate)}.pkl')
    model_path = f"models/backup/model_b{int(balance)}_w{int(winrate)}.keras"
    model.save(model_path)

    if balance > best_balance:
        best_balance = balance
        print(f"✅ Model mới tốt hơn: {balance:.2f} USDT")

print(f"\n📊 Tốt nhất sau 30 lần: {best_balance:.2f} USDT")

# ============================
# 📦 Nén và gửi file zip
# ============================
os.system('zip -r models_futures_all.zip models/backup backtest_results')
shutil.copy('models_futures_all.zip', '/content/drive/MyDrive/models_futures_all.zip')

with open('models_futures_all.zip', 'rb') as f:
    requests.post(
        f'https://api.telegram.org/bot{telegram_token}/sendDocument?chat_id={telegram_chat_id}',
        files={'document': f}
    )

report = f"[Backtest Futures]\nBalance tốt nhất: {best_balance:.2f} USDT"
requests.post(
    f'https://api.telegram.org/bot{telegram_token}/sendMessage',
    data={'chat_id': telegram_chat_id, 'text': report}
)