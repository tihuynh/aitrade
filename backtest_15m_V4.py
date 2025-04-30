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
    df.reset_index(inplace=True)  # 👉 Giải phóng timestamp ra khỏi index nếu có
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

    # Tạo tên theo timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"ai15m_model_{timestamp}"
    model_path = f"models/{model_name}.keras"
    scaler_path = f"models/{model_name}.pkl"
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

    # Lưu model và scaler
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"✅ Đã lưu model tại: {model_path}")
    print(f"✅ Đã lưu scaler tại: {scaler_path}")

    return model, scaler, model_name

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

    volume = df["volume"].iloc[lookback:].values
    volume_ma10 = df["volume"].rolling(10).mean().iloc[lookback:].values
    volume_breakout = volume > volume_ma10

    buy_condition = ai_confidence & macd_bullish & rsi_ok & price_near_bottom & adx_ok & volume_breakout

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
        predicted_close = predictions[i]
        if position == 0 and buy_condition[i]:
            position = 1
            buy_price = price
            mode = "fixed"
            if mode == "fixed":
                take_profit = round(buy_price * 1.004, 2)
                stop_loss = round(buy_price * 0.996, 2)
            elif mode == "adaptive":
                take_profit = min(predicted_close, buy_price + atr[i] * 4)
                stop_loss = buy_price - atr[i] * 1.5
            trade_log.append(f"Mua tại {buy_price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")

        elif position == 1:
            if price >= take_profit:
                gain_pct = (take_profit - buy_price) / buy_price
                gain_usdt = balance * gain_pct
                old_balance = balance
                balance += gain_usdt
                trade_log.append({
                    "timestamp": timestamp,
                    "action": "TP",
                    "buy_price": round(buy_price, 2),
                    "sell_price": round(price, 2),
                    "gain_pct": round(gain_pct * 100, 2),
                    "gain_usdt": round(gain_usdt, 2),
                    "balance_before": round(old_balance, 2),
                    "balance_after": round(balance, 2)
                })
                win_count += 1
                position = 0
            elif price <= stop_loss:
                loss_pct = (buy_price - stop_loss) / buy_price
                loss_usdt = balance * loss_pct
                old_balance = balance
                balance -= loss_usdt
                trade_log.append({
                    "timestamp": timestamp,
                    "action": "SL",
                    "buy_price": round(buy_price, 2),
                    "sell_price": round(price, 2),
                    "gain_pct": round(-loss_pct * 100, 2),
                    "gain_usdt": round(-loss_usdt, 2),
                    "balance_before": round(old_balance, 2),
                    "balance_after": round(balance, 2)
                })
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

    if balance >= 5500:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"models/backup/balance{int(balance)}_{timestamp}"
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy(f"models/{model_name}.keras", os.path.join(backup_dir, "model.keras"))
        shutil.copy(f"models/{model_name}.pkl", os.path.join(backup_dir, "scaler.pkl"))

    return balance, winrate

# ============================
# 🏁 Chạy train + backtest 30 lần và auto gửi báo cáo
# ============================

df = load_and_prepare_data(data_file)
best_balance = 0

for i in range(30):
    print(f"\nVòng train-backtest {i + 1}/30")
    model, scaler, model_name = train_model(df, lookback=100)
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
