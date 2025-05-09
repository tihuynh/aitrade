# ============================
# 📦 Cài thư viện cần thiết
# ============================
!pip install ta tensorflow matplotlib scikit-learn requests python-dotenv paramiko

# ============================
# 🚀 Train & Backtest LSTM Futures AI (30 models) - Colab Version
# ============================

import os
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paramiko
import time
import requests
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from dotenv import load_dotenv
import joblib  # nếu muốn save scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# ============================
# 🔗 Mount Google Drive
# ============================
from google.colab import drive
drive.mount('/content/drive')

# ============================
# 🔧 Setup Path
# ============================
RAW_FILE = "/content/drive/MyDrive/data/btc_futures_raw.csv"
OUTPUT_FILE = "/content/drive/MyDrive/data/btc_futures_15m.csv"

# Tạo thư mục nếu chưa có
os.makedirs("/content/drive/MyDrive/data", exist_ok=True)

# ============================
# 📥 Fetch Futures data từ Binance
# ============================
def fetch_binance_futures_klines(symbol="BTCUSDT", interval="15m", total_limit=30000):
    url = "https://fapi.binance.com/fapi/v1/klines"
    limit = 1000
    end_time = int(time.time() * 1000)
    all_data = []

    while len(all_data) < total_limit:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, total_limit - len(all_data)),
            "endTime": end_time
        }
        res = requests.get(url, params=params)
        try:
            data = res.json()
            # Kiểm tra nếu data không phải list → API lỗi
            if not isinstance(data, list):
                print("❌ API trả về lỗi:", data)
                break
        except Exception as e:
            print("❌ Lỗi khi parse JSON:", e)
            break

        if not data:
            break

        all_data = data + all_data  # 🛠 data là list OK rồi
        end_time = data[0][0] - 1
        time.sleep(0.2)

    # Convert to DataFrame nếu có dữ liệu
    if not all_data:
        raise ValueError("Không có dữ liệu trả về từ Binance")

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    float_cols = ["open", "high", "low", "close", "volume"]
    df[float_cols] = df[float_cols].astype(float)

    df = df[["timestamp"] + float_cols]
    df.to_csv("/content/drive/MyDrive/data/btc_futures_raw.csv", index=False)
    print("✅ Đã lưu raw data thành công")

    time.sleep(1)

fetch_binance_futures_klines()

# ============================
# 📂 Load raw data
# ============================
print("👀 Loading raw data...")

df = pd.read_csv(RAW_FILE)

# timestamp đã là datetime → chỉ cần convert utc
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Đảm bảo float
df = df.astype({"open": float, "high": float, "low": float, "close": float})

# ============================
# 📊 Tính các chỉ báo kỹ thuật
# ============================
print("📊 Calculating technical indicators...")

df["sma"] = SMAIndicator(df["close"], window=14).sma_indicator()
df["ema"] = EMAIndicator(df["close"], window=14).ema_indicator()
macd = MACD(df["close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_diff"] = macd.macd_diff()
df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()

df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

bb = BollingerBands(df["close"], window=20)
df["bb_bbm"] = bb.bollinger_mavg()
df["bb_bbh"] = bb.bollinger_hband()
df["bb_bbl"] = bb.bollinger_lband()
df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

# ============================
# 🗑️ Clean & Save
# ============================
print("🗑️ Cleaning data...")
df.dropna(inplace=True)

final_cols = ["timestamp", "close", "sma", "ema", "macd", "macd_signal", "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]
df[final_cols].to_csv(OUTPUT_FILE, index=False)

print(f"🚀 Data prepared and saved to {OUTPUT_FILE}")

# Tạo thư mục nếu chưa có
os.makedirs("/content/drive/MyDrive/AI_Models_Futures", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ============================
# 🔧 Setup
# ============================
LOOKBACK = 100
EPOCHS = 100
BATCH_SIZE = 32
REPEATS = 5

# ============================
# 📂 Load Data
# ============================
print("👀 Loading data...")
df = pd.read_csv("/content/drive/MyDrive/data/btc_futures_15m.csv")
feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff", "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx"]

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[feature_cols])

X, y = [], []
for i in range(LOOKBACK, len(df_scaled)):
    X.append(df_scaled[i-LOOKBACK:i])
    y.append(df_scaled[i, 0])

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ============================
# 📊 Equity Curve Plot
# ============================
def plot_equity_curve(balance_list, model_idx):
    plt.figure()
    plt.plot(balance_list)
    plt.title(f"Equity Curve Model {model_idx}")
    plt.xlabel("Trade")
    plt.ylabel("Balance")
    plt.grid()
    plt.savefig(f"logs/equity_curve_{model_idx}.png")
    plt.close()

# ============================
# 🧵 Train + Backtest Loop
# ============================
best_winrate = 0
best_balance = 0
best_model_path = ""

log = []

for model_idx in range(REPEATS):
    print(f"\n👉 Training model {model_idx+1}/{REPEATS}")

    random_seed = random.randint(0, 10000)
    np.random.seed(random_seed)
    random.seed(random_seed)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, len(feature_cols))),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stop])

    model_path = f"models/model_{model_idx}.keras"
    model.save(model_path)

    # Predict
    preds = model.predict(X_test, verbose=0)
    dummy = np.zeros((len(preds), len(feature_cols)))
    dummy[:, 0] = preds.flatten()
    preds_real = scaler.inverse_transform(dummy)[:, 0]

    y_real = scaler.inverse_transform(np.hstack((y_test.reshape(-1,1), np.zeros((len(y_test), len(feature_cols)-1)))))[:, 0]

    # Backtest logic
    balance = 1000
    wins = 0
    losses = 0
    trades = 0
    balance_list = [balance]

    for pred_price, real_price in zip(preds_real, y_real):
        if pred_price > real_price * 1.0005:  # giảm ngưỡng vào lệnh Long
            entry = real_price
            tp = entry * 1.004
            sl = entry * 0.996
            exit_price = real_price
            if exit_price >= tp:
                balance *= 1.004
                wins += 1
            elif exit_price <= sl:
                balance *= 0.996
                losses += 1
            else:
                balance *= (exit_price / entry)
            trades += 1
            balance_list.append(balance)

        elif pred_price < real_price * 0.9995:  # giảm ngưỡng vào lệnh Short
            entry = real_price
            tp = entry * 0.996
            sl = entry * 1.004
            exit_price = real_price
            if exit_price <= tp:
                balance *= 1.004
                wins += 1
            elif exit_price >= sl:
                balance *= 0.996
                losses += 1
            else:
                balance *= (entry / exit_price)
            trades += 1
            balance_list.append(balance)

    winrate = (wins/trades)*100 if trades > 0 else 0
    print(f"Model {model_idx}: Trades={trades}, Win={wins}, Loss={losses}, Winrate={winrate:.2f}%, Final Balance={balance:.2f}")

    plot_equity_curve(balance_list, model_idx)

    log.append((model_idx, trades, wins, losses, winrate, balance))

    if (winrate > best_winrate) or (winrate == best_winrate and balance > best_balance):
        best_winrate = winrate
        best_balance = balance
        best_model_path = model_path

# ============================
# 📚 Save Best Model
# ============================
if best_model_path:
    model = load_model(best_model_path)
    model.save("models/best_model.keras")
    print(f"\n🚀 Best model saved: {best_model_path} with Winrate={best_winrate:.2f}%, Balance={best_balance:.2f}")

# ============================
# 📄 Save training log
# ============================
log_df = pd.DataFrame(log, columns=["Model", "Trades", "Wins", "Losses", "Winrate", "Final Balance"])
log_df.to_csv("logs/training_log.csv", index=False)
print("\n📈 Training completed!")




# ============================
# 🚀 Upload best_model.keras + scaler.pkl từ Colab lên VPS - Đọc từ .env
# ============================



# ============================
# 🔧 Load biến môi trường từ .env
# ============================

# Đọc file .env lưu trong Google Drive
load_dotenv("/content/drive/MyDrive/data/.env")

vps_ip = os.getenv("VPS_IP")
vps_username = os.getenv("VPS_USERNAME")
vps_password = os.getenv("VPS_PASSWORD")
remote_folder = "/root/ai-trading-bot/models_backup/"

local_best_model = "/content/drive/MyDrive/AI_Models_Futures/best_model.keras"
local_scaler = "/content/drive/MyDrive/AI_Models_Futures/scaler.pkl"

# ============================
# 🔧 Hàm tạo kết nối SSH
# ============================
def create_ssh_client(server, port, user, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, port, user, password)
    return ssh

# ============================
# 📦 Upload file
# ============================
try:
    ssh = create_ssh_client(vps_ip, 22, vps_username, vps_password)
    sftp = ssh.open_sftp()

    # Tạo thư mục trên VPS nếu chưa có
    try:
        sftp.chdir(remote_folder)
    except IOError:
        sftp.mkdir(remote_folder)
        sftp.chdir(remote_folder)

    # Upload best_model
    print("🚀 Uploading best_model.keras...")
    sftp.put(local_best_model, remote_folder + "best_model.keras")

    # Upload scaler
    print("🚀 Uploading scaler.pkl...")
    sftp.put(local_scaler, remote_folder + "scaler.pkl")

    sftp.close()
    ssh.close()
    print("✅ Upload thành công lên VPS!")
except Exception as e:
    print(f"❌ Lỗi khi upload: {e}")
