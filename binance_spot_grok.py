# ============================
# 📦 Cài thư viện cần thiết
# ============================
!pip install ta tensorflow matplotlib scikit-learn requests python-dotenv ccxt joblib keras-tuner

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
import ccxt
import requests
import keras_tuner as kt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Attention, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from google.colab import drive, files
from dotenv import load_dotenv
from joblib import Parallel, delayed

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
# 🗂 Mount Google Drive
# ============================
drive.mount('/content/drive')

# ============================
# 🔧 Load biến môi trường từ .env
# ============================
env_path = "/content/drive/MyDrive/trading_bot/.env"
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    uploaded_env = files.upload()
    for filename in uploaded_env:
        if filename.endswith('.env'):
            os.rename(filename, ".env")
    load_dotenv(".env")

telegram_token = os.getenv("TELEGRAM_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

# ============================
# 📡 Hàm gửi thông báo Telegram
# ============================
def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        payload = {"chat_id": telegram_chat_id, "text": message}
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Failed to send Telegram message: {response.text}")
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

# ============================
# 📊 Load dữ liệu từ Binance API hoặc CSV
# ============================
def load_and_prepare_data(file_path=None, symbol="BTC/USDT", timeframe="15m", limit=30000):
    try:
        if file_path:
            df = pd.read_csv(file_path)
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Dữ liệu thiếu một trong các cột: {required_cols}")
        else:
            binance = ccxt.binance()
            ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        df.columns = df.columns.str.strip()
        df.reset_index(drop=True, inplace=True)
        if df["timestamp"].dtype in ["int64", "float64"]:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        df.sort_values("timestamp", inplace=True)

        # Tính toán chỉ báo kỹ thuật
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
        df["stoch"] = StochasticOscillator(df["high"], df["low"], df["close"], window=14).stoch()
        df["obv"] = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# ============================
# 🔧 Tạo các mô hình
# ============================
def create_transformer_model(input_shape, head_size=256, num_heads=4, ff_dim=4):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(2):  # 2 Transformer blocks
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        ff_output = Dense(ff_dim, activation="relu")(x)
        ff_output = Dense(input_shape[-1])(ff_output)
        x = LayerNormalization(epsilon=1e-6)(x + ff_output)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="huber")
    return model

def create_cnn_lstm_model(input_shape):
    inp = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inp)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="huber")
    return model

def create_lstm_attention_model(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = BatchNormalization()(x)
    attention = Attention()([x, x])
    x = tf.keras.layers.GlobalAveragePooling1D()(attention)
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="huber")
    return model

# ============================
# 🔧 Keras Tuner để tìm hyperparameter
# ============================
def build_model(hp, model_type="transformer", input_shape=(100, 14)):
    if model_type == "transformer":
        head_size = hp.Int('head_size', 128, 512, step=128)
        num_heads = hp.Int('num_heads', 2, 8, step=2)
        ff_dim = hp.Int('ff_dim', 4, 16, step=4)
        model = create_transformer_model(input_shape, head_size, num_heads, ff_dim)
    elif model_type == "cnn_lstm":
        filters = hp.Int('filters', 32, 128, step=32)
        lstm_units = hp.Int('lstm_units', 64, 256, step=64)
        inp = Input(shape=input_shape)
        x = Conv1D(filters=filters, kernel_size=3, activation='relu')(inp)
        x = MaxPooling1D(pool_size=2)(x)
        x = LSTM(lstm_units, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1))(x)
        x = LSTM(64)(x)
        out = Dense(1)(x)
        model = Model(inputs=inp, outputs=out)
    else:  # lstm_attention
        lstm_units = hp.Int('lstm_units', 64, 256, step=64)
        inp = Input(shape=input_shape)
        x = LSTM(lstm_units, return_sequences=True)(inp)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1))(x)
        x = LSTM(64, return_sequences=True)(x)
        x = BatchNormalization()(x)
        attention = Attention()([x, x])
        x = tf.keras.layers.GlobalAveragePooling1D()(attention)
        out = Dense(1)(x)
        model = Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')),
        loss="huber"
    )
    return model

# ============================
# 🤖 Train các mô hình AI, TP, SL
# ============================
def train_model(df, lookback=100, model_type="transformer"):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"ai15m_model_{model_type}_{timestamp}"
        feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff",
                        "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx", "stoch", "obv"]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[feature_cols])

        X = np.zeros((len(scaled) - lookback, lookback, len(feature_cols)))
        y = np.zeros((len(scaled) - lookback,))
        for i in range(lookback, len(scaled)):
            X[i - lookback] = scaled[i - lookback:i]
            y[i - lookback] = scaled[i][0]

        # Sử dụng Keras Tuner
        tuner = kt.Hyperband(
            lambda hp: build_model(hp, model_type, input_shape=(lookback, len(feature_cols))),
            objective='val_loss',
            max_epochs=50,
            factor=3,
            directory='tuner',
            project_name=model_name
        )
        tuner.search(X, y, epochs=50, validation_split=0.2, callbacks=[EarlyStopping(patience=5)])
        model = tuner.get_best_models(num_models=1)[0]

        model.save(f"models/{model_name}.keras")
        joblib.dump(scaler, f"models/{model_name}.pkl")
        return model, scaler, model_name
    except Exception as e:
        print(f"Error training AI model: {e}")
        return None, None, None

def train_tp_model(df, lookback=100):
    try:
        feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff",
                        "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx", "stoch", "obv"]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_cols])

        X = np.zeros((len(scaled_data) - lookback - 1, lookback, len(feature_cols)))
        y_tp = np.zeros((len(scaled_data) - lookback - 1,))
        for i in range(lookback, len(scaled_data) - 1):
            X[i - lookback] = scaled_data[i - lookback:i]
            close_now = df["close"].iloc[i]
            close_next = df["close"].iloc[i + 1]
            atr_now = df["atr"].iloc[i]
            y_tp[i - lookback] = min(close_next, close_now + atr_now * 4) / close_now

        model = Sequential([
            Input(shape=(lookback, len(feature_cols))),
            LSTM(32, return_sequences=True),
            BatchNormalization(),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="huber")
        model.fit(X, y_tp, epochs=20, batch_size=32, validation_split=0.2, verbose=0,
                  callbacks=[EarlyStopping(patience=5)])
        return model, scaler
    except Exception as e:
        print(f"Error training TP model: {e}")
        return None, None

def train_sl_model(df, lookback=100):
    try:
        feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff",
                        "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx", "stoch", "obv"]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_cols])

        X = np.zeros((len(scaled_data) - lookback - 1, lookback, len(feature_cols)))
        y_sl = np.zeros((len(scaled_data) - lookback - 1,))
        for i in range(lookback, len(scaled_data) - 1):
            X[i - lookback] = scaled_data[i - lookback:i]
            close_now = df["close"].iloc[i]
            close_next = df["close"].iloc[i + 1]
            atr_now = df["atr"].iloc[i]
            y_sl[i - lookback] = max(close_next, close_now - atr_now * 1.5) / close_now

        model = Sequential([
            Input(shape=(lookback, len(feature_cols))),
            LSTM(32, return_sequences=True),
            BatchNormalization(),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="huber")
        model.fit(X, y_sl, epochs=20, batch_size=32, validation_split=0.2, verbose=0,
                  callbacks=[EarlyStopping(patience=5)])
        return model, scaler
    except Exception as e:
        print(f"Error training SL model: {e}")
        return None, None

# ============================
# 📈 Backtest kết hợp AI + TP + SL
# ============================
def backtest(df, ai_model, ai_scaler, tp_model, tp_scaler, sl_model, sl_scaler, lookback=100):
    try:
        feature_cols = ["close", "sma", "ema", "macd", "macd_signal", "macd_diff",
                        "rsi", "bb_bbm", "bb_bbh", "bb_bbl", "atr", "adx", "stoch", "obv"]
        fee_rate = 0.001
        risk_per_trade = 0.02

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
                trailing_sl = max(trailing_sl, row["close"] - row["atr"] * 1.5)
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
# 🔁 Train và backtest song song
# ============================
def run_single_train_backtest(df, lookback=100, model_type="transformer"):
    try:
        ai_model, ai_scaler, model_name = train_model(df, lookback, model_type)
        if ai_model is None:
            return 0, 0, 0, None, None
        tp_model, tp_scaler = train_tp_model(df, lookback)
        if tp_model is None:
            return 0, 0, 0, None, None
        sl_model, sl_scaler = train_sl_model(df, lookback)
        if sl_model is None:
            return 0, 0, 0, None, None
        balance, win_rate, profit_factor = backtest(df, ai_model, ai_scaler, tp_model, tp_scaler, sl_model, sl_scaler, lookback)
        return balance, win_rate, profit_factor, model_name, ai_scaler
    except Exception as e:
        print(f"Error in single train-backtest: {e}")
        return 0, 0, 0, None, None

# ============================
# 🚀 Main execution
# ============================
def main():
    try:
        # Load dữ liệu từ Binance API hoặc file CSV
        data_file = None
        try:
            uploaded = files.upload()
            data_file = list(uploaded.keys())[0]
        except:
            print("No CSV file uploaded, fetching data from Binance API")
        df = load_and_prepare_data(data_file, limit=30000)
        if df is None:
            raise ValueError("Failed to load data")

        # Chia dữ liệu train/test
        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        # Thử cả hai mô hình: Transformer và CNN-LSTM
        model_types = ["transformer", "cnn_lstm"]
        best_balance = 0
        best_model_name = None
        best_scaler = None
        best_model_type = None

        for model_type in model_types:
            print(f"\nTraining and backtesting with {model_type} model")
            results = Parallel(n_jobs=-1)(delayed(run_single_train_backtest)(train_df, model_type=model_type) for _ in range(15))  # 15 lần cho mỗi mô hình
            for balance, win_rate, profit_factor, model_name, scaler in results:
                if balance > best_balance:
                    best_balance = balance
                    best_model_name = model_name
                    best_scaler = scaler
                    best_model_type = model_type

        # Backtest mô hình tốt nhất trên tập test
        if best_model_name and best_scaler:
            ai_model = tf.keras.models.load_model(f"models/{best_model_name}.keras")
            tp_model, tp_scaler = train_tp_model(test_df)
            sl_model, sl_scaler = train_sl_model(test_df)
            final_balance, final_win_rate, final_profit_factor = backtest(test_df, ai_model, best_scaler, tp_model, tp_scaler, sl_model, sl_scaler)
            print(f"\n📊 Kết quả tốt nhất trên tập test (model: {best_model_type}): Balance = {final_balance:.2f} USDT, Win Rate = {final_win_rate:.2%}, Profit Factor = {final_profit_factor:.2f}")

            # Lưu mô hình tốt nhất
            ai_model.save("models/best_model.keras")
            joblib.dump(best_scaler, "models/best_scaler.pkl")

            # Gửi thông báo Telegram
            send_telegram_message(f"Backtest completed. Best Model: {best_model_type}, Balance: {final_balance:.2f} USDT, Win Rate: {final_win_rate:.2%}, Profit Factor: {final_profit_factor:.2f}")
        else:
            print("No valid model found")
    except Exception as e:
        print(f"Error in main execution: {e}")
        send_telegram_message(f"Error in backtest: {e}")

if __name__ == "__main__":
    main()