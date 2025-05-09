# ============================
# 🚀 Live Trading Futures Binance
# ============================
import time
import requests
import hmac
import hashlib
import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# ============================
# 🔧 Load biến môi trường
# ============================
load_dotenv()
API_KEY = os.getenv("BINANCE_FUTURES_API_KEY")
API_SECRET = os.getenv("BINANCE_FUTURES_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN_FUTURES")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID_FUTURES")
entry_price = 93450
tp_price = 95100
sl_price = 93000
atr = 130
balance_before = 2100
# ============================
# 🛠 Hàm gửi Telegram
# ============================
def send_tele(text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        print("Lỗi gửi Telegram:", e)
send_tele(
    f"🔰 Mở LONG tại {entry_price:.2f}\n"
    f"📈 TP : {tp_price:.2f}\n"
    f"🛡️ SL : {sl_price:.2f}\n"
    f"📏 ATR hiện tại: {atr:.2f}\n"
    f"💵 Balance trước lệnh: {balance_before:.2f} USDT"
)
