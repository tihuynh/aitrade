# ======================================
# 📊 Bot Altcoin Season - Phân tích xu hướng thị trường + Gửi Telegram
# ======================================

import requests
import datetime
import time
import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN_ALTCOIN_SEASON")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID_ALTCOIN_SEASON")

# Hàm gửi báo cáo qua Telegram
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Lỗi gửi Telegram: {e}")

# Hàm lấy dữ liệu từ TradingView hoặc nguồn thay thế
def get_index_data():
    try:
        # DXY từ Yahoo Finance
        import yfinance as yf
        dxy = yf.Ticker("DX-Y.NYB")
        hist = dxy.history(period="1d")
        dxy_value = hist['Close'].iloc[-1] if not hist.empty else None

        # Các chỉ số crypto từ CoinGecko
        response = requests.get("https://api.coingecko.com/api/v3/global").json()
        btc_d = response['data']['market_cap_percentage']['btc']
        eth_d = response['data']['market_cap_percentage']['eth']
        total_market_cap = response['data']['total_market_cap']['usd']
        total2 = total_market_cap * (100 - btc_d) / 100
        total3 = total2 * (100 - eth_d) / 100
        others_d = 100 - btc_d - eth_d

        return {
            "DXY": round(dxy_value, 3),
            "BTC.D": round(btc_d, 2),
            "ETH.D": round(eth_d, 2),
            "OTHERS.D": round(others_d, 2),
            "TOTAL2": round(total2 / 1e9, 2),   # Bn USD
            "TOTAL3": round(total3 / 1e9, 2)    # Bn USD
        }

    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu chỉ số: {e}")
        return None


# Hàm đọc dữ liệu hôm qua từ file log (nếu có)
def load_yesterday_data():
    try:
        with open("logs/altseason_log.csv", "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None
            last_line = lines[-1].strip().split(',')
            return {
                "date": last_line[0],
                "DXY": float(last_line[1]),
                "BTC.D": float(last_line[2]),
                "ETH.D": float(last_line[3]),
                "OTHERS.D": float(last_line[4]),
                "TOTAL2": float(last_line[5]),
                "TOTAL3": float(last_line[6])
            }
    except:
        return None

# Hàm lưu dữ liệu hôm nay vào file log
def save_today_data(today_data):
    with open("logs/altseason_log.csv", "a") as f:
        line = f"{datetime.date.today()},{today_data['DXY']},{today_data['BTC.D']},{today_data['ETH.D']},{today_data['OTHERS.D']},{today_data['TOTAL2']},{today_data['TOTAL3']}\n"
        f.write(line)

# Hàm phân tích xu hướng

def analyze_trend(today, yesterday):
    def emoji(change):
        return "✅" if change < 0 else "⚠️"

    changes = {
        "DXY": today["DXY"] - yesterday["DXY"],
        "BTC.D": today["BTC.D"] - yesterday["BTC.D"],
        "ETH.D": today["ETH.D"] - yesterday["ETH.D"],
        "OTHERS.D": today["OTHERS.D"] - yesterday["OTHERS.D"],
        "TOTAL2": today["TOTAL2"] - yesterday["TOTAL2"],
        "TOTAL3": today["TOTAL3"] - yesterday["TOTAL3"]
    }

    message = "*📊 Dự báo xu hướng Crypto hôm nay:*\n\n"
    message += f"{emoji(changes['DXY'])} DXY: {today['DXY']} ({changes['DXY']:+.2f})\n"
    message += f"{emoji(-changes['BTC.D'])} BTC Dominance: {today['BTC.D']} ({changes['BTC.D']:+.2f})\n"
    message += f"{emoji(-changes['ETH.D'])} ETH Dominance: {today['ETH.D']} ({changes['ETH.D']:+.2f})\n"
    message += f"{emoji(-changes['OTHERS.D'])} Others Dominance: {today['OTHERS.D']} ({changes['OTHERS.D']:+.2f})\n"
    message += f"{emoji(-changes['TOTAL2'])} TOTAL2: {today['TOTAL2']}B ({changes['TOTAL2']:+.2f})\n"
    message += f"{emoji(-changes['TOTAL3'])} TOTAL3: {today['TOTAL3']}B ({changes['TOTAL3']:+.2f})\n\n"

    score = 0
    if changes['DXY'] < 0: score += 1
    if changes['BTC.D'] < 0: score += 1
    if changes['ETH.D'] > 0: score += 1
    if changes['OTHERS.D'] > 0: score += 1
    if changes['TOTAL2'] > 0: score += 1
    if changes['TOTAL3'] > 0: score += 1

    if score >= 4:
        message += "🎯 *Tín hiệu tích cực cho Altcoin.* Có thể giải ngân 20–30% USDT vào Altcoin mạnh."
    else:
        message += "🔎 *Thị trường đang phân vân.* Quan sát thêm trước khi giải ngân lớn."

    return message

# =============================
# Chạy bot mỗi sáng
# =============================
if __name__ == '__main__':
    today_data = get_index_data()
    yesterday_data = load_yesterday_data()

    if yesterday_data:
        report = analyze_trend(today_data, yesterday_data)
        send_telegram(report)
    else:
        send_telegram("📌 Bot khởi tạo dữ liệu thị trường. Báo cáo đầy đủ sẽ bắt đầu từ ngày mai.")

    save_today_data(today_data)
