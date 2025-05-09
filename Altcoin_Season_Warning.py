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
        error_msg = f"❌ Lỗi trong make_decision: {e}"
        send_telegram(error_msg)
        print(error_msg)
        return None


# Hàm đọc dữ liệu hôm qua từ file log (nếu có)
def load_yesterday_data():
    try:
        if not os.path.exists("logs/altseason_log.csv"):
            return None  # Không có file -> return luôn
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
    except Exception as e:
        error_msg = f"❌ Lỗi trong make_decision: {e}"
        send_telegram(error_msg)
        print(error_msg)
        return None

# Hàm lưu dữ liệu hôm nay vào file log
def save_today_data(today_data, score):
    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/altseason_log.csv", "a") as f:
            line = f"{datetime.date.today()},{today_data['DXY']},{today_data['BTC.D']},{today_data['ETH.D']},{today_data['OTHERS.D']},{today_data['TOTAL2']},{today_data['TOTAL3']},{score}\n"
            f.write(line)
    except Exception as e:
        error_msg = f"❌ Lỗi ghi log hôm nay: {e}"
        send_telegram(error_msg)
        print(error_msg)

# Hàm phân tích xu hướng

def analyze_trend(today, yesterday):
    try:
        def emoji_and_note(key, change):
            is_positive = {
                "DXY": change < 0,
                "BTC.D": change < 0,
                "ETH.D": change > 0,
                "OTHERS.D": change > 0,
                "TOTAL2": change > 0,
                "TOTAL3": change > 0
            }
            note = "✅ (tốt cho Altcoin)" if is_positive[key] else "⚠️ (xấu cho Altcoin)"
            emoji = "✅" if is_positive[key] else "⚠️"
            return emoji, note

        changes = {
            "DXY": today["DXY"] - yesterday["DXY"],
            "BTC.D": today["BTC.D"] - yesterday["BTC.D"],
            "ETH.D": today["ETH.D"] - yesterday["ETH.D"],
            "OTHERS.D": today["OTHERS.D"] - yesterday["OTHERS.D"],
            "TOTAL2": (today["TOTAL2"] - yesterday["TOTAL2"]) / yesterday["TOTAL2"] * 100,
            "TOTAL3": (today["TOTAL3"] - yesterday["TOTAL3"]) / yesterday["TOTAL3"] * 100,
        }

        message = "*📊 Dự báo xu hướng Crypto hôm nay:*\n\n"
        for key in ["DXY", "BTC.D", "ETH.D", "OTHERS.D", "TOTAL2", "TOTAL3"]:
            emoji, note = emoji_and_note(key, changes[key])
            unit = "%" if key.endswith("D") else "%"
            value = f"{today[key]:.2f}"
            delta = f"{changes[key]:+.2f}{unit}"
            label = key.replace(".", " Dominance").replace("TOTAL2", "TOTAL2").replace("TOTAL3", "TOTAL3")
            message += f"{emoji} {label}: {value}{unit} ({delta}) {note}\n"

        message += "\n"

        # Chấm điểm xu hướng Altcoin Season
        score = 0
        if changes['DXY'] < 0: score += 1
        if changes['BTC.D'] < 0: score += 1
        if changes['ETH.D'] > 0: score += 1
        if changes['OTHERS.D'] > 0: score += 1
        if changes['TOTAL2'] > 0: score += 1
        if changes['TOTAL3'] > 0: score += 1

        if score >= 5:
            message += "🔥 *Altcoin Season mạnh!* Có thể giải ngân mạnh vào Altcoin tiềm năng.\n"
        elif score >= 3:
            message += "✨ *Tín hiệu tích cực nhẹ cho Altcoin.* Cân nhắc giải ngân 10–30%.\n"
        else:
            message += "💤 *Thị trường chưa rõ ràng.* Nên quan sát thêm, chưa vội giải ngân.\n"

        message += f"\n*Score Altcoin Season hôm nay: {score}/6*"

        return message

    except Exception as e:
        error_msg = f"❌ Lỗi trong make_decision: {e}"
        send_telegram(error_msg)
        print(error_msg)
def emoji_and_note(key, change):
    is_good = (
        (key == "DXY" and change < 0) or
        (key == "BTC.D" and change < 0) or
        (key == "ETH.D" and change > 0) or
        (key == "OTHERS.D" and change > 0) or
        (key == "TOTAL2" and change > 0) or
        (key == "TOTAL3" and change > 0)
    )
    emoji = "✅" if is_good else "⚠️"
    note = "✅ (tốt cho Altcoin)" if is_good else "⚠️ (xấu cho Altcoin)"
    return emoji, note

def calculate_score(today, yesterday):
    try:
        score = 0
        if today["DXY"] < yesterday["DXY"]: score += 1
        if today["BTC.D"] < yesterday["BTC.D"]: score += 1
        if today["ETH.D"] > yesterday["ETH.D"]: score += 1
        if today["OTHERS.D"] > yesterday["OTHERS.D"]: score += 1
        if today["TOTAL2"] > yesterday["TOTAL2"]: score += 1
        if today["TOTAL3"] > yesterday["TOTAL3"]: score += 1
        return score
    except Exception as e:
        error_msg = f"❌ Lỗi trong make_decision: {e}"
        send_telegram(error_msg)
        print(error_msg)

# =============================
# Chạy bot mỗi sáng
# =============================
if __name__ == '__main__':
    try:
        today_data = get_index_data()
        if today_data is None:
            send_telegram("❌ Lỗi: Không thể lấy dữ liệu thị trường hôm nay. API có thể đang lỗi.")
            exit(1)
        yesterday_data = load_yesterday_data()

        if yesterday_data:
            report = analyze_trend(today_data, yesterday_data)
            send_telegram(report)
        else:
            send_telegram("📌 Bot khởi tạo dữ liệu thị trường. Báo cáo đầy đủ sẽ bắt đầu từ ngày mai.")

        score = calculate_score(today_data, yesterday_data) if yesterday_data else 0
        save_today_data(today_data, score)

    except Exception as e:
        error_msg = f"❌ Lỗi trong make_decision: {e}"
        send_telegram(error_msg)
        print(error_msg)
