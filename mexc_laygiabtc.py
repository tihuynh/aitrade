import websocket
import json

def on_open(ws):
    subscribe_message = {
        "method": "SUBSCRIPTION",
        "params": ["spot@public.deals.v3.api@BTCUSDT"],
        "id": 1
    }
    ws.send(json.dumps(subscribe_message))
    print("Đã gửi yêu cầu đăng ký nhận giao dịch BTC/USDT từ MEXC.")

def on_message(ws, message):
    data = json.loads(message)
    print("Nhận được dữ liệu:", data)
    # Kiểm tra nếu "d" có tồn tại và có danh sách "deals"
    if "d" in data and "deals" in data["d"] and len(data["d"]["deals"]) > 0:
        btc_price = data["d"]["deals"][0]["p"]  # Lấy giá từ deal đầu tiên
        print(f"✅ BTC Price MEXC: {btc_price} USDT")
    else:
        print("❌ Không tìm thấy trường 'p' trong dữ liệu MEXC!")

def on_error(ws, error):
    print("Đã xảy ra lỗi:", error)

def on_close(ws):
    print("Kết nối đã đóng.")

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("wss://wbs.mexc.com/ws",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()