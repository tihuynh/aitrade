import threading
import websocket
import json

# Websocket Binance & kucoin
binance_ws = "wss://stream.binance.com:9443/ws/btcusdt@trade"
okx_ws = "wss://ws.okx.com:8443/ws/v5/public"
bybit_ws = "wss://stream.bybit.com/v5/public/spot"

binance_price = 0
okx_price = 0
bybit_price = 0


def check_arbitrage():
    prices = {
        "Binance": binance_price,
        "OKX": okx_price,
        "Bybit": bybit_price
    }
    prices = {k: v for k, v in prices.items() if v > 0}
    sorted_prices = sorted(prices.items(), key=lambda x: x[1])
    if len(sorted_prices) >= 2:
        cheapest = sorted_prices[0]
        most_expensive = sorted_prices[-1]
        diff = most_expensive[1] - cheapest[1]
        print(f" Giá thấp nhất: {cheapest}| Giá cao nhất: {most_expensive}| Chênh lệch: {diff} USDT")
        if diff > 50:
            print(f"cơ hội arbitrage mua tại{cheapest[0]}, bán tại {most_expensive[0]}")


# Lấy giá từ Binance
def on_message_binance(ws, message):
    global binance_price
    data = json.loads(message)
    binance_price = float(data['p'])
    print(f"Binance BTC price {binance_price} USDT")
    check_arbitrage()


def on_open_okx(ws):
    subscribe_message = {
        "op": "subscribe",
        "args": [{"channel": "tickers", "instId": "BTC-USDT"}]
    }
    ws.send(json.dumps(subscribe_message))
    print("đã gửi yêu cầu đăng ký giá BTC-USDT từ Okx")


# Lấy giá từ OKX
def on_message_okx(ws, message):
    global okx_price
    data = json.loads(message)
    if "data" in data and len(data["data"]) > 0 and "last" in data["data"][0]:
        okx_price = float(data['data'][0]['last'])
        print(f"OKX BTC price {okx_price} USDT")
    check_arbitrage()


def on_open_bybit(ws):
    subsribe_message = {
        "op": "subscribe",
        "args": ["tickers.BTCUSDT"]
    }
    ws.send(json.dumps(subsribe_message))
    print("gửi yêu cầu đăng ký giá từ bybit")


# Lấy giá từ Bybit
def on_message_bybit(ws, message):
    global bybit_price
    data = json.loads(message)
    if "data" in data and "lastPrice" in data["data"]:
        bybit_price = float(data["data"]["lastPrice"])
        print(f"Bybit BTC price {bybit_price} USDT")
    check_arbitrage()


def run_binance():
    try:
        ws = websocket.WebSocketApp(binance_ws, on_message=on_message_binance)
        ws.run_forever()
    except Exception as e:
        print(f"Lỗi websocket binance: {e}")


def run_okx():
    print("🚀 Đang kết nối OKX WebSocket...")
    try:
        ws = websocket.WebSocketApp(okx_ws, on_open=on_open_okx, on_message=on_message_okx)
        ws.run_forever()
    except Exception as e:
        print(f"Lỗi websocket OKX: {e}")


def run_bybit():
    try:
        print("🚀 Đang kết nối bybit WebSocket...")
        ws = websocket.WebSocketApp(bybit_ws, on_open=on_open_bybit, on_message=on_message_bybit)
        ws.run_forever()
    except Exception as e:
        print(f"Lỗi websocket Bybit: {e}")


# chạy mỗi websocket trong 1 thread riêng
binance_thread = threading.Thread(target=run_binance, daemon=True)
okx_thread = threading.Thread(target=run_okx, daemon=True)
bybit_thread = threading.Thread(target=run_bybit, daemon=True)
binance_thread.start()
okx_thread.start()
bybit_thread.start()

# Giữ chương trình chạy mãi để thread không kết thúc
binance_thread.join()
okx_thread.join()
bybit_thread.join()
