import functools
import threading
import requests
import websocket
import json
import logging

# cấu hình logging
logging.basicConfig(filename="log.text", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# API lấy danh sách top 50 Altcoin từ CoinGecko
COINGECKO_API = ("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=50"
                 "&page=1&sparkline=false")

# Lấy 50 coin có volume thấp nhất (spread rộng hơn) COINGECKO_API =
# "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=volume_asc&per_page=50&page=1&sparkline=false"

# HOẶC (Lấy coin biến động mạnh nhất 24h) COINGECKO_API =
# "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=price_change_percentage_24h_desc&per_page=50
# &page=1&sparkline=false"

okx_ws = "wss://ws.okx.com:8443/ws/v5/public"
bybit_ws = "wss://stream.bybit.com/v5/public/spot"

token_prices = {
    "Binance": {},
    "OKX": {},
    "Bybit": {}
}


def catch_exception(func):
    """Decorator để tự động bắt lỗi trong mọi hàm"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Lỗi trong {func.__name__}: {str(e)}", exc_info=True)
            print(f"Lỗi trong {func.__name__} : {e}")
            return None

    return wrapper


@catch_exception
def get_top10_altcoin():
    response = requests.get(COINGECKO_API)
    coins = response.json()
    return [coin['symbol'].upper() for coin in coins]


altcoins = get_top10_altcoin()
print("danh sách 50 top altcoins:", altcoins)
if not altcoins or len(altcoins) == 0:
    print("Lỗi: Không lấy được danh sách Altcoin!")
    exit()  # Dừng chương trình nếu không có dữ liệu


@catch_exception
def check_arbitrage():
    print("check arbitrage")
    for token in altcoins:
        prices = {exchange: float(token_prices[exchange].get(token, 0)) for exchange in token_prices if
                  token_prices[exchange].get(token, 0)}
        prices = {k: v for k, v in prices.items() if v > 0}
        sorted_prices = sorted(prices.items(), key=lambda x: x[1])
        if len(sorted_prices) >= 2:
            cheapest = sorted_prices[0]
            most_expensive = sorted_prices[-1]
            diff = most_expensive[1] - cheapest[1]
            percentage_diff = (diff / cheapest[1]) * 100
            print(
                f" Coin {token}| Giá thấp nhất: {cheapest}| Giá cao nhất: {most_expensive}| Chênh lệch: {diff} USDT| {percentage_diff} %")
            if percentage_diff > 1:
                print(f"cơ hội arbitrage mua tại {cheapest[0]}, bán tại {most_expensive[0]}")


# Lấy giá từ Binance
binance_ws = [f"wss://stream.binance.com:9443/ws/{symbol.lower()}usdt@trade" for symbol in altcoins]


@catch_exception
def on_message_binance(ws, message):
    print("on_message_binance")
    data = json.loads(message)
    symbol = data['s'].replace('USDT', '')
    print(f"Binance {symbol}")
    if symbol in altcoins:
        print("symbol in altcoins Binance")
        token_prices["Binance"][symbol] = float(data['p'])
        check_arbitrage()


@catch_exception
def on_open_okx(ws):
    print("on_open_okx")
    subscribe_message = {
        "op": "subscribe",
        "args": [{"channel": "tickers", "instId": f"{symbol}-USDT"} for symbol in altcoins]
    }
    ws.send(json.dumps(subscribe_message))
    print("đã gửi yêu cầu đăng ký giá BTC-USDT từ Okx")


# Lấy giá từ OKX
@catch_exception
def on_message_okx(ws, message):
    data = json.loads(message)
    if "data" in data and len(data["data"]) > 0 and "last" in data["data"][0]:
        for entry in data["data"]:
            symbol = entry["instId"].replace('-USDT', '')
            print(symbol)
            if symbol in altcoins:
                print("symbol in altcoins OKX")
                token_prices["OKX"][symbol] = float(data['data'][0]['last'])
                check_arbitrage()


@catch_exception
def on_open_bybit(ws):
    subsribe_message = {
        "op": "subscribe",
        "args": [f"tickers.{symbol}USDT" for symbol in altcoins]
    }
    ws.send(json.dumps(subsribe_message))
    print("gửi yêu cầu đăng ký giá từ bybit")


# Lấy giá từ Bybit
@catch_exception
def on_message_bybit(ws, message):
    data = json.loads(message)
    if "data" in data and "lastPrice" in data["data"]:
        symbol = data["data"]["symbol"].replace('USDT', '')
        print(symbol)
        if symbol in altcoins:
            print("symbol in altcoins Bybit")
            token_prices["Bybit"][symbol] = float(data["data"]["lastPrice"])
            check_arbitrage()


@catch_exception
def run_websocket(ws_url, on_message, on_open=None):
    ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_open=on_open)
    ws.run_forever()


# Tạo luồng chạy WebSocket
print("Danh sách Binance WebSocket:")
for ws_url in binance_ws:
    print(ws_url)
for ws_url in binance_ws:
    if not ws_url.startswith("wss://"):
        print(f"Lỗi URL WebSocket: {ws_url}")
binance_thread = []
for ws_url in binance_ws:
    thread = threading.Thread(target=run_websocket, args=(ws_url, on_message_binance))
    thread.start()
    binance_thread.append(thread)

okx_thread = threading.Thread(target=run_websocket, args=(okx_ws, on_message_okx, on_open_okx))
bybit_thread = threading.Thread(target=run_websocket, args=(bybit_ws, on_message_bybit, on_open_bybit))

okx_thread.start()
bybit_thread.start()

# Giữ chương trình chạy mãi để thread không kết thúc
for thread in binance_thread:
    thread.join()
okx_thread.join()
bybit_thread.join()
