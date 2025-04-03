import functools
import threading
import requests
import websocket
import json
import logging
import  time

# Cấu hình logging
logging.basicConfig(filename="log.text", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s", filemode='w')
log_prices = logging.getLogger("log_prices")
log_prices.propagate = False
# Xóa các handler cũ nếu có
if log_prices.hasHandlers():
    log_prices.handlers.clear()

log_prices_handler = logging.FileHandler("log_top100Altcoin.txt", mode = 'w', encoding='utf-8')
log_prices_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
log_prices.addHandler(log_prices_handler)
log_prices.setLevel(logging.INFO)
# API lấy danh sách top 100 coin theo vốn hóa từ CoinGecko
COINGECKO_API = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false"

# API lấy danh sách coin trên Binance
BINANCE_API = "https://api.binance.com/api/v3/exchangeInfo"

# WebSocket URL của các sàn giao dịch
okx_ws = "wss://ws.okx.com:8443/ws/v5/public"
bybit_ws = "wss://stream.bybit.com/v5/public/spot"

# Biến lưu giá token trên các sàn
token_prices = {
    "Binance": {},
    "OKX": {},
    "Bybit": {}
}

# Decorator bắt lỗi
def catch_exception(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Lỗi trong {func.__name__}: {str(e)}", exc_info=True)
            print(f"Lỗi trong {func.__name__} : {e}")
            return None
    return wrapper

# Lấy danh sách top 100 Altcoin từ CoinGecko
@catch_exception
def get_top_altcoins():
    response = requests.get(COINGECKO_API)
    coins = response.json()
    return [coin['symbol'].upper() for coin in coins]

# Lấy danh sách coin hỗ trợ trên Binance
@catch_exception
def get_binance_symbols():
    response = requests.get(BINANCE_API)
    data = response.json()
    return {symbol['symbol'].replace('USDT', '') for symbol in data['symbols'] if symbol['symbol'].endswith('USDT')}

# Lấy danh sách Altcoin từ CoinGecko và lọc chỉ còn những coin có trên Binance
altcoins = get_top_altcoins()
binance_symbols = get_binance_symbols()
altcoins = [coin for coin in altcoins if coin in binance_symbols]

print("✅ Danh sách Altcoin hợp lệ trên Binance:", altcoins)

if not altcoins:
    print("❌ Lỗi: Không tìm thấy Altcoin hợp lệ trên Binance!")
    exit()

# Hàm kiểm tra arbitrage
@catch_exception
def check_arbitrage():
    print("🔍 Kiểm tra chênh lệch giá:")

    all_coins = set()
    for exchange in token_prices:
        all_coins.update(token_prices[exchange].keys())  # Thêm tất cả coin từ mọi sàn

    for coin in all_coins:  # Kiểm tra từng coin
        prices = {ex: token_prices[ex].get(coin, None) for ex in token_prices}  # Lấy giá từ từng sàn
        print(f"📊 Giá {coin}: {prices}")  # In ra tất cả giá của coin trên từng sàn

        # Loại bỏ các giá trị None (nếu có)
        valid_prices = {ex: price for ex, price in prices.items() if price is not None}

        # Nếu thiếu giá của Bybit, kiểm tra lại tại sao
        if "Bybit" not in valid_prices:
            print(f"⚠️ Không có giá Bybit cho {coin}!")

        # Tính chênh lệch giá
        if len(valid_prices) >= 2:  # Chỉ tính nếu có từ 2 sàn trở lên
            min_exchange, min_price = min(valid_prices.items(), key=lambda x: x[1])
            max_exchange, max_price = max(valid_prices.items(), key=lambda x: x[1])

            spread = max_price - min_price
            spread_percent = (spread / min_price) * 100

            print(f"📊 Coin {coin} | Giá thấp nhất: {min_exchange} ({min_price}) | Giá cao nhất: {max_exchange} ({max_price}) | Chênh lệch: {spread:.6f} USDT | {spread_percent:.2f}%")


            log_message = (f"📊 Coin {coin} | Giá thấp nhất: {min_exchange} ({min_price}) | Giá cao nhất: {max_exchange} ({max_price}) | Chênh lệch: {spread:.6f} USDT | {spread_percent:.2f}%")
            print(log_message)
            log_prices.info(log_message)
            log_prices_handler.flush()  # Đảm bảo log được ghi ngay lập tức

            if spread_percent > 1:
                arbitrage_message = (f"🚀 Cơ hội arbitrage: Mua tại {min_price}, Bán tại {max_price}!")
                print (arbitrage_message)
                log_prices.info(arbitrage_message)
                log_prices_handler.flush()  # Đảm bảo log được ghi ngay lập tức

# Kết nối WebSocket Binance
binance_ws = [f"wss://stream.binance.com:9443/ws/{symbol.lower()}usdt@trade" for symbol in altcoins]

@catch_exception
def on_message_binance(ws, message):
    print("on_message_binance")

    data = json.loads(message)
    symbol = data['s'].replace('USDT','')
    print(f"Binance {symbol}")
    if symbol in altcoins:
        token_prices["Binance"][symbol]= float(data['p'])
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
                print ("symbol in altcoins OKX")
                token_prices["OKX"][symbol] = float(data['data'][0]['last'])
                check_arbitrage()

def restart_websocket(exchange):
    print(f"🔄 Đang khởi động lại WebSocket của {exchange}...")
    time.sleep(5)  # Đợi 5 giây trước khi kết nối lại
    if  exchange == "Bybit":
        run_websocket(bybit_ws, on_message_bybit, on_open_bybit)



@catch_exception
def get_bybit_symbols():
    url = "https://api.bybit.com/v5/market/instruments-info?category=spot"  # API mới của Bybit
    response = requests.get(url)

    # Kiểm tra nếu response có lỗi
    if response.status_code != 200:
        print(f"❌ Lỗi API Bybit: HTTP {response.status_code}")
        return set()

    try:
        data = response.json()  # Chuyển đổi sang JSON
        # print(f"📩 Dữ liệu từ Bybit API: {data}")  # Debug API trả về gì

        if "result" not in data or "list" not in data["result"]:
            print("⚠️ API Bybit không trả về dữ liệu hợp lệ.")
            return set()

        return {symbol['symbol'].replace('USDT', '') for symbol in data["result"]["list"] if "USDT" in symbol['symbol']}

    except json.JSONDecodeError:
        print("❌ Lỗi: Không thể parse JSON từ API Bybit.")
        return set()


bybit_symbols = get_bybit_symbols()
altcoins_bybit = [coin for coin in altcoins if coin in bybit_symbols]
print(f"✅ Danh sách Altcoin hợp lệ trên Bybit: {altcoins_bybit}")

@catch_exception
def on_open_bybit(ws):
    print("on_open_bybit")
    max_per_request = 10
    altcoin_batches = [altcoins_bybit[i:i + max_per_request] for i in range(0, len(altcoins_bybit), max_per_request)]

    for batch in altcoin_batches:
        subscribe_message = {
            "op": "subscribe",
            "args": [f"tickers.{symbol}USDT" for symbol in batch]  # Chuyển từ 'publicTrade' sang 'tickers'
        }
        ws.send(json.dumps(subscribe_message))
        print(f"📩 Đã gửi yêu cầu đăng ký giá từ Bybit cho nhóm: {batch}")


# Lấy giá từ Bybit
@catch_exception
def on_message_bybit(ws, message):
    data = json.loads(message)
    # print(f"📩 Dữ liệu từ Bybit: {data}")

    if "topic" in data and "data" in data:
        topic = data["topic"]

        # Xử lý dữ liệu từ 'tickers'
        # if "tickers" in topic and isinstance(data["data"], list) and len(data["data"]) > 0:
        entries = data["data"] if isinstance(data["data"],list) else [data["data"]]
        for entry in entries:
            symbol = entry["symbol"]
            price = float(entry["lastPrice"])
            print(f"1.Bybit {symbol} = {price}")
            symbol = symbol.replace("USDT","")
            if symbol in altcoins_bybit:
                token_prices["Bybit"][symbol] = price
                print(f"✅ Bybit Cập nhật token_prices['Bybit'][{symbol}] = {price}")
                check_arbitrage()
            else:
                print(f"{symbol} not in {altcoins_bybit}")
                log_prices.info(f"{symbol} not in {altcoins_bybit}")
    else:
        print("⚠️ Không nhận được dữ liệu giá hợp lệ từ Bybit!")


@catch_exception
def run_websocket(ws_url, on_message, on_open = None):
    print(f"🔌 Đang kết nối WebSocket: {ws_url}")
    ws = websocket.WebSocketApp(ws_url, on_message = on_message, on_open= on_open)
    ws.run_forever()


# Tạo luồng chạy WebSocket

bybit_thread = threading.Thread(target=run_websocket, args=(bybit_ws, on_message_bybit, on_open_bybit))
okx_thread = threading.Thread(target=run_websocket, args=(okx_ws,on_message_okx,on_open_okx))

bybit_thread.start()
okx_thread.start()
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

#Giữ chương trình chạy mãi để thread không kết thúc
bybit_thread.join()
okx_thread.join()
for thread in binance_thread:
    thread.join()

