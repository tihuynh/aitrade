import json
import time

import websocket

coins = ["btcusdt", "ethusdt", "bnbusdt", "solusdt", "xrpusdt"]
# socket = "wss://stream.binance.com:9443/ws/btcusdt@trade"
socket = f"wss://stream.binance.com:9443/ws/{'/'.join([coin+'@trade' for coin in coins])}"
last_print_time={}
def on_message(ws, message):
    global last_print_time
    data = json.loads(message)
    symbol = data['s']
    price = data['p']
    current_time = time.time()
    if symbol not in last_print_time or (current_time - last_print_time[symbol] > 5):
        print(f"{symbol}:{price} USDT")
        last_print_time[symbol] = current_time

ws = websocket.WebSocketApp(socket,on_message=on_message)
ws.run_forever()
