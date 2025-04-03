import requests
url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
# while True:
response = requests.get(url)

data = response.json()

print (f"giá bitcoin hiện tại:{data['price']} USDT")
print (data)