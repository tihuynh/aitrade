# ======================================
# 🤖 BOT ALTSEASON FULL COMBO (Giai đoạn 2 + 3 + 4)
# ======================================

import requests
import time
import os
import json
from dotenv import load_dotenv
from web3 import Web3
from eth_account import Account
from datetime import datetime
# === Load biến môi trường ===
load_dotenv()
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
RPC_URL = os.getenv("RPC_URL")
WALLET_ADDRESS = Web3.to_checksum_address(Account.from_key(PRIVATE_KEY).address)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN_ALTCOIN_SEASON")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID_ALTCOIN_SEASON")

# === Kết nối Web3 ===
w3 = Web3(Web3.HTTPProvider(RPC_URL))
assert w3.is_connected(), "❌ Không thể kết nối RPC"

# === Router PancakeSwap V2 ===
ROUTER_ADDRESS = Web3.to_checksum_address("0x10ED43C718714eb63d5aA57B78B54704E256024E")
bscscan_api = os.getenv("BSCSCAN_API_KEY")
ROUTER_ABI = json.loads(requests.get(f"https://api.bscscan.com/api?module=contract&action=getabi&address=0x10ED43C718714eb63d5aA57B78B54704E256024E&apikey={bscscan_api}").json()["result"])
router = w3.eth.contract(address=ROUTER_ADDRESS, abi=ROUTER_ABI)

# === Gửi Telegram ===
def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Lỗi gửi Telegram: {e}")

# === Giai đoạn 2: Quét Dexscreener ===
def fetch_gecko_trending():
    try:
        url = "https://api.geckoterminal.com/api/v2/networks/bsc/pools?page=1"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"❌ GeckoTerminal lỗi HTTP {response.status_code}")
            return []

        data = response.json()["data"]
        tokens = []

        for item in data:
            attr = item["attributes"]
            tokens.append({
                "name": attr["base_token"]["name"],
                "symbol": attr["base_token"]["symbol"],
                "address": attr["base_token"]["address"],
                "price": float(attr["price_usd"]),
                "volume24h": float(attr["volume_usd"]["h24"]),
                "liquidity": float(attr["reserve_in_usd"]),
                "url": f"https://www.geckoterminal.com/bsc/pools/{item['id']}"
            })

        return tokens[:5]  # lấy top 5 token
    except Exception as e:
        error_msg = f"❌ Lỗi trong fetch_gecko_trending: {e}"
        send_telegram(error_msg)
        print(error_msg)
        return []


# def filter_top_tokens(pairs, top_n=5):
#     try:
#         filtered = []
#         for p in pairs:
#             try:
#                 if p.get("liquidity", {}).get("usd", 0) >= 20000 and p.get("txCount", {}).get("h1", 0) > 20:
#                     filtered.append({
#                         "name": p.get("baseToken", {}).get("name", "???"),
#                         "symbol": p.get("baseToken", {}).get("symbol", "???"),
#                         "address": p.get("baseToken", {}).get("address"),
#                         "price": float(p.get("priceUsd", 0)),
#                         "volume24h": float(p.get("volume", {}).get("h24", 0)),
#                         "liquidity": float(p.get("liquidity", {}).get("usd", 0)),
#                         "dex": p.get("dexId", "???"),
#                         "url": p.get("url", "")
#                     })
#             except:
#                 continue
#
#         return sorted(filtered, key=lambda x: x["volume24h"], reverse=True)[:top_n]
#     except Exception as e:
#         error_msg = f"❌ Lỗi trong filter_top_tokens: {e}"
#         send_telegram(error_msg)
#         print(error_msg)
# === Giai đoạn 3: Mua token ===
def auto_buy_token(token_address):
    try:
        token_address = Web3.to_checksum_address(token_address)

        # Lấy số dư BNB hiện tại và tính 90%
        balance = w3.eth.get_balance(WALLET_ADDRESS)
        bnb_total = w3.from_wei(balance, 'ether')
        bnb_amount = round(bnb_total * 0.9, 6)

        if bnb_amount < 0.002:
            print("⚠️ Số dư quá thấp, bỏ qua swap")
            return
        entry_price = get_token_price_usd(token_address)
        deadline = int(time.time()) + 60 * 10
        path = [Web3.to_checksum_address("0xBB4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"), token_address]

        gas_price = w3.eth.gas_price
        if gas_price > w3.to_wei(10, 'gwei'):
            print(f"❌ Gas price cao ({w3.from_wei(gas_price, 'gwei')} Gwei), huỷ lệnh!")
            return

        txn = router.functions.swapExactETHForTokens(
            0, path, WALLET_ADDRESS, deadline
        ).build_transaction({
            'from': WALLET_ADDRESS,
            'value': w3.to_wei(bnb_amount, 'ether'),
            'gas': 300000,
            'gasPrice': gas_price,
            'nonce': w3.eth.get_transaction_count(WALLET_ADDRESS),
        })

        signed_txn = w3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)

        send_telegram(f"🚀 Đã mua: `{token_address}`\n💰 Sử dụng {bnb_amount:.6f} BNB\n🔗 https://bscscan.com/tx/{tx_hash.hex()}")

        with open("buy_log.csv", "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp},{token_address},{entry_price},{bnb_amount},{tx_hash.hex()}\n")

        return tx_hash.hex()
    except Exception as e:
        error_msg = f"❌ Lỗi trong auto_buy_token: {e}"
        send_telegram(error_msg)
        print(error_msg)
        return None

# === Giai đoạn 4: Theo dõi TP/SL ===
def check_take_profit():
    try:
        if not os.path.exists("buy_log.csv"): return
        with open("buy_log.csv", "r") as f:
            lines = f.readlines()

        for line in lines:
            ts, address, entry_price, bnb, tx = line.strip().split(",")
            price_now = get_token_price_usd(address)
            entry_price = float(entry_price)

            if price_now >= entry_price * 2:
                send_telegram(f"🎯 TP đạt X2 với `{address}`\nGiá vào: ${entry_price:.4f} → Hiện tại: ${price_now:.4f}")
            elif price_now <= entry_price * 0.7:
                send_telegram(f"⚠️ Coin `{address}` tụt mạnh\nGiá vào: ${entry_price:.4f} → Hiện tại: ${price_now:.4f}")
    except Exception as e:
        error_msg = f"❌ Lỗi trong check_take_profit: {e}"
        send_telegram(error_msg)
        print(error_msg)

# === Lấy giá token hiện tại từ GeckoTerminal API ===
def get_token_price_usd(address):
    try:
        url = f"https://api.geckoterminal.com/api/v2/simple/networks/bsc/token_price/{address}"
        response = requests.get(url).json()
        price = float(response['data']['attributes']['price_usd'])
        return price
    except Exception as e:
        error_msg = f"❌ Lỗi trong get_token_price_usd: {e}"
        send_telegram(error_msg)
        print(error_msg)
        return 0

# =============================
# Chạy toàn bộ luồng mỗi 4 giờ
# =============================
if __name__ == '__main__':
    print("🚀 Đang chạy bot Altcoin Season Combo...")
    # pairs = fetch_dexscreener_trending()
    top_tokens = fetch_gecko_trending()

    for token in top_tokens:
        msg = f"🧠 Phát hiện coin mới: [{token['symbol']}]({token['url']})\n💧 Liquidity: ${token['liquidity']:,}\n📊 Volume 24h: ${token['volume24h']:,}\n💰 Giá: ${token['price']:.6f}"
        send_telegram(msg)
        auto_buy_token(token['address'])

    check_take_profit()
