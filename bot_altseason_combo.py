# ======================================
# 🤖 BOT ALTSEASON FULL COMBO (Giai đoạn 2 + 3 + 4 + 5)
# ======================================

import os
import json
import time
import requests
from dotenv import load_dotenv
from web3 import Web3
from eth_account import Account
from datetime import datetime

# === Load .env ===
load_dotenv()
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN_ALTCOIN_SEASON")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID_ALTCOIN_SEASON")

# === Các wrapped token cho 4 chain ===
WRAPPED_TOKENS = {
    "bsc": "0xBB4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",  # WBNB
    "eth": "0xC02aaa39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
    "arbitrum": "0x82af49447d8a07e3bd95bd0d56f35241523fbab1",  # WETH
    "base": "0x4200000000000000000000000000000000000006"   # WETH
}

# === Router PancakeSwap, Uniswap, v.v. cho 4 chain ===
ROUTERS = {
    "bsc": {
        "pancakeswap": {
            "address": "0x10ED43C718714eb63d5aA57B78B54704E256024E",
            "rpc": os.getenv("RPC_URL")
        }
    },
    "eth": {
        "uniswap": {
            "address": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            "rpc": os.getenv("RPC_URL_ETH")
        }
    },
    "arbitrum": {
        "sushiswap": {
            "address": "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506",
            "rpc": os.getenv("RPC_URL_ARBITRUM")
        }
    },
    "base": {
        "baseswap": {
            "address": "0x327Df1E6de05895d2ab08513aaDD9313Fe505d86",
            "rpc": os.getenv("RPC_URL_BASE")
        }
    }
}

# === Địa chỉ ví lạnh ===
COLD_WALLETS = {
    "bsc": os.getenv("COLD_WALLET_BSC"),
    "eth": os.getenv("COLD_WALLET_ETH"),
    "arbitrum": os.getenv("COLD_WALLET_ARBITRUM"),
    "base": os.getenv("COLD_WALLET_BASE")
}

# === Gửi Telegram ===
def send_telegram(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": "Markdown"
        })
    except:
        pass
# === Lọc scam nâng cao ===
def is_token_safe(chain, token_address):
    try:
        res = requests.get(f"https://api.honeypot.is/v1/TokenIsHoneypot", params={
            "network": chain,
            "token": token_address
        })
        data = res.json()
        if data.get("honeypot", True):
            send_telegram(f"🚫 Phát hiện honeypot ({chain}): {token_address}")
            return False
        return True
    except:
        return True
# === Quét top token từ GeckoTerminal cho nhiều chain ===
import requests

def fetch_top_tokens_super(top_n_per_chain=3, min_liquidity=10000, min_volume=10000):
    networks = {
        "bsc": "pancakeswap",
        "eth": "uniswap",
        "arbitrum": "sushiswap",
        "base": "baseswap"
    }
    super_tokens = []

    for chain in networks:
        try:
            url = f"https://api.geckoterminal.com/api/v2/networks/{chain}/pools?page=1"
            headers = {"Accept": "application/json"}
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                send_telegram(f"❌ GeckoTerminal lỗi {response.status_code} ({chain})")
                continue

            pools = response.json().get("data", [])
            count = 0

            for item in pools:
                attr = item["attributes"]
                liquidity = float(attr["reserve_in_usd"])
                volume = float(attr["volume_usd"]["h24"])

                if liquidity >= min_liquidity and volume >= min_volume:
                    token_info = attr["base_token"]
                    token_address = token_info["address"]

                    # Kiểm tra scam trước khi thêm
                    if is_token_safe(chain, token_address):
                        super_tokens.append({
                            "chain": chain,
                            "dex": networks[chain],
                            "token": token_address,
                            "name": token_info["name"],
                            "symbol": token_info["symbol"],
                            "price": float(attr["price_usd"]),
                            "liquidity": liquidity,
                            "volume24h": volume,
                            "url": f"https://www.geckoterminal.com/{chain}/pools/{item['id']}"
                        })
                        count += 1

                    if count >= top_n_per_chain:
                        break

        except Exception as e:
            send_telegram(f"❌ Lỗi fetch GeckoTerminal ({chain}): {e}")

    return super_tokens


# === Swap token theo chain ===
def auto_buy_token_from_usdt(chain, dex, token_address, amount_in_usdt):
    try:
        from web3 import Web3
        import json, time

        rpc = ROUTERS[chain][dex]["rpc"]
        router_address = ROUTERS[chain][dex]["address"]
        usdt_address = USDT_ADDRESS[chain]
        weth_address = WETH_ADDRESS[chain]

        w3 = Web3(Web3.HTTPProvider(rpc))
        account = Account.from_key(PRIVATE_KEY)
        wallet_address = account.address

        with open("abis/router_abi.json", "r") as f:
            router_abi = json.load(f)
        router = w3.eth.contract(address=Web3.to_checksum_address(router_address), abi=router_abi)

        # Kiểm tra cặp trực tiếp trước
        if check_pair_exists(chain, token_address, usdt_address):
            path = [Web3.to_checksum_address(usdt_address), Web3.to_checksum_address(token_address)]
        else:
            path = [
                Web3.to_checksum_address(usdt_address),
                Web3.to_checksum_address(weth_address),
                Web3.to_checksum_address(token_address)
            ]

        tx = router.functions.swapExactTokensForTokensSupportingFeeOnTransferTokens(
            w3.to_wei(amount_in_usdt, 'ether'),
            0,
            path,
            wallet_address,
            int(time.time()) + 600
        ).build_transaction({
            'from': wallet_address,
            'gas': 300000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(wallet_address),
        })

        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

        send_telegram(f"🛒 Đã mua {token_address} bằng USDT trên {dex.upper()} ({chain})\n🔗 https://explorer.{chain}.org/tx/{tx_hash.hex()}")
        with open("buy_log.csv", "a") as f:
            f.write(f"{datetime.now()},{token['chain']},{token['dex']},{token['token']},{token['price']}\n")

    except Exception as e:
        send_telegram(f"❌ Lỗi khi mua token từ USDT ({chain}): {e}")

def check_pair_exists(chain, tokenA, tokenB):
    try:
        from web3 import Web3
        import json

        dex_name = list(ROUTERS[chain].keys())[0]  # Lấy dex đầu tiên cho chain đó
        factory_address = ROUTERS[chain][dex_name]["factory"]
        rpc = ROUTERS[chain][dex_name]["rpc"]
        w3 = Web3(Web3.HTTPProvider(rpc))

        with open("abis/factory_abi.json", "r") as f:
            factory_abi = json.load(f)

        factory = w3.eth.contract(address=Web3.to_checksum_address(factory_address), abi=factory_abi)
        pair_address = factory.functions.getPair(tokenA, tokenB).call()

        return pair_address != "0x0000000000000000000000000000000000000000"

    except Exception as e:
        print(f"⚠️ Lỗi khi kiểm tra cặp {tokenA}/{tokenB}: {e}")
        return False



# === Bán token và chuyển 50% về ví lạnh ===
from web3 import Web3

def auto_sell_to_usdt(chain, dex, token_address, amount):
    try:
        from web3 import Web3
        import json, time

        rpc = ROUTERS[chain][dex]["rpc"]
        router_address = ROUTERS[chain][dex]["address"]
        usdt_address = USDT_ADDRESS[chain]

        w3 = Web3(Web3.HTTPProvider(rpc))
        account = Account.from_key(PRIVATE_KEY)
        wallet_address = account.address

        with open("abis/router_abi.json", "r") as f:
            router_abi = json.load(f)
        router = w3.eth.contract(address=Web3.to_checksum_address(router_address), abi=router_abi)

        path = [Web3.to_checksum_address(token_address), Web3.to_checksum_address(usdt_address)]

        tx = router.functions.swapExactTokensForTokensSupportingFeeOnTransferTokens(
            w3.to_wei(amount, 'ether'),
            0,
            path,
            wallet_address,
            int(time.time()) + 600
        ).build_transaction({
            'from': wallet_address,
            'gas': 300000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(wallet_address),
        })

        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

        send_telegram(f"💸 Đã bán token {token_address} → USDT trên {dex.upper()} ({chain})\n🔗 https://explorer.{chain}.org/tx/{tx_hash.hex()}")
        with open("buy_log.csv", "w") as f:
            for l in lines:
                if token_address not in l:
                    f.write(l)

    except Exception as e:
        send_telegram(f"❌ Lỗi khi bán token {token_address} → USDT ({chain}): {e}")


def auto_transfer_usdt_to_cold_wallet(chain):
    try:
        from web3 import Web3
        import json

        rpc = ROUTERS[chain]["pancakeswap"]["rpc"]  # chọn mặc định
        if chain == "eth": rpc = ROUTERS[chain]["uniswap"]["rpc"]
        elif chain == "arbitrum": rpc = ROUTERS[chain]["sushiswap"]["rpc"]
        elif chain == "base": rpc = ROUTERS[chain]["baseswap"]["rpc"]

        w3 = Web3(Web3.HTTPProvider(rpc))
        account = Account.from_key(PRIVATE_KEY)
        wallet_address = account.address
        cold_wallet = COLD_WALLETS[chain]
        usdt_address = USDT_ADDRESS[chain]

        # Load ABI ERC20 chuẩn
        with open("abis/erc20_abi.json", "r") as f:
            erc20_abi = json.load(f)

        usdt = w3.eth.contract(address=Web3.to_checksum_address(usdt_address), abi=erc20_abi)
        balance = usdt.functions.balanceOf(wallet_address).call()
        if balance == 0:
            send_telegram(f"⚠️ Không có USDT để chuyển ({chain})")
            return

        tx = usdt.functions.transfer(
            Web3.to_checksum_address(cold_wallet),
            int(balance * 0.5)  # chuyển 50%
        ).build_transaction({
            'from': wallet_address,
            'gas': 100000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(wallet_address),
        })

        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

        send_telegram(f"📤 Đã chuyển 50% USDT về ví lạnh ({chain})\n🔗 https://explorer.{chain}.org/tx/{tx_hash.hex()}")

    except Exception as e:
        send_telegram(f"❌ Lỗi chuyển USDT về ví lạnh ({chain}): {e}")


# === Theo dõi và bán nếu TP/SL ===
import os

def monitor_and_sell():
    if not os.path.exists("buy_log.csv"):
        return

    with open("buy_log.csv", "r") as f:
        lines = f.readlines()

    for line in lines:
        try:
            ts, chain, dex, token_address, entry_price = line.strip().split(",")
            entry_price = float(entry_price)
            price_now = get_token_price_usd(chain, token_address)
            if price_now == 0:
                continue

            amount = get_wallet_balance_eth(chain)

            if price_now >= entry_price * 2:
                print(f"🚀 X2: Bán và chuyển token {token_address} trên {chain}")
                auto_sell_to_usdt(chain, dex, token_address, amount)
                auto_transfer_usdt_to_cold_wallet(chain)  # Chỉ gọi khi X2

            elif price_now <= entry_price * 0.7:
                print(f"⚠️ Giảm 30%: Chỉ bán token {token_address} trên {chain}")
                auto_sell_to_usdt(chain, dex, token_address, amount)

        except Exception as e:
            print(f"❌ Lỗi monitor: {e}")

# === Hàm lấy giá token từ GeckoTerminal ===
def get_token_price_usd(chain, address):
    try:
        url = f"https://api.geckoterminal.com/api/v2/simple/networks/{chain}/token_price/{address}"
        res = requests.get(url).json()
        return float(res['data']['attributes']['price_usd'])
    except:
        return 0

# === Lấy số dư ETH trên từng chain ===
def get_wallet_balance_eth(chain):
    rpc = ROUTERS[chain][list(ROUTERS[chain].keys())[0]]["rpc"]
    w3 = Web3(Web3.HTTPProvider(rpc))
    return w3.from_wei(w3.eth.get_balance(Account.from_key(PRIVATE_KEY).address), 'ether')

def is_token_already_bought(token_address):
    if not os.path.exists("buy_log.csv"):
        return False
    with open("buy_log.csv", "r") as f:
        return token_address in f.read()


# === Vòng lặp 4h kiểm tra token và thực hiện giao dịch ===
def main_loop():
    while True:
        print("\n🚀 Bắt đầu vòng quét Altseason")
        tokens = fetch_top_tokens_super()
        if is_token_already_bought(tokens["tokens"]):
            continue

        for token in tokens:
            send_telegram(
                f"🧠 Phát hiện token `{token['symbol']}` trên *{token['chain']}*\n"
                f"💧 Liquidity: ${token['liquidity']:,}\n"
                f"📊 Volume 24h: ${token['volume24h']:,}\n"
                f"💰 Giá: ${token['price']:.6f}\n🔗 [Link]({token['url']})"
            )
            auto_buy_token_from_usdt(token["chain"], token["dex"], token["token"], 100)  # ví dụ 100 USDT
        monitor_and_sell()
        print("⏳ Ngủ 4 tiếng...")
        time.sleep(60 * 60 * 4)

if __name__ == "__main__":
    main_loop()
