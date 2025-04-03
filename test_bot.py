# ===== KẾT NỐI API BYBIT =====
session = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=testnet)

# ===== HÀM KIỂM TRA API HOẠT ĐỘNG =====
def test_api_connection():
    try:
        response = session.get_wallet_balance(accountType="SPOT")
        print("✅ [PASSED] Kết nối API Bybit thành công.")
        return True
    except Exception as e:
        print(f"❌ [FAILED] Lỗi kết nối API: {e}")
        return False