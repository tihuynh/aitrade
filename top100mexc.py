import requests
import pandas as pd


def get_mexc_top100():
    url = "https://api.mexc.com/api/v3/ticker/price"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df.columns = ['symbol', 'price']

        # Chuyển đổi giá sang kiểu float
        df['price'] = df['price'].astype(float)

        # Lọc chỉ lấy top 100 cặp giao dịch có giá trị cao nhất
        df = df.sort_values(by='price', ascending=False).head(100)

        return df
    else:
        print("Lỗi khi lấy dữ liệu từ MEXC API")
        return None


if __name__ == "__main__":
    top_100 = get_mexc_top100()
    if top_100 is not None:
        print(top_100)
