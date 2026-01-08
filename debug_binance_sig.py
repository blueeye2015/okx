import time
import hmac
import hashlib
import os
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv('/data/okx/.env')

API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()

def generate_curl():
    # 1. 准备参数
    params = {
        'timestamp': int(time.time() * 1000),
        'recvWindow': 20000
    }
    
    # 2. 生成签名
    query_string = urlencode(params)
    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # 3. 构造 curl 命令
    endpoint = "/fapi/v2/account"
    full_url = f"https://fapi.binance.com{endpoint}?{query_string}&signature={signature}"
    
    curl_cmd = f'curl -x http://127.0.0.1:7890 -H "X-MBX-APIKEY: {API_KEY}" "{full_url}"'
    
    print("\n" + "="*50)
    print("🚀 请复制并运行下面的命令进行最终测试:")
    print("="*50 + "\n")
    print(curl_cmd)
    print("\n" + "="*50)

if __name__ == "__main__":
    if not API_KEY or not SECRET_KEY:
        print("❌ 错误: .env 中缺少 API_KEY 或 SECRET_KEY")
    else:
        generate_curl()
