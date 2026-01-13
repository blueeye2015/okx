import time
import requests
import json
import os
import base64
import math
from urllib.parse import urlencode, quote
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# 加载环境变量
load_dotenv('/data/okx/.env')

# ================= 配置 =================
API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
BASE_URL = 'https://fapi.binance.com'
SYMBOL = 'BTCUSDT'
PROXIES = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}
# =======================================

TIME_OFFSET = 0

def sync_time():
    global TIME_OFFSET
    print("⏳ 正在校准时间...")
    try:
        res = requests.get(f"{BASE_URL}/fapi/v1/time", proxies=PROXIES, timeout=5)
        server_time = res.json()['serverTime']
        local_time = int(time.time() * 1000)
        TIME_OFFSET = server_time - local_time
        print(f"✅ 校准完成，偏移: {TIME_OFFSET}ms")
    except Exception as e:
        print(f"❌ 时间校准失败: {e}")

def get_signature(payload):
    try:
        private_key_str = SECRET_KEY.strip()
        if not private_key_str.startswith("-----BEGIN"):
            private_key_str = f"-----BEGIN PRIVATE KEY-----\n{private_key_str}\n-----END PRIVATE KEY-----"
        
        private_key = load_pem_private_key(private_key_str.encode('utf-8'), password=None)
        signature = private_key.sign(payload.encode('utf-8'))
        return base64.b64encode(signature).decode('utf-8')
    except Exception as e:
        print(f"🔑 签名生成失败: {e}")
        return None

def send_request(method, endpoint, params=None):
    if params is None: params = {}
    params['timestamp'] = int(time.time() * 1000) + TIME_OFFSET
    params['recvWindow'] = 60000
    
    query_string = urlencode(params)
    signature = get_signature(query_string)
    if not signature: return None
    
    full_url = f"{BASE_URL}{endpoint}?{query_string}&signature={quote(signature)}"
    headers = {'X-MBX-APIKEY': API_KEY, 'Content-Type': 'application/json'}
    
    try:
        response = requests.request(method, full_url, headers=headers, proxies=PROXIES)
        if response.status_code >= 400:
            print(f"❌ API 报错 ({response.status_code}): {response.text}")
            return None
        return response.json()
    except Exception as e:
        print(f"❌ 网络请求失败: {e}")
        return None

def get_price():
    res = send_request('GET', '/fapi/v1/ticker/price', {'symbol': SYMBOL})
    return float(res['price']) if res else None

def run_test():
    print("🚀 开始 API 连通性与下单测试...")
    
    # 1. 校准时间
    sync_time()
    
    # 2. 获取当前价格
    price = get_price()
    if not price:
        print("❌ 无法获取价格，测试终止")
        return
    
    print(f"💰 当前 {SYMBOL} 价格: {price}")
    
    # 3. 计算一个安全的挂单价 (当前价的 80%，确保不成交)
    safe_price = math.floor(price * 0.8)
    quantity = 0.002 # 最小数量
    
    print(f"🧪 准备挂一个测试单:")
    print(f"   方向: 买入 (BUY)")
    print(f"   价格: {safe_price} (远低于市价，不会成交)")
    print(f"   数量: {quantity} BTC")
    
    # 4. 下单 (LIMIT 订单)
    order_params = {
        'symbol': SYMBOL,
        'side': 'BUY',
        'type': 'LIMIT',
        'timeInForce': 'GTC', # Good Till Cancel
        'quantity': quantity,
        'price': safe_price
    }
    
    print("\n👉 正在发送订单...")
    order_res = send_request('POST', '/fapi/v1/order', order_params)
    
    if order_res and 'orderId' in order_res:
        order_id = order_res['orderId']
        print(f"✅ 挂单成功！订单 ID: {order_id}")
        print("👀 请在 10秒 内去币安 APP -> 合约 -> 当前委托 查看，应该能看到这个单子。")
        
        # 5. 倒计时撤单
        for i in range(10, 0, -1):
            print(f"⏰ {i} 秒后自动撤单...", end='\r')
            time.sleep(1)
        print("\n")
        
        # 6. 撤单
        print(f"🧹 正在撤销测试订单 {order_id}...")
        cancel_res = send_request('DELETE', '/fapi/v1/order', {'symbol': SYMBOL, 'orderId': order_id})
        
        if cancel_res:
            print(f"✅ 撤单成功！状态: {cancel_res.get('status', 'Unknown')}")
            print("🎉 测试通过！API 配置无误。")
        else:
            print("❌ 撤单失败，请手动去 APP 撤销！")
            
    else:
        print("❌ 下单失败，请检查上面的错误信息。")

if __name__ == "__main__":
    run_test()