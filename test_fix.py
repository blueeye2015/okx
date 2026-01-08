import time
import requests
import os
import base64
import json
from urllib.parse import urlencode, quote  # <--- 必须导入 quote
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# 加载环境变量
load_dotenv('/data/okx/.env')

# ==========================================
# 配置
# ==========================================
API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
BASE_URL = 'https://fapi.binance.com' 
SYMBOL = 'BTCUSDT'

PROXIES = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}

# ==========================================
# 核心逻辑 (集成所有修复)
# ==========================================
TIME_OFFSET = 0

def sync_server_time():
    global TIME_OFFSET
    print("⏳ 校准时间...")
    try:
        res = requests.get(f"{BASE_URL}/fapi/v1/time", proxies=PROXIES, timeout=5)
        server_time = res.json()['serverTime']
        local_time = int(time.time() * 1000)
        TIME_OFFSET = server_time - local_time
        print(f"✅ 时间偏移: {TIME_OFFSET} ms")
    except Exception as e:
        print(f"❌ 时间校准失败: {e}")

def get_signature(payload):
    """Ed25519 签名 (强力清洗版)"""
    # 清洗 Key
    raw_key = SECRET_KEY.replace("-----BEGIN PRIVATE KEY-----", "") \
                        .replace("-----END PRIVATE KEY-----", "") \
                        .replace("\n", "").replace(" ", "").replace("'", "").replace('"', "").strip()
    
    # 重组 PEM
    pem_key_str = f"-----BEGIN PRIVATE KEY-----\n{raw_key}\n-----END PRIVATE KEY-----"
    
    try:
        private_key = load_pem_private_key(pem_key_str.encode('utf-8'), password=None)
        signature = private_key.sign(payload.encode('utf-8'))
        return base64.b64encode(signature).decode('utf-8')
    except Exception as e:
        print(f"💥 Key 解析失败: {e}")
        raise

def send_request(method, endpoint, params=None):
    if params is None: params = {}
    
    # 1. 时间戳与窗口
    params['timestamp'] = int(time.time() * 1000) + TIME_OFFSET
    params['recvWindow'] = 60000
    
    # 2. 生成 Query String
    query_string = urlencode(params)
    
    # 3. 签名
    signature = get_signature(query_string)
    
    # 4. 构造 URL (关键：使用 quote 编码签名)
    full_url = f"{BASE_URL}{endpoint}?{query_string}&signature={quote(signature)}"
    
    headers = {
        'X-MBX-APIKEY': API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.request(method, full_url, headers=headers, proxies=PROXIES, timeout=10)
        # 即使报错也要返回内容，以便判断错误类型
        return response.status_code, response.json()
    except Exception as e:
        print(f"❌ 网络错误: {e}")
        return 0, {}

# ==========================================
# 主测试逻辑
# ==========================================
def test_order_permissions():
    sync_server_time()
    
    print("\n🚀 开始【下单权限】最终测试...")
    print(f"🎯 目标: 尝试在 {SYMBOL} 下一笔最小市价单")
    print("⚠️ 预期结果: 应该报错 '余额不足' (-2019)，而不是 '签名错误' (-1022)")
    
    # 构造下单参数
    params = {
        'symbol': SYMBOL,
        'side': 'BUY',
        'type': 'MARKET',
        'quantity': 0.001, # 最小数量
    }
    
    status, res = send_request('POST', '/fapi/v1/order', params)
    
    print("\n" + "="*40)
    print(f"📡 币安响应代码: {status}")
    print(f"📄 响应内容: {json.dumps(res, indent=2)}")
    print("="*40 + "\n")
    
    # 自动判题
    if status == 200:
        print("🎉 竟然下单成功了！(说明你账户里其实有钱？)")
    else:
        code = res.get('code')
        msg = res.get('msg')
        
        if code == -2019: # Margin is insufficient
            print("✅✅✅ 测试通过！ ✅✅✅")
            print("原因: 币安拒绝了订单，因为没钱。")
            print("结论: API Key 权限正常，签名正常，网络正常。你可以放心充值了。")
        elif code == -1022:
            print("❌ 测试失败: 签名仍然无效。")
        elif code == -2015:
            print("❌ 测试失败: API Key 权限不足 (未开启'合约交易'权限 或 IP限制)。")
        else:
            print(f"❓ 其他错误: {msg} (只要不是签名错误，通常都说明连接是通的)")

if __name__ == "__main__":
    test_order_permissions()