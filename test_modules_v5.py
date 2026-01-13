import time
import requests
import json
import os
import logging
import math
import csv
import base64
from urllib.parse import urlencode, quote
from datetime import datetime
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# 加载环境变量
load_dotenv('/data/okx/.env')

# ================= 配置 =================
API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
BASE_URL = 'https://fapi.binance.com'
SYMBOL = 'BTCUSDT'
TEST_LOG_PATH = '/data/okx/bot_trade_history_TEST.csv' # 测试用的账本

PROXIES = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}
# =======================================

TIME_OFFSET = 0

# --- 基础工具 (直接从 V5 复制) ---
def sync_server_time():
    global TIME_OFFSET
    try:
        res = requests.get(f"{BASE_URL}/fapi/v1/time", proxies=PROXIES, timeout=5)
        server_time = res.json()['serverTime']
        local_time = int(time.time() * 1000)
        TIME_OFFSET = server_time - local_time
        print(f"✅ 时间校准完成，偏移: {TIME_OFFSET}ms")
    except Exception as e:
        print(f"❌ 时间校准失败: {e}")

def get_signature(payload):
    private_key_str = SECRET_KEY.strip()
    if not private_key_str.startswith("-----BEGIN"):
        private_key_str = f"-----BEGIN PRIVATE KEY-----\n{private_key_str}\n-----END PRIVATE KEY-----"
    try:
        private_key = load_pem_private_key(private_key_str.encode('utf-8'), password=None)
        signature = private_key.sign(payload.encode('utf-8'))
        return base64.b64encode(signature).decode('utf-8')
    except Exception as e:
        print(f"🔑 签名失败: {e}")
        return None

def send_request(method, endpoint, params=None):
    if params is None: params = {}
    params['timestamp'] = int(time.time() * 1000) + TIME_OFFSET
    params['recvWindow'] = 60000
    query_string = urlencode(params)
    signature = get_signature(query_string)
    full_url = f"{BASE_URL}{endpoint}?{query_string}&signature={quote(signature)}"
    headers = {'X-MBX-APIKEY': API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.request(method, full_url, headers=headers, proxies=PROXIES)
        if response.status_code >= 400:
            print(f"❌ API 报错: {response.text}")
            return None
        return response.json()
    except Exception as e:
        print(f"❌ 网络错误: {e}")
        return None

# ==========================================
# 🎯 待测试的核心函数
# ==========================================

# 1. 记账函数测试
def record_transaction(strategy_source, action, price, quantity, pnl=0, note=""):
    print(f"📝 测试写入 CSV: {TEST_LOG_PATH} ...")
    file_exists = os.path.exists(TEST_LOG_PATH)
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        with open(TEST_LOG_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Time', 'Strategy', 'Action', 'Price', 'Quantity', 'Fee(Est)', 'PnL', 'Balance', 'Note'])
            writer.writerow([now_str, strategy_source, action, price, quantity, 0.0, pnl, 0.0, note])
        print("✅ CSV 写入成功!")
        return True
    except Exception as e:
        print(f"❌ CSV 写入失败: {e}")
        return False

# 2. 获取持仓测试
def get_position(symbol):
    print(f"🔍 查询 {symbol} 持仓...")
    res = send_request('GET', '/fapi/v2/positionRisk', {'symbol': symbol})
    if res:
        for pos in res:
            amt = float(pos['positionAmt'])
            if amt != 0:
                print(f"   当前持仓: {pos['side']} {abs(amt)} 个 (均价 {pos['entryPrice']})")
                return pos
    print("   当前空仓 (无持仓)")
    return None

# 3. 挂单测试
def place_order_test(symbol):
    # 先获取价格
    ticker = send_request('GET', '/fapi/v1/ticker/price', {'symbol': symbol})
    if not ticker: return None
    price = float(ticker['price'])
    
    # 计算一个安全价格 (现价的 80%)，防止成交
    safe_price = math.floor(price * 0.8)
    quantity = 0.002
    
    print(f"🚀 测试挂单: LIMIT BUY {quantity} BTC @ {safe_price} (防成交)")
    
    params = {
        'symbol': symbol,
        'side': 'BUY',
        'type': 'LIMIT',
        'timeInForce': 'GTC',
        'quantity': quantity,
        'price': safe_price
    }
    
    res = send_request('POST', '/fapi/v1/order', params)
    if res and 'orderId' in res:
        print(f"✅ 挂单成功! Order ID: {res['orderId']}")
        return res['orderId']
    else:
        print("❌ 挂单失败")
        return None

# 4. 撤单测试
def cancel_open_orders(symbol):
    print(f"🧹 测试撤销 {symbol} 所有挂单...")
    res = send_request('DELETE', '/fapi/v1/allOpenOrders', {'symbol': symbol})
    if res is not None:
        print("✅ 撤单指令发送成功 (返回内容通常是成功消息)")
    else:
        print("❌ 撤单失败")

# ==========================================
# 🔥 主测试流程
# ==========================================
def run_full_test():
    print("="*40)
    print("🛠️ Auto Trader V5 模块功能自检")
    print("="*40)
    
    # 0. 基础环境
    sync_server_time()
    
    # 1. 测试记账 (最简单，先测)
    print("\n[Step 1] 测试本地记账功能...")
    if record_transaction("TEST_MODULE", "TEST_ENTRY", 90000, 0.001, 0, "Function Check"):
        # 验证文件是否存在
        if os.path.exists(TEST_LOG_PATH):
            print(f"   文件已生成: {TEST_LOG_PATH}")
        else:
            print("❌ 文件未找到，权限问题？")
            
    # 2. 测试获取持仓
    print("\n[Step 2] 测试获取账户持仓...")
    get_position(SYMBOL)
    
    # 3. 测试挂单 (组合拳)
    print("\n[Step 3] 测试 挂单 -> 等待 -> 撤单 流程...")
    order_id = place_order_test(SYMBOL)
    
    if order_id:
        print("⏳ 等待 5 秒，请去 APP 看一眼是否有挂单...")
        time.sleep(5)
        
        # 4. 测试撤单
        print("\n[Step 4] 测试全撤挂单...")
        cancel_open_orders(SYMBOL)
        
        print("⏳ 再等待 2 秒确认撤单...")
        time.sleep(2)
        
        # 验证挂单是否真的没了
        open_orders = send_request('GET', '/fapi/v1/openOrders', {'symbol': SYMBOL})
        if open_orders == []:
            print("✅ 验证通过：当前无挂单！")
        else:
            print(f"⚠️ 警告：当前仍有 {len(open_orders)} 个挂单，请手动检查！")

    print("\n" + "="*40)
    print("🎉 测试结束")

if __name__ == "__main__":
    run_full_test()