import time
import requests
import json
import os
import logging
import math
import base64
from urllib.parse import urlencode, quote
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# 加载环境变量
load_dotenv('/data/okx/.env')

# ================= 配置区域 =================
API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
BASE_URL = 'https://fapi.binance.com'
SYMBOL = 'BTCUSDT'

# 这里的代理要和主程序保持一致
PROXIES = {
    'http': 'http://127.0.0.1:7890', 
    'https': 'http://127.0.0.1:7890'
}

# 补单的风控参数 (建议与主程序保持一致)
TP_RATE = 0.04  # 止盈 +4%
SL_RATE = 0.01  # 止损 -1%
# ===========================================

def get_signature(payload):
    private_key_str = SECRET_KEY.strip()
    if not private_key_str.startswith("-----BEGIN"):
        private_key_str = f"-----BEGIN PRIVATE KEY-----\n{private_key_str}\n-----END PRIVATE KEY-----"
    try:
        private_key = load_pem_private_key(private_key_str.encode('utf-8'), password=None)
        signature = private_key.sign(payload.encode('utf-8'))
        return base64.b64encode(signature).decode('utf-8')
    except Exception as e:
        print(f"❌ 签名失败: {e}")
        return None

def send_request(method, endpoint, params=None):
    if params is None: params = {}
    params['timestamp'] = int(time.time() * 1000)
    query_string = urlencode(params)
    signature = get_signature(query_string)
    full_url = f"{BASE_URL}{endpoint}?{query_string}&signature={quote(signature)}"
    headers = {'X-MBX-APIKEY': API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.request(method, full_url, headers=headers, proxies=PROXIES, timeout=10)
        return response.json()
    except Exception as e:
        print(f"❌ 网络请求失败: {e}")
        return None

def fix_position():
    print("="*40)
    print("🚑 裸奔仓位修复工具 (Fix Legacy Position)")
    print("="*40)
    
    # 1. 查询当前持仓
    print("🔍 正在查询账户持仓...")
    res = send_request('GET', '/fapi/v2/positionRisk', {'symbol': SYMBOL})
    pos = None
    
    if res:
        for p in res:
            amt = float(p['positionAmt'])
            if amt != 0:
                pos = p
                break
    
    if not pos:
        print("✅ 当前账户是空仓，无需修复。")
        return

    amt = abs(float(pos['positionAmt']))
    entry_price = float(pos['entryPrice'])
    raw_side = float(pos['positionAmt'])
    side = 'LONG' if raw_side > 0 else 'SHORT'
    
    print(f"🚨 发现持仓: [{side}] {amt} BTC")
    print(f"💰 开仓均价: {entry_price}")
    
    # 2. 查询当前挂单
    print("🔍 检查现有挂单...")
    open_orders = send_request('GET', '/fapi/v1/openOrders', {'symbol': SYMBOL})
    
    if len(open_orders) > 0:
        print(f"⚠️ 警告: 当前已有 {len(open_orders)} 个挂单。")
        for o in open_orders:
            print(f"   - {o['type']} {o['side']} @ {o.get('stopPrice', o.get('price'))}")
            
        confirm = input("\n🤔 是否撤销所有旧挂单并重新挂风控？(y/n): ")
        if confirm.lower() == 'y':
            print("🧹 正在撤销旧挂单...")
            send_request('DELETE', '/fapi/v1/allOpenOrders', {'symbol': SYMBOL})
            time.sleep(1) # 等一等
        else:
            print("🚫 操作已取消，程序退出。")
            return
    else:
        print("✅ 当前无挂单，该仓位正在“裸奔”！")

    # 3. 计算止盈止损价格
    # 注意：如果当前价格已经跌破止损价，挂单可能会立即触发或报错。
    # 这里我们只负责按比例挂单。
    
    if side == 'LONG':
        tp_price = round(entry_price * (1 + TP_RATE), 1)
        sl_price = round(entry_price * (1 - SL_RATE), 1)
        close_side = 'SELL'
    else: # SHORT
        tp_price = round(entry_price * (1 - TP_RATE), 1)
        sl_price = round(entry_price * (1 + SL_RATE), 1)
        close_side = 'BUY'
        
    print(f"\n🛡️ 准备补挂风控单 (基于开仓价 {entry_price}):")
    print(f"   🛑 止损 (SL): {sl_price} ({-SL_RATE*100}%)")
    print(f"   🎉 止盈 (TP): {tp_price} (+{TP_RATE*100}%)")
    
    confirm_fix = input("\n👉 确认执行挂单吗？(y/n): ")
    if confirm_fix.lower() != 'y':
        print("🚫 操作已取消。")
        return

    # 4. 执行挂单
    # 挂止损
    p1 = {
        'symbol': SYMBOL,
        'side': close_side,
        'type': 'STOP_MARKET',
        'stopPrice': sl_price,
        'closePosition': 'true' # 关键：只平仓，不乱开
    }
    print("   正在挂止损单...", end=" ")
    res1 = send_request('POST', '/fapi/v1/order', p1)
    if res1 and 'orderId' in res1:
        print(f"✅ 成功 (ID: {res1['orderId']})")
    else:
        print(f"❌ 失败: {res1}")

    # 挂止盈
    p2 = {
        'symbol': SYMBOL,
        'side': close_side,
        'type': 'TAKE_PROFIT_MARKET',
        'stopPrice': tp_price,
        'closePosition': 'true'
    }
    print("   正在挂止盈单...", end=" ")
    res2 = send_request('POST', '/fapi/v1/order', p2)
    if res2 and 'orderId' in res2:
        print(f"✅ 成功 (ID: {res2['orderId']})")
    else:
        print(f"❌ 失败: {res2}")

    print("\n🎉 修复完成！现在可以把账户交给机器人托管了。")

if __name__ == "__main__":
    fix_position()