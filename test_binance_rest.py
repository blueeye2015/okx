import time
import hmac
import hashlib
import requests
import os
import json
from urllib.parse import urlencode
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('/data/okx/.env')

# ==========================================
# 配置
# ==========================================
API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
BASE_URL = 'https://fapi.binance.com' # U本位合约地址

PROXIES = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}

SYMBOL = 'BTCUSDT'
TEST_AMOUNT_USDT = 6.0 # 测试下单金额
LEVERAGE = 5

# ==========================================
# 调试辅助
# ==========================================
def debug_keys():
    print("\n🔍 [Key 诊断]")
    print(f"API_KEY (repr): {repr(API_KEY[:5] + '...' + API_KEY[-5:])}")
    print(f"SECRET  (repr): {repr(SECRET_KEY[:5] + '...' + SECRET_KEY[-5:])}")
    
    if '\\n' in repr(API_KEY) or '\\r' in repr(API_KEY):
        print("⚠️ 警告: API Key 中包含换行符！请检查 .env 文件。")
    if '\\n' in repr(SECRET_KEY) or '\\r' in repr(SECRET_KEY):
        print("⚠️ 警告: Secret Key 中包含换行符！请检查 .env 文件。")
    
    if len(API_KEY) != 64:
        print(f"⚠️ 提示: 标准币安 API Key 长度通常为 64，你的是 {len(API_KEY)}。")
    if len(SECRET_KEY) != 64:
        print(f"⚠️ 提示: 标准币安 Secret Key 长度通常为 64，你的是 {len(SECRET_KEY)}。")

# ==========================================
# 核心工具函数：签名与请求
# ==========================================
def get_signature(params):
    """生成签名"""
    query_string = urlencode(params)
    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def send_request(method, endpoint, params=None):
    """发送请求封装"""
    if params is None:
        params = {}
    
    # 加上时间戳
    params['timestamp'] = int(time.time() * 1000)
    # 放宽接收窗口，避免网络延迟导致的问题
    params['recvWindow'] = 20000 
    # 加上签名
    params['signature'] = get_signature(params)
    
    headers = {
        'X-MBX-APIKEY': API_KEY
    }
    
    url = f"{BASE_URL}{endpoint}"
    
    try:
        response = requests.request(
            method, 
            url, 
            params=params, 
            headers=headers, 
            proxies=PROXIES,
            timeout=10
        )
        response.raise_for_status() # 检查 HTTP 错误
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败 [{endpoint}]: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"   服务器返回: {e.response.text}")
             # 打印发送的参数帮助调试（脱敏签名）
             debug_params = params.copy()
             if 'signature' in debug_params: debug_params['signature'] = '***'
             print(f"   发送参数: {debug_params}")
        raise

# ==========================================
# 业务功能
# ==========================================
def set_leverage(symbol, leverage):
    print(f"⚙️ 设置杠杆: {leverage}x ...")
    try:
        res = send_request('POST', '/fapi/v1/leverage', {'symbol': symbol, 'leverage': leverage})
        print(f"✅ 设置成功: {res['leverage']}x")
    except Exception:
        print("⚠️ 设置杠杆失败 (可能已设置)")

def get_balance():
    print("💰 查询余额...")
    res = send_request('GET', '/fapi/v2/balance')
    for asset in res:
        if asset['asset'] == 'USDT':
            return float(asset['availableBalance'])
    return 0.0

def get_price(symbol):
    res = send_request('GET', '/fapi/v1/ticker/price', {'symbol': symbol})
    return float(res['price'])

def place_order(symbol, side, quantity, reduce_only=False):
    """
    下单
    side: BUY or SELL
    quantity: 币的数量 (float)
    """
    params = {
        'symbol': symbol,
        'side': side.upper(),
        'type': 'MARKET',
        'quantity': quantity,
    }
    if reduce_only:
        params['reduceOnly'] = 'true'
        
    print(f"🚀 下单: {side} {quantity} {symbol} (ReduceOnly={reduce_only})...")
    res = send_request('POST', '/fapi/v1/order', params)
    print(f"✅ 订单成功! ID: {res['orderId']} | 状态: {res['status']}")
    return res

def get_position(symbol):
    """获取单向持仓"""
    res = send_request('GET', '/fapi/v2/positionRisk', {'symbol': symbol})
    for pos in res:
        # 币安可能返回双向持仓数据，我们要找那个有数量的，或者合并判断
        amt = float(pos['positionAmt'])
        if amt != 0:
            return {
                'amount': amt,
                'entryPrice': float(pos['entryPrice']),
                'unrealizedProfit': float(pos['unrealizedProfit'])
            }
    return None

def main():
    print("="*50)
    print("🦈 币安原生 REST API 测试")
    print("="*50)
    
    # 运行 Key 诊断
    debug_keys()
    
    if not API_KEY or not SECRET_KEY:
        print("❌ 错误: 请在 .env 文件中设置 BINANCE_API_KEY 和 BINANCE_SECRET_KEY")
        return

    try:
        # 1. 查余额
        bal = get_balance()
        print(f"✅ 可用余额: {bal} USDT")
        
        if bal < 2:
            print("❌ 余额不足，无法测试")
            return
            
        # 2. 查价格并计算数量
        price = get_price(SYMBOL)
        print(f"📊 当前 {SYMBOL} 价格: {price}")
        
        # 计算下单数量 (名义价值 / 价格)
        # 币安 BTC 最小精度通常是 0.001，我们这里简单保留3位小数
        # 这里的TEST_AMOUNT_USDT是名义价值 (即 本金x杠杆)
        qty = round((TEST_AMOUNT_USDT * LEVERAGE) / price, 3)
        if qty == 0: qty = 0.001 # 兜底
        
        print(f"🧮 计划交易数量: {qty} BTC (名义价值 ~{qty*price:.2f} USDT)")
        
        # 3. 设置杠杆
        set_leverage(SYMBOL, LEVERAGE)
        
        # 4. 开多单
        input("👉 按回车键开始 [开多单] 测试 (产生真实交易)...")
        place_order(SYMBOL, 'BUY', qty)
        
        time.sleep(2)
        
        # 5. 查持仓
        pos = get_position(SYMBOL)
        if pos:
            print(f"✅ 当前持仓: {pos['amount']} BTC | 未实现盈亏: {pos['unrealizedProfit']}")
        else:
            print("❌ 未查询到持仓！")
            
        # 6. 平仓
        input("👉 按回车键开始 [平仓] 测试...")
        # 如果持仓是正数，卖出平仓；如果是负数，买入平仓
        if pos:
            side = 'SELL' if pos['amount'] > 0 else 'BUY'
            abs_qty = abs(pos['amount'])
            place_order(SYMBOL, side, abs_qty, reduce_only=True)
            print("✅ 平仓完成")
            
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")

if __name__ == "__main__":
    main()
