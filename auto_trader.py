import time
import requests
import json
import os
import logging
import math
from urllib.parse import urlencode, quote
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import base64
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# 加载环境变量
load_dotenv('/data/okx/.env')

# ==========================================
# 1. 配置区域
# ==========================================
API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()

BASE_URL = 'https://fapi.binance.com'
SIGNAL_CSV_PATH = '/data/okx/reversal_signals2.csv'

# 交易参数
SYMBOL = 'BTCUSDT'
LEVERAGE = 5
USDT_AMOUNT = 20.0     # 每次投入本金

# 风控参数 (优先使用CSV信号里的价格，如果没有则用下面的默认比例)
DEFAULT_TP_RATE = 0.040  # 止盈扩大到 4.0% (吃大肉)
DEFAULT_SL_RATE = 0.010  # 止损保持 1.0% (严格风控)
BREAKEVEN_TRIGGER = 0.012 # [新增] 浮盈达到 1.2% 时，触发保本损

CHECK_INTERVAL = 3
PROXIES = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}

# ==========================================
# 2. 日志配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("auto_trader_v3.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ==========================================
# 3. 核心 API 封装 (无需变动)
# ==========================================
TIME_OFFSET = 0

def sync_server_time():
    global TIME_OFFSET
    logger.info("⏳ 校准服务器时间...")
    try:
        res = requests.get(f"{BASE_URL}/fapi/v1/time", proxies=PROXIES, timeout=5)
        res.raise_for_status()
        server_time = res.json()['serverTime']
        local_time = int(time.time() * 1000)
        TIME_OFFSET = server_time - local_time
        logger.info(f"✅ 校准完成，偏移量: {TIME_OFFSET}ms")
    except Exception as e:
        logger.error(f"❌ 时间校准失败: {e}")

def get_signature(payload):
    private_key_str = SECRET_KEY.strip()
    if not private_key_str.startswith("-----BEGIN"):
        private_key_str = f"-----BEGIN PRIVATE KEY-----\n{private_key_str}\n-----END PRIVATE KEY-----"
    
    try:
        private_key = load_pem_private_key(private_key_str.encode('utf-8'), password=None)
        signature = private_key.sign(payload.encode('utf-8'))
        return base64.b64encode(signature).decode('utf-8')
    except Exception as e:
        logger.error(f"签名失败: {e}")
        raise

def send_request(method, endpoint, params=None):
    if params is None: params = {}
    params['timestamp'] = int(time.time() * 1000) + TIME_OFFSET
    params['recvWindow'] = 60000
    
    query_string = urlencode(params)
    signature = get_signature(query_string)
    full_url = f"{BASE_URL}{endpoint}?{query_string}&signature={quote(signature)}"
    
    headers = {'X-MBX-APIKEY': API_KEY, 'Content-Type': 'application/json'}
    
    try:
        response = requests.request(method, full_url, headers=headers, proxies=PROXIES, timeout=10)
        if response.status_code >= 400:
            logger.error(f"API Error ({response.status_code}): {response.text}")
            return None
        return response.json()
    except Exception as e:
        logger.error(f"Request Exception: {e}")
        return None

# ==========================================
# 4. 业务逻辑函数 (增强健壮性)
# ==========================================
def init_exchange():
    """初始化杠杆和持仓模式"""
    logger.info("⚙️ 初始化交易所设置...")
    send_request('POST', '/fapi/v1/leverage', {'symbol': SYMBOL, 'leverage': LEVERAGE})
    try:
        # 强制单向持仓
        res = send_request('GET', '/fapi/v1/positionSide/dual')
        if res and res['dualSidePosition']:
            logger.info("⚠️ 切换为单向持仓模式...")
            send_request('POST', '/fapi/v1/positionSide/dual', {'dualSidePosition': 'false'})
    except Exception as e:
        logger.warning(f"持仓模式检查跳过: {e}")

def get_current_price(symbol):
    res = send_request('GET', '/fapi/v1/ticker/price', {'symbol': symbol})
    return float(res['price']) if res else None

def get_position(symbol):
    """获取当前持仓"""
    res = send_request('GET', '/fapi/v2/positionRisk', {'symbol': symbol})
    if res:
        for pos in res:
            amt = float(pos['positionAmt'])
            if amt != 0:
                return {
                    'side': 'LONG' if amt > 0 else 'SHORT',
                    'amount': abs(amt),
                    'pnl': float(pos['unrealizedProfit'])
                }
    return None

def cancel_open_orders(symbol):
    """【关键】取消所有挂单"""
    logger.info("🧹 撤销所有挂单(止盈止损)...")
    send_request('DELETE', '/fapi/v1/allOpenOrders', {'symbol': symbol})

def place_order(symbol, side, quantity, order_type='MARKET', price=None, reduce_only=False, stop_price=None):
    """通用下单函数"""
    params = {
        'symbol': symbol,
        'side': side.upper(),
        'type': order_type,
        'quantity': quantity,
    }
    if reduce_only: params['reduceOnly'] = 'true'
    if price: params['price'] = price
    if stop_price: params['stopPrice'] = stop_price
    
    if order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
        params['closePosition'] = 'true'
        del params['quantity']
    
    res = send_request('POST', '/fapi/v1/order', params)
    if res and 'orderId' in res:
        logger.info(f"✅ 订单成功 ({side} {order_type}): ID {res['orderId']}")
        return res
    return None

def check_and_move_sl_to_breakeven():
    """
    巡航监控：如果浮盈达标，将止损上移至开仓价 (保本)
    """
    try:
        # 1. 获取当前持仓
        pos = get_position(SYMBOL)
        if not pos: return # 空仓不处理

        price = get_current_price(SYMBOL)
        if not price: return

        # 2. 计算当前浮盈比例 (不带杠杆)
        entry_price = float(pos['entry_price'])
        if pos['side'] == 'LONG':
            pnl_pct = (price - entry_price) / entry_price
        else:
            return # 我们只做多，忽略空单逻辑

        # 3. 判断是否触发保本逻辑
        if pnl_pct > BREAKEVEN_TRIGGER:
            # 获取当前所有挂单
            open_orders = send_request('GET', '/fapi/v1/openOrders', {'symbol': SYMBOL})
            if not open_orders: return

            # 找到现有的止损单 (STOP_MARKET)
            current_sl_order = None
            for order in open_orders:
                if order['type'] == 'STOP_MARKET':
                    current_sl_order = order
                    break
            
            # 如果没有止损单，或者止损单价格已经比开仓价高了(已经保护过了)，就跳过
            if current_sl_order:
                current_stop_price = float(current_sl_order['stopPrice'])
                
                # 稍微加一点点手续费 buffer (比如 entry * 1.001)
                new_stop_price = round(entry_price * 1.001, 1)

                # 只有当 新止损 > 旧止损 时才修改 (只上移，不下移)
                if new_stop_price > current_stop_price:
                    logger.info(f"💰 浮盈达标 ({pnl_pct*100:.2f}%)，触发保本逻辑！")
                    logger.info(f"🛡️ 修改止损: {current_stop_price} -> {new_stop_price} (保本+微利)")
                    
                    # 币安修改订单接口 (PUT /fapi/v1/order) - 也可以先撤后挂，修改接口更稳
                    # 但为了代码简单通用，我们采用：撤销旧止损 -> 挂新止损
                    
                    # 1. 撤销旧止损
                    send_request('DELETE', '/fapi/v1/order', {
                        'symbol': SYMBOL, 
                        'orderId': current_sl_order['orderId']
                    })
                    
                    # 2. 挂新止损 (STOP_MARKET)
                    # 注意：止损单不需要 quantity 参数如果 reduceOnly=true 不支持，
                    # 最好是用 place_order 的 closePosition=true 模式
                    place_order(SYMBOL, 'SELL', pos['amount'], order_type='STOP_MARKET', stop_price=new_stop_price)

    except Exception as e:
        logger.error(f"保本监控出错: {e}")

def close_position(pos):
    """
    【封装】安全平仓逻辑
    1. 撤销所有挂单 (防止平仓后止损单被误触)
    2. 市价全平
    """
    logger.info(f"🛡️ 正在平掉 {pos['side']} 仓位...")
    
    # 1. 先撤单
    cancel_open_orders(SYMBOL)
    
    # 2. 再平仓
    side = 'SELL' if pos['side'] == 'LONG' else 'BUY'
    
    # 简单的重试机制
    for i in range(3):
        res = place_order(SYMBOL, side, pos['amount'], reduce_only=True)
        if res:
            logger.info("🎉 平仓成功，落袋为安。")
            return True
        else:
            logger.warning(f"⚠️ 平仓失败，第 {i+1} 次重试...")
            time.sleep(1)
            
    logger.error("❌❌❌ 严重警告：平仓失败，请手动检查！")
    return False

def calculate_quantity(price, usdt_amt, leverage):
    if price <= 0: return 0.0
    notional = usdt_amt * leverage
    raw_qty = notional / price
    qty = math.floor(raw_qty * 1000) / 1000
    if qty < 0.001: qty = 0.001
    return qty

# ==========================================
# 5. 信号执行逻辑 (只做多 + 平仓)
# ==========================================
def execute_trade(signal_row):
    sig_val = int(signal_row['Signal']) 
    sig_type = signal_row.get('Type', 'Unknown')
    
    price = get_current_price(SYMBOL)
    if not price: return
    
    pos = get_position(SYMBOL)
    
    # ==========================================
    # 场景 A: 做多信号 (Signal = 1) -> 开仓 / 持有
    # ==========================================
    if sig_val == 1:
        logger.info(f"\n⚡ 收到开多信号: {sig_type}")
        
        # 1. 如果已有多单 -> 保持不动 (或者可以加仓，这里暂且保持)
        if pos and pos['side'] == 'LONG':
            logger.info("🍵 当前已持有优质多单，继续持有。")
            return
            
        # 2. 如果有空单 (异常情况) -> 立即平掉，准备反手
        if pos and pos['side'] == 'SHORT':
            logger.info("🔄 发现空单，立即平仓反手...")
            close_position(pos)
        
        # 3. 执行开仓
        qty = calculate_quantity(price, USDT_AMOUNT, LEVERAGE)
        logger.info(f"🚀 执行开多: {qty} BTC")
        
        # 开单前先清理可能的残留挂单
        cancel_open_orders(SYMBOL) 
        
        if place_order(SYMBOL, 'BUY', qty):
            # 4. 挂止盈止损 (OTOCO)
            csv_tp = float(signal_row.get('TP_Price', 0))
            csv_sl = float(signal_row.get('SL_Price', 0))
            
            # 优先用 CSV 里的价格，没有则用默认比例
            tp_price = csv_tp if csv_tp > price else price * (1 + DEFAULT_TP_RATE)
            sl_price = csv_sl if csv_sl < price and csv_sl > 0 else price * (1 - DEFAULT_SL_RATE)
            
            # 精度修正
            tp_price = round(tp_price, 1)
            sl_price = round(sl_price, 1)
            
            logger.info(f"🛡️ 部署风控: 止盈 {tp_price} | 止损 {sl_price}")
            
            # 挂单
            place_order(SYMBOL, 'SELL', qty, order_type='STOP_MARKET', stop_price=sl_price)
            place_order(SYMBOL, 'SELL', qty, order_type='TAKE_PROFIT_MARKET', stop_price=tp_price)

    # ==========================================
    # 场景 B: 做空信号 (Signal = -1) -> 仅仅平仓 / 空仓
    # ==========================================
    elif sig_val == -1:
        logger.info(f"\n🛑 收到做空(顶部)信号: {sig_type}")
        
        # 1. 如果有多单 -> 立即平仓逃顶
        if pos and pos['side'] == 'LONG':
            logger.info("🏃‍♂️ 触发逃顶逻辑，平掉多单...")
            close_position(pos)
            logger.info("✅ 已空仓，等待回调。")
            
        # 2. 如果为空仓 -> 保持观望，不开空
        elif not pos:
            logger.info("👀 当前空仓，信号指示顶部风险，继续观望 (不执行开空)。")
            
        # 3. 如果有空单 -> 保持
        else:
            logger.info("🍵 持有空单中。")

# ==========================================
# 6. 主程序
# ==========================================
def get_latest_signal():
    try:
        if not os.path.exists(SIGNAL_CSV_PATH): return None
        df = pd.read_csv(SIGNAL_CSV_PATH)
        if df.empty: return None
        return df.iloc[-1]
    except Exception as e:
        logger.error(f"CSV 读取错误: {e}")
        return None

def main():
    logger.info("🦈 自动交易机器人 V3 (Long-Only + Escape Mode) 启动...")
    logger.info(f"💰 单笔本金: {USDT_AMOUNT} U | 杠杆: {LEVERAGE}x")
    logger.info("📝 策略模式: 收到做多信号开多，收到做空信号平多 (不开空)")
    
    sync_server_time()
    init_exchange()
    
    last_sig = get_latest_signal()
    last_processed_time = last_sig['Time'] if last_sig is not None else None
    
    logger.info(f"⏳ 监控中... (最后信号: {last_processed_time})")

    while True:
        try:
            latest_sig = get_latest_signal()
            
            if latest_sig is not None:
                curr_time = latest_sig['Time']
                
                # 检查新信号
                if curr_time != last_processed_time:
                    # 检查时效 (15分钟内)
                    sig_dt = pd.to_datetime(curr_time)
                    now_dt = datetime.now(sig_dt.tz)
                    
                    if (now_dt - sig_dt).total_seconds() < 900:
                        execute_trade(latest_sig)
                        last_processed_time = curr_time
                    else:
                        logger.warning(f"⚠️ 信号已过期，跳过: {curr_time}")
                        last_processed_time = curr_time
                        
            # ==============================
            # [新增] 每轮循环都检查一次保本
            # ==============================
            check_and_move_sl_to_breakeven()

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("程序停止。")
            break
        except Exception as e:
            logger.error(f"异常: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()