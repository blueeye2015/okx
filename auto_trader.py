import time
import requests
import json
import os
import logging
import math
import csv
from urllib.parse import urlencode, quote
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import base64
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import traceback

# 加载环境变量
load_dotenv('/data/okx/.env')

# ==========================================
# 1. 配置区域
# ==========================================
API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()

BASE_URL = 'https://fapi.binance.com'

# 信号文件路径
SIGNAL_PATH = '/data/okx/reversal_signals2.csv'  # 只读取 Reversal 2

# 交易账本路径
HISTORY_LOG_PATH = '/data/okx/bot_trade_history.csv'

# 交易参数
SYMBOL = 'BTCUSDT'
LEVERAGE = 5
USDT_AMOUNT = 200.0    # 单次开仓本金 (既然只跑一个策略，可以把资金集中起来，比如200U)

# 风控参数 (全局兜底)
DEFAULT_TP_RATE = 0.040  # 止盈 4.0%
DEFAULT_SL_RATE = 0.010  # 止损 1.0%
BREAKEVEN_TRIGGER = 0.012 # 保本触发

CHECK_INTERVAL = 5
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
        logging.FileHandler("auto_trader_v5.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ==========================================
# 3. 核心 API 封装 (Ed25519)
# ==========================================
TIME_OFFSET = 0

# 【新增】创建一个全局 Session 对象，复用 TCP 连接
# 这能极大减少握手次数，解决 "Connection reset" 问题
session = requests.Session()
retries = Retry(
    total=3,                # 最大重试次数
    backoff_factor=0.5,     # 失败后等待时间: 0.5s, 1s, 2s...
    status_forcelist=[500, 502, 503, 504, 104], # 遇到这些状态码重试
    allowed_methods=["GET", "POST", "DELETE"]
)
# 挂载重试适配器
session.mount('https://', HTTPAdapter(max_retries=retries))
# 设置代理到 Session 级别
session.proxies.update(PROXIES)

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
        response = session.request(method, full_url, headers=headers, timeout=(5, 10))
        if response.status_code >= 400:
            logger.error(f"API Error ({response.status_code}): {response.text}")
            return None
        return response.json()
    except requests.exceptions.ConnectionError as e:
        # 专门捕获连接重置错误，并不再打印堆栈，只打印警告
        logger.warning(f"⚠️ 网络抖动 (Connection Reset)，将自动重试... {e}")
    except Exception as e:
        logger.error(f"Request Exception: {e}")
        return None

# ==========================================
# 4. 账户记录模块 (New!)
# ==========================================
def record_transaction(strategy_source, action, price, quantity, pnl=0, note=""):
    """
    记录交易流水到 CSV
    """
    file_exists = os.path.exists(HISTORY_LOG_PATH)
    
    # 估算手续费 (Taker 0.05%)
    fee = (price * quantity) * 0.0005
    
    # 获取当前账户余额 (为了记录资金曲线)
    balance = 0
    try:
        res = send_request('GET', '/fapi/v2/balance')
        if res:
            for b in res:
                if b['asset'] == 'USDT':
                    balance = float(b['balance'])
                    break
    except: pass

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(HISTORY_LOG_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写表头
        if not file_exists:
            writer.writerow(['Time', 'Strategy', 'Action', 'Price', 'Quantity', 'Fee(Est)', 'PnL', 'Balance', 'Note'])
        
        writer.writerow([now_str, strategy_source, action, price, quantity, round(fee, 4), round(pnl, 4), round(balance, 2), note])
    
    logger.info(f"📝 交易已记录: {action} {quantity} @ {price} (PnL: {pnl})")

# ==========================================
# 5. 业务逻辑函数
# ==========================================
def init_exchange():
    logger.info("⚙️ 初始化交易所...")
    send_request('POST', '/fapi/v1/leverage', {'symbol': SYMBOL, 'leverage': LEVERAGE})
    try:
        res = send_request('GET', '/fapi/v1/positionSide/dual')
        if res and res['dualSidePosition']:
            send_request('POST', '/fapi/v1/positionSide/dual', {'dualSidePosition': 'false'})
    except: pass

def get_current_price(symbol):
    res = send_request('GET', '/fapi/v1/ticker/price', {'symbol': symbol})
    return float(res['price']) if res else None

def get_position(symbol):
    res = send_request('GET', '/fapi/v2/positionRisk', {'symbol': symbol})
    if res:
        for pos in res:
            amt = float(pos['positionAmt'])
            if amt != 0:
                return {
                    'side': 'LONG' if amt > 0 else 'SHORT',
                    'amount': abs(amt),
                    'entry_price': float(pos['entryPrice']),
                    'pnl': float(pos['unRealizedProfit'])
                }
    return None

def cancel_open_orders(symbol):
    logger.info("🧹 撤销挂单...")
    send_request('DELETE', '/fapi/v1/allOpenOrders', {'symbol': symbol})

def place_order(symbol, side, quantity, order_type='MARKET', price=None, reduce_only=False, stop_price=None):
    params = {
        'symbol': symbol, 'side': side.upper(), 'type': order_type, 'quantity': quantity
    }
    if reduce_only: params['reduceOnly'] = 'true'
    if price: params['price'] = price
    if stop_price: params['stopPrice'] = stop_price
    if order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
        params['closePosition'] = 'true'
        del params['quantity']
    
    res = send_request('POST', '/fapi/v1/order', params)
    if res and 'orderId' in res:
        return res
    return None

def calculate_quantity(price, usdt_amt, leverage):
    if price <= 0: return 0.0
    notional = usdt_amt * leverage
    raw_qty = notional / price
    qty = math.floor(raw_qty * 1000) / 1000
    if qty < 0.001: qty = 0.001
    return qty

# ==========================================
# 6. 核心交易执行逻辑 (V5 并行版)
# ==========================================

def execute_trade(signal_row, price):
    # 1. 获取当前仓位状态
    pos = get_position(SYMBOL)
    
    sig_val = int(signal_row['Signal'])
    
    # === 场景 A: 做多信号 (1) ===
    if sig_val == 1:
        # 如果空仓，开多
        if not pos:
            logger.info(f"🚀 [Signal 1] 发现做多信号，当前空仓，执行开仓！")
            qty = calculate_quantity(price, USDT_AMOUNT, LEVERAGE)
            
            cancel_open_orders(SYMBOL)
            if place_order(SYMBOL, 'BUY', qty):
                record_transaction("Reversal 2 Escape","OPEN_LONG", price, qty)
                
                # 挂止盈止损
                csv_tp = float(signal_row.get('TP_Price', 0))
                csv_sl = float(signal_row.get('SL_Price', 0))
                # 优先用 CSV 价格，没有则用默认
                tp = csv_tp if csv_tp > price else price * (1 + DEFAULT_TP_RATE)
                sl = csv_sl if csv_sl < price and csv_sl > 0 else price * (1 - DEFAULT_SL_RATE)
                
                logger.info(f"🛡️ 挂单风控: SL {round(sl,1)} | TP {round(tp,1)}")
                # 等1秒确保成交
                time.sleep(1) 
                # 这里我们直接用 qty 挂止损，因为刚开仓，持仓量等于 qty
                # 如果怕网络延迟导致持仓没更新，可以用 closePosition=true 模式(不需要qty)
                place_order(SYMBOL, 'SELL', qty, order_type='STOP_MARKET', stop_price=round(sl,1))
                place_order(SYMBOL, 'SELL', qty, order_type='TAKE_PROFIT_MARKET', stop_price=round(tp,1))
        
        # 如果已持有做多，则忽略 (防止重复加仓)
        elif pos['side'] == 'LONG':
            logger.info("🍵 [Signal 1] 收到做多信号，但已持仓，继续持有。")
        
        # 如果持有空单(异常情况)，先平仓再反手(或者直接平仓)
        elif pos['side'] == 'SHORT':
            logger.info("🔄 [Signal 1] 持有空单，平仓反手...")
            cancel_open_orders(SYMBOL)
            place_order(SYMBOL, 'BUY', pos['amount'], reduce_only=True)
            # 这里简单处理，先平仓，下一轮循环再开多
    
    # === 场景 B: 做空信号 (-1) ===
    elif sig_val == -1:
        # 如果持有多单，必须平仓逃顶
        if pos and pos['side'] == 'LONG':
            logger.info(f"🏃‍♂️ [Signal -1] 触发逃顶信号！平掉多单...")
            cancel_open_orders(SYMBOL)
            if place_order(SYMBOL, 'SELL', pos['amount'], reduce_only=True):
                record_transaction("ESCAPE_LONG", pos['entry_price'], pos['amount'], pos['pnl'])
                logger.info("✅ 逃顶成功，已空仓。")
        
        # 如果空仓，则观望
        else:
            logger.info("🛑 [Signal -1] 收到做空信号，当前空仓，保持观望。")

def close_position(source_name, pos):
    """
    执行平仓逻辑
    """
    logger.info(f"🏃‍♂️ [{source_name}] 触发平仓/逃顶...")
    cancel_open_orders(SYMBOL)
    
    side = 'SELL' if pos['side'] == 'LONG' else 'BUY'
    res = place_order(SYMBOL, side, pos['amount'], reduce_only=True)
    
    if res:
        # 记录账本
        # 注意：这里记录的是大致的 Realized PnL，准确值需要查 UserTrades，这里用未结盈亏近似记录
        record_transaction(source_name, "CLOSE_ALL", pos['entry_price'], pos['amount'], pos['pnl'], "Escape/Exit")
        logger.info("✅ 平仓成功。")

def check_breakeven():
    """保本逻辑"""
    try:
        pos = get_position(SYMBOL)
        if not pos or pos['side'] != 'LONG': return
        
        price = get_current_price(SYMBOL)
        if not price: return
        
        entry = pos['entry_price']
        pnl_pct = (price - entry) / entry
        
        if pnl_pct > BREAKEVEN_TRIGGER:
            open_orders = send_request('GET', '/fapi/v1/openOrders', {'symbol': SYMBOL})
            if not open_orders: return
            
            curr_sl = None
            for o in open_orders: 
                if o['type'] == 'STOP_MARKET': 
                    curr_sl = float(o['stopPrice'])
                    break
            
            new_sl = round(entry * 1.001, 1)
            
            # 如果没有止损单，或者新止损 > 旧止损
            if curr_sl is None or new_sl > curr_sl:
                logger.info(f"💰 浮盈 {pnl_pct*100:.2f}%，触发保本 (SL -> {new_sl})")
                cancel_open_orders(SYMBOL)
                # 重新挂止损，止盈
                # 注意：这里简单暴力全撤全挂，防止漏单
                place_order(SYMBOL, 'SELL', pos['amount'], order_type='STOP_MARKET', stop_price=new_sl)
                # 止盈单如果被撤了要补回去吗？为了简单，这里建议只管止损，止盈可以不补或者补一个默认的
                # 实盘中最好只修改止损单，不撤止盈。但API修改复杂。
                # 简单处理：补一个 4% 的默认止盈
                tp_default = round(entry * (1 + DEFAULT_TP_RATE), 1)
                place_order(SYMBOL, 'SELL', pos['amount'], order_type='TAKE_PROFIT_MARKET', stop_price=tp_default)
                
    except Exception as e:
        logger.error(f"保本检查错: {e}")

# ==========================================
# 7. 信号读取器
# ==========================================
def get_signal(path):
    try:
        if not os.path.exists(path): return None
        df = pd.read_csv(path)
        if df.empty: return None
        return df.iloc[-1]
    except: return None

# ==========================================
# 8. 主循环
# ==========================================
def main():
    logger.info("🦈 Auto Trader V5 (双核并行 + 独立账本) 启动...")
    logger.info(f"📂 监听策略2: {SIGNAL_PATH} (Reversal 2)")
    logger.info(f"📒 交易账本: {HISTORY_LOG_PATH}")
    
    sync_server_time()
    init_exchange()
    
    # 初始化信号时间状态
    last_time_2 = None
    

    
    s2 = get_signal(SIGNAL_PATH)
    if s2 is not None: last_time_2 = s2['Time']

    logger.info("⏳ 监控开始...")

    while True:
        try:
            current_price = get_current_price(SYMBOL)
            if not current_price:
                time.sleep(CHECK_INTERVAL)
                continue

            # --- 2. 处理 Reversal 2 (多 + 逃顶) ---
            sig2 = get_signal(SIGNAL_PATH)
            if sig2 is not None and sig2['Time'] != last_time_2:
                t2 = sig2['Time']
                # 时效检查
                if (datetime.now(pd.to_datetime(t2).tz) - pd.to_datetime(t2)).total_seconds() < 1800:
                    val = int(sig2['Signal'])
                    
                    if val == 1: # 做多信号 -> 加仓
                        execute_trade(sig2, current_price)
                        
                    elif val == -1: # 做空信号 -> 逃顶 (平全仓)
                        pos = get_position(SYMBOL)
                        if pos and pos['side'] == 'LONG':
                            close_position("Reversal 2 Escape", pos)
                        else:
                            logger.info("🛑 Reversal 2 逃顶信号，但当前空仓，跳过。")
                            
                    last_time_2 = t2
                else:
                    last_time_2 = t2

            # --- 3. 保本巡航 ---
            check_breakeven()
            
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())  # 这行代码会告诉你是第几行报错
            time.sleep(5)

if __name__ == "__main__":
    main()