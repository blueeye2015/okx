import ccxt
import pandas as pd
import time
import json
import os
from datetime import datetime

# ================= 配置区 (币安专用) =================
API_CONFIG = {
    'apiKey': '你的币安API_KEY',
    'secret': '你的币安SECRET_KEY',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',  # ⚠️ 核心：告诉CCXT我们要操作"U本位合约"
    }
}

# 币安 U本位永续合约通常直接用 BTC/USDT
SYMBOL = 'BTC/USDT' 

# ⚠️ 币安下单单位是【币的个数】
# OKX 的 1张 可能是 100美元，但币安的 0.01 就是 0.01个BTC (价值约900美元)
# 请务必确认你的下单量！
POSITION_AMOUNT = 0.002  # 每次买 0.002 BTC
LEVERAGE = 5             # 杠杆倍数

# 信号确认阈值
CONFIRM_THRESHOLD = 2 
SIGNAL_FILE = '/data/okx/universal_signals.csv'

# ================= 初始化 =================
exchange = ccxt.binance(API_CONFIG) 
# exchange.set_sandbox_mode(True) # 币安也有测试网，如果需要请解开

# 全局变量
last_signal = 0
signal_count = 0

def init_exchange_settings():
    """初始化币安合约设置"""
    print("⚙️ 正在初始化币安设置...")
    try:
        # 1. 设置杠杆
        # 币安需要先加载市场信息才能设杠杆
        exchange.load_markets()
        exchange.set_leverage(LEVERAGE, SYMBOL)
        print(f"✅ 杠杆已设置为: {LEVERAGE}x")
        
        # 2. 强制设置为【单向持仓】(One-way Mode)
        # 币安 API: set_position_mode(hedged=False)
        try:
            exchange.set_position_mode(False, SYMBOL)
            print("✅ 已确认为单向持仓模式")
        except Exception as e:
            # 如果已经是单向模式，API可能会报错，忽略即可
            # print(f"ℹ️ 持仓模式检查: {e}") 
            pass
            
    except Exception as e:
        print(f"⚠️ 初始化设置警告: {e}")
        print("💡 提示: 如果报错 'No need to change' 可以忽略")

def get_contract_position():
    """获取币安合约持仓"""
    try:
        # fetch_positions 在币安会返回一堆币的持仓，需要过滤
        positions = exchange.fetch_positions([SYMBOL])
        for pos in positions:
            # 币安的数据结构里，symbol 可能是 BTCUSDT
            # 且我们只关心持仓量 > 0 的
            if pos['symbol'] == SYMBOL and float(pos['contracts']) > 0:
                return {
                    'side': pos['side'], # 'long' 或 'short'
                    'amount': float(pos['contracts']), # 持仓数量 (币数)
                    'entry_price': float(pos['entryPrice']),
                    'unrealized_pnl': float(pos['unrealizedPnl'])
                }
        return None # 空仓
    except Exception as e:
        print(f"❌ 获取持仓失败: {e}")
        return None

def execute_trade(action, amount=None):
    """
    执行交易 (逻辑与 OKX 通用)
    """
    try:
        size = amount if amount else POSITION_AMOUNT
        order = None
        
        if action == 'open_long':
            print(f"🚀 [币安开多] 买入 {size} BTC...")
            order = exchange.create_market_buy_order(SYMBOL, size)
            
        elif action == 'open_short':
            print(f"🚀 [币安开空] 卖出 {size} BTC...")
            order = exchange.create_market_sell_order(SYMBOL, size)
            
        elif action == 'close_long':
            print(f"🛑 [币安平多] 卖出平仓 {size}...")
            # 币安单向模式下，直接反向卖出即可平仓，参数与 OKX 类似
            order = exchange.create_market_sell_order(SYMBOL, size, params={'reduceOnly': True})
            
        elif action == 'close_short':
            print(f"🛑 [币安平空] 买入平仓 {size}...")
            order = exchange.create_market_buy_order(SYMBOL, size, params={'reduceOnly': True})
            
        print(f"✅ 订单成交ID: {order['id']}")
        return True
        
    except Exception as e:
        print(f"❌ 下单失败: {e}")
        return False

def main():
    print(f"🤖 币安交易机器人启动 (需连续 {CONFIRM_THRESHOLD} 次信号)...")
    init_exchange_settings()
    
    global last_signal, signal_count
    
    while True:
        try:
            if not os.path.exists(SIGNAL_FILE):
                time.sleep(5); continue
                
            # 读取最后一行信号
            df = pd.read_csv(SIGNAL_FILE).tail(1)
            if df.empty: continue
            row = df.iloc[0]
            
            # 时间检查 (900秒过期)
            sig_time = pd.to_datetime(row['Time'])
            if (datetime.now() - sig_time).total_seconds() > 900:
                print(f"⏳ 信号过期...", end='\r')
                time.sleep(10); continue

            # 信号解析
            signal = int(row['Signal']) 
            
            # --- 信号确认逻辑 (同之前) ---
            print(f"\n🕒 {datetime.now().strftime('%H:%M:%S')} | 信号:{signal}", end='')
            if signal == last_signal:
                signal_count += 1
                print(f" | 确认次数: {signal_count}")
            else:
                print(f" | 变化 -> 重置")
                signal_count = 1
                last_signal = signal
            
            effective_signal = signal if signal_count >= CONFIRM_THRESHOLD else 0
            
            # 获取持仓
            pos = get_contract_position()
            pos_side = pos['side'] if pos else 'none'
            
            # --- 状态机 (同之前) ---
            if effective_signal == 1: # 做多
                if pos_side == 'none':
                    execute_trade('open_long')
                elif pos_side == 'short':
                    execute_trade('close_short', pos['amount'])
                    time.sleep(1)
                    execute_trade('open_long')
                    
            elif effective_signal == 2: # 做空
                if pos_side == 'none':
                    execute_trade('open_short')
                elif pos_side == 'long':
                    execute_trade('close_long', pos['amount'])
                    time.sleep(1)
                    execute_trade('open_short')
            
            elif effective_signal == 0:
                 # 保持不动 (Wait)
                 if pos_side != 'none':
                    print(f"☕ 保持持仓 {pos_side}...")
                 else:
                    print("💤 空仓...")

            time.sleep(60)

        except Exception as e:
            print(f"⚠️ 循环错误: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()