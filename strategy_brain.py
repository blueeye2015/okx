import pandas as pd
import json
import csv
import os
import time
from datetime import datetime
import clickhouse_connect 

# ================= 配置区域 =================
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'ETHUSDT'
COMMAND_FILE = '/data/okx/trade_commands.csv'  # 指令管道
STATE_FILE = '/data/okx/strategy_state.json'   # 策略记忆

# 策略参数 (高波动+均值回归)
CONFIG = {
    'SYMBOL': 'ETHUSDT',
    'ENTRY_RATIO': -0.5,
    'ENTRY_CVD': -2_000_000,
    'VOL_MULTIPLIER': 1.0,
    
    # === 资金管理 (10x 杠杆专用) ===
    # 假设你有 1000U 本金，这里填的是"名义价值"
    # 建议：首单名义价值 = 总本金 * 20% (即实际占用 2% 本金)
    'BASE_AMOUNT': 1000,         # 名义价值 200U (实际占用 20U)
    
    # === 补仓逻辑 (宽间距) ===
    'DCA_MULTIPLIER': 1.3,      # 每次只加 1.3 倍
    'MAX_DCA_COUNT': 4,         # 最多补 3 次
    
    # 间距：2% -> 5% -> 11% (累计跌幅)
    # 这里的逻辑是：第一枪跌2%补，第二枪要再跌3%，第三枪要再跌6%
    'DCA_STEPS': [0.015, 0.025, 0.040, 0.060],
    
    # === 离场 ===
    'HARD_TP': 0.024,           # 10x杠杆下，币价涨 1.2%，本金就赚 12%，够了
    'HARD_SL': -0.08            # 灾难止损：如果总跌幅超过 15%，全平 (此时剩点渣)
}
# ===========================================

def get_client():
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    return client

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: return json.load(f)
    return {
        'status': 'EMPTY', 'avg_price': 0.0, 'total_qty': 0.0, 
        'dca_count': 0, 'total_invest': 0.0
    }

def save_state(state):
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=4)

def send_command(action, symbol, quantity, price, reason):
    """把指令写入管道文件"""
    file_exists = os.path.exists(COMMAND_FILE)
    with open(COMMAND_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'action', 'symbol', 'quantity', 'price', 'reason', 'status'])
        
        # 写入一条新指令 (Status 默认为 PENDING)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            action, symbol, quantity, price, reason, 'PENDING'
        ])
    print(f"📡 [指令发出] {action} {symbol} Qty:{quantity:.4f} ({reason})")

def fetch_market_data():
    """从 ClickHouse 获取最新K线和指标"""
    client = get_client()
    query = f"""
    SELECT time, close_price, net_cvd, cvd_ratio 
    FROM marketdata.features_15m 
    WHERE symbol = '{SYMBOL}' ORDER BY time DESC LIMIT 100
    """
    df = client.query_df(query)
    # --- [修正1] DataFrame 专用的判空方式 ---
    if df.empty:
        return None
    
    # --- [修正2] 统一列名 ---
    # 数据库列名是 close_price, net_cvd...
    # 但策略逻辑里用的是 close, cvd, ratio
    # 我们这里做一个映射，防止后面报错
    df = df.rename(columns={
        'close_price': 'close',
        'net_cvd': 'cvd',
        'cvd_ratio': 'ratio'
    })
    
    # --- [修正3] 排序 ---
    # SQL 取的是倒序(DESC)为了拿最新数据，但计算波动率需要正序(ASC)
    df = df.sort_values('time').reset_index(drop=True)
    
    # 计算波动率 (1小时 vs 24小时)
    df['ret'] = df['close'].pct_change()
    current_vol = df['ret'].tail(4).std()
    baseline_vol = df['ret'].tail(96).std()
    
    # 提取最新一行
    row = df.iloc[-1]
    
    return {
        'price': row['close'],
        'cvd': row['cvd'],
        'ratio': row['ratio'],
        # 增加容错：如果基准波动率为0或空，默认视为满足条件
        'is_high_vol': current_vol > (baseline_vol * CONFIG['VOL_MULTIPLIER']) if pd.notna(baseline_vol) and baseline_vol > 0 else True
    }

def run_brain():
    print(f"🧠 策略大脑启动 ({SYMBOL})...")
    
    data = fetch_market_data()
    if not data: return
    
    price = data['price']
    state = load_state()
    
    print(f"   当前价格: {price} | CVD: {data['cvd']/10000:.0f}万 | Ratio: {data['ratio']:.2f}")

    # --- 1. 空仓时：寻找开首单机会 ---
    if state['status'] == 'EMPTY':
        if (data['ratio'] < CONFIG['ENTRY_RATIO']) and \
           (data['cvd'] < CONFIG['ENTRY_CVD']) and \
           (data['is_high_vol']):
            
            # 计算首单数量
            invest_amt = CONFIG['BASE_AMOUNT']
            qty = round(invest_amt / price, 4) # ETH 保留3-4位小数
            
            send_command('OPEN_LONG', SYMBOL, qty, price, 'Panic Entry (Base)')
            
            # 更新状态
            state.update({
                'status': 'HOLDING',
                'avg_price': price,
                'total_qty': qty,
                'total_invest': invest_amt,
                'dca_count': 0
            })
            save_state(state)

    # --- 2. 持仓时：马丁格尔管理 ---
    elif state['status'] == 'HOLDING':
        avg_price = state['avg_price']
        # 计算相对于【均价】的盈亏
        pnl_pct = (price - avg_price) / avg_price
        
        # A. 离场判断 (止盈/回正抛出)
        # 逻辑：(指标回暖 AND 微利) OR (达到硬止盈)
        is_reversal = (data['ratio'] > 0) or (data['cvd'] > 0)
        
        if (is_reversal and pnl_pct > 0.001) or (pnl_pct > CONFIG['HARD_TP']):
            # 全平：卖出当前所有持仓
            send_command('CLOSE_LONG', SYMBOL, state['total_qty'], price, 'Take Profit')
            state['status'] = 'EMPTY'
            save_state(state)
            return
            
        # B. 灾难止损 (防止归零)
        if pnl_pct < CONFIG['HARD_SL']:
            send_command('CLOSE_LONG', SYMBOL, state['total_qty'], price, 'Hard Stop Loss')
            state['status'] = 'EMPTY'
            save_state(state)
            return

        # ======================================
        # C. 马丁补仓逻辑 (DCA Logic)
        # ======================================
        current_step = state['dca_count']
        
        # 只有在次数未满时才检查
        if current_step < CONFIG['MAX_DCA_COUNT']:
            # 获取当前级别的触发跌幅 (例如第1次是 -1.5%)
            target_drop = -CONFIG['DCA_STEPS'][current_step]
            
            # 只有跌幅达标才补
            if pnl_pct <= target_drop:
                # --- [关键] 计算倍投金额 ---
                # 第1次(idx=0): 1000 * 1.5^1 = 1500 U
                # 第2次(idx=1): 1000 * 1.5^2 = 2250 U
                next_invest = CONFIG['BASE_AMOUNT'] * (CONFIG['DCA_MULTIPLIER'] ** (current_step + 1))
                
                # 换算成币的数量
                next_qty = round(next_invest / price, 4)
                
                reason = f"DCA Buy #{current_step + 1} (Drop {pnl_pct*100:.1f}%)"
                send_command('DCA_BUY', SYMBOL, next_qty, price, reason)
                
                # --- [关键] 重新计算均价 ---
                new_total_invest = state['total_invest'] + next_invest
                new_total_qty = state['total_qty'] + next_qty
                new_avg_price = new_total_invest / new_total_qty
                
                # 更新状态
                state.update({
                    'total_invest': new_total_invest,
                    'total_qty': new_total_qty,
                    'avg_price': new_avg_price, # 均价被拉低了！
                    'dca_count': current_step + 1
                })
                save_state(state)
                print(f"📉 [补仓成功] 新均价: {new_avg_price:.2f} | 累计投入: {new_total_invest:.0f} U")

if __name__ == "__main__":
    while True:
        try:
            run_brain()
            time.sleep(60) # 10秒轮询一次
        except KeyboardInterrupt:
            print("\n程序已停止")
            break
        except Exception as e:
            print(f"⚠️ 主循环错误: {e}")
            time.sleep(60)
    