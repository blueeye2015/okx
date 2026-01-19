import pandas as pd
import numpy as np
import os
import json
import csv
from datetime import datetime
import clickhouse_connect

# ==========================================
# 1. 生产环境配置
# ==========================================
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'ETHUSDT'
SIGNAL_OUTPUT_PATH = '/data/okx/reversal_signals2.csv'
STATE_FILE_PATH = '/data/okx/martingale_state.json'  # 用于记录补仓状态

# --- 策略参数 (Light Martingale) ---
STRATEGY_CONFIG = {
    # 入场门槛
    'ENTRY_RATIO': -0.5,        # 卖单占比 > 50%
    'ENTRY_CVD_USDT': -2_000_000, # 净流出 > 200万 U (ETH合适)
    
    # 波动率过滤
    'VOL_WINDOW_SHORT': 4,      # 短期波动 (1小时)
    'VOL_WINDOW_LONG': 96,      # 长期基准 (24小时)
    'VOL_MULTIPLIER': 1.0,      # 当前波动 > 基准 * 1.0 才开单

    # 马丁补仓配置 (3-Step DCA)
    'MAX_DCA_COUNT': 3,
    'DCA_STEPS': [0.015, 0.030, 0.060], # 跌 1.5%, 3%, 6% 补仓
    
    # 离场配置
    'HARD_TP': 0.015,           # 硬止盈 1.5%
    'HARD_SL': -0.15,           # 硬止损 15% (防黑天鹅)
}

# ==========================================
# 2. 基础设施函数
# ==========================================
def get_clickhouse_client():
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    return client(**CLICKHOUSE)

def load_state():
    """读取当前持仓状态"""
    if os.path.exists(STATE_FILE_PATH):
        try:
            with open(STATE_FILE_PATH, 'r') as f:
                return json.load(f)
        except:
            pass
    # 默认空仓状态
    return {
        'status': 'EMPTY',      # EMPTY, HOLDING
        'entry_price': 0.0,     # 第一笔开仓价
        'avg_price': 0.0,       # 当前持仓均价
        'total_qty': 0.0,       # 持仓数量 (以 ETH 计)
        'total_invest': 0.0,    # 总投入 (以 USDT 计)
        'dca_count': 0,         # 当前补仓次数
        'last_action_time': None
    }

def save_state(state):
    """保存状态"""
    with open(STATE_FILE_PATH, 'w') as f:
        json.dump(state, f, indent=4)

def append_signal_to_csv(action, price, reason, indicators):
    """写入信号文件"""
    file_exists = os.path.exists(SIGNAL_OUTPUT_PATH)
    
    with open(SIGNAL_OUTPUT_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        # 写表头
        if not file_exists:
            writer.writerow(['time', 'symbol', 'action', 'price', 'reason', 'cvd_ratio', 'net_cvd', 'volatility'])
        
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            SYMBOL,
            action,
            price,
            reason,
            round(indicators.get('cvd_ratio', 0), 2),
            round(indicators.get('net_cvd', 0), 0),
            round(indicators.get('vol_score', 0), 2)
        ])
    print(f"🚀 [SIGNAL] {action} triggered at {price} ({reason})")

# ==========================================
# 3. 核心计算逻辑
# ==========================================
def fetch_and_process_data():
    client = get_clickhouse_client()
    
    # 获取最近 24小时 + 1小时的数据 (用于计算波动率基准)
    # 我们需要 features_15m 表中的 close_price, net_cvd, cvd_ratio
    query = f"""
    SELECT time, close_price, net_cvd, cvd_ratio 
    FROM marketdata.features_15m 
    WHERE symbol = '{SYMBOL}' 
    ORDER BY time DESC 
    LIMIT 150
    """
    
    data = client.execute(query)
    if not data:
        print("⚠️ No data found in ClickHouse.")
        return None

    df = pd.DataFrame(data, columns=['time', 'close_price', 'net_cvd', 'cvd_ratio'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # 计算技术指标
    df['returns'] = df['close_price'].pct_change()
    
    # 短期波动率 (过去1小时)
    df['vol_short'] = df['returns'].rolling(STRATEGY_CONFIG['VOL_WINDOW_SHORT']).std()
    
    # 长期波动率基准 (过去24小时)
    df['vol_baseline'] = df['vol_short'].rolling(STRATEGY_CONFIG['VOL_WINDOW_LONG']).mean()
    
    return df.iloc[-1] # 返回最新的一行数据

# ==========================================
# 4. 决策引擎 (Decision Engine)
# ==========================================
def run_strategy():
    # 1. 获取数据
    row = fetch_and_process_data()
    if row is None: return

    current_price = row['close_price']
    
    # 2. 读取状态
    state = load_state()
    
    # 3. 计算关键指标
    # 波动率得分: 当前波动 / 基准波动 (大于1.0说明风浪大)
    vol_score = 0
    if row['vol_baseline'] > 0:
        vol_score = row['vol_short'] / row['vol_baseline']
    
    indicators = {
        'cvd_ratio': row['cvd_ratio'],
        'net_cvd': row['net_cvd'],
        'vol_score': vol_score
    }
    
    print(f"🔍 监控中 | Price: {current_price} | Ratio: {row['cvd_ratio']:.2f} | CVD: {row['net_cvd']/10000:.0f}万 | VolScore: {vol_score:.2f}")

    # ----------------------------------------------------
    # 场景 A: 当前空仓 (寻找开仓机会)
    # ----------------------------------------------------
    if state['status'] == 'EMPTY':
        # 逻辑: 恐慌抛售 (Ratio < -0.5) + 大额流出 + 高波动环境
        is_panic = (row['cvd_ratio'] < STRATEGY_CONFIG['ENTRY_RATIO']) and \
                   (row['net_cvd'] < STRATEGY_CONFIG['ENTRY_CVD_USDT'])
        
        is_high_vol = vol_score > STRATEGY_CONFIG['VOL_MULTIPLIER']
        
        if is_panic and is_high_vol:
            # 触发开仓
            append_signal_to_csv('OPEN_LONG', current_price, 'Panic+HighVol Entry', indicators)
            
            # 更新状态为持仓
            state['status'] = 'HOLDING'
            state['entry_price'] = current_price
            state['avg_price'] = current_price
            state['dca_count'] = 0
            state['total_qty'] = 1.0 # 这里只是标记，实际数量由下单程序决定
            save_state(state)

    # ----------------------------------------------------
    # 场景 B: 当前持仓 (管理 补仓/止盈)
    # ----------------------------------------------------
    elif state['status'] == 'HOLDING':
        avg_price = state['avg_price']
        pnl_pct = (current_price - avg_price) / avg_price
        
        # --- B1. 检查离场 (回正抛出) ---
        # 逻辑: 只要指标回正 (Ratio>0 或 CVD>0) 且 不亏太多，就跑
        is_reversal = (row['cvd_ratio'] > 0) or (row['net_cvd'] > 0)
        
        should_close = False
        close_reason = ""
        
        if is_reversal:
            should_close = True
            close_reason = "Sentiment Reversal"
        elif pnl_pct >= STRATEGY_CONFIG['HARD_TP']:
            should_close = True
            close_reason = "Hard TP Hit"
        elif pnl_pct <= STRATEGY_CONFIG['HARD_SL']:
            should_close = True
            close_reason = "Hard Stop Loss"
            
        if should_close:
            append_signal_to_csv('CLOSE_LONG', current_price, close_reason, indicators)
            # 重置为空仓
            state = load_state() # 重新初始化一个空的
            # 实际上 load_state 的默认值就是空，这里手动重置更安全
            state['status'] = 'EMPTY'
            state['dca_count'] = 0
            save_state(state)
            return

        # --- B2. 检查马丁补仓 (DCA) ---
        # 如果还没跑，看看是不是跌深了需要补仓
        current_dca = state['dca_count']
        
        if current_dca < STRATEGY_CONFIG['MAX_DCA_COUNT']:
            # 获取下一次补仓的跌幅阈值
            target_drop = -STRATEGY_CONFIG['DCA_STEPS'][current_dca]
            
            # 计算相对于均价的跌幅
            # 注意: 马丁通常是基于"均价"跌幅，或者是"上一次成交价"跌幅。
            # 这里使用最稳健的: 相对于均价跌了多少
            drop_from_avg = (current_price - avg_price) / avg_price
            
            if drop_from_avg <= target_drop:
                append_signal_to_csv(f'DCA_BUY_{current_dca+1}', current_price, f'Drop {drop_from_avg*100:.1f}%', indicators)
                
                # 模拟更新均价 (实际交易中需要读取交易所真实成交)
                # 假设倍投 1.5倍，这里做简单估算更新状态
                # 真实场景建议由执行器回写状态，或者这里只发信号，状态由 Balance Monitor 更新
                state['dca_count'] += 1
                
                # 简单均价更新模拟 (假设每次买 1, 1.5, 2.25...)
                # New Avg = (Old_Cost + New_Cost) / (Old_Qty + New_Qty)
                # 这里只更新次数，均价更新最好由下单机器人完成，或者下次循环读取交易所
                # 为防止重复信号，必须增加计数
                save_state(state)

if __name__ == "__main__":
    print(f"[{datetime.now()}] Running Martingale Signal Engine for {SYMBOL}...")
    run_strategy()