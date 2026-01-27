import clickhouse_connect
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

# ================= 配置区 =================
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'ETHUSDT'
SIGNAL_OUTPUT_PATH = '/data/okx/eth_signals.csv'

# 策略参数 (根据 12/26 行情调优)
LOOKBACK_WINDOW = 8       # 回看过去 8 根K线 (2小时) 寻找恐慌
PANIC_CVD_THRES = -20.0   # 定义什么是恐慌：单根 CVD 流出超过 20M
RSI_OVERSOLD = 45         # RSI 阈值 (宽松一点，因为我们要抓启动瞬间)
IGNITION_CVD = -10000000        # 启动信号：CVD 必须大于 1kw
IGNITION_WALL = -1.0       # 启动信号：墙必须增加 (不能撤单)

# 止盈止损 (反转单盈亏比通常很好)
TP_PCT = 0.015  # 止盈 1.5% (吃反弹)
SL_PCT = 0.008  # 止损 0.8% (跌破前低就跑)

# 全局变量防重复
last_processed_data_time = None

def get_latest_market_data():
    """获取最近 3 小时的数据用于计算指标"""
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    
    # 多拉一点数据算 RSI
    query = f"""
    SELECT 
        time, close_price, 
        wall_shift_pct, net_cvd
    FROM marketdata.features_15m
    WHERE symbol = '{SYMBOL}'
    ORDER BY time DESC
    LIMIT 50
    """
    df = client.query_df(query)
    # 转为正序 (旧 -> 新)
    df = df.sort_values('time').reset_index(drop=True)
    
    return df

def calculate_signals(df):
    """核心逻辑：计算背离和反转"""
    # 1. 计算 RSI (14)
    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 2. 提取当前 K 线 (最新一根)
    curr = df.iloc[-1]
    
    # 3. 提取过去 N 根 K 线 (不包含当前) 用于对比
    history = df.iloc[-LOOKBACK_WINDOW-1 : -1] 
    
    # --- 逻辑判断 ---
    
    # A. 恐慌判定: 过去一段时间，是否有过暴跌式流出？
    # 检查历史中最小的 CVD 是否小于阈值 (例如 -34)
    #has_panic_history = history['net_cvd'].min() < PANIC_CVD_THRES
    
    # B. 磨底判定: 当前价格是否在低位？
    # 当前价格 <= 历史最低价 * 1.003 (允许 0.3% 的误差，即没有飞得太高)
    lowest_price = history['close_price'].min()
    is_at_bottom = curr['close_price'] <= lowest_price 
    
    # C. 启动判定 (Trigger): 现在的资金和墙怎么样？
    # 12/26 08:45 的情况：CVD +16, Wall +0.10
    is_ignition = (curr['net_cvd'] < IGNITION_CVD) and (curr['wall_shift_pct'] < IGNITION_WALL)
    
    # D. RSI 辅助
    is_oversold = curr['rsi'] < RSI_OVERSOLD
    
    # 综合信号
    signal = 0
    reason = "WAIT"
    
    # 只有当：有过恐慌 + 现在还在底部 + 突然资金进场 + RSI不高 -> 买入！
    if  is_at_bottom and is_ignition and is_oversold:
        signal = 1
        reason = "🚀 REVERSAL (背离启动)"
    
    # 调试日志 (方便你观察当前状态)
    debug_info = {
        'crrent_rsi': curr['rsi'] ,
        'Panic_Min_CVD': history['net_cvd'].min(),
        'Price_vs_Low': f"{curr['close_price']:.1f}/{lowest_price:.1f}",
        'Cur_CVD': curr['net_cvd'],
        'Cur_Wall': curr['wall_shift_pct']
    }
    
    return signal, reason, debug_info

def run_monitor():
    global last_processed_data_time
    
    system_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        df = get_latest_market_data()
    except Exception as e:
        print(f"[{system_time}] ❌ 数据获取失败: {e}")
        return

    # 获取最新数据时间
    data_time = df.iloc[-1]['time']
    current_price = df.iloc[-1]['close_price']
    
    # 防重复
    if data_time == last_processed_data_time:
        print(f"[{system_time}] ⏳ K线未更新 ({data_time}) ...", end='\r')
        return
    
    last_processed_data_time = data_time
    
    # 计算信号
    signal, reason, debug = calculate_signals(df)
    
    # 打印看板
    print(f"\n[{system_time}] ⚡ K线更新: {data_time}")
    print(f"   价格: {current_price:.2f} | 信号: {reason}")
    print(f"   rsi:{debug['crrent_rsi']}")
    print(f"   状态: 历史恐慌CVD({debug['Panic_Min_CVD']:.1f}) | 离底幅度({debug['Price_vs_Low']})")
    print(f"   触发: 当前CVD({debug['Cur_CVD']:.1f}) | 当前Wall({debug['Cur_Wall']:.4f})")
    
    # 如果有信号，写入 CSV
    if signal == 1:
        tp_price = current_price * (1 + TP_PCT)
        sl_price = current_price * (1 - SL_PCT)
        
        output_row = {
            'Log_Time': system_time,
            'Time': data_time,
            'Price': current_price,
            'Signal': signal,
            'Type': reason,
            'TP_Price': round(tp_price, 2),
            'SL_Price': round(sl_price, 2),
            'Prob': 0.88, # 这种形态胜率通常很高，给个假概率方便 executor 读取
            'Desc': f"CVD:{debug['Cur_CVD']:.1f}"
        }
        
        df_out = pd.DataFrame([output_row])
        if not os.path.exists(SIGNAL_OUTPUT_PATH):
            df_out.to_csv(SIGNAL_OUTPUT_PATH, index=False)
        else:
            df_out.to_csv(SIGNAL_OUTPUT_PATH, mode='a', header=False, index=False)
        
        print(f"✅ 信号已发送至: {SIGNAL_OUTPUT_PATH}")

if __name__ == "__main__":
    print("🦈 反转猎手监控程序启动 (Target: 12/26 Pattern)...")
    print(f"   配置: 寻找过去 {LOOKBACK_WINDOW} 根K线内的恐慌 (CVD < {PANIC_CVD_THRES})")
    print(f"   触发: CVD > {IGNITION_CVD} 且 Wall > 0")
    
    while True:
        try:
            run_monitor()
            time.sleep(60) # 10秒刷一次，等待 ClickHouse 数据更新
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ 错误: {e}")
            time.sleep(60)