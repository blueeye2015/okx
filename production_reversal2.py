import sys
import time
import os
import pandas as pd
import numpy as np
import clickhouse_connect
from datetime import datetime

# ==========================================
# 1. 基础配置与修复
# ==========================================

CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTCUSDT'
SIGNAL_OUTPUT_PATH = '/data/okx/reversal_signals2.csv'

# 全局变量防重复
last_processed_data_time = None

# ==========================================
# 2. 核心逻辑参数 (基于 12/26 & 12/29 复盘)
# ==========================================
# 通用参数
RSI_PERIOD = 14

# --- 多头策略 (Long) 参数 ---
# 逻辑: 恐慌抛售后 + 强力资金回补
LONG_RSI_THRES = 45       # RSI 必须超卖 (宽松点给 45)
LONG_MIN_CVD = 10.0       # 启动资金至少 +10M
LONG_IGNITION_RATIO = 0.4 # 启动资金 / 恐慌资金 >= 0.4 (回补力度要够)
LONG_TP = 0.015           # 止盈 1.5%
LONG_SL = 0.008           # 止损 0.8%

# --- 空头策略 (Short) 参数 ---
# 逻辑: 价格创新高 + 资金巨量流出 (背离)
SHORT_MIN_SELL_CVD = -10.0 # 砸盘资金至少 -10M
SHORT_TP = 0.015           # 止盈 1.5%
SHORT_SL = 0.008           # 止损 0.8%

def get_latest_market_data():
    """获取最近 50 根K线用于计算指标"""
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    
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
    """
    核心信号计算逻辑
    同时监测: 🚀 底部反转 (Long) 和 🔻 顶部背离 (Short)
    """
    # 1. 计算 RSI
    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss.replace(0, 1)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 获取当前K线 (Latest) 和 历史K线
    curr = df.iloc[-1]
    # 取过去 4 根 K 线 (不含当前) 用于对比
    history_4bar = df.iloc[-5:-1] 
    
    signal = 0
    signal_type = "WAIT"
    desc = "观察中..."
    tp_pct = 0.0
    sl_pct = 0.0
    prob = 0.0

    # ==========================================
    # 策略 A: 底部恐慌反转 (Long Logic)
    # ==========================================
    # 1. RSI 超卖
    is_oversold = curr['rsi'] < LONG_RSI_THRES
    
    # 2. 寻找过去的恐慌盘 (过去4根里的最小 CVD)
    # 如果没负值，给个默认 -1 防止除零
    recent_panic_cvd = history_4bar['net_cvd'].min()
    if recent_panic_cvd > 0: recent_panic_cvd = -1.0
    
    # 3. 计算回补力度 (Ignition Ratio)
    # 当前CVD / |历史恐慌CVD|
    ignition_ratio = curr['net_cvd'] / abs(recent_panic_cvd)
    
    # 4. 判定启动
    is_strong_ignition = (curr['net_cvd'] > LONG_MIN_CVD) and (ignition_ratio > LONG_IGNITION_RATIO)
    
    if is_oversold and is_strong_ignition:
        signal = 1 # 1 代表做多信号 (Executor需适配)
        signal_type = "🚀 LONG (Strong Reversal)"
        desc = f"CVD:{curr['net_cvd']:.1f} vs Panic:{recent_panic_cvd:.1f} (Ratio {ignition_ratio:.2f})"
        tp_pct = LONG_TP
        sl_pct = LONG_SL
        prob = 0.88 # 高胜率形态

    # ==========================================
    # 策略 B: 顶部资金背离 (Short Logic)
    # ==========================================
    # 1. 价格创新高 (当前收盘价 > 过去4根的最高收盘价)
    local_high_price = history_4bar['close_price'].max()
    is_new_high = curr['close_price'] > local_high_price
    
    # 2. 资金流出 (CVD 为负且有一定规模)
    is_selling = curr['net_cvd'] < SHORT_MIN_SELL_CVD
    
    # 只有在没有 Long 信号时才检查 Short (避免冲突，Long 优先抄底)
    if signal == 0 and is_new_high and is_selling:
        signal = -1 # -1 代表做空信号 (Executor需适配)
        signal_type = "🔻 SHORT (Bearish Div)"
        desc = f"New High({curr['close_price']:.0f}) but CVD:{curr['net_cvd']:.1f}"
        tp_pct = SHORT_TP
        sl_pct = SHORT_SL
        prob = 0.85 

    # 调试状态返回 (用于打印日志)
    debug_info = {
        'RSI': curr['rsi'],
        'CVD': curr['net_cvd'],
        'Panic_CVD': recent_panic_cvd,
        'Local_High': local_high_price,
        'Ratio': ignition_ratio if 'ignition_ratio' in locals() else 0
    }
    
    return signal, signal_type, desc, tp_pct, sl_pct, prob, debug_info

def run_monitor():
    global last_processed_data_time
    
    system_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        df = get_latest_market_data()
    except Exception as e:
        print(f"[{system_time}] ❌ 数据获取失败: {e}")
        return

    # 检查是否是新 K 线
    data_time = df.iloc[-1]['time']
    current_price = df.iloc[-1]['close_price']
    
    if data_time == last_processed_data_time:
        # 为了不刷屏，这里用 end='\r'
        # print(f"[{system_time}] ⏳ 等待新K线 ({data_time}) ...", end='\r')
        return
    
    # 更新时间戳
    last_processed_data_time = data_time
    
    # 计算信号
    signal, sig_type, desc, tp_pct, sl_pct, prob, debug = calculate_signals(df)
    
    # ==========================================
    # 控制台日志 (Dashboard)
    # ==========================================
    print(f"\n[{system_time}] ⚡ K线更新: {data_time}")
    print(f"   价格: {current_price:.2f} | 信号: {sig_type}")
    print(f"   数据: RSI({debug['RSI']:.1f}) | CVD({debug['CVD']:.1f})")
    
    if signal == 0:
        print(f"   状态: 恐慌CVD({debug['Panic_CVD']:.1f}) | 启动比率({debug['Ratio']:.2f})")
    elif signal == 1:
        print(f"   🔥 触发多头: {desc}")
    elif signal == -1:
        print(f"   ❄️ 触发空头: {desc}")

    # ==========================================
    # 写信号文件
    # ==========================================
    if signal != 0:
        # 计算具体价格
        if signal == 1: # Long
            tp_price = current_price * (1 + tp_pct)
            sl_price = current_price * (1 - sl_pct)
        else: # Short
            tp_price = current_price * (1 - tp_pct)
            sl_price = current_price * (1 + sl_pct)
        
        output_row = {
            'Log_Time': system_time,
            'Time': data_time,
            'Price': current_price,
            'Signal': signal,     # 1 或 -1
            'Type': sig_type,
            'TP_Price': round(tp_price, 2),
            'SL_Price': round(sl_price, 2),
            'Prob': prob,
            'Desc': desc
        }
        
        df_out = pd.DataFrame([output_row])
        
        # 写入 CSV (追加模式)
        if not os.path.exists(SIGNAL_OUTPUT_PATH):
            df_out.to_csv(SIGNAL_OUTPUT_PATH, index=False)
        else:
            df_out.to_csv(SIGNAL_OUTPUT_PATH, mode='a', header=False, index=False)
        
        print(f"✅ 信号已写入: {SIGNAL_OUTPUT_PATH}")

if __name__ == "__main__":
    print("🦈 全能反转猎手 (Enhanced Long + Bearish Short) 启动...")
    print(f"   LONG规则: RSI<{LONG_RSI_THRES} + CVD>{LONG_MIN_CVD} + Ratio>{LONG_IGNITION_RATIO}")
    print(f"   SHORT规则: 新高 + CVD<{SHORT_MIN_SELL_CVD}")
    print(f"   日志模式: 无缓冲 (实时刷新)")
    
    while True:
        try:
            run_monitor()
            time.sleep(10) # 10秒轮询一次
        except KeyboardInterrupt:
            print("\n程序已停止")
            break
        except Exception as e:
            print(f"⚠️ 主循环错误: {e}")
            time.sleep(10)