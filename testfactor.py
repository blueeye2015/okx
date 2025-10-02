# debug_factor.py

import os
import random
import numpy as np
import pandas as pd
import clickhouse_connect

# --- 全局随机性控制 (与主脚本完全一致) ---
# 必须在导入任何其他库之前执行
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
# ---

# 导入你的因子计算函数
# 确保 momentum_scanner.py 和这个脚本在同一个文件夹
from momentum_scanner import _calculate_features_and_factor

# --- 模拟主脚本的环境 ---
CLICKHOUSE_CONFIG = {'host': '127.0.0.1', 'port': 8123, 'user': 'default', 'password': '12'}
# 从你的日志中选择一个在不同运行中，因子值明显不同的币
# 例如 backtest2.log 中的 NEO-USDT 和 backtest3.log 中的 LDO-USDT
TARGET_SYMBOL = 'NEO-USDT' 
END_DATE = '2024-01-06'    # 第一次调仓日的日期

def get_single_symbol_data(symbol, end_date_str):
    """获取用于单次因子计算的、干净的历史数据"""
    client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)
    query = f"""
    SELECT timestamp, symbol, open, high, low, close, volume
    FROM marketdata.okx_klines_1d
    WHERE symbol = '{symbol}' AND timestamp <= toDateTime('{end_date_str}')
    ORDER BY timestamp
    """
    df = client.query_df(query)
    # 确保数据类型和时区处理与主脚本完全一致
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    return df

if __name__ == '__main__':
    print(f"--- 正在为 {TARGET_SYMBOL} 在 {END_DATE} 进行单一因子计算测试 ---")
    
    # 1. 获取和主脚本完全一致的输入数据
    historical_data = get_single_symbol_data(TARGET_SYMBOL, END_DATE)
    print(f"获取到 {len(historical_data)} 行历史数据，数据尾部:")
    print(historical_data.tail(3))
    
    # 2. 调用核心的因子计算函数
    factor_result = _calculate_features_and_factor(historical_data)
    
    # 3. 打印决定性的结果
    print("\n" + "="*50)
    print("--- 因子计算结果 ---")
    if factor_result is not None:
        # 我们打印出尽可能多的小数位，以观察微小的差异
        print(f"因子 (pred_proba): {factor_result['factor']:.20f}")
    else:
        print("因子计算返回 None")
    print("="*50)