#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STEP 1: 特征工程与数据切片 (完美适配数据库Schema版)
功能：
1. 从 stock_history 读取行情(含换手率) 和 daily_basic 读取市值
2. 引入 Benchmark (000300.SH) 计算特质波动率 (IV)
3. 计算 D-MOM 增强因子 (IV, Streaks, Size, Momentum)
4. 严格执行 shift(1) 防止未来函数
5. 按月切片存储，供下一步全市场训练使用
"""
import os, pandas as pd, numpy as np, psycopg2
import multiprocessing as mp
import numba
from datetime import datetime
from dotenv import load_dotenv

load_dotenv('.env')
DSN = os.getenv('DB_DSN1')
OUTPUT_DIR = 'cache/monthly_chunks'
os.makedirs(OUTPUT_DIR, exist_ok=True)
BENCHMARK_SYMBOL = '000300.SH'

# --- 0. Numba 加速函数 (不变) ---
@numba.jit(nopython=True)
def calc_rolling_iv_numba(y_arr, x_arr, window=20):
    """Numba加速计算滚动特质波动率"""
    n = len(y_arr)
    out = np.full(n, np.nan)
    if n < window: return out
    
    # 预分配矩阵 [1, market_ret]
    X = np.ones((window, 2)) 
    
    for i in range(window, n + 1):
        y_slice = y_arr[i-window : i]
        x_slice = x_arr[i-window : i]
        X[:, 1] = x_slice
        
        # OLS: beta = (X'X)^-1 X'Y
        xtx_00 = window
        xtx_01 = np.sum(x_slice)
        xtx_11 = np.sum(x_slice ** 2)
        
        det = xtx_00 * xtx_11 - xtx_01 * xtx_01
        if abs(det) < 1e-8: continue
            
        inv_00, inv_01 = xtx_11 / det, -xtx_01 / det
        inv_11 = xtx_00 / det
        
        xty_0, xty_1 = np.sum(y_slice), np.sum(x_slice * y_slice)
        
        beta_0 = inv_00 * xty_0 + inv_01 * xty_1
        beta_1 = inv_01 * xty_0 + inv_11 * xty_1
        
        # 计算残差平方和
        rss = 0.0
        for j in range(window):
            y_pred = beta_0 + beta_1 * x_slice[j]
            resid = y_slice[j] - y_pred
            rss += resid * resid
        
        out[i-1] = np.sqrt(rss / (window - 2))
    return out

@numba.jit(nopython=True)
def calc_streaks(price_arr, window=20):
    """计算过去N天最长连涨/连跌天数"""
    n = len(price_arr)
    up_streak = np.zeros(n)
    down_streak = np.zeros(n)
    
    changes = np.zeros(n)
    for i in range(1, n):
        if price_arr[i] > price_arr[i-1]: changes[i] = 1
        elif price_arr[i] < price_arr[i-1]: changes[i] = -1
    
    for i in range(window, n):
        win = changes[i-window+1 : i+1]
        max_up = 0
        curr_up = 0
        max_down = 0
        curr_down = 0
        
        for c in win:
            if c == 1:
                curr_up += 1
                curr_down = 0
            elif c == -1:
                curr_down += 1
                curr_up = 0
            else:
                curr_up = 0
                curr_down = 0
            
            if curr_up > max_up: max_up = curr_up
            if curr_down > max_down: max_down = curr_down
            
        up_streak[i] = max_up
        down_streak[i] = max_down
        
    return up_streak, down_streak

# --- 1. 获取全量股票代码 ---
def get_all_symbols():
    conn = psycopg2.connect(DSN)
    # 直接从 stock_history 获取有数据的股票
    sql = "SELECT DISTINCT symbol FROM public.stock_history WHERE adjust_type = 'hfq'"
    res = pd.read_sql(sql, conn)
    conn.close()
    return res['symbol'].tolist()

# --- 2. 获取基准指数数据 ---
def get_benchmark_data():
    print("正在加载基准指数数据...")
    conn = psycopg2.connect(DSN)
    # 你的 index_daily 表结构里有 ts_code, trade_date, close
    df = pd.read_sql(f"""
        SELECT trade_date, close 
        FROM public.index_daily 
        WHERE ts_code = '{BENCHMARK_SYMBOL}' 
        ORDER BY trade_date
    """, conn)
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()
    df['mkt_ret'] = df['close'].pct_change()
    return df['mkt_ret']

# 初始化全局变量
mkt_ret_global = None

# --- 3. 单只股票处理逻辑 ---
def process_single_stock(symbol):
    try:
        conn = psycopg2.connect(DSN)
        
        # 3.1 获取行情数据 (stock_history)
        # ✅ 修正：直接读取 turnover 字段
        pr = pd.read_sql(f"""
            SELECT trade_date, close, open, high, low, volume, turnover
            FROM public.stock_history
            WHERE symbol = '{symbol}'
            AND trade_date >= '2010-01-01'
            AND adjust_type = 'hfq'
        """, conn)
        
        if pr.empty:
            conn.close(); return None

        # 3.2 获取市值数据 (daily_basic)
        # 你的 daily_basic 表结构里有 security_code(对应symbol), trade_date, total_mv
        mv = pd.read_sql(f"""
            SELECT trade_date, total_mv
            FROM public.daily_basic
            WHERE security_code = '{symbol}'
            AND trade_date >= '2010-01-01'
        """, conn)
        conn.close()

        # 3.3 数据合并
        pr['trade_date'] = pd.to_datetime(pr['trade_date'])
        mv['trade_date'] = pd.to_datetime(mv['trade_date'])
        
        # Inner Join: 必须同时有价格和市值
        df = pd.merge(pr, mv, on='trade_date', how='inner')
        df = df.sort_values('trade_date').set_index('trade_date')
        
        # Join Benchmark
        df = df.join(mkt_ret_global, how='left')
        
        if len(df) < 250: return None
        
        # ==========================================
        # 4. 特征工程 (Factor Calculation)
        # ==========================================
        
        # 基础收益率
        df['ret'] = df['close'].pct_change()
        
        # --- A. 换手率因子 ---
        # ✅ 修正：直接使用数据库里的 turnover (假设单位是%，需不需要除100看数值大小，做排名因子无所谓)
        # 如果 turnover 有空值，用 0 填充
        df['turnover'] = df['turnover'].fillna(0)
        
        # --- B. D-MOM 核心因子 ---
        
        # 1. 特质波动率 IV
        valid_idx = (~np.isnan(df['ret'])) & (~np.isnan(df['mkt_ret']))
        df['IV_20d'] = np.nan
        if valid_idx.sum() > 30:
            iv_vals = calc_rolling_iv_numba(
                df.loc[valid_idx, 'ret'].values,
                df.loc[valid_idx, 'mkt_ret'].values,
                window=20
            )
            df.loc[valid_idx, 'IV_20d'] = iv_vals
            
        # 2. 连涨/连跌天数
        up_s, down_s = calc_streaks(df['close'].values, window=20)
        df['up_streak_20d'] = up_s
        df['down_streak_20d'] = down_s
        
        # 3. 基础动量与反转
        df['return_1m'] = df['close'].pct_change(20) # 月度反转
        df['return_6m'] = df['close'].pct_change(120) # 中期动量
        
        # 4. 市值因子
        df['log_mv'] = np.log(df['total_mv'])

        # ==========================================
        # 5. 防作弊处理：Shift(1)
        # ==========================================
        feature_cols = []
        
        # 映射关系：原始列 -> 滞后列 (T1)
        # 以后只用 _t1 结尾的列训练
        raw_features = {
            'log_mv': 'log_mv_t1',
            'turnover': 'turnover_1m_t1', # 直接用turnover字段
            'IV_20d': 'IV_20d_t1',
            'up_streak_20d': 'up_streak_t1',
            'down_streak_20d': 'down_streak_t1',
            'return_1m': 'return_1m_t1',
            'return_6m': 'return_6m_t1'
        }
        
        for raw, t1 in raw_features.items():
            df[t1] = df[raw].shift(1)
            feature_cols.append(t1)
            
        # 对换手率做个平滑处理 (20日均值)，更稳定
        df['turnover_1m_t1'] = df['turnover_1m_t1'].rolling(20).mean()

        # ==========================================
        # 6. 生成标签 (Target)
        # ==========================================
        # 预测未来 20 天收益率
        df['target_return'] = df['close'].shift(-20) / df['close'] - 1
        df['target_label'] = (df['target_return'] > 0).astype(int)
        
        # 7. 整理输出
        out_cols = feature_cols + ['target_label', 'close']
        
        # 截取有效时间段
        res = df.loc['2014-01-01':'2025-12-31', out_cols].copy()
        res['symbol'] = symbol
        
        return res.reset_index()

    except Exception as e:
        # print(f"Error {symbol}: {e}")
        return None

def init_worker(mkt_data):
    """多进程初始化"""
    global mkt_ret_global
    mkt_ret_global = mkt_data

def main_etl():
    # 1. 准备基准数据
    mkt_ret = get_benchmark_data()
    print(f"基准数据加载完成，共 {len(mkt_ret)} 天")
    
    # 2. 获取股票列表
    symbols = get_all_symbols()
    print(f"开始处理 {len(symbols)} 只股票...")
    
    # 3. 并行计算
    results = []
    with mp.Pool(processes=mp.cpu_count(), initializer=init_worker, initargs=(mkt_ret,)) as pool:
        for i, res in enumerate(pool.imap_unordered(process_single_stock, symbols, chunksize=20)):
            if res is not None:
                results.append(res)
            if i > 0 and i % 500 == 0:
                print(f"已处理 {i}/{len(symbols)}...")
    
    if not results:
        print("❌ 未获取到任何数据，请检查数据库连接")
        return

    # 4. 合并与存储
    print("正在合并全量数据...")
    full_df = pd.concat(results, ignore_index=True)
    full_df['trade_date'] = pd.to_datetime(full_df['trade_date'])
    
    print(f"正在按月切片存储至 {OUTPUT_DIR} ...")
    full_df['month_str'] = full_df['trade_date'].dt.strftime('%Y-%m')
    
    saved_count = 0
    for month, group in full_df.groupby('month_str'):
        save_path = os.path.join(OUTPUT_DIR, f"{month}.parquet")
        group.drop(columns=['month_str']).to_parquet(save_path, compression='snappy')
        saved_count += 1
        
    print(f"✅ ETL完成！已保存 {saved_count} 个月度文件。")

if __name__ == '__main__':
    main_etl()