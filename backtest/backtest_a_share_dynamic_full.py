#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态逐年扩样因子计算（2014-2025）
1. 每年重新计算「当年已上市且未退市」的股票
2. 单票单文件缓存，已存在自动跳过（增量）
3. 多进程并行，CPU 跑满
4. 价格/市值/业绩对齐、停牌过滤、未来函数规避已内置
"""
import os, multiprocessing as mp, logging, pandas as pd, psycopg2
from datetime import datetime
from momentum_scanner_a_share_dynamic import calculate_factors_for_single_symbol  # 你的原函数
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

load_dotenv('.env')
DSN = os.getenv('DB_DSN1')
START_YEAR, END_YEAR = 2014, 2025
CACHE_DIR = 'factor_cache_per_stock'
os.makedirs(CACHE_DIR, exist_ok=True)
BENCHMARK_SYMBOL='000300.SH'

# ------------------- 工具：逐年股票池 -------------------
def stocks_for_year(y: int):
    cut = f'{y}-12-31'
    conn = psycopg2.connect(DSN)
    sql = f"""
    SELECT DISTINCT symbol
    FROM public.stock_basic
    WHERE list_date <= '{cut}'
      AND (delist_date IS NULL OR delist_date > '{cut}')
    ORDER BY symbol;
    """
    sym = pd.read_sql(sql, conn)['symbol'].tolist()
    conn.close()
    return sym

# ------------------- 工具：单票因子（含对齐+过滤） -------------------
MIN_DAYS = 756          # 3 年最少日线

def calc_one(symbol: str) -> bool:
    """计算单只股票的因子并缓存"""
    cache_file = os.path.join(CACHE_DIR, f'{symbol}.parquet')
    if os.path.exists(cache_file):
        logging.info(f'{symbol} 因子已存在，跳过')
        return True
    
    conn = None
    try:
        # 建立数据库连接
        conn = psycopg2.connect(DSN)
        
        # 1. 获取价格数据（后复权）
        pr = pd.read_sql(f"""
            SELECT trade_date, close
            FROM public.stock_history
            WHERE symbol = '{symbol}'
            AND trade_date BETWEEN '2010-01-01' AND '2025-12-31'
            AND adjust_type = 'hfq'
        """, conn)
        
        if pr.empty:
            logging.warning(f'{symbol} 无价格数据，跳过')
            return False

        # 2. 获取市值数据
        mv = pd.read_sql(f"""
            SELECT trade_date, total_mv
            FROM public.daily_basic
            WHERE security_code = '{symbol}'
            AND trade_date BETWEEN '2010-01-01' AND '2025-12-31'
        """, conn)

        # 3. 获取业绩数据（扣非净利润）
        pro = pd.read_sql(f"""
            SELECT report_date, deduct_parent_netprofit
            FROM public.profit_sheet
            WHERE security_code = '{symbol}'
        """, conn)
        
        # 4. **获取基准指数数据（用于计算市场收益）**
        benchmark = pd.read_sql(f"""
            SELECT trade_date, close
            FROM public.index_daily
            WHERE ts_code = '{BENCHMARK_SYMBOL}'
            AND trade_date BETWEEN '2010-01-01' AND '2025-12-31'
        """, conn)
        
    except Exception as e:
        logging.error(f'{symbol} 数据获取失败: {e}', exc_info=True)
        if conn:
            conn.close()
        return False
    finally:
        if conn:
            conn.close()

    # --- 数据清洗与对齐 ---
    try:
        # 转换日期格式
        pr['trade_date'] = pd.to_datetime(pr['trade_date'])
        mv['trade_date'] = pd.to_datetime(mv['trade_date'])
        pro['trade_date'] = pd.to_datetime(pro['report_date'])
        benchmark['trade_date'] = pd.to_datetime(benchmark['trade_date'])
        
        # 使用基准指数的交易日历
        cal = benchmark['trade_date'].sort_values()
        
        # 价格数据对齐到交易日历（停牌日用前值填充）
        pr = pr.sort_values('trade_date').set_index('trade_date').reindex(cal, method='ffill').reset_index()
        pr['symbol'] = symbol
        
        # 合并价格、市值、业绩数据
        df = pd.merge(pr, mv, on='trade_date', how='left')
        df = pd.merge_asof(df.sort_values('trade_date'),
                           pro.sort_values('trade_date')[['trade_date', 'deduct_parent_netprofit']],
                           on='trade_date', direction='backward')
        df = df.dropna(subset=['close']).ffill()  # 市值/业绩允许前向填充
        
        # 数据长度检查
        if len(df) < MIN_DAYS:
            logging.warning(f'{symbol} 数据长度不足 {MIN_DAYS} 天，跳过')
            return False
        
        # 交易日占比过滤（年均交易日≥30%）
        yr = df['trade_date'].dt.year
        market_days_per_year = df.groupby(yr)['trade_date'].count().mean()
        if len(df) < market_days_per_year * 0.30:
            logging.warning(f'{symbol} 交易日占比不足30%，跳过')
            return False
        
        # --- 计算市场收益序列 ---
        benchmark = benchmark.set_index('trade_date')['close'].sort_index()
        mkt_ret_series = benchmark.pct_change().dropna()
        mkt_ret_series.name = 'mkt_ret'
        
        # --- 构建因子计算所需的数据结构 ---
        # calculate_factors_for_single_symbol 需要包含 symbol 列的 DataFrame
        df_for_calc = df[['trade_date', 'close', 'total_mv', 'deduct_parent_netprofit']].copy()
        df_for_calc['symbol'] = symbol
        
        # --- 调用单票因子计算函数 ---
        fac = calculate_factors_for_single_symbol(symbol, df_for_calc, mkt_ret_series)
        
        if fac is None or fac.empty:
            logging.warning(f'{symbol} 因子计算结果为None或空，跳过')
            return False
        
        # 保存因子到缓存
        fac.to_parquet(cache_file)
        logging.info(f'{symbol} 因子 {len(fac)} 行已缓存')
        return True
        
    except Exception as e:
        logging.error(f'{symbol} 因子计算失败: {e}', exc_info=True)
        return False

# ------------------- 主流程：逐年滚动 + 多进程 -------------------
# backtest_a_share_dynamic_full.py

def main():
    """
    一次性计算所有股票2014-2025年的全量因子
    """
    # 获取截至2025年所有符合条件的股票（包含2014-2025年期间上市的全部股票）
    all_symbols = stocks_for_year(END_YEAR)
    
    logging.info(f'==== 全量股票池 {len(all_symbols)} 只 ====')
    
    # 一次性并行计算
    with mp.Pool(processes=mp.cpu_count()) as pool:
        ok = sum(pool.imap(calc_one, all_symbols))
    
    logging.info(f'全量计算完成，成功 {ok}/{len(all_symbols)} 只')

if __name__ == '__main__':
    # 运行前可选择清空缓存
    if os.path.exists(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        logging.info(f"已清空旧缓存目录: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    main()

if __name__ == '__main__':
    main()