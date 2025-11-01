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
from momentum_scanner_a_share import calc_monthly_ic   # 你的原函数
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
    cache_file = os.path.join(CACHE_DIR, f'{symbol}.parquet')
    if os.path.exists(cache_file):
        return True
    try:
        conn = psycopg2.connect(DSN)
        try:
        
            # 1. 价格
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

            # 2. 交易日历（沪深300）
            cal = pd.read_sql(f"""
                SELECT DISTINCT trade_date
                FROM public.index_daily
                WHERE ts_code = '000300.SH'
                AND trade_date BETWEEN '2010-01-01' AND '2025-12-31'
            """, conn)['trade_date']

            # 2. 市值
            mv = pd.read_sql(f"""
                SELECT trade_date, total_mv
                FROM public.daily_basic
                WHERE security_code = '{symbol}'
                AND trade_date BETWEEN '2010-01-01' AND '2025-12-31'
            """, conn)

            # 3. 业绩
            pro = pd.read_sql(f"""
                SELECT report_date, deduct_parent_netprofit
                FROM public.profit_sheet
                WHERE security_code = '{symbol}'
            """, conn)
        
        finally:
            conn.close() 

        # --- 日历对齐 + 停牌补全 ---
        pr['trade_date'] = pd.to_datetime(pr['trade_date'])
        cal = pd.to_datetime(cal)
        # 按日历 reindex，缺日用前收填充（停牌/周末）
        # --- 日历对齐 + 停牌补全（保证单调）---
        pr = pr.sort_values('trade_date')          # 价格升序
        cal = pd.to_datetime(cal).sort_values()    # 日历升序
        pr = pr.set_index('trade_date').reindex(cal, method='ffill').reset_index()
        pr['symbol'] = symbol

        # --- 合并 & asof ---
        pr['trade_date'] = pd.to_datetime(pr['trade_date'])
        mv['trade_date'] = pd.to_datetime(mv['trade_date'])
        pro['trade_date'] = pd.to_datetime(pro['report_date'])

        df = pd.merge(pr, mv, on='trade_date', how='left')
        df = pd.merge_asof(df.sort_values('trade_date'),
                        pro.sort_values('trade_date')[['trade_date', 'deduct_parent_netprofit']],
                        on='trade_date', direction='backward')
        df = df.dropna(subset=['close']).ffill()   # 市值/业绩允许前向填充

        # 交易日占比过滤
        if len(df) < 756:
            return False
        
        # --- 交易日占比过滤（年均可交易日 ≥30%）---
        yr = df['trade_date'].dt.year
        market_days_per_year = df.groupby(yr)['trade_date'].count().mean()
        if len(df) < market_days_per_year * 0.30:
            return False

        # --- 算因子 ---
        ic, fac = calc_monthly_ic(df, [symbol])
        if fac is None or fac.empty:
            return False
        fac.to_parquet(cache_file)
        logging.info(f'{symbol} 因子 {len(fac)} 行已缓存')
        return True
    except Exception as e:
        logging.error(f'{symbol} 失败: {e}', exc_info=True)
        return False

# ------------------- 主流程：逐年滚动 + 多进程 -------------------
def main():
    for y in range(START_YEAR, END_YEAR + 1):
        sym = stocks_for_year(y)[:10]
        logging.info(f'==== {y} 年 股票池 {len(sym)} 只 ====')
        with mp.Pool(processes=mp.cpu_count()) as pool:
            ok = sum(pool.map(calc_one, sym))
        logging.info(f'{y} 年 成功 {ok}/{len(sym)} 只')

if __name__ == '__main__':
    main()