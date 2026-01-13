# -*- coding: utf-8 -*-
"""
逐日动态因子生成主程序
1. 按交易日循环，每天从数据库拉取“当日可交易”股票
2. 调用 momentum_scanner_a_share.calc_factor_one_day 生成当日因子
3. 结果写入 daily_factor/yyyy-mm-dd_factor.parquet
4. 最后合并成 factor_cache/daily_factor_all.parquet 供 visual_backtest.py 使用
"""
import os
import glob
import pandas as pd
import numpy as np
import psycopg2
import logging
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

# 导入你写好的“单日因子函数”
from momentum_scanner_a_share_dynamic import calc_factor_one_day

load_dotenv('.env')
POSTGRES_CONFIG = os.getenv('DB_DSN1')   # 形如 host=xxx dbname=xxx user=xxx password=xxx
BENCHMARK_SYMBOL = '000300.SH'
ADJUST_TYPE = 'hfq'
CACHE_DIR = 'factor_cache'
DAILY_FACTOR_DIR = 'daily_factor'

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DAILY_FACTOR_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ---------- 1. 取交易日序列 ----------
def get_trade_days(start: str, end: str) -> pd.DatetimeIndex:
    sql = f"""
    SELECT DISTINCT trade_date
    FROM index_daily
    WHERE ts_code = '{BENCHMARK_SYMBOL}'
      AND trade_date BETWEEN '{start}' AND '{end}'
    ORDER BY trade_date
    """
    with psycopg2.connect(POSTGRES_CONFIG) as conn:
        days = pd.read_sql(sql, conn)['trade_date']
    return pd.to_datetime(days)

# ---------- 2. 取“当日可交易”股票列表 ----------
def get_universe_one_day(trade_date: pd.Timestamp) -> list:
    sql = f"""
    SELECT DISTINCT symbol
    FROM stock_history
    WHERE trade_date = %(td)s
      AND adjust_type = '{ADJUST_TYPE}'
      AND volume > 0
    """
    with psycopg2.connect(POSTGRES_CONFIG) as conn:
        cur = conn.cursor()
        cur.execute(sql, {'td': trade_date.date()})
        univ = [r[0] for r in cur.fetchall()]
    return univ

# ---------- 3. 取单日原始数据（价格+市值+业绩） ----------
def get_daily_raw(trade_date: pd.Timestamp, universe: list) -> pd.DataFrame:
    if not universe:
        return pd.DataFrame()

    td_str = trade_date.date()
    # 价格
    price_sql = f"""
    SELECT trade_date, symbol, close
    FROM stock_history
    WHERE trade_date <= '{td_str}'
      AND symbol IN %(sym)s
      AND adjust_type = '{ADJUST_TYPE}'
    """
    # 市值
    mv_sql = f"""
    SELECT trade_date, security_code AS symbol, total_mv
    FROM daily_basic
    WHERE trade_date <= '{td_str}'
      AND security_code IN %(sym)s
    """
    # 业绩
    profit_sql = f"""
    SELECT report_date AS trade_date, security_code AS symbol, deduct_parent_netprofit
    FROM profit_sheet
    WHERE security_code IN %(sym)s
    """
    with psycopg2.connect(POSTGRES_CONFIG) as conn:
        price = pd.read_sql(price_sql, conn, params={'sym': tuple(universe)})
        mv = pd.read_sql(mv_sql, conn, params={'sym': tuple(universe)})
        profit = pd.read_sql(profit_sql, conn, params={'sym': tuple(universe)})

    # 合并
    price['trade_date'] = pd.to_datetime(price['trade_date'])
    mv['trade_date'] = pd.to_datetime(mv['trade_date'])
    profit['trade_date'] = pd.to_datetime(profit['trade_date'])

    df = price.merge(mv, on=['trade_date', 'symbol'], how='left')
    df = pd.merge_asof(df.sort_values('trade_date'),
                       profit.sort_values('trade_date'),
                       on='trade_date', by='symbol', direction='backward')
    return df

def fill_weekends(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    对原始日线数据（含 trade_date, symbol, close, total_mv, ...）
    1. 生成完整日历索引（含周末）
    2. 每个 symbol 单独 reindex → 前向填充价格/市值
    3. 业绩字段仍保持 NaN，由后续 merge_asof 处理
    返回：补齐后的 DataFrame，列与输入完全一致
    """
    if df_raw.empty:
        return df_raw

    # 1. 建立完整日历
    date_idx = pd.date_range(
        df_raw['trade_date'].min(),
        df_raw['trade_date'].max(),
        freq='D'
    )

    # 2. 每个 symbol 单独补全
    res = []
    for sym, g in df_raw.groupby('symbol'):
        g = g.set_index('trade_date').sort_index()
        # 只 ffill 价格类、市值类；业绩类不 fill，防止未来函数
        cols_ffill = ['close', 'total_mv']  # 需要补的列
        cols_other = [c for c in g.columns if c not in cols_ffill + ['symbol']]
        g_ff = g[cols_ffill].reindex(date_idx, method='ffill')
        g_oth = g[cols_other].reindex(date_idx)  # 不 forward fill
        g_new = pd.concat([g_ff, g_oth], axis=1)
        g_new['symbol'] = sym
        g_new.index.name = 'trade_date'
        res.append(g_new.reset_index())

    return pd.concat(res, ignore_index=True)

# ---------- 4. 主循环 ----------
def main():
    START_DATE = '2014-04-01'
    END_DATE = '2018-12-31'
    trade_days = get_trade_days(START_DATE, END_DATE)

    logging.info(f'共 {len(trade_days)} 个交易日，开始逐日生成因子...')

    for td in tqdm(trade_days, ncols=80):
        ffile = os.path.join(DAILY_FACTOR_DIR, f'{td.date()}_factor.parquet')
        if os.path.exists(ffile):
            continue          # 已算过跳过

        univ = get_universe_one_day(td)
        if not univ:
            continue

        raw = get_daily_raw(td, univ)
        if raw.empty:
            continue
        
        df_aligned_chunk = fill_weekends(raw)
        # 计算市场日收益
        df_aligned_chunk['return'] = df_aligned_chunk.groupby('symbol')['close'].pct_change()
        mkt_ret = df_aligned_chunk.groupby('trade_date')['return'].mean().rename('mkt_ret')
        
        # 调用单日因子函数
        fac_df = calc_factor_one_day(td, df_aligned_chunk.set_index(['trade_date', 'symbol']).sort_index(), mkt_ret)
        if not fac_df.empty:
            fac_df.to_parquet(ffile, index=False)

    # ---------- 5. 合并总表 ----------
    all_files = glob.glob(os.path.join(DAILY_FACTOR_DIR, '*.parquet'))
    if not all_files:
        logging.error('未生成任何日因子，请检查数据库/函数')
        return

    daily_factor = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)
    out_path = os.path.join(CACHE_DIR, 'daily_factor_all.parquet')
    daily_factor.to_parquet(out_path, index=False)
    logging.info(f'合并完成 -> {out_path}  共 {len(daily_factor)} 行')

if __name__ == '__main__':
    main()