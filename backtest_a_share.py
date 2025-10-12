import pandas as pd
import numpy as np
import psycopg2
import logging
import os
from momentum_scanner_a_share import calc_monthly_ic

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- 1. 配置项 ---
POSTGRES_CONFIG = dict(
    host='127.0.0.1', port=5432, user='postgres', password='12', dbname='Financialdata'
)
START_DATE = '2017-01-03'
END_DATE = '2025-09-30'
ADJUST_TYPE = 'hfq'
BENCHMARK_SYMBOL = '000300.SH'
CHUNK_SIZE = 100
CACHE_DIR = 'factor_cache'

# --- 2. 数据获取 ---
try:
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    logging.info("PostgreSQL数据库连接成功。")
except Exception as e:
    logging.error(f"数据库连接失败: {e}")
    exit()

sql_symbols = f"""
SELECT symbol FROM public.stock_history WHERE adjust_type = '{ADJUST_TYPE}' AND symbol != '{BENCHMARK_SYMBOL}'
GROUP BY symbol HAVING min(trade_date) <= '{START_DATE}'::date AND count(*) >= 756 + 252 ORDER BY symbol
"""
# <<<--- 核心修改：修正获取股票列表的逻辑 ---<<<
stock_symbols_list = pd.read_sql_query(sql_symbols, conn)['symbol'].tolist()

logging.info(f"可用A股标的 {len(stock_symbols_list)} 只 (复权类型: {ADJUST_TYPE})")

# ... (后续代码与上一版完全相同，为简洁此处省略，请确保您本地是完整的) ...
# 您只需修改上面这一行，文件的其余部分保持不变即可。
# 为保证万无一失，下面是完整的代码。

all_stocks_query = f"SELECT trade_date, symbol, close FROM public.stock_history WHERE symbol IN {tuple(stock_symbols_list)} AND trade_date BETWEEN '{START_DATE}'::date AND '{END_DATE}'::date AND adjust_type = '{ADJUST_TYPE}'"
df_stocks = pd.read_sql_query(all_stocks_query, conn)
logging.info(f"已从 stock_history 表获取 {len(df_stocks)} 行股票数据。")

index_price_query = f"SELECT trade_date, ts_code, close FROM public.index_daily WHERE ts_code = '{BENCHMARK_SYMBOL}' AND trade_date BETWEEN '{START_DATE}'::date AND '{END_DATE}'::date"
df_index = pd.read_sql_query(index_price_query, conn)
df_index.rename(columns={'ts_code': 'symbol'}, inplace=True)
logging.info(f"已从 index_daily 表获取 {len(df_index)} 行指数数据。")

df = pd.concat([df_stocks, df_index], ignore_index=True)
symbols_list = stock_symbols_list + [BENCHMARK_SYMBOL]
conn.close()
logging.info("数据库连接已关闭。股票与指数数据合并完毕。")
df['trade_date'] = pd.to_datetime(df['trade_date'])

# --- 3. 因子计算（采用分块和缓存新架构） ---
logging.info('开始计算每日因子（采用分块缓存策略）...')
os.makedirs(CACHE_DIR, exist_ok=True)
all_daily_factors = []

symbol_chunks = [symbols_list[i:i + CHUNK_SIZE] for i in range(0, len(symbols_list), CHUNK_SIZE)]

for i, chunk in enumerate(symbol_chunks):
    chunk_filename = os.path.join(CACHE_DIR, f'daily_factor_chunk_{i}.parquet')
    logging.info(f"--- 处理批次 {i+1}/{len(symbol_chunks)} ---")
    
    chunk_factor_df = pd.DataFrame()

    if os.path.exists(chunk_filename):
        logging.info(f"找到缓存文件 {chunk_filename}，直接加载。")
        chunk_factor_df = pd.read_parquet(chunk_filename)
    else:
        logging.info(f"未找到缓存，为批次中的 {len(chunk)} 只股票计算因子...")
        df_chunk_raw = df[df['symbol'].isin(chunk)]
        
        if df_chunk_raw.empty:
            logging.warning(f"批次 {i+1} 中未找到任何股票数据，已跳过。涉及股票代码: {chunk}")
            continue

        date_index = pd.date_range(df_chunk_raw['trade_date'].min(), df_chunk_raw['trade_date'].max(), freq='D')
        aligned_list = []
        for symbol in chunk:
            g = df_chunk_raw[df_chunk_raw['symbol'] == symbol].set_index('trade_date')
            if g.empty: continue
            g = g.reindex(date_index).ffill()
            g['symbol'] = symbol
            aligned_list.append(g.reset_index().rename(columns={'index': 'trade_date'}))
        
        if not aligned_list: continue
        df_aligned_chunk = pd.concat(aligned_list, ignore_index=True).dropna(subset=['close'])

        _, chunk_factor_df = calc_monthly_ic(df_aligned_chunk, chunk)
        
        if chunk_factor_df is not None and not chunk_factor_df.empty:
            chunk_factor_df.to_parquet(chunk_filename)
            logging.info(f"批次计算完成，结果已缓存至 {chunk_filename}")

    if chunk_factor_df is not None and not chunk_factor_df.empty:
        all_daily_factors.append(chunk_factor_df)

if not all_daily_factors:
    logging.error("因子计算未能生成任何数据，程序退出。")
    exit()

daily_factor = pd.concat(all_daily_factors, ignore_index=True).dropna()
logging.info("所有批次的因子数据已合并。")

# --- 4. IC计算与回测 ---
df_aligned = df 
logging.info("开始执行策略回测...")

index_price = df_aligned[df_aligned['symbol']==BENCHMARK_SYMBOL].set_index('trade_date')['close'].sort_index()
index_sma200 = index_price.rolling(200, min_periods=100).mean()
is_bull_market_daily = (index_price > index_sma200).dropna()
regime_filter_monthly = is_bull_market_daily.resample('ME').last().shift(1).dropna()

def top_ret_from_factor(daily_df: pd.DataFrame, factor_df: pd.DataFrame) -> pd.Series:
    price = daily_df.copy()
    price['ret_1d'] = price.groupby('symbol')['close'].pct_change()
    df = price.merge(factor_df, on=['trade_date', 'symbol'], how='inner')
    top_ret_by_day = []
    for month, sub in df.groupby(pd.Grouper(key='trade_date', freq='ME')):
        if sub.empty: continue
        eod = sub.groupby('symbol').last()
        top_syms = eod.nlargest(int(len(eod)*0.1)+1, 'factor').index
        
        nxt_month = month + pd.DateOffset(months=1)
        nxt_df = df[df['trade_date'].between(nxt_month.replace(day=1), nxt_month + pd.offsets.MonthEnd(1))]
        if nxt_df.empty: continue
            
        top_day_ret = (nxt_df[nxt_df['symbol'].isin(top_syms)].groupby('trade_date')['ret_1d'].mean())
        top_ret_by_day.append(top_day_ret)
    return pd.concat(top_ret_by_day) if top_ret_by_day else pd.Series(dtype=float)

top_day_ret = top_ret_from_factor(df_aligned, daily_factor)

if top_day_ret.empty:
    logging.info('Top10% 日收益为空')
else:
    ANNUAL_FACTOR = 252
    def cagr(r): return (1+r).prod()**(ANNUAL_FACTOR/len(r))-1 if len(r) > 0 else 0
    def sharpe(r): return r.mean()/r.std()*np.sqrt(ANNUAL_FACTOR) if r.std() > 0 else 0
    cagr_val = cagr(top_day_ret)
    sharp_val = sharpe(top_day_ret)
    logging.info(f'【原始策略】CAGR : {cagr_val*100:.2f}%')
    logging.info(f'【原始策略】夏普 : {sharp_val:.2f}')

    top_df = top_day_ret.to_frame('ret')
    top_df['month_end'] = top_df.index.to_period('M').to_timestamp('M')
    
    #top_df['weight'] = top_df['month_end'].map(regime_filter_monthly.map({True: 0.5, False: 1.0})).fillna(1.0)
    top_df['weight'] = top_df['month_end'].map(regime_filter_monthly.map({True: 1.0, False: 0.0})).fillna(1.0)
    top_ret_adj = top_df['ret'] * top_df['weight']

    print('\n--- 择时策略回测结果 ---')
    if not top_ret_adj.empty and top_ret_adj.std() > 0:
        cagr_adj_val = cagr(top_ret_adj)
        sharpe_adj_val = sharpe(top_ret_adj)
        logging.info(f'【择时后】CAGR: {cagr_adj_val*100:.2f}%')
        logging.info(f'【择时后】夏普: {sharpe_adj_val:.2f}')
        
        # 修正后的诊断代码
        bull_ret = top_df[top_df['weight']==1.0]['ret'].mean()
        bear_ret = top_df[top_df['weight']==0.0]['ret'].mean() # 策略在熊市的原始收益
        print(f"保留月份(牛市)策略原始日均收益: {bull_ret}")
        print(f"屏蔽月份(熊市)策略原始日均收益: {bear_ret}")