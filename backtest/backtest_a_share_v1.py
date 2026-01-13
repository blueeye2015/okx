import pandas as pd
import numpy as np
import psycopg2
import logging
import os
from datetime import datetime
from momentum_scanner_a_share import calc_monthly_ic   # 仅训练/预测

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ==========  配置（只改这里）  ==========
POSTGRES_CONFIG = dict(host='127.0.0.1', port=5432, user='postgres', password='12', dbname='Financialdata')
START_DATE    = '2010-01-04'
END_DATE      = '2019-12-31'
ADJUST_TYPE   = 'hfq'
BENCHMARK_SYM = '000300.SH'
CHUNK_SIZE    = 100
CACHE_DIR     = 'factor_cache'
# ========================================

BENCHMARK_SYMBOL = BENCHMARK_SYM
os.makedirs(CACHE_DIR, exist_ok=True)

def load_trade_cal(conn, start: str, end: str) -> pd.DataFrame:
    sql = """
        SELECT cal_date::date AS trade_date,
               prev_year_date::date
        FROM   public.v_trade_cal_ly
        WHERE  is_open = 1
          AND  cal_date BETWEEN %s AND %s
        ORDER  BY cal_date
    """
    cal = pd.read_sql(sql, conn, params=(start, end), parse_dates=['trade_date', 'prev_year_date'])
    cal.columns = cal.columns.str.strip()
    cal.set_index('trade_date', inplace=True)
    return cal

def build_feature_engine(conn, symbols_list, start_date, end_date):
    """统一特征工厂：利润同比 + 市值 + 底部形态 + 波动率 → 落盘 feat_cache"""
    # 1. 一次性拉所有基础数据（无时间限制）
    price_q = f"""
        SELECT trade_date, symbol, close
        FROM   public.stock_history
        WHERE  symbol IN {tuple(symbols_list)}
          AND  adjust_type = '{ADJUST_TYPE}'
        ORDER  BY trade_date
    """
    mv_q = f"""
        SELECT trade_date, security_code AS symbol, total_mv
        FROM   public.daily_basic
        WHERE  security_code IN {tuple(symbols_list)}
        ORDER  BY trade_date
    """
    profit_q = f"""
        SELECT report_date AS trade_date, security_code AS symbol, deduct_parent_netprofit
        FROM   public.profit_sheet
        WHERE  security_code IN {tuple(symbols_list)}
        ORDER  BY trade_date
    """
    df_p = pd.read_sql(price_q, conn, parse_dates=['trade_date'])
    df_m = pd.read_sql(mv_q,   conn, parse_dates=['trade_date'])
    df_f = pd.read_sql(profit_q, conn, parse_dates=['trade_date'])

    # 2. 交易日历（只拉一次）
    cal = load_trade_cal(conn,
                         (df_p['trade_date'].min() - pd.DateOffset(years=1)).date().strftime('%Y-%m-%d'),
                         end_date)

    # 3. 统一对齐到全日历骨架（含去年 bar）
    date_idx = pd.date_range(cal.index.min(), end_date, freq='D')
    skeleton = pd.MultiIndex.from_product([date_idx, symbols_list],
                                          names=['trade_date', 'symbol']).to_frame(index=False)

    # 4. 把市值、利润 asof 到骨架
    df_m = pd.merge_asof(skeleton, df_m.sort_values('trade_date'), on='trade_date', by='symbol', direction='backward')
    df_f = pd.merge_asof(skeleton, df_f.sort_values('trade_date'), on='trade_date', by='symbol', direction='backward')

    # 5. 合并到价格（价格从 start_date 开始，但特征带前一年）
    base = pd.merge_asof(df_p[df_p['trade_date'].between(start_date, end_date)],
                         df_m[['trade_date', 'symbol', 'total_mv']],
                         on='trade_date', by='symbol', direction='backward')
    base = pd.merge_asof(base,
                         df_f[['trade_date', 'symbol', 'deduct_parent_netprofit']],
                         on='trade_date', by='symbol', direction='backward')

    # 6. 特征工程（在同一张表上算）
    base = base.merge(cal[['prev_year_date']], left_on='trade_date', right_index=True, how='left')
    lookup = base[['deduct_parent_netprofit']].rename(columns={'deduct_parent_netprofit': 'profit_ly'})
    base = pd.merge_asof(base.sort_values('prev_year_date'),
                         lookup.sort_index(),
                         left_on='prev_year_date',
                         right_index=True,
                         direction='backward',
                         suffixes=('', '_ly'))
    base['profit_yoy']   = (base['deduct_parent_netprofit'] / base['profit_ly_ly'] - 1)
    base['log_mv']       = np.log(base['total_mv'].clip(lower=1))
    base['price_pos_1y'] = (base['close'] - base['close'].rolling(252, min_periods=60).min()) \
                           / (base['close'].rolling(252, min_periods=60).max() - base['close'].rolling(252, min_periods=60).min())
    base['volatility_3m'] = base['close'].pct_change().rolling(63, min_periods=20).std()

    # 7. 只保留需要的列并落盘
    feat = base[['trade_date', 'symbol', 'profit_yoy', 'log_mv', 'price_pos_1y', 'volatility_3m']].dropna(subset=['profit_yoy'])
    feat_path = os.path.join(CACHE_DIR, f'feat_{START_DATE}_{END_DATE}.parquet')
    feat.to_parquet(feat_path)
    logging.info(f'特征落盘完成 {feat_path}  {len(feat)} 行')
    return feat

def main():
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    # 股票列表（你已写好）
    symbols = pd.read_sql(f"""
        SELECT symbol FROM public.stock_history
        WHERE adjust_type='{ADJUST_TYPE}' AND symbol!='{BENCHMARK_SYMBOL}'
        GROUP BY symbol HAVING COUNT(*)>252
    """, conn)['symbol'].tolist()

    feat_df = build_feature_engine(conn, symbols, START_DATE, END_DATE)

    # 市场收益序列
    mkt = pd.read_sql(f"""
        SELECT trade_date, close FROM public.index_daily
        WHERE ts_code='{BENCHMARK_SYMBOL}' AND trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """, conn, parse_dates=['trade_date']).set_index('trade_date')['close'].pct_change().rename('mkt_ret')

    # 因子计算（只训练/预测）
    ic, factor = calc_monthly_ic(feat_df, symbols, mkt)
    factor_path = os.path.join(CACHE_DIR, f'factor_{START_DATE}_{END_DATE}.parquet')
    factor.to_parquet(factor_path)
    logging.info(f'因子已保存 {factor_path}  IC均值={ic.mean():.3f}')

    conn.close()

if __name__ == '__main__':
    main()