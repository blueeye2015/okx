import pandas as pd
import numpy as np
import clickhouse_connect
import logging
from momentum_scanner_ic import calc_monthly_ic,_calculate_features_and_factor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 配置
CLICKHOUSE_CONFIG = dict(host='127.0.0.1', port=8123, user='default', password='12')
START_DATE = '2019-01-01'
END_DATE = '2025-09-30'

ch = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)

# 1. 取全市场日K

sql = f"""
SELECT symbol
FROM marketdata.okx_klines_1d
WHERE symbol LIKE '%-USDT'
GROUP BY symbol
HAVING min(timestamp) <= toDateTime('{START_DATE}')
    AND count(*) >= 756 + 252      -- 36 个月训练 + 12 个月回测
ORDER BY symbol
"""
symbols_list = [r[0] for r in ch.query(sql).result_rows]
if 'BTC-USDT' not in symbols_list:
    symbols_list.insert(0, 'BTC-USDT')
logging.info(f"复用 Backtrader 口径，可用币种 {len(symbols_list)} 只")


# 统一日期轴
date_index = pd.date_range(START_DATE, END_DATE, freq='D')

all_prices_query = f"""
SELECT timestamp, symbol, close
FROM marketdata.okx_klines_1d
WHERE symbol IN {tuple(symbols_list)}
    AND timestamp BETWEEN toDateTime('{START_DATE}') AND toDateTime('{END_DATE}')
ORDER BY timestamp, symbol
"""
df = ch.query_df(all_prices_query)
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

# === 下面是 Backtrader 同款对齐 ===
aligned_list = []
for symbol in symbols_list:
    g = df[df['symbol'] == symbol].set_index('timestamp')
    first_valid = g.index.min()
    g = g.reindex(date_index)
    g = g[g.index >= first_valid]
    g = g[~g.index.duplicated(keep='last')]
    g = g.ffill()
    g = g.dropna(subset=['close'])
    g['symbol'] = symbol
    aligned_list.append(g.reset_index())

df_aligned = pd.concat(aligned_list, ignore_index=False)
df_aligned = df_aligned.reset_index().rename(columns={'index': 'timestamp'})
logging.info(f"对齐后总 bar 数: {len(df_aligned)}")

# 2. 计算月度IC
logging.info('开始计算月度IC...')
ic_s, daily_factor = calc_monthly_ic(df_aligned, symbols_list)
if ic_s.empty:
    logging.warning("IC 序列为空，请检查数据或训练期长度")
else:
    logging.info('月度IC均值: %.3f', ic_s.mean())
    logging.info('IC>0 占比: %.1f%%', (ic_s > 0).mean() * 100)
daily_factor = daily_factor.dropna()  

# === 市场状态分析与过滤器定义 ===
btc_price = df_aligned[df_aligned['symbol']=='BTC-USDT'].set_index('timestamp')['close']

logging.info("正在生成基于BTC 200日均线的市场状态过滤器...")
btc_sma200 = btc_price.rolling(200, min_periods=100).mean()
is_bull_market_daily = (btc_price > btc_sma200).dropna()
regime_filter_monthly = is_bull_market_daily.resample('ME').last().shift(1).dropna()

# 原有的分析逻辑可以保留，用于观察
btc_month = btc_price.resample('ME').last().pct_change().shift(-1)
bear_mask = btc_month < -0.05
print('（分析用）熊市月份 IC 均值:', ic_s[bear_mask].mean())
print('（分析用）牛市月份 IC 均值:', ic_s[~bear_mask].mean())


# 3. 复用因子表 → Top10% 组合日收益
def top_ret_from_factor(daily_df: pd.DataFrame,
                        factor_df: pd.DataFrame) -> pd.Series:
    price = daily_df.copy()
    price['ret_1d'] = price.groupby('symbol')['close'].pct_change()
    df = price.merge(factor_df, on=['timestamp', 'symbol'], how='inner')

    top_ret_by_day = []
    for month, sub in df.groupby(pd.Grouper(key='timestamp', freq='ME')):
        if sub.empty:
            continue
        eod = sub.groupby('symbol').last()
        top_syms = eod.nlargest(int(len(eod)*0.1)+1, 'factor').index
        
        nxt_month = month + pd.DateOffset(months=1)
        nxt_df = df[df['timestamp'].between(nxt_month.replace(day=1),
                                            nxt_month + pd.offsets.MonthEnd(1))]
        if nxt_df.empty:
            continue
            
        top_day_ret = (nxt_df[nxt_df['symbol'].isin(top_syms)]
                           .groupby('timestamp')['ret_1d'].mean())
                           
        # <<<--- 这里是修正的地方 ---<<<
        top_ret_by_day.append(top_day_ret)

    return pd.concat(top_ret_by_day) if top_ret_by_day else pd.Series(dtype=float)


# -------------- 主流程 --------------
top_day_ret = top_ret_from_factor(df_aligned, daily_factor)

if top_day_ret.empty:
    logging.info('Top10% 日收益为空')
else:
    def cagr(r): return (1+r).prod()**(365/len(r))-1 if len(r) > 0 else 0
    def sharpe(r): return r.mean()/r.std()*np.sqrt(365) if r.std() > 0 else 0

    cagr_val = cagr(top_day_ret)
    sharp_val = sharpe(top_day_ret)
    logging.info(f'【原始策略】CAGR : {cagr_val*100:.2f}%')
    logging.info(f'【原始策略】夏普 : {sharp_val:.2f}')

# --- 2) 给日收益加「月末日期」列 & 权重 ---
top_df = top_day_ret.to_frame('ret')
top_df['month_end'] = top_df.index.to_period('M').to_timestamp('M')

top_df['weight'] = top_df['month_end'].map(regime_filter_monthly.map({True: 0.5, False: 1.0})).fillna(1.0)

# --- 3) 加权收益 ---
top_ret_adj = top_df['ret'] * top_df['weight']

# === 择时策略结果打印 ===
print('\n--- 择时策略回测结果 ---')
print('策略权重分布:\n', top_df['weight'].value_counts())
active_months = regime_filter_monthly.sum()
total_months = regime_filter_monthly.count()
print(f'策略开启月份: {active_months} / 总月份: {total_months} (占比: {active_months/total_months:.1%})')

if not top_ret_adj.empty and top_ret_adj.std() > 0:
    cagr_adj_val = cagr(top_ret_adj)
    sharpe_adj_val = sharpe(top_ret_adj)
    logging.info(f'【择时后】CAGR: {cagr_adj_val*100:.2f}%')
    logging.info(f'【择时后】夏普: {sharpe_adj_val:.2f}')
else:
    logging.info('择时后收益为空/恒定')

shield_month_ret = top_df[top_df['weight']==0.5]['ret'].mean()
normal_month_ret = top_df[top_df['weight']==1.0]['ret'].mean()
print('被屏蔽月份(熊市)策略原始日均收益:', shield_month_ret)
print('保留月份(牛市)策略原始日均收益:', normal_month_ret)

nav_orig = (1 + top_day_ret).cumprod()
nav_adj = (1 + top_ret_adj).cumprod()

print('原始策略最终净值:', nav_orig.iloc[-1] if not nav_orig.empty else 1)
print('择时后策略最终净值:', nav_adj.iloc[-1] if not nav_adj.empty else 1)

# 4. 打印前10条IC
print('\n月度IC前10条:')
print(ic_s.head(24))