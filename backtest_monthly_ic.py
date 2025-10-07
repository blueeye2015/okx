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
   AND count(*) >= 756 + 252     -- 36 个月训练 + 12 个月回测
ORDER BY symbol
"""
symbols_list = [r[0] for r in ch.query(sql).result_rows]
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
    # 头部缺失直接砍掉（防止未来函数）
    g = g[g.index >= first_valid]
    g = g[~g.index.duplicated(keep='last')]
    g = g.ffill()                       # 仅尾部补空
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
 
btc_price = df_aligned[df_aligned['symbol']=='BTC-USDT'].set_index('timestamp')['close']
btc_month = btc_price.resample('ME').last().pct_change().shift(-1)  # 下月收益当牛熊标签
month_vol = (
    btc_price.pct_change().rolling(21).std()
    .resample('ME').last()          # 月末值
    .dropna()
)

bear_mask = btc_month < -0.05      # 单月跌超 5% 视为熊市

# --- 1) 双条件：年化波动 > 35%  OR  BTC 下月跌超 5% ---
high_vol  = (month_vol > 0.35) | bear_mask               # 任一成立就空仓
print('熊市月份 IC 均值:', ic_s[bear_mask].mean())
print('牛市月份 IC 均值:', ic_s[~bear_mask].mean())

# 3. 复用因子表 → Top10% 组合日收益
def top_ret_from_factor(daily_df: pd.DataFrame,
                        factor_df: pd.DataFrame) -> pd.Series:
    price = daily_df.copy()
    price['ret_1d'] = price.groupby('symbol')['close'].pct_change()
    # 把因子拼到价格表
    df = price.merge(factor_df, on=['timestamp', 'symbol'], how='inner')

    top_ret_by_day = []
    for month, sub in df.groupby(pd.Grouper(key='timestamp', freq='ME')):
        if sub.empty:
            continue
        # 当月最后一天截面
        eod = sub.groupby('symbol').last()
        top_syms = eod.nlargest(int(len(eod)*0.1)+1, 'factor').index
        # 下月每日收益
        nxt_month = month + pd.DateOffset(months=1)
        nxt_df = df[df['timestamp'].between(nxt_month.replace(day=1),
                                           nxt_month + pd.offsets.MonthEnd(1))]
        if nxt_df.empty:
            continue
        top_day_ret = (nxt_df[nxt_df['symbol'].isin(top_syms)]
                       .groupby('timestamp')['ret_1d'].mean())
        top_ret_by_day.append(top_day_ret)

    return pd.concat(top_ret_by_day) if top_ret_by_day else pd.Series(dtype=float)


# -------------- 主流程 --------------
top_day_ret = top_ret_from_factor(df_aligned, daily_factor)

if top_day_ret.empty:
    logging.info('Top10% 日收益为空')
else:
    def cagr(r): return (1+r).prod()**(252/len(r))-1
    def sharpe(r): return r.mean()/r.std()*np.sqrt(252)

    cagr_val = cagr(top_day_ret)
    sharp_val = sharpe(top_day_ret)
    logging.info(f'Top10% CAGR : {cagr_val*100:.2f}%')
    logging.info(f'Top10% 夏普 : {sharp_val:.2f}')

# --- 2) 给日收益加「月末日期」列 & 权重 ---
top_df = top_day_ret.to_frame('ret')
top_df['month_end'] = top_df.index.to_period('M').to_timestamp('M')

# 首先定义双重过滤条件
extreme_market_mask = high_vol & bear_mask

# 【修改在这里】将 high_vol 替换为 extreme_market_mask
top_df['weight'] = top_df['month_end'].map(extreme_market_mask.map({True: 0.5, False: 1.0})).fillna(1.0)


# --- 3) 加权收益 ---
top_ret_adj = top_df['ret'] * top_df['weight']

# === 波动率分层结果打印 ===
print('高波动+熊市双过滤后 权重分布:\n', top_df['weight'].value_counts())
print('高波动+熊市双过滤 月份数量:', extreme_market_mask.sum(), '/ 总月份:', extreme_market_mask.count())
if not top_ret_adj.empty and top_ret_adj.std() > 0:
    def cagr(r): return (1+r).prod()**(252/len(r))-1
    def sharpe(r): return r.mean()/r.std()*np.sqrt(252)
    # 这里的描述现在就和代码逻辑一致了
    logging.info(f'绝对波动+熊市分层后 CAGR: {cagr(top_ret_adj)*100:.2f}%')
    logging.info(f'绝对波动+熊市分层后夏普: {sharpe(top_ret_adj):.2f}')
else:
    logging.info('分层后仍为空/恒定')
shield_month = top_df[top_df['weight']==0.5]['ret'].mean()
normal_month = top_df[top_df['weight']==1]['ret'].mean()
print('被屏蔽月份日均收益:', shield_month)
print('保留月份日均收益:', normal_month)
nav_orig = (1 + top_day_ret).cumprod()
nav_half = (1 + top_ret_adj).cumprod()

print('原始最终净值:', nav_orig.iloc[-1])
print('降半仓最终净值:', nav_half.iloc[-1])
# 4. 打印前10条IC
print('\n月度IC前10条:')
print(ic_s.head(12))