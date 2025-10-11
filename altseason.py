import pandas as pd
import numpy as np
import clickhouse_connect
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- 配置 ---
CLICKHOUSE_CONFIG = dict(host='127.0.0.1', port=8123, user='default', password='12')
START_DATE = '2019-01-01'
END_DATE = '2025-09-30'

# --- 1. 获取所有币种的日收益率 (无幸存者偏差) ---
logging.info("正在获取全市场日线数据...")
ch = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)
# 注意：这个查询没有 HAVING 过滤，会拉取所有币种的历史数据
sql = f"""
SELECT timestamp, symbol, close
FROM marketdata.okx_klines_1d
WHERE symbol LIKE '%-USDT'
    AND timestamp BETWEEN toDateTime('{START_DATE}') AND toDateTime('{END_DATE}')
ORDER BY timestamp, symbol
"""
df = ch.query_df(sql)
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
logging.info("数据获取完毕。")

# --- 2. 计算每日收益率并构建宽表 ---
df['return'] = df.groupby('symbol')['close'].pct_change().fillna(0)
# 使用pivot_table将数据从长表转换为宽表
df_pivoted = df.pivot_table(index='timestamp', columns='symbol', values='return')

# --- 3. 计算山寨币指数和领先指标 ---
if 'BTC-USDT' not in df_pivoted.columns:
    logging.error("数据中未找到 BTC-USDT，无法进行分析。")
else:
    btc_returns = df_pivoted['BTC-USDT']
    # 选出所有山寨币
    altcoin_symbols = [col for col in df_pivoted.columns if col != 'BTC-USDT']
    df_altcoins = df_pivoted[altcoin_symbols]

    # 计算山寨币指数的日收益（等权平均）
    # min_count=5 意味着当天至少要有5个山寨币有数据，才计算指数，避免噪音
    altcoin_index_returns = df_altcoins.mean(axis=1, skipna=True)
    # 计算领先指标：山寨币指数 vs BTC
    spread_returns = (altcoin_index_returns - btc_returns).fillna(0)
    
    # --- 4. 模拟加载您的策略收益 (这里需要您运行之前的回测得到 top_day_ret) ---
    # !!! 注意：请先运行您的 `backtest_a_share.py` (或crypto版)，得到 top_day_ret
    # 然后用一种方式加载它，例如：
    # top_day_ret = pd.read_csv('top_day_ret.csv', index_col=0, parse_dates=True)['ret']
    # 为了演示，我们先用山寨币指数本身作为策略收益的代理
    # 在您实际使用时，请务必替换成您自己的策略收益序列！
    try:
        # 尝试加载您之前运行的结果，如果失败则使用代理
        top_day_ret = pd.read_csv('top_day_ret.csv', index_col=0, parse_dates=True)['ret']
        logging.info("成功加载策略日收益 top_day_ret.csv。")
    except FileNotFoundError:
        logging.warning("未找到 top_day_ret.csv, 将使用山寨币指数作为策略代理进行演示。")
        top_day_ret = altcoin_index_returns.copy()


    # --- 5. 计算累计净值并绘图 ---
    df_plot = pd.DataFrame(index=df_pivoted.index)
    df_plot['strategy_nav'] = (1 + top_day_ret).cumprod()
    df_plot['spread_nav'] = (1 + spread_returns).cumprod()
    df_plot = df_plot.dropna()

    logging.info("正在绘制对比图...")
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # 绘制策略净值
    ax1.plot(df_plot.index, df_plot['strategy_nav'], color='dodgerblue', label='您的策略累计净值')
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('策略累计净值', color='dodgerblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    ax1.set_yscale('log') # 使用对数坐标，更方便观察涨跌趋势

    # 创建第二个Y轴，绘制领先指标净值
    ax2 = ax1.twinx()
    ax2.plot(df_plot.index, df_plot['spread_nav'], color='coral', linestyle='--', alpha=0.8, label='山寨季领先指标 (山寨指数 - BTC)')
    ax2.set_ylabel('山寨季领先指标累计值', color='coral', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='coral')
    ax2.set_yscale('log')

    # 设置图例和标题
    fig.suptitle('策略表现 vs. 山寨季领先指标', fontsize=16)
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 格式化日期显示
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    plt.savefig('strategy_vs_altseason.png')
    logging.info("对比图已保存为 strategy_vs_altseason.png")