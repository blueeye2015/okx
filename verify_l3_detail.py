#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
Level 3 胜率 79.3% 详细拆解
展示每一个样本的计算过程
"""

import os
import sys
import logging
import psycopg2
import pandas as pd
import numpy as np

DATABASE_URL = os.getenv('DB_DSN', 'postgresql://postgres:12@127.0.0.1:5432/market_data')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def get_db_conn():
    return psycopg2.connect(DATABASE_URL)


def get_symbols(market_type: str):
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT symbol FROM binance_daily_klines WHERE market_type = %s ORDER BY symbol",
                (market_type,)
            )
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def fetch_klines(symbol: str, market_type: str):
    conn = get_db_conn()
    query = """
        SELECT open_time, open, high, low, close, volume, quote_volume
        FROM binance_daily_klines
        WHERE symbol = %s AND market_type = %s AND open_time >= '2018-01-01'
        ORDER BY open_time ASC
    """
    try:
        df = pd.read_sql(query, conn, params=(symbol, market_type))
        if not df.empty:
            df['open_time'] = pd.to_datetime(df['open_time'])
            df = df.set_index('open_time')
        return df
    finally:
        conn.close()


def main():
    symbols = get_symbols('spot')
    logger.info(f"共 {len(symbols)} 个币")

    cases = []  # 收集所有 Level 3 案例

    for i, symbol in enumerate(symbols):
        if i % 50 == 0:
            logger.info(f"进度: {i}/{len(symbols)}")

        df = fetch_klines(symbol, 'spot')
        if df.empty or len(df) < 400:
            continue

        df['ma24'] = df['close'].rolling(window=24, min_periods=24).mean()
        weekly = df['close'].resample('W-FRI').last().dropna()
        weekly_ma = weekly.rolling(window=60, min_periods=60).mean()
        df['weekly_ma60'] = weekly_ma.reindex(df.index, method='ffill')
        df['rolling_high_500d'] = df['high'].rolling(window=500, min_periods=300).max()
        df['drawdown'] = (df['rolling_high_500d'] - df['close']) / df['rolling_high_500d']
        df['below_weekly_ma'] = df['close'] < df['weekly_ma60']
        df['weak_days_250'] = df['below_weekly_ma'].rolling(window=250, min_periods=200).sum()
        df['weak_ratio'] = df['weak_days_250'] / 250
        df['above_ma24'] = df['close'] > df['ma24']
        df['prev_above_ma24'] = df['above_ma24'].shift(1)
        df['is_breakout'] = (~df['prev_above_ma24'].fillna(True)) & df['above_ma24']
        df['pre_return_20d'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)

        for idx in df[df['is_breakout']].index:
            i = df.index.get_loc(idx)
            if i < 60 or i >= len(df) - 60:
                continue
            row = df.iloc[i]
            if pd.isna(row['drawdown']) or pd.isna(row['pre_return_20d']):
                continue

            dd = float(row['drawdown'])
            weak_ratio = float(row['weak_ratio']) if not pd.isna(row['weak_ratio']) else 0
            pre_r20d = float(row['pre_return_20d'])

            # 严格超跌条件
            is_strict_oversold = (dd >= 0.80) and (weak_ratio >= 0.80)
            is_l3 = pre_r20d > 0.30

            if is_l3 and is_strict_oversold:
                future = df.iloc[i+1:min(i+31, len(df))]
                if len(future) < 30:
                    continue

                post_min_30d = (future['close'].iloc[:30].min() - row['close']) / row['close']
                post_max_30d = (future['close'].iloc[:30].max() - row['close']) / row['close']
                post_final_30d = (future['close'].iloc[29] - row['close']) / row['close']

                cases.append({
                    'symbol': symbol,
                    'date': idx.strftime('%Y-%m-%d'),
                    'price': float(row['close']),
                    'drawdown': dd,
                    'weak_ratio': weak_ratio,
                    'pre_r20d': pre_r20d,
                    'post_min_30d': post_min_30d,
                    'post_max_30d': post_max_30d,
                    'post_final_30d': post_final_30d,
                    'is_success': post_min_30d <= -0.20,
                })

    df_cases = pd.DataFrame(cases)

    print("\n" + "="*100)
    print("  Level 3 胜率 79.3% 详细拆解")
    print("="*100)

    print(f"\n【筛选条件】")
    print(f"  1. drawdown >= 80%  (从500日高点回撤≥80%)")
    print(f"  2. weak_ratio >= 80% (250天内低于60周均线的天数占比≥80%)")
    print(f"  3. pre_r20d > 30%   (突破前20天涨幅>30%)")
    print(f"  4. 站上MA24         (当日收盘价 > MA24)")
    print(f"\n【样本总数】{len(df_cases)} 个")

    success = df_cases[df_cases['is_success'] == True]
    failure = df_cases[df_cases['is_success'] == False]

    print(f"\n【结果分类】")
    print(f"  做空成功 (30天内最大跌幅 ≥20%): {len(success)} 个")
    print(f"  做空失败 (30天内最大跌幅 <20%):  {len(failure)} 个")
    print(f"  胜率: {len(success)} / {len(df_cases)} = {len(success)/len(df_cases)*100:.1f}%")

    # 成功组详细统计
    print(f"\n{'='*100}")
    print("  【成功组】30天内跌≥20%")
    print(f"{'='*100}")
    print(f"  样本数: {len(success)}")
    if len(success) > 0:
        print(f"  平均最大跌幅: {success['post_min_30d'].mean()*100:.1f}%")
        print(f"  中位数最大跌幅: {success['post_min_30d'].median()*100:.1f}%")
        print(f"  平均最终收益: {success['post_final_30d'].mean()*100:.1f}%")
        print(f"  中位数最终收益: {success['post_final_30d'].median()*100:.1f}%")
        print(f"\n  最大跌幅分布:")
        print(f"    10%分位: {success['post_min_30d'].quantile(0.10)*100:.1f}%")
        print(f"    25%分位: {success['post_min_30d'].quantile(0.25)*100:.1f}%")
        print(f"    50%分位: {success['post_min_30d'].quantile(0.50)*100:.1f}%")
        print(f"    75%分位: {success['post_min_30d'].quantile(0.75)*100:.1f}%")
        print(f"    90%分位: {success['post_min_30d'].quantile(0.90)*100:.1f}%")

    # 失败组详细统计
    print(f"\n{'='*100}")
    print("  【失败组】30天内跌<20%（即反弹或横盘）")
    print(f"{'='*100}")
    print(f"  样本数: {len(failure)}")
    if len(failure) > 0:
        print(f"  平均最大跌幅: {failure['post_min_30d'].mean()*100:.1f}%")
        print(f"  中位数最大跌幅: {failure['post_min_30d'].median()*100:.1f}%")
        print(f"  平均最大涨幅: {failure['post_max_30d'].mean()*100:.1f}%")
        print(f"  中位数最大涨幅: {failure['post_max_30d'].median()*100:.1f}%")
        print(f"  平均最终收益: {failure['post_final_30d'].mean()*100:.1f}%")
        print(f"  中位数最终收益: {failure['post_final_30d'].median()*100:.1f}%")

    # 盈亏比
    print(f"\n{'='*100}")
    print("  【盈亏比计算】")
    print(f"{'='*100}")
    avg_loss = abs(df_cases[df_cases['is_success']]['post_min_30d'].median()) if len(success) > 0 else 0
    avg_gain = abs(df_cases[~df_cases['is_success']]['post_max_30d'].median()) if len(failure) > 0 else 0
    rr = avg_loss / avg_gain if avg_gain > 0 else 0
    print(f"  成功组中位数最大跌幅 (盈利): {avg_loss*100:.1f}%")
    print(f"  失败组中位数最大涨幅 (亏损): {avg_gain*100:.1f}%")
    print(f"  盈亏比: {avg_loss:.3f} / {avg_gain:.3f} = {rr:.2f}")

    # 展示部分样本
    print(f"\n{'='*100}")
    print("  【部分样本展示】成功组 Top 10 最大跌幅")
    print(f"{'='*100}")
    top_success = success.nsmallest(10, 'post_min_30d')[['symbol', 'date', 'drawdown', 'pre_r20d', 'post_min_30d', 'post_final_30d']]
    for _, row in top_success.iterrows():
        print(f"  {row['symbol']:<12} {row['date']}  回撤:{row['drawdown']*100:>5.1f}%  pre_r20d:{row['pre_r20d']*100:>6.1f}%  最大跌:{row['post_min_30d']*100:>7.1f}%  最终:{row['post_final_30d']*100:>7.1f}%")

    print(f"\n{'='*100}")
    print("  【部分样本展示】失败组 Top 10 最大涨幅")
    print(f"{'='*100}")
    top_failure = failure.nlargest(10, 'post_max_30d')[['symbol', 'date', 'drawdown', 'pre_r20d', 'post_min_30d', 'post_max_30d', 'post_final_30d']]
    for _, row in top_failure.iterrows():
        print(f"  {row['symbol']:<12} {row['date']}  回撤:{row['drawdown']*100:>5.1f}%  pre_r20d:{row['pre_r20d']*100:>6.1f}%  最大跌:{row['post_min_30d']*100:>7.1f}%  最大涨:{row['post_max_30d']*100:>7.1f}%  最终:{row['post_final_30d']*100:>7.1f}%")

    print(f"\n{'='*100}")


if __name__ == '__main__':
    main()
