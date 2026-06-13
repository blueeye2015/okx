#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
研究问题：山寨币超跌反弹后下跌，后续有多少能涨回反弹高点？花了多久？

针对 Level 3 成功案例（30天内跌≥20%），进一步研究：
1. 后续是否涨回突破日收盘价（做空入场点）
2. 涨回的话花了多少天
3. 没有涨回的话，最大跌幅和最终跌幅
4. 不同时间窗口（60天、90天、180天、1年、2年）的恢复率
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

    l3_success_cases = []  # Level 3 成功案例（30天内跌≥20%）
    l3_failure_cases = []  # Level 3 失败案例（30天内跌<20%）

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
            if i < 60 or i >= len(df) - 90:  # 需要至少 90 天后续数据
                continue
            row = df.iloc[i]
            if pd.isna(row['drawdown']) or pd.isna(row['pre_return_20d']):
                continue

            dd = float(row['drawdown'])
            weak_ratio = float(row['weak_ratio']) if not pd.isna(row['weak_ratio']) else 0
            pre_r20d = float(row['pre_return_20d'])

            is_strict_oversold = (dd >= 0.80) and (weak_ratio >= 0.80)
            is_l3 = pre_r20d > 0.30

            if is_l3 and is_strict_oversold:
                future = df.iloc[i+1:]
                if len(future) < 30:
                    continue

                entry_price = row['close']
                post_min_30d = future['close'].iloc[:30].min()
                post_max_30d = future['close'].iloc[:30].max()
                post_min_60d = future['close'].iloc[:60].min() if len(future) >= 60 else np.nan
                post_max_60d = future['close'].iloc[:60].max() if len(future) >= 60 else np.nan
                post_min_90d = future['close'].iloc[:90].min() if len(future) >= 90 else np.nan
                post_max_90d = future['close'].iloc[:90].max() if len(future) >= 90 else np.nan

                # 看是否涨回 entry_price，以及花了多少天
                recovered_days = None
                future_high_all = future['high'].max()
                future_high_idx = future['high'].idxmax()
                ever_recovered = future_high_all >= entry_price
                if ever_recovered:
                    recovered_days = (future_high_idx - idx).days

                case = {
                    'symbol': symbol,
                    'date': idx.strftime('%Y-%m-%d'),
                    'entry_price': entry_price,
                    'drawdown': dd,
                    'pre_r20d': pre_r20d,
                    'post_min_30d_pct': (post_min_30d - entry_price) / entry_price,
                    'post_max_30d_pct': (post_max_30d - entry_price) / entry_price,
                    'post_min_60d_pct': (post_min_60d - entry_price) / entry_price if not pd.isna(post_min_60d) else np.nan,
                    'post_max_60d_pct': (post_max_60d - entry_price) / entry_price if not pd.isna(post_max_60d) else np.nan,
                    'post_min_90d_pct': (post_min_90d - entry_price) / entry_price if not pd.isna(post_min_90d) else np.nan,
                    'post_max_90d_pct': (post_max_90d - entry_price) / entry_price if not pd.isna(post_max_90d) else np.nan,
                    'future_high_pct': (future_high_all - entry_price) / entry_price,
                    'ever_recovered': ever_recovered,
                    'recovered_days': recovered_days,
                    'data_end': df.index[-1].strftime('%Y-%m-%d'),
                }

                is_success = post_min_30d <= entry_price * 0.80
                if is_success:
                    l3_success_cases.append(case)
                else:
                    l3_failure_cases.append(case)

    df_success = pd.DataFrame(l3_success_cases)
    df_failure = pd.DataFrame(l3_failure_cases)

    print("\n" + "="*100)
    print("  超跌反弹后下跌的后续恢复研究")
    print("="*100)

    print(f"\n【样本说明】")
    print(f"  Level 3 严格超跌反弹总数: {len(df_success) + len(df_failure)} 个")
    print(f"  - 30天内跌≥20%（做空成功）: {len(df_success)} 个")
    print(f"  - 30天内跌<20%（做空失败）: {len(df_failure)} 个")

    print(f"\n{'='*100}")
    print("  一、做空成功组（30天内跌≥20%）后续恢复情况")
    print(f"{'='*100}")

    if len(df_success) > 0:
        print(f"\n  1. 是否涨回做空入场价（突破日收盘价）？")
        recovered = df_success['ever_recovered'].sum()
        print(f"     最终涨回: {recovered} / {len(df_success)} = {recovered/len(df_success)*100:.1f}%")
        print(f"     未涨回: {len(df_success) - recovered} / {len(df_success)} = {(len(df_success)-recovered)/len(df_success)*100:.1f}%")

        print(f"\n  2. 涨回所需时间分布（成功涨回的 {recovered} 个案例）")
        recovered_cases = df_success[df_success['ever_recovered'] == True]
        if len(recovered_cases) > 0:
            print(f"     平均天数: {recovered_cases['recovered_days'].mean():.0f} 天")
            print(f"     中位数天数: {recovered_cases['recovered_days'].median():.0f} 天")
            print(f"     25%分位: {recovered_cases['recovered_days'].quantile(0.25):.0f} 天")
            print(f"     50%分位: {recovered_cases['recovered_days'].quantile(0.50):.0f} 天")
            print(f"     75%分位: {recovered_cases['recovered_days'].quantile(0.75):.0f} 天")
            print(f"     90%分位: {recovered_cases['recovered_days'].quantile(0.90):.0f} 天")
            print(f"     最长: {recovered_cases['recovered_days'].max():.0f} 天")
            print(f"     最短: {recovered_cases['recovered_days'].min():.0f} 天")

        print(f"\n  3. 后续最大涨幅/跌幅分布（全部 {len(df_success)} 个案例）")
        print(f"     后续最大涨幅中位数: {df_success['future_high_pct'].median()*100:.1f}%")
        print(f"     后续最大涨幅平均: {df_success['future_high_pct'].mean()*100:.1f}%")
        print(f"     后续30天最大跌幅中位数: {df_success['post_min_30d_pct'].median()*100:.1f}%")
        print(f"     后续60天最大跌幅中位数: {df_success['post_min_60d_pct'].median()*100:.1f}%")
        print(f"     后续90天最大跌幅中位数: {df_success['post_min_90d_pct'].median()*100:.1f}%")

        print(f"\n  4. 不同时间窗口内涨回入场价的概率")
        for days in [30, 60, 90, 180, 365, 730]:
            # 从后续数据中找第 days 天是否 >= entry_price
            # 这里用 recovered_days <= days 来近似
            valid = df_success[df_success['recovered_days'].notna() | (df_success['ever_recovered'] == False)]
            pct = (valid['ever_recovered'] & (valid['recovered_days'] <= days)).mean()
            print(f"     {days} 天内涨回: {pct*100:.1f}%")

        print(f"\n  5. 涨回最快的案例 Top 10")
        if len(recovered_cases) > 0:
            fast = recovered_cases.nsmallest(10, 'recovered_days')[['symbol', 'date', 'post_min_30d_pct', 'future_high_pct', 'recovered_days']]
            for _, row in fast.iterrows():
                print(f"     {row['symbol']:<14} {row['date']}  30天最大跌:{row['post_min_30d_pct']*100:>7.1f}%  最高反弹:{row['future_high_pct']*100:>7.1f}%  涨回天数:{row['recovered_days']:>3.0f}")

        print(f"\n  6. 涨回最慢的案例 Top 10")
        if len(recovered_cases) > 0:
            slow = recovered_cases.nlargest(10, 'recovered_days')[['symbol', 'date', 'post_min_30d_pct', 'future_high_pct', 'recovered_days']]
            for _, row in slow.iterrows():
                print(f"     {row['symbol']:<14} {row['date']}  30天最大跌:{row['post_min_30d_pct']*100:>7.1f}%  最高反弹:{row['future_high_pct']*100:>7.1f}%  涨回天数:{row['recovered_days']:>4.0f}")

        print(f"\n  7. 始终未涨回的典型案例 Top 10（最终跌幅最大）")
        not_recovered = df_success[df_success['ever_recovered'] == False].sort_values('future_high_pct')
        for _, row in not_recovered.head(10).iterrows():
            print(f"     {row['symbol']:<14} {row['date']}  30天最大跌:{row['post_min_30d_pct']*100:>7.1f}%  后续最高:{row['future_high_pct']*100:>7.1f}%  数据截止:{row['data_end']}")

    print(f"\n{'='*100}")
    print("  二、做空失败组（30天内跌<20%）作为对照")
    print(f"{'='*100}")

    if len(df_failure) > 0:
        recovered_f = df_failure['ever_recovered'].sum()
        print(f"\n  最终涨回: {recovered_f} / {len(df_failure)} = {recovered_f/len(df_failure)*100:.1f}%")
        print(f"  后续最大涨幅中位数: {df_failure['future_high_pct'].median()*100:.1f}%")
        recovered_f_cases = df_failure[df_failure['ever_recovered'] == True]
        if len(recovered_f_cases) > 0:
            print(f"  涨回中位数天数: {recovered_f_cases['recovered_days'].median():.0f} 天")

    print(f"\n{'='*100}")


if __name__ == '__main__':
    main()
