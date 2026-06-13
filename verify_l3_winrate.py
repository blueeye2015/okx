#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
验证 Level 3 胜率差异：49% vs 79.3%
对比三种条件下的胜率
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

    results = {
        'all_breakout_l3': [],
        'loose_oversold_l3': [],
        'strict_oversold_l3': [],
    }

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

            post_min_30d = ((df['close'].iloc[i+1:min(i+31, len(df))].min() - row['close']) / row['close'])

            dd = float(row['drawdown'])
            weak_days = int(row['weak_days_250']) if not pd.isna(row['weak_days_250']) else 0
            weak_ratio = float(row['weak_ratio']) if not pd.isna(row['weak_ratio']) else 0
            pre_r20d = float(row['pre_return_20d'])

            # 三种"超跌"定义
            is_loose_oversold = (dd >= 0.50) or (weak_days >= 100)  # breakout_quality.py 定义
            is_strict_oversold = (dd >= 0.80) and (weak_ratio >= 0.80)  # backtest_validator.py 定义
            is_l3 = pre_r20d > 0.30
            is_short_success = post_min_30d <= -0.20

            if is_l3:
                results['all_breakout_l3'].append(is_short_success)
                if is_loose_oversold:
                    results['loose_oversold_l3'].append(is_short_success)
                if is_strict_oversold:
                    results['strict_oversold_l3'].append(is_short_success)

    print("\n" + "="*80)
    print("  Level 3 胜率差异验证")
    print("="*80)

    for k, v in results.items():
        if v:
            winrate = sum(v) / len(v) * 100
            print(f"\n{k}:")
            print(f"  样本数: {len(v)}")
            print(f"  胜率(30天跌≥20%): {winrate:.1f}%")
        else:
            print(f"\n{k}: 无样本")

    print("\n" + "="*80)
    print("差异分析:")
    print("="*80)
    print("""
条件A: all_breakout_l3
  = 所有MA24突破中 pre_r20d>0.3 的币（不限制超跌背景）

条件B: loose_oversold_l3
  = breakout_quality.py 定义: drawdown>=50% OR weak_days>=100

条件C: strict_oversold_l3
  = backtest_validator.py 定义: drawdown>=80% AND weak_ratio>=80%

核心差异:
  - 条件C最严格，只保留"跌80%+弱势80%"的极端币
  - 这类币本身已极度虚弱，pre_r20d>0.3的"暴涨后突破"假突破概率极高
  - 条件B较宽松，包含了更多"中度超跌"币，稀释了胜率
""")


if __name__ == '__main__':
    main()
