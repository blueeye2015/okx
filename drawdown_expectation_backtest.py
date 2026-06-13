#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
单币跌幅预期值回测（方案1）
按当前回撤分桶，统计各桶后续30天的表现分布

回撤桶定义:
  [0.50, 0.60)  轻度回调
  [0.60, 0.70)  中度回调
  [0.70, 0.80)  深度回调
  [0.80, 0.90)  极度超跌
  [0.90, 1.00]  濒死状态

跟踪指标:
  - 后续最大跌幅 (从当前价格到后续最低点的跌幅)
  - 后续最终收益 (30天后收盘价 vs 当前价格)
  - 是否创出新低 (后续最低价 < 当前价格)
"""

import os
import sys
import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict

DATABASE_URL = os.getenv('DB_DSN', 'postgresql://postgres:12@127.0.0.1:5432/market_data')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def get_db_conn():
    return psycopg2.connect(DATABASE_URL)


def get_symbols(market_type: str) -> List[str]:
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


def fetch_daily_klines(symbol: str, market_type: str) -> pd.DataFrame:
    conn = get_db_conn()
    query = """
        SELECT open_time, open, high, low, close
        FROM binance_daily_klines
        WHERE symbol = %s AND market_type = %s AND open_time >= '2020-01-01'
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


def compute_symbol_drawdown_events(symbol: str, market_type: str) -> pd.DataFrame:
    """
    计算单个币所有历史时点的回撤及后续表现
    返回 DataFrame，每行是一个"处于某回撤桶"的时点
    """
    df = fetch_daily_klines(symbol, market_type)
    if df.empty or len(df) < 60:
        return pd.DataFrame()

    # 计算回撤（从上市以来高点）
    df['rolling_high'] = df['high'].expanding().max()
    df['drawdown'] = (df['rolling_high'] - df['close']) / df['rolling_high']

    # 后续 30 天的表现
    future_low = df['low'].rolling(window=30, min_periods=1).min().shift(-29)
    future_close = df['close'].shift(-30)

    # 计算指标
    df['max_fall_30d'] = (df['close'] - future_low) / df['close']  # 从当前价格到后续最低点的跌幅
    df['final_return_30d'] = (future_close - df['close']) / df['close']  # 30天后收益
    df['new_low_30d'] = future_low < df['close']  # 是否创出新低

    # 只保留有后续数据的行
    df = df.dropna(subset=['max_fall_30d', 'final_return_30d', 'new_low_30d'])
    if df.empty:
        return pd.DataFrame()

    # 分桶
    def bucket(dd):
        if dd >= 0.90:
            return '90%+'
        elif dd >= 0.80:
            return '80-90%'
        elif dd >= 0.70:
            return '70-80%'
        elif dd >= 0.60:
            return '60-70%'
        elif dd >= 0.50:
            return '50-60%'
        else:
            return None

    df['bucket'] = df['drawdown'].apply(bucket)
    df = df[df['bucket'].notna()]

    if df.empty:
        return pd.DataFrame()

    return df[['bucket', 'drawdown', 'max_fall_30d', 'final_return_30d', 'new_low_30d']].copy()


def run_backtest(market_type: str = 'spot'):
    symbols = get_symbols(market_type)
    logger.info(f"开始回测 {market_type} 市场，共 {len(symbols)} 个币...")

    all_events = []
    for i, symbol in enumerate(symbols, 1):
        try:
            events = compute_symbol_drawdown_events(symbol, market_type)
            if not events.empty:
                all_events.append(events)
            if i % 100 == 0:
                logger.info(f"进度: {i}/{len(symbols)}")
        except Exception as e:
            logger.debug(f"[{symbol}] 异常: {e}")

    if not all_events:
        logger.error("没有有效数据")
        return pd.DataFrame()

    logger.info(f"合并 {len(all_events)} 个币的数据...")
    big_df = pd.concat(all_events, ignore_index=True)
    logger.info(f"共收集 {len(big_df)} 个有效时点")
    return big_df


def print_report(df: pd.DataFrame):
    if df.empty:
        print("❌ 无数据")
        return

    buckets = ['50-60%', '60-70%', '70-80%', '80-90%', '90%+']

    print(f"\n{'='*110}")
    print(f"  单币跌幅预期值回测报告")
    print(f"  样本: {len(df)} 个历史时点")
    print(f"{'='*110}")

    # 总体分布
    print(f"\n📊 各回撤桶样本分布:")
    for b in buckets:
        cnt = len(df[df['bucket'] == b])
        pct = cnt / len(df) * 100
        print(f"  {b}: {cnt} 个时点 ({pct:.1f}%)")

    # 核心指标
    print(f"\n{'='*110}")
    print("  【指标1】后续30天最大跌幅（从当前价格到最低点的跌幅）")
    print(f"{'='*110}")
    print(f"\n{'回撤桶':<10} {'样本数':>8} {'10%分位':>10} {'25%分位':>10} {'50%分位':>10} {'75%分位':>10} {'90%分位':>10} {'均值':>10}")
    print("-" * 110)

    for b in buckets:
        sub = df[df['bucket'] == b]['max_fall_30d']
        if len(sub) == 0:
            continue
        print(f"{b:<10} {len(sub):>8} {sub.quantile(0.10)*100:>9.1f}% {sub.quantile(0.25)*100:>9.1f}% {sub.quantile(0.50)*100:>9.1f}% {sub.quantile(0.75)*100:>9.1f}% {sub.quantile(0.90)*100:>9.1f}% {sub.mean()*100:>9.1f}%")

    print(f"\n{'='*110}")
    print("  【指标2】后续30天最终收益（30天后收盘价 vs 当前价格）")
    print(f"{'='*110}")
    print(f"\n{'回撤桶':<10} {'样本数':>8} {'10%分位':>10} {'25%分位':>10} {'50%分位':>10} {'75%分位':>10} {'90%分位':>10} {'均值':>10}")
    print("-" * 110)

    for b in buckets:
        sub = df[df['bucket'] == b]['final_return_30d']
        if len(sub) == 0:
            continue
        print(f"{b:<10} {len(sub):>8} {sub.quantile(0.10)*100:>9.1f}% {sub.quantile(0.25)*100:>9.1f}% {sub.quantile(0.50)*100:>9.1f}% {sub.quantile(0.75)*100:>9.1f}% {sub.quantile(0.90)*100:>9.1f}% {sub.mean()*100:>9.1f}%")

    print(f"\n{'='*110}")
    print("  【指标3】后续30天创出新低概率")
    print(f"{'='*110}")
    print(f"\n{'回撤桶':<10} {'样本数':>8} {'创新低概率':>12}")
    print("-" * 50)

    for b in buckets:
        sub = df[df['bucket'] == b]['new_low_30d']
        if len(sub) == 0:
            continue
        print(f"{b:<10} {len(sub):>8} {sub.mean()*100:>11.1f}%")

    # 做空目标价建议
    print(f"\n{'='*110}")
    print("  【做空目标价参考】基于历史50%分位最大跌幅")
    print(f"{'='*110}")
    print(f"\n{'当前回撤':<12} {'假设当前价格':>12} {'50%概率再跌':>14} {'目标价(剩余价值)':>18}")
    print("-" * 70)

    for b in buckets:
        sub = df[df['bucket'] == b]['max_fall_30d']
        if len(sub) == 0:
            continue
        median_fall = sub.quantile(0.50)
        # 假设当前价格 = 1，当前回撤 = bucket中值
        bucket_mid = {'50-60%': 0.55, '60-70%': 0.65, '70-80%': 0.75, '80-90%': 0.85, '90%+': 0.95}[b]
        current_price = 1 - bucket_mid  # 当前价格是原高点的 (1-drawdown)
        target_price = current_price * (1 - median_fall)  # 目标价
        print(f"{b:<12} {current_price*100:>11.1f}%      {median_fall*100:>13.1f}%      {target_price*100:>17.1f}%")

    print(f"\n{'='*110}")
    print("  回测完成")
    print(f"{'='*110}")


def main():
    df = run_backtest('spot')
    print_report(df)


if __name__ == '__main__':
    main()
