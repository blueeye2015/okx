#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
策略胜率历史验证器
验证两个策略：
  1. 做空增强版：pre_r20d>0.3 + 缩量突破 + 小幅度突破 → 30天内跌20%+胜率
  2. 做多底部反转：超跌 + pre_r20d<0.05 + 放量突破 + 有力突破 → 30天内涨20%+胜率
同时对比 5天/10天跟踪期的表现差异
"""

import os
import sys
import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass

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


def calculate_weekly_ma(df_daily: pd.DataFrame, weekly_period: int = 60) -> pd.Series:
    weekly = df_daily['close'].resample('W-FRI').last().dropna()
    weekly_ma = weekly.rolling(window=weekly_period, min_periods=weekly_period).mean()
    weekly_ma_daily = weekly_ma.reindex(df_daily.index, method='ffill')
    return weekly_ma_daily


@dataclass
class BreakoutEvent:
    symbol: str
    date: datetime
    price: float
    drawdown: float
    weak_days: int
    weak_ratio: float
    pre_r5d: float
    pre_r20d: float
    pre_r60d: float
    vol_ratio: float
    quote_vol_ratio: float
    volatility: float
    dist_ma24: float
    post_r5d: float
    post_r10d: float
    post_r20d: float
    post_r30d: float
    post_min5d: float
    post_min10d: float
    post_min30d: float
    post_max5d: float
    post_max10d: float
    post_max30d: float


def analyze_symbol(symbol: str, market_type: str) -> List[BreakoutEvent]:
    """分析单个币的所有MA24突破事件"""
    df = fetch_daily_klines(symbol, market_type)
    if df.empty or len(df) < 400:
        return []

    df['ma6'] = df['close'].rolling(window=6, min_periods=6).mean()
    df['ma24'] = df['close'].rolling(window=24, min_periods=24).mean()
    df['weekly_ma60'] = calculate_weekly_ma(df, 60)
    df['rolling_high_500d'] = df['high'].rolling(window=500, min_periods=300).max()
    df['drawdown'] = (df['rolling_high_500d'] - df['close']) / df['rolling_high_500d']
    df['below_weekly_ma'] = df['close'] < df['weekly_ma60']
    df['above_ma24'] = df['close'] > df['ma24']

    df['volume_ma20'] = df['volume'].rolling(window=20, min_periods=15).mean()
    df['quote_volume_ma20'] = df['quote_volume'].rolling(window=20, min_periods=15).mean()
    df['return_1d'] = df['close'].pct_change()
    df['volatility_20d'] = df['return_1d'].rolling(window=20, min_periods=15).std()

    df['pre_return_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['pre_return_20d'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    df['pre_return_60d'] = (df['close'] - df['close'].shift(60)) / df['close'].shift(60)
    df['weak_days_250'] = df['below_weekly_ma'].rolling(window=250, min_periods=200).sum()
    df['weak_ratio'] = df['weak_days_250'] / 250

    df['prev_above_ma24'] = df['above_ma24'].shift(1)
    df['is_breakout'] = (~df['prev_above_ma24'].fillna(True)) & df['above_ma24']

    events = []
    for idx in df[df['is_breakout']].index:
        i = df.index.get_loc(idx)
        if i < 60 or i >= len(df) - 60:
            continue

        row = df.iloc[i]
        if pd.isna(row['ma24']) or pd.isna(row['drawdown']) or pd.isna(row['volume_ma20']):
            continue
        if row['volume_ma20'] == 0:
            continue

        future = df.iloc[i+1:min(i+61, len(df))]
        if len(future) < 30:
            continue

        post_r5d = (future['close'].iloc[4] - row['close']) / row['close'] if len(future) >= 5 else np.nan
        post_r10d = (future['close'].iloc[9] - row['close']) / row['close'] if len(future) >= 10 else np.nan
        post_r20d = (future['close'].iloc[19] - row['close']) / row['close'] if len(future) >= 20 else np.nan
        post_r30d = (future['close'].iloc[29] - row['close']) / row['close'] if len(future) >= 30 else np.nan

        post_min5d = ((future['close'].iloc[:5].min() - row['close']) / row['close']) if len(future) >= 5 else np.nan
        post_min10d = ((future['close'].iloc[:10].min() - row['close']) / row['close']) if len(future) >= 10 else np.nan
        post_min30d = ((future['close'].iloc[:30].min() - row['close']) / row['close']) if len(future) >= 30 else np.nan

        post_max5d = ((future['close'].iloc[:5].max() - row['close']) / row['close']) if len(future) >= 5 else np.nan
        post_max10d = ((future['close'].iloc[:10].max() - row['close']) / row['close']) if len(future) >= 10 else np.nan
        post_max30d = ((future['close'].iloc[:30].max() - row['close']) / row['close']) if len(future) >= 30 else np.nan

        events.append(BreakoutEvent(
            symbol=symbol,
            date=idx,
            price=row['close'],
            drawdown=row['drawdown'],
            weak_days=int(row['weak_days_250']) if not pd.isna(row['weak_days_250']) else 0,
            weak_ratio=row['weak_ratio'] if not pd.isna(row['weak_ratio']) else 0,
            pre_r5d=row['pre_return_5d'] if not pd.isna(row['pre_return_5d']) else 0,
            pre_r20d=row['pre_return_20d'] if not pd.isna(row['pre_return_20d']) else 0,
            pre_r60d=row['pre_return_60d'] if not pd.isna(row['pre_return_60d']) else 0,
            vol_ratio=(row['volume'] / row['volume_ma20']) if row['volume_ma20'] > 0 else 1.0,
            quote_vol_ratio=(row['quote_volume'] / row['quote_volume_ma20']) if row['quote_volume_ma20'] > 0 else 1.0,
            volatility=row['volatility_20d'] if not pd.isna(row['volatility_20d']) else 0,
            dist_ma24=(row['close'] - row['ma24']) / row['ma24'],
            post_r5d=post_r5d,
            post_r10d=post_r10d,
            post_r20d=post_r20d,
            post_r30d=post_r30d,
            post_min5d=post_min5d,
            post_min10d=post_min10d,
            post_min30d=post_min30d,
            post_max5d=post_max5d,
            post_max10d=post_max10d,
            post_max30d=post_max30d,
        ))

    return events


def run_validation(market_type: str = 'spot'):
    symbols = get_symbols(market_type)
    logger.info(f"开始回测 {market_type} 市场，共 {len(symbols)} 个币...")

    all_events = []
    for i, symbol in enumerate(symbols, 1):
        try:
            events = analyze_symbol(symbol, market_type)
            all_events.extend(events)
            if i % 50 == 0:
                logger.info(f"进度: {i}/{len(symbols)}，已收集 {len(all_events)} 个突破事件")
        except Exception as e:
            logger.debug(f"[{symbol}] 异常: {e}")

    logger.info(f"回测完成! 共 {len(all_events)} 个突破事件")
    return all_events


def print_report(events: List[BreakoutEvent]):
    if not events:
        print("❌ 无数据")
        return

    df = pd.DataFrame([{
        'drawdown': e.drawdown,
        'weak_days': e.weak_days,
        'weak_ratio': e.weak_ratio,
        'pre_r5d': e.pre_r5d,
        'pre_r20d': e.pre_r20d,
        'pre_r60d': e.pre_r60d,
        'vol_ratio': e.vol_ratio,
        'quote_vol_ratio': e.quote_vol_ratio,
        'volatility': e.volatility,
        'dist_ma24': e.dist_ma24,
        'post_r5d': e.post_r5d,
        'post_r10d': e.post_r10d,
        'post_r20d': e.post_r20d,
        'post_r30d': e.post_r30d,
        'post_min5d': e.post_min5d,
        'post_min10d': e.post_min10d,
        'post_min30d': e.post_min30d,
        'post_max5d': e.post_max5d,
        'post_max10d': e.post_max10d,
        'post_max30d': e.post_max30d,
    } for e in events])

    # 过滤异常值
    df = df[df['vol_ratio'] < 50]
    df = df[df['volatility'] < 0.5]

    print(f"\n{'='*100}")
    print(f"  策略胜率历史验证报告  (总样本: {len(df)} 个MA24突破事件)")
    print(f"{'='*100}")

    # ========== 做空策略验证 ==========
    print(f"\n{'='*100}")
    print("  【做空策略验证】目标: 30天内跌 20%+")
    print(f"{'='*100}")

    # 基础条件：超跌背景
    oversold_df = df[(df['drawdown'] >= 0.80) & (df['weak_ratio'] >= 0.80)]
    print(f"\n📌 基础筛选 (跌≥80% 且 弱势≥80%): {len(oversold_df)} 个事件")
    if len(oversold_df) > 0:
        short_success = (oversold_df['post_min30d'] <= -0.20).mean()
        print(f"   30天内跌≥20% 概率: {short_success*100:.1f}%")
        print(f"   30天平均最大跌幅: {oversold_df['post_min30d'].median()*100:.1f}%")
        print(f"   30天平均最大涨幅: {oversold_df['post_max30d'].median()*100:.1f}%")

    # Level 3 单条件: pre_r20d > 0.3
    l3_df = oversold_df[oversold_df['pre_r20d'] > 0.30]
    print(f"\n📌 Level 3 单条件 (pre_r20d > 30%): {len(l3_df)} 个事件")
    if len(l3_df) > 0:
        short_success = (l3_df['post_min30d'] <= -0.20).mean()
        rr = abs(l3_df['post_min30d'].median()) / abs(l3_df['post_max30d'].median()) if l3_df['post_max30d'].median() != 0 else 0
        print(f"   30天内跌≥20% 概率: {short_success*100:.1f}%")
        print(f"   盈亏比 (中位数): {rr:.2f}")
        print(f"   30天平均最大跌幅: {l3_df['post_min30d'].median()*100:.1f}%")
        print(f"   30天平均最大涨幅: {l3_df['post_max30d'].median()*100:.1f}%")

    # Level 3+ 增强版: pre_r20d > 0.3 + 缩量 + 小幅度突破
    l3_plus_df = l3_df[(l3_df['vol_ratio'] < 0.80) & (l3_df['dist_ma24'] < 0.02)]
    print(f"\n📌 Level 3+ 增强版 (pre_r20d>30% + 缩量<0.8x + 突破幅度<2%): {len(l3_plus_df)} 个事件")
    if len(l3_plus_df) > 0:
        short_success = (l3_plus_df['post_min30d'] <= -0.20).mean()
        rr = abs(l3_plus_df['post_min30d'].median()) / abs(l3_plus_df['post_max30d'].median()) if l3_plus_df['post_max30d'].median() != 0 else 0
        print(f"   30天内跌≥20% 概率: {short_success*100:.1f}%")
        print(f"   盈亏比 (中位数): {rr:.2f}")
        print(f"   30天平均最大跌幅: {l3_plus_df['post_min30d'].median()*100:.1f}%")
        print(f"   30天平均最大涨幅: {l3_plus_df['post_max30d'].median()*100:.1f}%")

    # 5天 vs 10天跟踪期对比（做空）
    print(f"\n{'='*100}")
    print("  【做空跟踪期对比】Level 3 单条件下")
    print(f"{'='*100}")
    if len(l3_df) > 0:
        short_5d = (l3_df['post_min5d'] <= -0.10).mean()
        short_10d = (l3_df['post_min10d'] <= -0.15).mean()
        short_30d = (l3_df['post_min30d'] <= -0.20).mean()
        print(f"   5天内跌≥10%: {short_5d*100:.1f}%")
        print(f"   10天内跌≥15%: {short_10d*100:.1f}%")
        print(f"   30天内跌≥20%: {short_30d*100:.1f}%")

    # ========== 做多策略验证 ==========
    print(f"\n{'='*100}")
    print("  【做多策略验证】目标: 30天内涨 20%+")
    print(f"{'='*100}")

    # 底部反转条件: 跌≥70% + 弱势≥80% + pre_r20d<5% + 放量突破 + 有力突破
    long_df = df[
        (df['drawdown'] >= 0.70) &
        (df['weak_ratio'] >= 0.80) &
        (df['pre_r20d'] < 0.05) &
        (df['vol_ratio'] > 1.20) &
        (df['dist_ma24'] > 0.02)
    ]
    print(f"\n📌 底部反转条件 (跌≥70% + 弱势≥80% + pre_r20d<5% + 放量>1.2x + 突破>2%): {len(long_df)} 个事件")
    if len(long_df) > 0:
        long_success = (long_df['post_max30d'] >= 0.20).mean()
        rr = abs(long_df['post_max30d'].median()) / abs(long_df['post_min30d'].median()) if long_df['post_min30d'].median() != 0 else 0
        print(f"   30天内涨≥20% 概率: {long_success*100:.1f}%")
        print(f"   盈亏比 (中位数): {rr:.2f}")
        print(f"   30天平均最大涨幅: {long_df['post_max30d'].median()*100:.1f}%")
        print(f"   30天平均最大跌幅: {long_df['post_min30d'].median()*100:.1f}%")

    # 放宽条件: 跌≥60% + pre_r20d<10% + 放量突破
    long_relaxed_df = df[
        (df['drawdown'] >= 0.60) &
        (df['weak_ratio'] >= 0.70) &
        (df['pre_r20d'] < 0.10) &
        (df['vol_ratio'] > 1.20)
    ]
    print(f"\n📌 放宽版 (跌≥60% + 弱势≥70% + pre_r20d<10% + 放量>1.2x): {len(long_relaxed_df)} 个事件")
    if len(long_relaxed_df) > 0:
        long_success = (long_relaxed_df['post_max30d'] >= 0.20).mean()
        print(f"   30天内涨≥20% 概率: {long_success*100:.1f}%")
        print(f"   30天平均最大涨幅: {long_relaxed_df['post_max30d'].median()*100:.1f}%")
        print(f"   30天平均最大跌幅: {long_relaxed_df['post_min30d'].median()*100:.1f}%")

    # 5天 vs 10天跟踪期对比（做多）
    print(f"\n{'='*100}")
    print("  【做多跟踪期对比】底部反转严格条件下")
    print(f"{'='*100}")
    if len(long_df) > 0:
        long_5d = (long_df['post_max5d'] >= 0.10).mean()
        long_10d = (long_df['post_max10d'] >= 0.15).mean()
        long_30d = (long_df['post_max30d'] >= 0.20).mean()
        print(f"   5天内涨≥10%: {long_5d*100:.1f}%")
        print(f"   10天内涨≥15%: {long_10d*100:.1f}%")
        print(f"   30天内涨≥20%: {long_30d*100:.1f}%")

    # 止损分析：如果按突破日低点下方3%止损，实际盈亏如何？
    print(f"\n{'='*100}")
    print("  【止损模拟】做多严格条件，止损=突破日低点-3%")
    print(f"{'='*100}")
    if len(long_df) > 0:
        # 简化：假设突破日低点 ≈ open 或前一日 low，这里用 post_min5d 近似
        # 如果5天内跌幅达到3%即止损
        stopped = long_df['post_min5d'] <= -0.03
        not_stopped = ~stopped
        print(f"   5天内触发3%止损比例: {stopped.mean()*100:.1f}%")
        if not_stopped.sum() > 0:
            print(f"   未止损案例中，30天涨≥20%概率: {(long_df.loc[not_stopped, 'post_max30d'] >= 0.20).mean()*100:.1f}%")
        if stopped.sum() > 0:
            print(f"   止损案例中，30天涨≥20%概率: {(long_df.loc[stopped, 'post_max30d'] >= 0.20).mean()*100:.1f}%")

    print(f"\n{'='*100}")
    print("  验证完成")
    print(f"{'='*100}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', choices=['spot', 'swap'], default='spot')
    args = parser.parse_args()

    events = run_validation(args.market)
    print_report(events)
