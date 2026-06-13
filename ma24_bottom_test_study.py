#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
MA24 底部多次测试突破研究

目标：量化"长期处于底部 + 多次上穿MA24又回落 + 最终突破"这种形态的后续表现

用法：
  python ma24_bottom_test_study.py --market swap --bottom-window 30 --min-below-ratio 0.50 --min-cross-ups 2
"""

import os
import sys
import time
import logging
import argparse
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('ma24_bottom_test')

DB_DSN = os.getenv('DB_DSN', 'postgresql://postgres:12@127.0.0.1:5432/market_data')


@dataclass
class MA24BottomBreakoutEvent:
    symbol: str
    date: datetime
    price: float
    ma24: float
    dist_ma24: float
    below_ratio: float  # 窗口内低于MA24的比例
    cross_up_count: int  # 窗口内上穿次数
    prev_cross_up_date: Optional[datetime]  # 最近一次上穿日期
    days_since_prev_cross: int
    window_return: float  # 窗口内收益率
    window_drawdown: float  # 窗口内最大回撤
    volume_ratio: float
    post_r5d: Optional[float]
    post_r10d: Optional[float]
    post_r20d: Optional[float]
    post_r30d: Optional[float]
    post_max5d: Optional[float]
    post_max10d: Optional[float]
    post_max20d: Optional[float]
    post_max30d: Optional[float]
    post_min5d: Optional[float]
    post_min10d: Optional[float]
    post_drawdown5d: Optional[float]
    post_drawdown10d: Optional[float]


def get_db_conn():
    return psycopg2.connect(DB_DSN)


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


def fetch_klines(symbol: str, market_type: str, min_history: int = 120) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        df = pd.read_sql("""
            SELECT open_time, open, high, low, close, volume, quote_volume
            FROM binance_daily_klines
            WHERE symbol = %s AND market_type = %s
            ORDER BY open_time ASC
        """, conn, params=(symbol, market_type))
        if len(df) < min_history:
            return pd.DataFrame()
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.set_index('open_time')
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
        return df
    finally:
        conn.close()


def analyze_symbol(
    symbol: str,
    market_type: str,
    bottom_window: int = 30,
    min_below_ratio: float = 0.50,
    min_cross_ups: int = 2,
    recent_below_days: int = 5,
    min_breakout_dist: float = 0.0,
    min_future_days: int = 5
) -> List[MA24BottomBreakoutEvent]:
    """分析单个币的所有MA24底部多次测试突破事件"""
    df = fetch_klines(symbol, market_type)
    if df.empty or len(df) < bottom_window + 60:
        return []

    df['ma24'] = df['close'].rolling(window=24, min_periods=20).mean()
    df['above_ma24'] = df['close'] > df['ma24']
    df['dist_ma24'] = (df['close'] - df['ma24']) / df['ma24']
    df['cross_up'] = (~df['above_ma24'].shift(1).fillna(False)) & df['above_ma24']
    df['cross_down'] = df['above_ma24'].shift(1).fillna(False) & (~df['above_ma24'])
    df['volume_ma20'] = df['quote_volume'].rolling(window=20, min_periods=15).mean()
    df['volume_ratio'] = df['quote_volume'] / df['volume_ma20']

    df_valid = df.dropna(subset=['ma24', 'volume_ma20'])
    if len(df_valid) < bottom_window + min_future_days + 1:
        return []

    events = []
    for i in range(bottom_window, len(df_valid) - min_future_days):
        row = df_valid.iloc[i]

        # 条件1: 当天收盘价 > MA24（突破）
        if not row['above_ma24']:
            continue

        # 条件2: 突破幅度 >= 阈值
        if row['dist_ma24'] < min_breakout_dist:
            continue

        # 条件3: 最近5天内曾经低于MA24（避免已经连续在上方）
        recent = df_valid.iloc[i-recent_below_days:i]
        if recent.empty or recent['above_ma24'].all():
            continue

        # 条件4: 底部窗口内大部分时间低于MA24
        window = df_valid.iloc[i-bottom_window:i]
        below_ratio = (~window['above_ma24']).sum() / len(window)
        if below_ratio < min_below_ratio:
            continue

        # 条件5: 窗口内上穿次数 >= 阈值
        cross_up_count = window['cross_up'].sum()
        if cross_up_count < min_cross_ups:
            continue

        # 计算窗口内收益和回撤
        window_start_price = window['close'].iloc[0]
        window_end_price = window['close'].iloc[-1]
        window_return = (window_end_price - window_start_price) / window_start_price
        window_drawdown = (window['low'].min() - window['high'].max()) / window['high'].max()

        # 最近一次上穿
        prev_cross_ups = window[window['cross_up']]
        if not prev_cross_ups.empty:
            prev_cross = prev_cross_ups.iloc[-1]
            days_since_prev = i - df_valid.index.get_loc(prev_cross.name)
        else:
            prev_cross = None
            days_since_prev = -1

        # 未来表现（基于可用数据动态计算）
        future = df_valid.iloc[i+1:min(i+31, len(df_valid))]
        price = row['close']

        post_r5d = (future['close'].iloc[4] - price) / price if len(future) >= 5 else np.nan
        post_r10d = (future['close'].iloc[9] - price) / price if len(future) >= 10 else np.nan
        post_r20d = (future['close'].iloc[19] - price) / price if len(future) >= 20 else np.nan
        post_r30d = (future['close'].iloc[29] - price) / price if len(future) >= 30 else np.nan

        post_max5d = (future['high'].iloc[:5].max() - price) / price if len(future) >= 5 else np.nan
        post_max10d = (future['high'].iloc[:10].max() - price) / price if len(future) >= 10 else np.nan
        post_max20d = (future['high'].iloc[:20].max() - price) / price if len(future) >= 20 else np.nan
        post_max30d = (future['high'].iloc[:30].max() - price) / price if len(future) >= 30 else np.nan

        post_min5d = (future['low'].iloc[:5].min() - price) / price if len(future) >= 5 else np.nan
        post_min10d = (future['low'].iloc[:10].min() - price) / price if len(future) >= 10 else np.nan
        post_drawdown5d = (future['low'].iloc[:5].min() - price) / price if len(future) >= 5 else np.nan
        post_drawdown10d = (future['low'].iloc[:10].min() - price) / price if len(future) >= 10 else np.nan

        events.append(MA24BottomBreakoutEvent(
            symbol=symbol,
            date=row.name,
            price=price,
            ma24=row['ma24'],
            dist_ma24=row['dist_ma24'],
            below_ratio=below_ratio,
            cross_up_count=cross_up_count,
            prev_cross_up_date=prev_cross.name if prev_cross is not None else None,
            days_since_prev_cross=days_since_prev,
            window_return=window_return,
            window_drawdown=window_drawdown,
            volume_ratio=row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1.0,
            post_r5d=post_r5d,
            post_r10d=post_r10d,
            post_r20d=post_r20d,
            post_r30d=post_r30d,
            post_max5d=post_max5d,
            post_max10d=post_max10d,
            post_max20d=post_max20d,
            post_max30d=post_max30d,
            post_min5d=post_min5d,
            post_min10d=post_min10d,
            post_drawdown5d=post_drawdown5d,
            post_drawdown10d=post_drawdown10d,
        ))

    return events


def run_study(symbols: List[str], market_type: str, **kwargs) -> List[MA24BottomBreakoutEvent]:
    all_events = []
    for i, sym in enumerate(symbols, 1):
        try:
            events = analyze_symbol(sym, market_type, **kwargs)
            all_events.extend(events)
            if i % 50 == 0:
                logger.info(f"进度: {i}/{len(symbols)}，已收集 {len(all_events)} 个事件")
        except Exception as e:
            logger.debug(f"[{sym}] 分析失败: {e}")
    return all_events


def print_report(events: List[MA24BottomBreakoutEvent]):
    if not events:
        print("\n❌ 未找到符合条件的事件\n")
        return

    df = pd.DataFrame([{
        'symbol': e.symbol,
        'date': e.date,
        'price': e.price,
        'dist_ma24': e.dist_ma24,
        'below_ratio': e.below_ratio,
        'cross_up_count': e.cross_up_count,
        'days_since_prev_cross': e.days_since_prev_cross,
        'window_return': e.window_return,
        'window_drawdown': e.window_drawdown,
        'volume_ratio': e.volume_ratio,
        'post_r5d': e.post_r5d,
        'post_r10d': e.post_r10d,
        'post_r20d': e.post_r20d,
        'post_r30d': e.post_r30d,
        'post_max5d': e.post_max5d,
        'post_max10d': e.post_max10d,
        'post_max20d': e.post_max20d,
        'post_max30d': e.post_max30d,
        'post_min5d': e.post_min5d,
        'post_min10d': e.post_min10d,
        'post_drawdown5d': e.post_drawdown5d,
        'post_drawdown10d': e.post_drawdown10d,
    } for e in events])

    print(f"\n{'='*100}")
    print(f"  MA24 底部多次测试突破研究  (总样本: {len(df)} 个事件)")
    print(f"{'='*100}")
    print(f"   含30天完整未来数据: {df['post_r30d'].notna().sum()} 个")

    print(f"\n📊 事件特征")
    print(f"   平均低于MA24比例: {df['below_ratio'].median()*100:.1f}%")
    print(f"   平均上穿次数: {df['cross_up_count'].median():.1f}")
    print(f"   平均窗口收益: {df['window_return'].median()*100:.1f}%")
    print(f"   平均窗口回撤: {df['window_drawdown'].median()*100:.1f}%")
    print(f"   平均量比: {df['volume_ratio'].median():.2f}")

    # 后续表现
    print(f"\n{'='*100}")
    print("  【后续表现】按收盘价计算")
    print(f"{'='*100}")

    for target in [0.10, 0.20, 0.30, 0.50]:
        r5 = (df['post_r5d'].dropna() >= target).mean()
        r10 = (df['post_r10d'].dropna() >= target).mean()
        r20 = (df['post_r20d'].dropna() >= target).mean()
        r30 = (df['post_r30d'].dropna() >= target).mean()
        print(f"\n📌 收盘涨≥{target*100:.0f}%概率:")
        print(f"   5天: {r5*100:.1f}%, 10天: {r10*100:.1f}%, 20天: {r20*100:.1f}%, 30天: {r30*100:.1f}%")

    # 最大涨幅（按高点）
    print(f"\n{'='*100}")
    print("  【最大涨幅】按未来高点计算")
    print(f"{'='*100}")

    for target in [0.20, 0.30, 0.50, 1.00]:
        m5 = (df['post_max5d'].dropna() >= target).mean()
        m10 = (df['post_max10d'].dropna() >= target).mean()
        m20 = (df['post_max20d'].dropna() >= target).mean()
        m30 = (df['post_max30d'].dropna() >= target).mean()
        print(f"\n📌 最大涨≥{target*100:.0f}%概率:")
        print(f"   5天: {m5*100:.1f}%, 10天: {m10*100:.1f}%, 20天: {m20*100:.1f}%, 30天: {m30*100:.1f}%")

    # 中位数收益
    print(f"\n{'='*100}")
    print("  【中位数收益】")
    print(f"{'='*100}")
    print(f"   5天收盘: {df['post_r5d'].median()*100:.1f}%, 最大: {df['post_max5d'].median()*100:.1f}%, 最低: {df['post_min5d'].median()*100:.1f}%")
    print(f"   10天收盘: {df['post_r10d'].median()*100:.1f}%, 最大: {df['post_max10d'].median()*100:.1f}%, 最低: {df['post_min10d'].median()*100:.1f}%")
    print(f"   20天收盘: {df['post_r20d'].median()*100:.1f}%, 最大: {df['post_max20d'].median()*100:.1f}%")
    print(f"   30天收盘: {df['post_r30d'].median()*100:.1f}%, 最大: {df['post_max30d'].median()*100:.1f}%")

    # 失败风险
    print(f"\n{'='*100}")
    print("  【下跌风险】")
    print(f"{'='*100}")
    print(f"   5天收盘跌≥10%: {(df['post_r5d'].dropna() <= -0.10).mean()*100:.1f}%")
    print(f"   5天最大回撤≥10%: {(df['post_drawdown5d'].dropna() <= -0.10).mean()*100:.1f}%")
    print(f"   10天收盘跌≥20%: {(df['post_r10d'].dropna() <= -0.20).mean()*100:.1f}%")
    print(f"   10天最大回撤≥15%: {(df['post_drawdown10d'].dropna() <= -0.15).mean()*100:.1f}%")

    # 最新事件
    print(f"\n{'='*100}")
    print("  【最近10个事件】")
    print(f"{'='*100}")
    recent = df.sort_values('date', ascending=False).head(10)
    print(recent[['date','symbol','dist_ma24','below_ratio','cross_up_count','post_r5d','post_max10d']].to_string(index=False))

    print(f"\n{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(description='MA24 底部多次测试突破研究')
    parser.add_argument('--market', type=str, default='swap', choices=['spot', 'swap'])
    parser.add_argument('--symbols', type=str, help='指定币种，逗号分隔')
    parser.add_argument('--bottom-window', type=int, default=30, help='底部窗口天数（默认30）')
    parser.add_argument('--min-below-ratio', type=float, default=0.50, help='窗口内低于MA24的最小比例（默认0.5）')
    parser.add_argument('--min-cross-ups', type=int, default=2, help='窗口内最小上穿次数（默认2）')
    parser.add_argument('--recent-below-days', type=int, default=5, help='突破前N天内需曾低于MA24（默认5）')
    parser.add_argument('--min-breakout-dist', type=float, default=0.0, help='最小突破幅度（默认0）')
    parser.add_argument('--min-future-days', type=int, default=5, help='最小未来数据天数（默认5）')
    args = parser.parse_args()

    kwargs = {
        'bottom_window': args.bottom_window,
        'min_below_ratio': args.min_below_ratio,
        'min_cross_ups': args.min_cross_ups,
        'recent_below_days': args.recent_below_days,
        'min_breakout_dist': args.min_breakout_dist,
        'min_future_days': args.min_future_days,
    }

    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols = get_symbols(args.market)

    logger.info(f"开始研究 {len(symbols)} 个币，参数: {kwargs}")
    events = run_study(symbols, args.market, **kwargs)
    print_report(events)


if __name__ == '__main__':
    main()
