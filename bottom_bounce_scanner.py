#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
超跌寻底后启动事件监控器 (Level3 风格)

功能:
1. 回测历史上所有"寻底后启动"事件的表现
2. 实时监控当前市场，发现新的事件

事件定义:
- 寻底期 + 最近30天内满足过 strict oversold（drawdown≥80%, weak_ratio≥80%）
- 寻底期（T-lookback-2 到 T-3）大部分时间收盘价 < MA6
- 最近2天启动：累计涨幅 ≥ min_return，且收盘价 > MA6
- 当前位置：MA6 < 收盘价 < MA24

用法:
  历史回测: python bottom_bounce_scanner.py --backtest --market spot
  实时监控: python bottom_bounce_scanner.py --scan-now --market all
"""

import os
import sys
import logging
import argparse
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

DB_DSN = os.getenv('DB_DSN', 'postgresql://postgres:12@127.0.0.1:5432/market_data')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('bottom_bounce')


# ==========================================
# 数据模型
# ==========================================
@dataclass
class BottomBounceEvent:
    symbol: str
    market_type: str
    date: datetime
    price: float
    ma6: float
    ma24: float
    drawdown: float
    weak_ratio: float
    two_day_return: float
    pre_launch_below_ma6_ratio: float
    pre_launch_days: int
    vol_ratio: float
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


# ==========================================
# 数据获取
# ==========================================
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


def fetch_daily_klines(symbol: str, market_type: str, min_date: str = '2018-01-01') -> pd.DataFrame:
    conn = get_db_conn()
    query = """
        SELECT open_time, open, high, low, close, volume, quote_volume
        FROM binance_daily_klines
        WHERE symbol = %s AND market_type = %s AND open_time >= %s
        ORDER BY open_time ASC
    """
    try:
        df = pd.read_sql(query, conn, params=(symbol, market_type, min_date))
        if not df.empty:
            df['open_time'] = pd.to_datetime(df['open_time'])
            df = df.set_index('open_time')
        return df
    finally:
        conn.close()


def calculate_weekly_ma(df: pd.DataFrame, period: int = 60) -> pd.Series:
    weekly = df['close'].resample('W-FRI').last().dropna()
    weekly_ma = weekly.rolling(window=period, min_periods=period).mean()
    return weekly_ma.reindex(df.index, method='ffill')


# ==========================================
# 核心分析
# ==========================================
def analyze_symbol(
    symbol: str,
    market_type: str,
    min_return: float = 0.20,
    lookback_days: int = 10,
    min_below_ma6_ratio: float = 0.50,
    only_latest: bool = False,
) -> List[BottomBounceEvent]:
    """
    分析单个币的所有"寻底后启动"事件
    
    only_latest=True: 只检查最新一天（实时监控模式）
    only_latest=False: 遍历历史所有日期（回测模式）
    """
    df = fetch_daily_klines(symbol, market_type)
    if df.empty or len(df) < 300:
        return []

    # 计算指标
    df['ma6'] = df['close'].rolling(window=6, min_periods=6).mean()
    df['ma24'] = df['close'].rolling(window=24, min_periods=24).mean()
    df['weekly_ma60'] = calculate_weekly_ma(df, 60)
    df['rolling_high_500d'] = df['high'].rolling(window=500, min_periods=300).max()
    df['drawdown'] = (df['rolling_high_500d'] - df['close']) / df['rolling_high_500d']
    df['below_weekly_ma'] = df['close'] < df['weekly_ma60']
    df['weak_ratio_250'] = df['below_weekly_ma'].rolling(window=250, min_periods=200).mean()
    df['below_ma6'] = df['close'] < df['ma6']
    df['volume_ma20'] = df['volume'].rolling(window=20, min_periods=15).mean()

    df_valid = df.dropna(subset=['ma6', 'ma24', 'drawdown', 'weak_ratio_250', 'volume_ma20']).copy()
    if len(df_valid) < lookback_days + 30:
        return []

    # 2天累计收益 (T-1和T相对于T-2)
    df_valid['two_day_return'] = (df_valid['close'] - df_valid['close'].shift(2)) / df_valid['close'].shift(2)

    events = []

    if only_latest:
        scan_indices = [df_valid.index[-1]]
    else:
        scan_indices = df_valid.index[lookback_days + 3:-30]

    for idx in scan_indices:
        i = df_valid.index.get_loc(idx)
        row = df_valid.iloc[i]

        price = float(row['close'])
        ma6 = float(row['ma6'])
        ma24 = float(row['ma24'])

        # ===== 条件1: 当前 MA6 < 收盘价 < MA24 =====
        if not (ma6 < price < ma24):
            continue

        # ===== 条件2: 最近2天累计涨幅 >= min_return =====
        two_day_return = float(row['two_day_return'])
        if two_day_return < min_return or np.isnan(two_day_return):
            continue

        # ===== 条件3: 最近2天中至少有1天收盘价 > MA6 =====
        recent_2d = df_valid.iloc[max(0, i-1):i+1]
        if (recent_2d['close'] <= recent_2d['ma6']).all():
            continue

        # ===== 条件4: 寻底期（T-lookback-2 到 T-3）大部分时间 < MA6 =====
        pre_launch_start = max(0, i - lookback_days - 2)
        pre_launch = df_valid.iloc[pre_launch_start:i-2]
        if len(pre_launch) < 3:
            continue

        below_ma6_count = pre_launch['below_ma6'].sum()
        below_ma6_ratio = below_ma6_count / len(pre_launch)
        if below_ma6_ratio < min_below_ma6_ratio:
            continue

        # ===== 条件5: 寻底期 + 最近30天内满足过 strict oversold =====
        check_start = max(0, i - lookback_days - 30)
        check_window = df_valid.iloc[check_start:i]
        ever_oversold = (
            (check_window['drawdown'] >= 0.80) & (check_window['weak_ratio_250'] >= 0.80)
        ).any()
        if not ever_oversold:
            continue

        # ===== 获取后续表现 =====
        future = df_valid.iloc[i+1:min(i+31, len(df_valid))]
        if len(future) < 5:
            continue

        post_r5d = (future['close'].iloc[4] - price) / price if len(future) >= 5 else np.nan
        post_r10d = (future['close'].iloc[9] - price) / price if len(future) >= 10 else np.nan
        post_r20d = (future['close'].iloc[19] - price) / price if len(future) >= 20 else np.nan
        post_r30d = (future['close'].iloc[29] - price) / price if len(future) >= 30 else np.nan

        post_min5d = (future['close'].iloc[:5].min() - price) / price if len(future) >= 5 else np.nan
        post_min10d = (future['close'].iloc[:10].min() - price) / price if len(future) >= 10 else np.nan
        post_min30d = (future['close'].iloc[:30].min() - price) / price if len(future) >= 30 else np.nan

        post_max5d = (future['close'].iloc[:5].max() - price) / price if len(future) >= 5 else np.nan
        post_max10d = (future['close'].iloc[:10].max() - price) / price if len(future) >= 10 else np.nan
        post_max30d = (future['close'].iloc[:30].max() - price) / price if len(future) >= 30 else np.nan

        vol_ratio = float(row['volume'] / row['volume_ma20']) if row['volume_ma20'] > 0 else 1.0

        events.append(BottomBounceEvent(
            symbol=symbol,
            market_type=market_type,
            date=idx,
            price=price,
            ma6=ma6,
            ma24=ma24,
            drawdown=float(row['drawdown']),
            weak_ratio=float(row['weak_ratio_250']),
            two_day_return=two_day_return,
            pre_launch_below_ma6_ratio=below_ma6_ratio,
            pre_launch_days=len(pre_launch),
            vol_ratio=vol_ratio,
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


# ==========================================
# 回测引擎
# ==========================================
def run_backtest(market_types: List[str], **kwargs) -> List[BottomBounceEvent]:
    all_events = []
    for mtype in market_types:
        symbols = get_symbols(mtype)
        logger.info(f"开始回测 {mtype} 市场，共 {len(symbols)} 个币...")
        for i, sym in enumerate(symbols, 1):
            try:
                events = analyze_symbol(sym, mtype, only_latest=False, **kwargs)
                all_events.extend(events)
                if i % 50 == 0:
                    logger.info(f"  进度: {i}/{len(symbols)}，已收集 {len(all_events)} 个事件")
            except Exception as e:
                logger.debug(f"[{sym}] 异常: {e}")
        logger.info(f"{mtype} 回测完成，累计 {len(all_events)} 个事件")
    return all_events


def scan_now(market_types: List[str], **kwargs) -> List[BottomBounceEvent]:
    current_events = []
    for mtype in market_types:
        symbols = get_symbols(mtype)
        logger.info(f"扫描 {mtype} 市场当前状态，共 {len(symbols)} 个币...")
        for sym in symbols:
            try:
                events = analyze_symbol(sym, mtype, only_latest=True, **kwargs)
                if events:
                    current_events.extend(events)
            except Exception as e:
                logger.debug(f"[{sym}] 异常: {e}")
    return current_events


# ==========================================
# 报告输出
# ==========================================
def print_report(events: List[BottomBounceEvent]):
    if not events:
        print("\n❌ 未找到符合条件的事件\n")
        return

    df = pd.DataFrame([{
        'symbol': e.symbol,
        'market_type': e.market_type,
        'date': e.date,
        'price': e.price,
        'drawdown': e.drawdown,
        'weak_ratio': e.weak_ratio,
        'two_day_return': e.two_day_return,
        'pre_launch_below_ma6_ratio': e.pre_launch_below_ma6_ratio,
        'vol_ratio': e.vol_ratio,
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

    print(f"\n{'='*100}")
    print(f"  超跌寻底后启动事件报告  (总样本: {len(df)} 个事件)")
    print(f"{'='*100}")

    # ===== 基础统计 =====
    print(f"\n📊 事件基础特征")
    print(f"   平均回撤: {df['drawdown'].median()*100:.1f}%")
    print(f"   平均弱势占比: {df['weak_ratio'].median()*100:.1f}%")
    print(f"   平均2天涨幅: {df['two_day_return'].median()*100:.1f}%")
    print(f"   寻底期<MA6占比: {df['pre_launch_below_ma6_ratio'].median()*100:.1f}%")
    print(f"   平均量比: {df['vol_ratio'].median():.2f}")

    # ===== 做多验证：目标 30天内涨 20%+ =====
    print(f"\n{'='*100}")
    print("  【做多验证】目标: 30天内涨 20%+")
    print(f"{'='*100}")

    long_success_5d = (df['post_max5d'] >= 0.10).mean()
    long_success_10d = (df['post_max10d'] >= 0.15).mean()
    long_success_30d = (df['post_max30d'] >= 0.20).mean()

    print(f"\n📌 全部样本 ({len(df)} 个):")
    print(f"   5天内涨≥10%:  {long_success_5d*100:.1f}%")
    print(f"   10天内涨≥15%: {long_success_10d*100:.1f}%")
    print(f"   30天内涨≥20%: {long_success_30d*100:.1f}%")
    print(f"   盈亏比 (中位数): {abs(df['post_max30d'].median()) / abs(df['post_min30d'].median()):.2f}")
    print(f"   30天最大涨幅中位数: {df['post_max30d'].median()*100:.1f}%")
    print(f"   30天最大跌幅中位数: {df['post_min30d'].median()*100:.1f}%")

    # 按2天涨幅分层
    high_momentum = df[df['two_day_return'] >= 0.30]
    if len(high_momentum) > 0:
        print(f"\n📌 高动量组 (2天涨幅≥30%, {len(high_momentum)} 个):")
        print(f"   30天内涨≥20%: {(high_momentum['post_max30d'] >= 0.20).mean()*100:.1f}%")
        print(f"   盈亏比: {abs(high_momentum['post_max30d'].median()) / abs(high_momentum['post_min30d'].median()):.2f}")

    # 按量比分层
    high_vol = df[df['vol_ratio'] >= 2.0]
    if len(high_vol) > 0:
        print(f"\n📌 高量组 (量比≥2.0, {len(high_vol)} 个):")
        print(f"   30天内涨≥20%: {(high_vol['post_max30d'] >= 0.20).mean()*100:.1f}%")
        print(f"   盈亏比: {abs(high_vol['post_max30d'].median()) / abs(high_vol['post_min30d'].median()):.2f}")

    # 按回撤分层
    deep_dd = df[df['drawdown'] >= 0.90]
    if len(deep_dd) > 0:
        print(f"\n📌 深跌组 (回撤≥90%, {len(deep_dd)} 个):")
        print(f"   30天内涨≥20%: {(deep_dd['post_max30d'] >= 0.20).mean()*100:.1f}%")
        print(f"   盈亏比: {abs(deep_dd['post_max30d'].median()) / abs(deep_dd['post_min30d'].median()):.2f}")

    # ===== 止损分析 =====
    print(f"\n{'='*100}")
    print("  【止损模拟】止损 = 突破日低点 - 5%")
    print(f"{'='*100}")
    stopped = df['post_min5d'] <= -0.05
    print(f"   5天内触发5%止损比例: {stopped.mean()*100:.1f}%")
    if (~stopped).sum() > 0:
        ns = df[~stopped]
        print(f"   未止损案例 30天涨≥20%: {(ns['post_max30d'] >= 0.20).mean()*100:.1f}%")
    if stopped.sum() > 0:
        s = df[stopped]
        print(f"   止损案例 30天涨≥20%:   {(s['post_max30d'] >= 0.20).mean()*100:.1f}%")

    print(f"\n{'='*100}\n")


def print_current_events(events: List[BottomBounceEvent]):
    if not events:
        print("\n📭 当前无符合条件的寻底后启动事件\n")
        return

    print(f"\n{'='*110}")
    print(f"  🚀 当前市场寻底后启动信号 ({len(events)} 个)")
    print(f"{'='*110}")
    print(f"\n{'日期':<12} {'币种':<16} {'市场':<6} {'价格':<12} {'MA6':<12} {'MA24':<12} {'2天涨幅':<10} {'寻底<MA6':<10} {'回撤':<8} {'量比':<8}")
    print("-" * 110)

    for e in sorted(events, key=lambda x: x.two_day_return, reverse=True):
        print(
            f"{e.date.strftime('%Y-%m-%d'):<12} {e.symbol:<16} {e.market_type:<6} "
            f"{e.price:<12.6f} {e.ma6:<12.6f} {e.ma24:<12.6f} "
            f"{e.two_day_return*100:<10.1f}% {e.pre_launch_below_ma6_ratio*100:<10.1f}% "
            f"{e.drawdown*100:<8.1f}% {e.vol_ratio:<8.2f}"
        )
    print(f"\n{'='*110}\n")


# ==========================================
# CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='超跌寻底后启动事件监控器')
    parser.add_argument('--backtest', action='store_true', help='历史回测模式')
    parser.add_argument('--scan-now', action='store_true', help='实时监控模式（只扫最新一天）')
    parser.add_argument('--market', type=str, default='all', choices=['spot', 'swap', 'all'])
    parser.add_argument('--min-return', type=float, default=0.20, help='最近2天最小累计涨幅（默认20%）')
    parser.add_argument('--lookback', type=int, default=10, help='寻底期天数（默认10天）')
    parser.add_argument('--min-below-ma6', type=float, default=0.50, help='寻底期收盘价<MA6的最小占比（默认50%）')
    args = parser.parse_args()

    if args.market == 'all':
        market_types = ['spot', 'swap']
    else:
        market_types = [args.market]

    kwargs = {
        'min_return': args.min_return,
        'lookback_days': args.lookback,
        'min_below_ma6_ratio': args.min_below_ma6,
    }

    if args.backtest:
        logger.info(f"启动历史回测: market={args.market}, min_return={args.min_return*100:.0f}%, lookback={args.lookback}天")
        events = run_backtest(market_types, **kwargs)
        print_report(events)
    elif args.scan_now:
        logger.info(f"启动实时监控: market={args.market}")
        events = scan_now(market_types, **kwargs)
        print_current_events(events)
    else:
        # 默认：先回测，再扫描当前
        logger.info(f"启动完整分析: market={args.market}")
        events = run_backtest(market_types, **kwargs)
        print_report(events)
        current = scan_now(market_types, **kwargs)
        print_current_events(current)


if __name__ == '__main__':
    main()
