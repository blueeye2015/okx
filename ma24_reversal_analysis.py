#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
MA24 历史穿越分析
目的：对已满足"超跌"条件的币（跌80%+弱势250天），分析它们在过去250天内与MA24的关系：
1. 曾经站上MA24后又回落至MA24之下（假突破）
2. 曾经回落至MA24之下后又震荡站上来（真反转）
3. 当前状态分类
"""

import os
import sys
import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv('/data/okx/.env')

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


def fetch_daily_klines(symbol: str, market_type: str, min_date: datetime) -> pd.DataFrame:
    conn = get_db_conn()
    query = """
        SELECT open_time, open, high, low, close, volume
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


def calculate_weekly_ma(df_daily: pd.DataFrame, weekly_period: int = 60) -> pd.Series:
    weekly = df_daily['close'].resample('W-FRI').last().dropna()
    weekly_ma = weekly.rolling(window=weekly_period, min_periods=weekly_period).mean()
    weekly_ma_daily = weekly_ma.reindex(df_daily.index, method='ffill')
    return weekly_ma_daily


@dataclass
class SymbolAnalysis:
    symbol: str
    market_type: str
    current_price: float
    drawdown: float
    below_weekly_ma_days: int
    current_above_ma24: bool
    # 穿越统计（基于最近250天）
    cross_events: List[Dict]  # 每次穿越事件
    ever_above_then_below: bool  # 曾经站上后又跌破
    ever_below_then_above: bool  # 曾经跌破后又站上
    days_above_ma24_250d: int    # 250天内站上天数
    days_below_ma24_250d: int    # 250天内跌破天数
    first_cross_to_above_date: Optional[datetime]  # 首次站上MA24日期
    last_cross_to_below_date: Optional[datetime]   # 最近一次跌破MA24日期
    last_cross_to_above_date: Optional[datetime]   # 最近一次站上MA24日期


def analyze_symbol(symbol: str, market_type: str, cfg: dict) -> Optional[SymbolAnalysis]:
    """
    分析单个交易对的超跌状态 + MA24历史穿越行为
    """
    buffer_days = max(cfg['long_bear_days'], cfg['weekly_ma_period'] * 7, cfg['medium_ma'] * 2)
    min_date = datetime.now() - timedelta(days=cfg['lookback_days'] + buffer_days)

    df = fetch_daily_klines(symbol, market_type, min_date)
    if df.empty or len(df) < cfg['min_history_days']:
        return None

    df['ma6'] = df['close'].rolling(window=cfg['short_ma'], min_periods=cfg['short_ma']).mean()
    df['ma24'] = df['close'].rolling(window=cfg['medium_ma'], min_periods=cfg['medium_ma']).mean()
    df['weekly_ma60'] = calculate_weekly_ma(df, cfg['weekly_ma_period'])

    recent_df = df.tail(cfg['lookback_days'] + cfg['medium_ma'])
    if len(recent_df) < cfg['lookback_days']:
        return None

    historical_high = recent_df['high'].max()

    # 检查长期弱势
    bear_window = recent_df.tail(cfg['long_bear_days']).dropna(subset=['weekly_ma60'])
    if len(bear_window) < cfg['long_bear_days'] * 0.8:
        return None
    below_ma_count = (bear_window['close'] < bear_window['weekly_ma60']).sum()
    if below_ma_count / len(bear_window) < 0.80:
        return None

    # 检查深度回撤
    current_price = recent_df['close'].iloc[-1]
    drawdown = (historical_high - current_price) / historical_high
    if drawdown < cfg['drawdown_threshold']:
        return None

    # ========== MA24 历史穿越分析 ==========
    # 用最近250天（不含最新一天）来看历史穿越行为
    analysis_window = recent_df.tail(cfg['long_bear_days']).copy()
    analysis_window = analysis_window.dropna(subset=['ma24'])
    if len(analysis_window) < 50:
        return None

    analysis_window['above_ma24'] = analysis_window['close'] > analysis_window['ma24']

    # 当前状态
    current_above = analysis_window['above_ma24'].iloc[-1]

    # 找状态转换点
    analysis_window['state_change'] = analysis_window['above_ma24'].astype(int).diff()
    # diff: 0->1 表示 below->above (站上), 0 表示没变, -1 表示 above->below (跌破)
    cross_events = []
    ever_above_then_below = False
    ever_below_then_above = False
    first_above_date = None
    last_below_date = None
    last_above_date = None

    # 遍历状态转换
    for i in range(1, len(analysis_window)):
        prev_state = analysis_window['above_ma24'].iloc[i-1]
        curr_state = analysis_window['above_ma24'].iloc[i]
        curr_date = analysis_window.index[i]

        if not prev_state and curr_state:
            # 站上MA24
            cross_events.append({'date': curr_date, 'type': 'up', 'price': analysis_window['close'].iloc[i]})
            ever_below_then_above = True
            if first_above_date is None:
                first_above_date = curr_date
            last_above_date = curr_date
        elif prev_state and not curr_state:
            # 跌破MA24
            cross_events.append({'date': curr_date, 'type': 'down', 'price': analysis_window['close'].iloc[i]})
            ever_above_then_below = True
            last_below_date = curr_date

    days_above = int(analysis_window['above_ma24'].sum())
    days_below = int((~analysis_window['above_ma24']).sum())

    return SymbolAnalysis(
        symbol=symbol,
        market_type=market_type,
        current_price=current_price,
        drawdown=drawdown,
        below_weekly_ma_days=below_ma_count,
        current_above_ma24=current_above,
        cross_events=cross_events,
        ever_above_then_below=ever_above_then_below,
        ever_below_then_above=ever_below_then_above,
        days_above_ma24_250d=days_above,
        days_below_ma24_250d=days_below,
        first_cross_to_above_date=first_above_date,
        last_cross_to_below_date=last_below_date,
        last_cross_to_above_date=last_above_date,
    )


def run_analysis(market_type: str = 'spot', drawdown: float = 0.80, weak_days: int = 250):
    cfg = {
        'market_type': market_type,
        'drawdown_threshold': drawdown,
        'long_bear_days': weak_days,
        'weekly_ma_period': 60,
        'short_ma': 6,
        'medium_ma': 24,
        'lookback_days': 500,
        'min_history_days': 400,
    }

    symbols = get_symbols(market_type)
    logger.info(f"开始分析 {market_type} 市场，共 {len(symbols)} 个交易对...")
    logger.info(f"筛选条件: 回撤>={drawdown*100:.0f}%, 低于60周均线>={weak_days}天")

    results: List[SymbolAnalysis] = []
    for i, symbol in enumerate(symbols, 1):
        try:
            res = analyze_symbol(symbol, market_type, cfg)
            if res:
                results.append(res)
            if i % 50 == 0:
                logger.info(f"进度: {i}/{len(symbols)}，已发现 {len(results)} 个满足超跌条件的币")
        except Exception as e:
            logger.warning(f"[{symbol}] 分析异常: {e}")

    logger.info(f"扫描完成! 共检查 {len(symbols)} 个交易对，发现 {len(results)} 个满足超跌条件的币")
    return results


def print_classification(results: List[SymbolAnalysis], market_type: str):
    if not results:
        print("\n❌ 未发现满足超跌条件的交易对")
        return

    # 分类
    # A. 当前在MA24之上
    above_now = [r for r in results if r.current_above_ma24]
    # B. 当前在MA24之下
    below_now = [r for r in results if not r.current_above_ma24]

    # 在A中细分：曾经跌破后又站上来 vs 一直在上面
    a_ever_below_then_above = [r for r in above_now if r.ever_below_then_above]
    a_always_above = [r for r in above_now if not r.ever_below_then_above]

    # 在A中再细分：曾经站上后又跌破过（震荡），然后现在又上来
    a_ever_above_then_below = [r for r in above_now if r.ever_above_then_below]
    a_never_above_then_below = [r for r in above_now if not r.ever_above_then_below]

    # 在B中细分：曾经站上过MA24然后又跌破（假突破） vs 从未站上过
    b_ever_above_then_below = [r for r in below_now if r.ever_above_then_below]
    b_never_above = [r for r in below_now if not r.ever_above_then_below]

    # 用户特别关心的两类：
    # 1. "站上MA24后又回落至MA24之下" → 当前below且曾经above_then_below
    # 2. "曾经回落后又震荡上来" → 当前above且曾经below_then_above（即曾经跌破又站上）

    print(f"\n{'='*90}")
    print(f"MA24 历史穿越分析 ({market_type}) — 共 {len(results)} 个满足超跌条件的币")
    print(f"{'='*90}")

    print(f"\n📊 总体分布（按当前与MA24关系）：")
    print(f"   当前站上MA24: {len(above_now)} 个")
    print(f"   当前跌破MA24: {len(below_now)} 个")

    print(f"\n{'='*90}")
    print(f"【第一类】站上MA24后又回落至MA24之下（假突破/反弹失败）— {len(b_ever_above_then_below)} 个")
    print(f"{'='*90}")
    print(f"{'Symbol':<12} {'Current':>10} {'Drawdown':>10} {'LastAbove':>12} {'LastDown':>12} {'DaysBelow':>10}")
    print("-" * 90)
    for r in sorted(b_ever_above_then_below, key=lambda x: x.drawdown, reverse=True):
        la = r.last_cross_to_above_date.strftime('%Y-%m-%d') if r.last_cross_to_above_date else 'N/A'
        lb = r.last_cross_to_below_date.strftime('%Y-%m-%d') if r.last_cross_to_below_date else 'N/A'
        print(f"{r.symbol:<12} {r.current_price:>10.4f} {r.drawdown*100:>9.1f}% {la:>12} {lb:>12} {r.days_below_ma24_250d:>10}")

    print(f"\n{'='*90}")
    print(f"【第二类】曾经跌破MA24后又震荡站上来（震荡筑底/真反转）— {len(a_ever_below_then_above)} 个")
    print(f"{'='*90}")
    print(f"{'Symbol':<12} {'Current':>10} {'Drawdown':>10} {'FirstUp':>12} {'LastDown':>12} {'DaysAbove':>10}")
    print("-" * 90)
    for r in sorted(a_ever_below_then_above, key=lambda x: x.drawdown, reverse=True):
        fa = r.first_cross_to_above_date.strftime('%Y-%m-%d') if r.first_cross_to_above_date else 'N/A'
        lb = r.last_cross_to_below_date.strftime('%Y-%m-%d') if r.last_cross_to_below_date else 'N/A'
        print(f"{r.symbol:<12} {r.current_price:>10.4f} {r.drawdown*100:>9.1f}% {fa:>12} {lb:>12} {r.days_above_ma24_250d:>10}")

    # 子分类：第二类中，曾经先突破再跌破再站上（多次震荡）
    a_double_cross = [r for r in a_ever_below_then_above if r.ever_above_then_below]
    print(f"\n   其中：经历'站上→跌破→再站上'多次震荡的: {len(a_double_cross)} 个")
    if a_double_cross:
        print(f"   {', '.join([r.symbol for r in a_double_cross])}")

    print(f"\n{'='*90}")
    print(f"【第三类】当前在MA24之上且从未跌破过（持续站上）— {len(a_always_above)} 个")
    print(f"{'='*90}")
    if a_always_above:
        for r in a_always_above:
            print(f"   {r.symbol}: 价格 {r.current_price:.4f}, 回撤 {r.drawdown*100:.1f}%, 250天内站上 {r.days_above_ma24_250d} 天")
    else:
        print("   无")

    print(f"\n{'='*90}")
    print(f"【第四类】当前在MA24之下且从未站上过（持续弱势）— {len(b_never_above)} 个")
    print(f"{'='*90}")
    if b_never_above:
        print(f"   共 {len(b_never_above)} 个，前5个:")
        for r in sorted(b_never_above, key=lambda x: x.drawdown, reverse=True)[:5]:
            print(f"   {r.symbol}: 价格 {r.current_price:.4f}, 回撤 {r.drawdown*100:.1f}%")
    else:
        print("   无")

    print(f"\n{'='*90}")
    print(f"📈 汇总统计")
    print(f"{'='*90}")
    print(f"  满足超跌条件的币总数: {len(results)}")
    print(f"  ├─ 当前站上MA24: {len(above_now)}")
    print(f"  │   ├─ 曾经跌破后又站上（震荡上来）: {len(a_ever_below_then_above)}")
    print(f"  │   │   └─ 其中经历过'站上→跌破→再站上': {len(a_double_cross)}")
    print(f"  │   └─ 一直站在MA24之上（从未跌破）: {len(a_always_above)}")
    print(f"  └─ 当前跌破MA24: {len(below_now)}")
    print(f"      ├─ 曾经站上后又跌破（假突破）: {len(b_ever_above_then_below)}")
    print(f"      └─ 从未站上过MA24: {len(b_never_above)}")
    print(f"{'='*90}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MA24历史穿越分析')
    parser.add_argument('--market', choices=['spot', 'swap'], default='spot')
    parser.add_argument('--drawdown', type=float, default=0.80)
    parser.add_argument('--weak-days', type=int, default=250)
    args = parser.parse_args()

    results = run_analysis(args.market, args.drawdown, args.weak_days)
    print_classification(results, args.market)
