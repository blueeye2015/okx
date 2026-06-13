#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
合约持仓量(OI)暴增与后续暴跌关系研究

目标：
1. 定义 OI 暴增事件（单日 OI 增长率超过阈值）
2. 统计 OI 暴增后未来 N 天出现暴跌的概率
3. 寻找最优阈值组合

用法：
  python oi_spike_study.py --market swap --days 500 --oi-thresholds 0.15,0.20,0.25,0.30
"""

import os
import sys
import time
import logging
import argparse
import requests
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
logger = logging.getLogger('oi_spike_study')

PROXY = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
BASE_FAPI = 'https://fapi.binance.com'


@dataclass
class OISpikeEvent:
    symbol: str
    date: datetime
    price: float
    oi: float
    oi_change_pct: float
    price_change_pct: float
    volume: float
    volume_ratio: float
    long_short_ratio_accounts: Optional[float]
    long_short_ratio_positions: Optional[float]
    post_max_drop_3d: float
    post_max_drop_5d: float
    post_max_drop_7d: float
    post_max_drop_10d: float
    post_close_return_3d: float
    post_close_return_5d: float
    post_close_return_7d: float
    post_close_return_10d: float
    post_max_rise_3d: float
    post_max_rise_5d: float


def binance_request(base_url: str, endpoint: str, params: Dict = None) -> List[Dict]:
    url = f"{base_url}{endpoint}"
    for attempt in range(5):
        try:
            time.sleep(0.3)  # 降低请求频率
            resp = requests.get(url, params=params, proxies=PROXY, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"请求失败 ({attempt+1}/5): {url} {e}，等待 {wait}s")
            time.sleep(wait)
    raise Exception(f"请求最终失败: {url}")


def get_swap_symbols() -> List[str]:
    """从本地数据库获取合约交易对列表"""
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT symbol FROM binance_daily_klines
                WHERE market_type = 'swap' ORDER BY symbol
            """)
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def fetch_klines_local(symbol: str, market_type: str = 'swap', limit: int = 500) -> pd.DataFrame:
    """从本地数据库获取K线数据"""
    conn = psycopg2.connect(DB_DSN)
    try:
        df = pd.read_sql("""
            SELECT open_time, open, high, low, close, volume, quote_volume
            FROM binance_daily_klines
            WHERE symbol = %s AND market_type = %s
            ORDER BY open_time DESC
            LIMIT %s
        """, conn, params=(symbol, market_type, limit))
        if df.empty:
            return pd.DataFrame()
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.set_index('open_time')
        df = df.sort_index()
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
        return df
    finally:
        conn.close()


def fetch_open_interest_hist(symbol: str, limit: int = 500) -> pd.DataFrame:
    data = binance_request(
        BASE_FAPI,
        '/futures/data/openInterestHist',
        {'symbol': symbol, 'period': '1d', 'limit': limit}
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
    return df


def fetch_ratio_data(endpoint: str, symbol: str, limit: int = 500) -> pd.DataFrame:
    try:
        data = binance_request(
            BASE_FAPI,
            f'/futures/data/{endpoint}',
            {'symbol': symbol, 'period': '1d', 'limit': limit}
        )
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        return df
    except Exception as e:
        logger.debug(f"[{symbol}] {endpoint} 获取失败: {e}")
        return pd.DataFrame()


def analyze_symbol(symbol: str, min_history: int = 30) -> List[OISpikeEvent]:
    """分析单个币的所有OI暴增事件"""
    # 获取K线和OI数据
    klines = fetch_klines_local(symbol, 'swap', 500)
    oi = fetch_open_interest_hist(symbol, 500)
    if klines.empty or oi.empty or len(klines) < min_history or len(oi) < min_history:
        return []

    # 合并数据
    df = klines[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].copy()
    df['oi'] = oi['sumOpenInterest']
    df = df.dropna(subset=['oi', 'close'])
    if len(df) < min_history:
        return []

    # 计算指标
    df['oi_change_pct'] = df['oi'].pct_change()
    df['price_change_pct'] = df['close'].pct_change()
    df['volume_ma20'] = df['quote_volume'].rolling(20, min_periods=10).mean()
    df['volume_ratio'] = df['quote_volume'] / df['volume_ma20']

    # 获取多空比数据（可选）
    acc_ratio = fetch_ratio_data('topLongShortAccountRatio', symbol, 500)
    pos_ratio = fetch_ratio_data('topLongShortPositionRatio', symbol, 500)
    if not acc_ratio.empty:
        df['ls_ratio_accounts'] = acc_ratio['longShortRatio'].astype(float)
    else:
        df['ls_ratio_accounts'] = np.nan
    if not pos_ratio.empty:
        df['ls_ratio_positions'] = pos_ratio['longShortRatio'].astype(float)
    else:
        df['ls_ratio_positions'] = np.nan

    events = []
    for i in range(1, len(df) - 10):
        row = df.iloc[i]
        if pd.isna(row['oi_change_pct']) or pd.isna(row['close']):
            continue

        # 未来10天数据
        future = df.iloc[i+1:i+11]

        events.append(OISpikeEvent(
            symbol=symbol,
            date=df.index[i],
            price=float(row['close']),
            oi=float(row['oi']),
            oi_change_pct=float(row['oi_change_pct']),
            price_change_pct=float(row['price_change_pct']),
            volume=float(row['quote_volume']),
            volume_ratio=float(row['volume_ratio']) if not pd.isna(row['volume_ratio']) else 1.0,
            long_short_ratio_accounts=float(row['ls_ratio_accounts']) if not pd.isna(row['ls_ratio_accounts']) else None,
            long_short_ratio_positions=float(row['ls_ratio_positions']) if not pd.isna(row['ls_ratio_positions']) else None,
            post_max_drop_3d=((future['low'].iloc[:3].min() - row['close']) / row['close']),
            post_max_drop_5d=((future['low'].iloc[:5].min() - row['close']) / row['close']),
            post_max_drop_7d=((future['low'].iloc[:7].min() - row['close']) / row['close']),
            post_max_drop_10d=((future['low'].iloc[:10].min() - row['close']) / row['close']),
            post_close_return_3d=((future['close'].iloc[2] - row['close']) / row['close']),
            post_close_return_5d=((future['close'].iloc[4] - row['close']) / row['close']),
            post_close_return_7d=((future['close'].iloc[6] - row['close']) / row['close']),
            post_close_return_10d=((future['close'].iloc[9] - row['close']) / row['close']),
            post_max_rise_3d=((future['high'].iloc[:3].max() - row['close']) / row['close']),
            post_max_rise_5d=((future['high'].iloc[:5].max() - row['close']) / row['close']),
        ))

    return events


def run_study(symbols: List[str]) -> List[OISpikeEvent]:
    all_events = []
    for i, sym in enumerate(symbols, 1):
        try:
            events = analyze_symbol(sym)
            all_events.extend(events)
            if i % 10 == 0:
                logger.info(f"进度: {i}/{len(symbols)}，已收集 {len(all_events)} 个事件")
            time.sleep(0.5)  # 每个币之间休息
        except Exception as e:
            logger.warning(f"[{sym}] 分析失败: {e}")
    return all_events


def print_report(events: List[OISpikeEvent], oi_thresholds: List[float], drop_thresholds: List[float]):
    if not events:
        print("\n❌ 无数据\n")
        return

    df = pd.DataFrame([{
        'symbol': e.symbol,
        'date': e.date,
        'price': e.price,
        'oi': e.oi,
        'oi_change_pct': e.oi_change_pct,
        'price_change_pct': e.price_change_pct,
        'volume_ratio': e.volume_ratio,
        'ls_ratio_accounts': e.long_short_ratio_accounts,
        'ls_ratio_positions': e.long_short_ratio_positions,
        'post_max_drop_3d': e.post_max_drop_3d,
        'post_max_drop_5d': e.post_max_drop_5d,
        'post_max_drop_7d': e.post_max_drop_7d,
        'post_max_drop_10d': e.post_max_drop_10d,
        'post_close_return_3d': e.post_close_return_3d,
        'post_close_return_5d': e.post_close_return_5d,
        'post_close_return_7d': e.post_close_return_7d,
        'post_close_return_10d': e.post_close_return_10d,
        'post_max_rise_3d': e.post_max_rise_3d,
        'post_max_rise_5d': e.post_max_rise_5d,
    } for e in events])

    print(f"\n{'='*100}")
    print(f"  OI 暴增与后续暴跌关系研究  (总样本: {len(df)} 个币日事件)")
    print(f"{'='*100}")

    # 基础分布
    print(f"\n📊 OI 日变化分布")
    print(f"   均值: {df['oi_change_pct'].mean()*100:.2f}%")
    print(f"   中位数: {df['oi_change_pct'].median()*100:.2f}%")
    print(f"   90分位: {df['oi_change_pct'].quantile(0.90)*100:.2f}%")
    print(f"   95分位: {df['oi_change_pct'].quantile(0.95)*100:.2f}%")
    print(f"   99分位: {df['oi_change_pct'].quantile(0.99)*100:.2f}%")

    # 不同 OI 阈值下的暴跌概率
    print(f"\n{'='*100}")
    print("  【核心结果】OI 暴增后未来 N 天最大跌幅 ≥ X 的概率")
    print(f"{'='*100}")

    print(f"\n{'OI阈值':<10} {'样本数':<8} {'3天跌≥10%':<12} {'5天跌≥15%':<12} {'7天跌≥20%':<12} {'10天跌≥25%':<12}")
    print("-" * 80)

    for oi_th in oi_thresholds:
        subset = df[df['oi_change_pct'] >= oi_th]
        if len(subset) < 5:
            continue
        d3 = (subset['post_max_drop_3d'] <= -0.10).mean()
        d5 = (subset['post_max_drop_5d'] <= -0.15).mean()
        d7 = (subset['post_max_drop_7d'] <= -0.20).mean()
        d10 = (subset['post_max_drop_10d'] <= -0.25).mean()
        print(f"{oi_th*100:>6.0f}%   {len(subset):<8} {d3*100:>8.1f}%      {d5*100:>8.1f}%      {d7*100:>8.1f}%      {d10*100:>8.1f}%")

    # 基准对比：所有币日的暴跌概率
    print(f"\n{'基准(全样本)':<10} {len(df):<8} {(df['post_max_drop_3d']<=-0.10).mean()*100:>8.1f}%      {(df['post_max_drop_5d']<=-0.15).mean()*100:>8.1f}%      {(df['post_max_drop_7d']<=-0.20).mean()*100:>8.1f}%      {(df['post_max_drop_10d']<=-0.25).mean()*100:>8.1f}%")

    # 加入价格涨幅条件：OI暴增 + 当天大涨
    print(f"\n{'='*100}")
    print("  【增强条件】OI 暴增 + 当天大涨（类似于 HMSTR 6/11 情况）")
    print(f"{'='*100}")

    print(f"\n{'OI阈值':<10} {'涨幅条件':<12} {'样本数':<8} {'3天跌≥10%':<12} {'5天跌≥15%':<12} {'7天跌≥20%':<12} {'10天跌≥25%':<12}")
    print("-" * 90)

    for oi_th in oi_thresholds:
        for price_th in [0.05, 0.10, 0.15, 0.20]:
            subset = df[(df['oi_change_pct'] >= oi_th) & (df['price_change_pct'] >= price_th)]
            if len(subset) < 5:
                continue
            d3 = (subset['post_max_drop_3d'] <= -0.10).mean()
            d5 = (subset['post_max_drop_5d'] <= -0.15).mean()
            d7 = (subset['post_max_drop_7d'] <= -0.20).mean()
            d10 = (subset['post_max_drop_10d'] <= -0.25).mean()
            print(f"{oi_th*100:>6.0f}%   当天+{price_th*100:.0f}%       {len(subset):<8} {d3*100:>8.1f}%      {d5*100:>8.1f}%      {d7*100:>8.1f}%      {d10*100:>8.1f}%")

    # 加入多空比条件
    print(f"\n{'='*100}")
    print("  【多空比条件】OI 暴增 + 全球多空比 < 1.0（散户偏空，大户拉盘）")
    print(f"{'='*100}")

    print(f"\n{'OI阈值':<10} {'样本数':<8} {'3天跌≥10%':<12} {'5天跌≥15%':<12} {'7天跌≥20%':<12} {'10天跌≥25%':<12}")
    print("-" * 80)

    for oi_th in oi_thresholds:
        subset = df[(df['oi_change_pct'] >= oi_th) & (df['ls_ratio_accounts'].notna()) & (df['ls_ratio_accounts'] < 1.5)]
        if len(subset) < 5:
            continue
        d3 = (subset['post_max_drop_3d'] <= -0.10).mean()
        d5 = (subset['post_max_drop_5d'] <= -0.15).mean()
        d7 = (subset['post_max_drop_7d'] <= -0.20).mean()
        d10 = (subset['post_max_drop_10d'] <= -0.25).mean()
        print(f"{oi_th*100:>6.0f}%   {len(subset):<8} {d3*100:>8.1f}%      {d5*100:>8.1f}%      {d7*100:>8.1f}%      {d10*100:>8.1f}%")

    # 最佳阈值分析：综合胜率*盈亏比
    print(f"\n{'='*100}")
    print("  【最佳阈值探索】OI 暴增后 5 天跌≥15% 的胜率与盈亏比")
    print(f"{'='*100}")

    best_score = 0
    best_params = None
    for oi_th in [x/100 for x in range(5, 51, 5)]:
        subset = df[df['oi_change_pct'] >= oi_th]
        if len(subset) < 10:
            continue
        win_rate = (subset['post_max_drop_5d'] <= -0.15).mean()
        avg_drop = subset['post_max_drop_5d'].median()
        avg_rise = subset['post_max_rise_5d'].median()
        rr = abs(avg_drop) / abs(avg_rise) if avg_rise != 0 else 0
        score = win_rate * rr
        if score > best_score:
            best_score = score
            best_params = (oi_th, len(subset), win_rate, avg_drop, avg_rise, rr)
        print(f"  OI≥{oi_th*100:.0f}%: 样本{len(subset)}, 胜率{win_rate*100:.1f}%, 中位跌幅{avg_drop*100:.1f}%, 中位涨幅{avg_rise*100:.1f}%, RR={rr:.2f}, 综合分={score:.2f}")

    if best_params:
        print(f"\n🏆 最佳阈值: OI 日增 ≥ {best_params[0]*100:.0f}%")
        print(f"   样本数: {best_params[1]}, 5天跌≥15%胜率: {best_params[2]*100:.1f}%")
        print(f"   中位跌幅: {best_params[3]*100:.1f}%, 中位涨幅: {best_params[4]*100:.1f}%, 盈亏比: {best_params[5]:.2f}")

    print(f"\n{'='*100}\n")


if __name__ == '__main__':
    import time
    parser = argparse.ArgumentParser(description='OI 暴增与后续暴跌关系研究')
    parser.add_argument('--market', type=str, default='swap', choices=['swap'])
    parser.add_argument('--symbols', type=str, help='指定币种，逗号分隔，默认全市场')
    parser.add_argument('--oi-thresholds', type=str, default='0.10,0.15,0.20,0.25,0.30',
                        help='OI暴增阈值，逗号分隔')
    parser.add_argument('--drop-thresholds', type=str, default='0.10,0.15,0.20,0.25',
                        help='暴跌阈值，逗号分隔')
    args = parser.parse_args()

    oi_thresholds = [float(x) for x in args.oi_thresholds.split(',')]
    drop_thresholds = [float(x) for x in args.drop_thresholds.split(',')]

    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols = get_swap_symbols()

    logger.info(f"开始研究 {len(symbols)} 个合约币...")
    events = run_study(symbols)
    print_report(events, oi_thresholds, drop_thresholds)
