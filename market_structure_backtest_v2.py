#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
市场结构转暖信号回测 V2
核心问题：当市场出现转暖信号时，符合做多条件的币表现如何？

做多筛选条件：
  - 跌 ≥60%（drawdown >= 0.60）
  - 站上 MA24
  - pre_r20d < 0.15（突破前20天涨幅 < 15%）
  - 成交量 > MA20 × 1.2

对比组：无信号日，同样筛选条件
"""

import os
import sys
import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

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
        SELECT open_time, open, high, low, close, volume
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


def compute_symbol_metrics(symbol: str, market_type: str) -> pd.DataFrame:
    """计算单个币的每日完整指标"""
    df = fetch_daily_klines(symbol, market_type)
    if df.empty or len(df) < 30:
        return pd.DataFrame()

    df['ma24'] = df['close'].rolling(window=24, min_periods=24).mean()
    df['rolling_high'] = df['high'].expanding().max()
    df['drawdown'] = (df['rolling_high'] - df['close']) / df['rolling_high']
    df['above_ma24'] = df['close'] > df['ma24']
    df['volume_ma20'] = df['volume'].rolling(window=20, min_periods=15).mean()
    df['pre_return_20d'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    df['vol_ratio'] = df['volume'] / df['volume_ma20']

    # 未来 N 天收益（用于跟踪）
    for days in [5, 10, 20, 30]:
        df[f'future_return_{days}d'] = (df['close'].shift(-days) - df['close']) / df['close']

    df['symbol'] = symbol
    return df[['symbol', 'close', 'drawdown', 'above_ma24', 'pre_return_20d', 'vol_ratio',
               'future_return_5d', 'future_return_10d', 'future_return_20d', 'future_return_30d']].copy()


def build_market_and_signals(market_type: str = 'spot') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回:
      big_df: 所有币的每日指标 (multi-index: date, symbol)
      market_df: 市场层面指标 (index: date)
    """
    symbols = get_symbols(market_type)
    logger.info(f"开始构建 {market_type} 市场数据，共 {len(symbols)} 个币...")

    all_metrics = []
    for i, symbol in enumerate(symbols, 1):
        try:
            metrics = compute_symbol_metrics(symbol, market_type)
            if not metrics.empty:
                all_metrics.append(metrics)
            if i % 100 == 0:
                logger.info(f"进度: {i}/{len(symbols)}")
        except Exception as e:
            logger.debug(f"[{symbol}] 异常: {e}")

    if not all_metrics:
        logger.error("没有有效数据")
        return pd.DataFrame(), pd.DataFrame()

    logger.info(f"合并 {len(all_metrics)} 个币的数据...")
    big_df = pd.concat(all_metrics)
    big_df = big_df.reset_index().rename(columns={'open_time': 'date'})

    # 计算市场层面指标
    daily = big_df.groupby('date').agg(
        total_symbols=('symbol', 'nunique'),
        oversold_count=('drawdown', lambda x: (x >= 0.80).sum()),
        above_ma24_count=('above_ma24', 'sum'),
    )
    daily['oversold_ratio'] = daily['oversold_count'] / daily['total_symbols']
    daily['above_ma24_ratio'] = daily['above_ma24_count'] / daily['total_symbols']

    # 确保日期连续
    daily = daily.asfreq('D')
    daily['oversold_ratio'] = daily['oversold_ratio'].ffill()
    daily['above_ma24_ratio'] = daily['above_ma24_ratio'].ffill()

    # 识别信号
    daily['oversold_decline_3d'] = (
        (daily['oversold_ratio'].diff(1) < 0) &
        (daily['oversold_ratio'].diff(2) < 0) &
        (daily['oversold_ratio'].diff(3) < 0)
    )
    daily['above24_rise_3d'] = (
        (daily['above_ma24_ratio'].diff(1) > 0) &
        (daily['above_ma24_ratio'].diff(2) > 0) &
        (daily['above_ma24_ratio'].diff(3) > 0)
    )
    daily['signal'] = daily['oversold_decline_3d'] & daily['above24_rise_3d']

    logger.info(f"市场指标构建完成，共 {len(daily)} 个交易日，信号日 {daily['signal'].sum()} 个")
    return big_df, daily


def filter_long_candidates(df_day: pd.DataFrame) -> pd.DataFrame:
    """筛选符合做多条件的币"""
    return df_day[
        (df_day['drawdown'] >= 0.60) &
        (df_day['above_ma24'] == True) &
        (df_day['pre_return_20d'] < 0.15) &
        (df_day['vol_ratio'] > 1.2) &
        (df_day['drawdown'].notna()) &
        (df_day['pre_return_20d'].notna()) &
        (df_day['vol_ratio'].notna())
    ]


def analyze_signal_performance(big_df: pd.DataFrame, market_df: pd.DataFrame) -> Dict:
    """分析信号日符合做多条件的币的表现"""
    signal_dates = market_df[market_df['signal'] == True].index.tolist()
    non_signal_dates = market_df[market_df['signal'] == False].index.tolist()
    logger.info(f"信号日: {len(signal_dates)} 个")

    results = {}

    for days in [5, 10, 20, 30]:
        col = f'future_return_{days}d'

        # 信号日表现
        sig_returns = []
        for date in signal_dates:
            day_df = big_df[big_df['date'] == date]
            candidates = filter_long_candidates(day_df)
            if not candidates.empty:
                sig_returns.extend(candidates[col].dropna().tolist())

        # 无信号日表现（随机采样同样数量的日期）
        non_sig_returns = []
        sample_dates = np.random.choice(non_signal_dates, min(len(signal_dates), len(non_signal_dates)), replace=False)
        for date in sample_dates:
            day_df = big_df[big_df['date'] == date]
            candidates = filter_long_candidates(day_df)
            if not candidates.empty:
                non_sig_returns.extend(candidates[col].dropna().tolist())

        if sig_returns:
            results[f'signal_{days}d_mean'] = np.mean(sig_returns)
            results[f'signal_{days}d_median'] = np.median(sig_returns)
            results[f'signal_{days}d_positive_rate'] = sum(1 for r in sig_returns if r > 0) / len(sig_returns)
            results[f'signal_{days}d_samples'] = len(sig_returns)

        if non_sig_returns:
            results[f'non_signal_{days}d_mean'] = np.mean(non_sig_returns)
            results[f'non_signal_{days}d_median'] = np.median(non_sig_returns)
            results[f'non_signal_{days}d_positive_rate'] = sum(1 for r in non_sig_returns if r > 0) / len(non_sig_returns)
            results[f'non_signal_{days}d_samples'] = len(non_sig_returns)

    return results


def print_report(results: Dict):
    if not results:
        print("❌ 无数据")
        return

    print(f"\n{'='*100}")
    print(f"  市场结构转暖信号回测 V2")
    print(f"  做多筛选: 跌≥60% + 站上MA24 + pre_r20d<15% + 成交量>1.2x")
    print(f"{'='*100}")

    print(f"\n{'周期':<10} {'信号日-平均':>12} {'信号日-中位数':>14} {'信号日-正收益比例':>16} {'样本数':>8} {'无信号日-平均':>12} {'无信号日-中位数':>14} {'无信号日-正收益':>12}")
    print("-" * 110)

    for days in [5, 10, 20, 30]:
        sig_mean = results.get(f'signal_{days}d_mean', 0)
        sig_med = results.get(f'signal_{days}d_median', 0)
        sig_pos = results.get(f'signal_{days}d_positive_rate', 0)
        sig_n = results.get(f'signal_{days}d_samples', 0)
        non_mean = results.get(f'non_signal_{days}d_mean', 0)
        non_med = results.get(f'non_signal_{days}d_median', 0)
        non_pos = results.get(f'non_signal_{days}d_positive_rate', 0)

        print(f"{days}天      {sig_mean*100:>11.2f}% {sig_med*100:>13.2f}% {sig_pos*100:>15.1f}% {sig_n:>8} {non_mean*100:>11.2f}% {non_med*100:>13.2f}% {non_pos*100:>11.1f}%")

    print(f"\n{'='*100}")


def main():
    big_df, market_df = build_market_and_signals('spot')
    if big_df.empty:
        return
    results = analyze_signal_performance(big_df, market_df)
    print_report(results)


if __name__ == '__main__':
    main()
