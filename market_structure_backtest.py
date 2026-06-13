#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
市场结构转暖信号历史回测
验证: oversold_ratio 连续3天下降 + above_ma24_ratio 连续3天上升
      出现后，市场后续 5/10/30 天的平均表现
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


def compute_symbol_daily_metrics(symbol: str, market_type: str) -> pd.DataFrame:
    """计算单个币的每日指标，返回 (date, is_oversold, above_ma24, close, return_1d)"""
    df = fetch_daily_klines(symbol, market_type)
    if df.empty or len(df) < 30:
        return pd.DataFrame()

    df['ma24'] = df['close'].rolling(window=24, min_periods=24).mean()
    # 用上市以来最高点计算回撤（对新币友好）
    df['rolling_high'] = df['high'].expanding().max()
    df['drawdown'] = (df['rolling_high'] - df['close']) / df['rolling_high']
    df['above_ma24'] = df['close'] > df['ma24']
    df['return_1d'] = df['close'].pct_change()

    # oversold: 简化版，只用 drawdown >= 0.80（不依赖 weekly_ma60，对新币友好）
    df['is_oversold'] = df['drawdown'] >= 0.80

    return df[['close', 'return_1d', 'is_oversold', 'above_ma24']].copy()


def build_market_daily_metrics(market_type: str = 'spot') -> pd.DataFrame:
    """构建市场层面的每日指标"""
    symbols = get_symbols(market_type)
    logger.info(f"开始构建 {market_type} 市场每日指标，共 {len(symbols)} 个币...")

    all_metrics = []
    for i, symbol in enumerate(symbols, 1):
        try:
            metrics = compute_symbol_daily_metrics(symbol, market_type)
            if metrics.empty:
                continue
            metrics['symbol'] = symbol
            all_metrics.append(metrics)
            if i % 100 == 0:
                logger.info(f"进度: {i}/{len(symbols)}")
        except Exception as e:
            logger.debug(f"[{symbol}] 异常: {e}")

    if not all_metrics:
        logger.error("没有有效数据")
        return pd.DataFrame()

    logger.info(f"合并 {len(all_metrics)} 个币的数据...")
    big_df = pd.concat(all_metrics)

    # 按日期聚合
    daily = big_df.groupby(big_df.index).agg(
        total_symbols=('symbol', 'nunique'),
        oversold_count=('is_oversold', 'sum'),
        above_ma24_count=('above_ma24', 'sum'),
        avg_return_1d=('return_1d', 'mean'),
        median_return_1d=('return_1d', 'median'),
    )

    daily['oversold_ratio'] = daily['oversold_count'] / daily['total_symbols']
    daily['above_ma24_ratio'] = daily['above_ma24_count'] / daily['total_symbols']

    # 确保日期连续（填充缺失日期）
    daily = daily.asfreq('D')
    daily['oversold_ratio'] = daily['oversold_ratio'].ffill()
    daily['above_ma24_ratio'] = daily['above_ma24_ratio'].ffill()
    daily['total_symbols'] = daily['total_symbols'].ffill()

    logger.info(f"市场指标构建完成，共 {len(daily)} 个交易日")
    return daily


def identify_signals(daily: pd.DataFrame) -> pd.DataFrame:
    """识别市场结构转暖信号"""
    # 连续3天 oversold_ratio 下降
    daily['oversold_1d'] = daily['oversold_ratio'].diff(1)
    daily['oversold_2d'] = daily['oversold_ratio'].diff(2)
    daily['oversold_3d'] = daily['oversold_ratio'].diff(3)
    daily['oversold_decline_3d'] = (
        (daily['oversold_1d'] < 0) &
        (daily['oversold_2d'] < 0) &
        (daily['oversold_3d'] < 0)
    )

    # 连续3天 above_ma24_ratio 上升
    daily['above24_1d'] = daily['above_ma24_ratio'].diff(1)
    daily['above24_2d'] = daily['above_ma24_ratio'].diff(2)
    daily['above24_3d'] = daily['above_ma24_ratio'].diff(3)
    daily['above24_rise_3d'] = (
        (daily['above24_1d'] > 0) &
        (daily['above24_2d'] > 0) &
        (daily['above24_3d'] > 0)
    )

    # 信号定义
    daily['signal'] = daily['oversold_decline_3d'] & daily['above24_rise_3d']

    return daily


def analyze_signal_performance(daily: pd.DataFrame) -> Dict:
    """分析信号出现后的市场表现"""
    signal_days = daily[daily['signal'] == True].index
    logger.info(f"共识别到 {len(signal_days)} 个信号日")

    if len(signal_days) == 0:
        return {}

    results = {
        'signal_count': len(signal_days),
        'signal_dates': signal_days.strftime('%Y-%m-%d').tolist(),
    }

    # 计算信号日后 N 天的市场累计收益
    for days in [5, 10, 20, 30]:
        post_returns = []
        for sig_date in signal_days:
            try:
                start_idx = daily.index.get_loc(sig_date)
                end_idx = min(start_idx + days, len(daily) - 1)
                if end_idx <= start_idx:
                    continue
                # 市场累计收益 = 所有币平均日收益的几何累积
                daily_rets = daily['avg_return_1d'].iloc[start_idx:end_idx]
                cum_ret = (1 + daily_rets.fillna(0)).prod() - 1
                post_returns.append(cum_ret)
            except Exception:
                continue

        if post_returns:
            results[f'post_{days}d_mean'] = np.mean(post_returns)
            results[f'post_{days}d_median'] = np.median(post_returns)
            results[f'post_{days}d_positive_rate'] = sum(1 for r in post_returns if r > 0) / len(post_returns)
            results[f'post_{days}d_samples'] = len(post_returns)

    # 对比：无信号日的表现
    non_signal_days = daily[daily['signal'] == False].index
    for days in [5, 10, 20, 30]:
        post_returns = []
        sample_dates = np.random.choice(non_signal_days, min(500, len(non_signal_days)), replace=False)
        for sig_date in sample_dates:
            try:
                start_idx = daily.index.get_loc(sig_date)
                end_idx = min(start_idx + days, len(daily) - 1)
                if end_idx <= start_idx:
                    continue
                daily_rets = daily['avg_return_1d'].iloc[start_idx:end_idx]
                cum_ret = (1 + daily_rets.fillna(0)).prod() - 1
                post_returns.append(cum_ret)
            except Exception:
                continue

        if post_returns:
            results[f'non_signal_post_{days}d_mean'] = np.mean(post_returns)
            results[f'non_signal_post_{days}d_median'] = np.median(post_returns)
            results[f'non_signal_post_{days}d_positive_rate'] = sum(1 for r in post_returns if r > 0) / len(post_returns)

    return results


def print_report(results: Dict):
    if not results:
        print("❌ 无信号数据")
        return

    print(f"\n{'='*100}")
    print(f"  市场结构转暖信号回测报告")
    print(f"{'='*100}")
    print(f"\n信号定义: oversold_ratio 连续3天下降 + above_ma24_ratio 连续3天上升")
    print(f"信号出现次数: {results['signal_count']}")
    print(f"信号日期: {', '.join(results['signal_dates'][:10])}{'...' if results['signal_count'] > 10 else ''}")

    print(f"\n{'='*100}")
    print("  信号日后市场表现 vs 无信号日随机样本")
    print(f"{'='*100}")
    print(f"\n{'周期':<10} {'信号日-平均':>12} {'信号日-中位数':>12} {'信号日-正收益比例':>16} {'无信号日-平均':>12} {'无信号日-正收益':>12}")
    print("-" * 100)

    for days in [5, 10, 20, 30]:
        sig_mean = results.get(f'post_{days}d_mean', 0)
        sig_med = results.get(f'post_{days}d_median', 0)
        sig_pos = results.get(f'post_{days}d_positive_rate', 0)
        non_mean = results.get(f'non_signal_post_{days}d_mean', 0)
        non_pos = results.get(f'non_signal_post_{days}d_positive_rate', 0)
        samples = results.get(f'post_{days}d_samples', 0)

        print(f"{days}天      {sig_mean*100:>11.2f}% {sig_med*100:>11.2f}% {sig_pos*100:>15.1f}% {non_mean*100:>11.2f}% {non_pos*100:>11.1f}%  (样本: {samples})")

    print(f"\n{'='*100}")
    print("  回测完成")
    print(f"{'='*100}")


def main():
    daily = build_market_daily_metrics('spot')
    if daily.empty:
        return

    daily = identify_signals(daily)
    results = analyze_signal_performance(daily)
    print_report(results)


if __name__ == '__main__':
    main()
