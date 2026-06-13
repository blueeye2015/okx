#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
做空策略优化器
从历史MA24突破案例中，系统搜索能显著提升做空胜率的条件组合
"""

import os
import sys
import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
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
            cur.execute("SELECT DISTINCT symbol FROM binance_daily_klines WHERE market_type = %s ORDER BY symbol", (market_type,))
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def fetch_daily_klines(symbol: str, market_type: str, min_date: datetime) -> pd.DataFrame:
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


def calculate_weekly_ma(df_daily: pd.DataFrame, weekly_period: int = 60) -> pd.Series:
    weekly = df_daily['close'].resample('W-FRI').last().dropna()
    weekly_ma = weekly.rolling(window=weekly_period, min_periods=weekly_period).mean()
    weekly_ma_daily = weekly_ma.reindex(df_daily.index, method='ffill')
    return weekly_ma_daily


def collect_cases(symbol: str, market_type: str) -> List[Dict]:
    min_date = datetime(2018, 1, 1)
    df = fetch_daily_klines(symbol, market_type, min_date)
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
    df['pre_return_10d'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['pre_return_20d'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    df['weak_days_250'] = df['below_weekly_ma'].rolling(window=250, min_periods=200).sum()
    df['weak_ratio'] = df['weak_days_250'] / 250
    df['prev_above_ma24'] = df['above_ma24'].shift(1)
    df['is_breakout'] = (~df['prev_above_ma24'].fillna(True)) & df['above_ma24']
    df['below_ma24'] = df['close'] <= df['ma24']
    df['below_ma24_group'] = (df['below_ma24'] != df['below_ma24'].shift()).cumsum()
    df['days_below_ma24'] = df.groupby('below_ma24_group')['below_ma24'].cumsum() * df['below_ma24']

    cases = []
    for idx in df[df['is_breakout']].index:
        i = df.index.get_loc(idx)
        if i < 60 or i >= len(df) - 60:
            continue
        row = df.iloc[i]
        if pd.isna(row['ma24']) or pd.isna(row['drawdown']) or pd.isna(row['volume_ma20']) or row['volume_ma20'] == 0:
            continue

        future_prices = df['close'].iloc[i+1:min(i+61, len(df))]
        future_ma24 = df['ma24'].iloc[i+1:min(i+61, len(df))]
        if len(future_prices) < 30:
            continue

        post_r5d = (future_prices.iloc[4] - row['close']) / row['close'] if len(future_prices) >= 5 else np.nan
        post_r10d = (future_prices.iloc[9] - row['close']) / row['close'] if len(future_prices) >= 10 else np.nan
        post_r20d = (future_prices.iloc[19] - row['close']) / row['close'] if len(future_prices) >= 20 else np.nan
        post_r30d = (future_prices.iloc[29] - row['close']) / row['close'] if len(future_prices) >= 30 else np.nan

        future_30 = future_prices.iloc[:30]
        post_max_30d = (future_30.max() - row['close']) / row['close']
        post_min_30d = (future_30.min() - row['close']) / row['close']

        above_30d = future_prices.iloc[29] > future_ma24.iloc[29] if len(future_prices) >= 30 else False

        cases.append({
            'symbol': symbol,
            'drawdown': row['drawdown'],
            'weak_days': int(row['weak_days_250']) if not pd.isna(row['weak_days_250']) else 0,
            'weak_ratio': row['weak_ratio'] if not pd.isna(row['weak_ratio']) else 0,
            'pre_r5d': row['pre_return_5d'] if not pd.isna(row['pre_return_5d']) else 0,
            'pre_r10d': row['pre_return_10d'] if not pd.isna(row['pre_return_10d']) else 0,
            'pre_r20d': row['pre_return_20d'] if not pd.isna(row['pre_return_20d']) else 0,
            'vol_ratio': row['volume'] / row['volume_ma20'],
            'quote_vol_ratio': row['quote_volume'] / row['quote_volume_ma20'],
            'volatility': row['volatility_20d'] if not pd.isna(row['volatility_20d']) else 0,
            'days_below_ma24': int(row['days_below_ma24']) if not pd.isna(row['days_below_ma24']) else 0,
            'dist_ma24': (row['close'] - row['ma24']) / row['ma24'],
            'post_r5d': post_r5d,
            'post_r10d': post_r10d,
            'post_r20d': post_r20d,
            'post_r30d': post_r30d,
            'post_max30': post_max_30d,
            'post_min30': post_min_30d,
            'above30': above_30d,
            'oversold_bg': (row['drawdown'] >= 0.50) or (row['weak_days_250'] >= 100),
        })
    return cases


def run_data_collection(market_type: str = 'spot'):
    symbols = get_symbols(market_type)
    logger.info(f"收集 {market_type} 市场数据，共 {len(symbols)} 个币...")
    all_cases = []
    for i, symbol in enumerate(symbols, 1):
        try:
            cases = collect_cases(symbol, market_type)
            all_cases.extend(cases)
            if i % 50 == 0:
                logger.info(f"进度: {i}/{len(symbols)}，已收集 {len(all_cases)} 个案例")
        except Exception as e:
            pass
    logger.info(f"完成! 共 {len(all_cases)} 个案例")
    return pd.DataFrame(all_cases)


def evaluate_short_strategy(df: pd.DataFrame, filters: Dict[str, Tuple], name: str):
    """评估一组做空过滤条件的胜率"""
    mask = pd.Series(True, index=df.index)
    for col, (op, val) in filters.items():
        if op == '>':
            mask &= df[col] > val
        elif op == '<':
            mask &= df[col] < val
        elif op == '>=':
            mask &= df[col] >= val
        elif op == '<=':
            mask &= df[col] <= val

    subset = df[mask]
    n = len(subset)
    if n < 50:
        return None

    # 做空指标
    below30 = (~subset['above30']).mean()  # 30天后跌破MA24
    drop10 = (subset['post_r30d'] < -0.10).mean()  # 跌>10%
    drop20 = (subset['post_r30d'] < -0.20).mean()  # 跌>20%
    drop30 = (subset['post_r30d'] < -0.30).mean()  # 跌>30%
    avg_return = subset['post_r30d'].mean()
    median_return = subset['post_r30d'].median()
    max_risk = subset['post_max30'].median()  # 做空最大风险（突破后最高涨幅）
    min_reward = subset['post_min30'].median()  # 做空最小收益（突破后最低跌幅）

    # 盈亏比 = |中位数跌幅| / 中位数涨幅
    rr_ratio = abs(min_reward) / max_risk if max_risk > 0 else 0

    return {
        'name': name,
        'n': n,
        'below30': below30,
        'drop10': drop10,
        'drop20': drop20,
        'drop30': drop30,
        'avg_return': avg_return,
        'median_return': median_return,
        'max_risk': max_risk,
        'min_reward': min_reward,
        'rr_ratio': rr_ratio,
        'score': drop20 * rr_ratio if rr_ratio > 0 else 0,  # 综合评分
    }


def grid_search(df: pd.DataFrame):
    """网格搜索高胜率做空条件组合"""
    logger.info("开始网格搜索高胜率做空条件...")

    strategies = []

    # 基础池：只考虑超跌背景
    base_df = df[df['oversold_bg']].copy()

    # 定义条件网格
    conditions = {
        'pre_r5d': [('>', 0.05), ('>', 0.10), ('>', 0.15), ('>', 0.20)],
        'pre_r20d': [('>', 0.00), ('>', 0.10), ('>', 0.20), ('>', 0.30)],
        'drawdown': [('>', 0.80), ('>', 0.85), ('>', 0.90)],
        'weak_days': [('>', 200), ('>', 220), ('>', 240)],
        'vol_ratio': [('>', 1.2), ('>', 1.5), ('>', 2.0)],
        'quote_vol_ratio': [('>', 1.2), ('>', 1.5), ('>', 2.0)],
        'volatility': [('>', 0.03), ('>', 0.05), ('>', 0.07)],
        'dist_ma24': [('>', 0.02), ('>', 0.03), ('>', 0.05)],
        'days_below_ma24': [('>', 30), ('>', 60), ('>', 90)],
    }

    # 单条件策略
    for col, ops_vals in conditions.items():
        for op, val in ops_vals:
            r = evaluate_short_strategy(base_df, {col: (op, val)}, f"{col}{op}{val}")
            if r:
                strategies.append(r)

    # 双条件组合（重点搜索）
    key_conditions = [
        ('pre_r5d', '>', 0.10),
        ('pre_r20d', '>', 0.20),
        ('drawdown', '>', 0.85),
        ('weak_days', '>', 200),
        ('vol_ratio', '>', 1.5),
        ('volatility', '>', 0.05),
        ('dist_ma24', '>', 0.03),
        ('days_below_ma24', '>', 60),
    ]

    for i, (c1, o1, v1) in enumerate(key_conditions):
        for c2, o2, v2 in key_conditions[i+1:]:
            r = evaluate_short_strategy(base_df, {c1: (o1, v1), c2: (o2, v2)}, f"{c1}{o1}{v1}&{c2}{o2}{v2}")
            if r:
                strategies.append(r)

    # 三条件组合（精选）
    triples = [
        {'pre_r5d': ('>', 0.10), 'drawdown': ('>', 0.85), 'vol_ratio': ('>', 1.5)},
        {'pre_r5d': ('>', 0.10), 'drawdown': ('>', 0.85), 'dist_ma24': ('>', 0.03)},
        {'pre_r20d': ('>', 0.20), 'weak_days': ('>', 200), 'vol_ratio': ('>', 1.5)},
        {'drawdown': ('>', 0.90), 'volatility': ('>', 0.05), 'dist_ma24': ('>', 0.03)},
        {'pre_r5d': ('>', 0.15), 'weak_days': ('>', 220), 'days_below_ma24': ('>', 60)},
    ]
    for t in triples:
        name = '&'.join([f"{k}{v[0]}{v[1]}" for k, v in t.items()])
        r = evaluate_short_strategy(base_df, t, name)
        if r:
            strategies.append(r)

    return pd.DataFrame(strategies)


def print_top_strategies(strategies_df: pd.DataFrame):
    print(f"\n{'='*100}")
    print("做空策略优化结果（按 跌>20%概率 × 盈亏比 综合评分排序）")
    print(f"{'='*100}")

    # 过滤掉样本太少的
    strategies_df = strategies_df[strategies_df['n'] >= 100]
    strategies_df = strategies_df.sort_values('score', ascending=False)

    print(f"\n{'策略条件':<45} {'样本':>6} {'跌破MA24':>10} {'跌>10%':>10} {'跌>20%':>10} {'跌>30%':>10} {'盈亏比':>8} {'综合分':>8}")
    print("-" * 100)

    for _, row in strategies_df.head(30).iterrows():
        print(f"{row['name']:<45} {row['n']:>6} {row['below30']*100:>9.1f}% {row['drop10']*100:>9.1f}% "
              f"{row['drop20']*100:>9.1f}% {row['drop30']*100:>9.1f}% {row['rr_ratio']:>8.2f} {row['score']:>8.3f}")

    # 最优策略详细分析
    if len(strategies_df) > 0:
        best = strategies_df.iloc[0]
        print(f"\n{'='*100}")
        print(f"🏆 最优策略: {best['name']}")
        print(f"{'='*100}")
        print(f"  历史样本数: {best['n']} 个")
        print(f"  30天后跌破MA24概率: {best['below30']*100:.1f}%")
        print(f"  30天跌>10%概率: {best['drop10']*100:.1f}%")
        print(f"  30天跌>20%概率: {best['drop20']*100:.1f}%")
        print(f"  30天跌>30%概率: {best['drop30']*100:.1f}%")
        print(f"  做空最大风险(中位数涨幅): {best['max_risk']*100:.1f}%")
        print(f"  做空最小收益(中位数跌幅): {best['min_reward']*100:.1f}%")
        print(f"  盈亏比: {best['rr_ratio']:.2f}")
        print(f"  平均30天收益: {best['avg_return']*100:.1f}%")
        print(f"  中位数30天收益: {best['median_return']*100:.1f}%")

    # 按单一指标排序：只看跌>20%概率最高的
    print(f"\n{'='*100}")
    print("📉 跌>20%概率最高的策略（纯胜率优先）")
    print(f"{'='*100}")
    top_drop20 = strategies_df.nlargest(15, 'drop20')
    print(f"{'策略条件':<45} {'样本':>6} {'跌>20%':>10} {'跌>30%':>10} {'盈亏比':>8}")
    print("-" * 100)
    for _, row in top_drop20.iterrows():
        print(f"{row['name']:<45} {row['n']:>6} {row['drop20']*100:>9.1f}% {row['drop30']*100:>9.1f}% {row['rr_ratio']:>8.2f}")

    # 按盈亏比排序
    print(f"\n{'='*100}")
    print("⚖️ 盈亏比最高的策略（赔率优先）")
    print(f"{'='*100}")
    top_rr = strategies_df[strategies_df['n'] >= 200].nlargest(15, 'rr_ratio')
    print(f"{'策略条件':<45} {'样本':>6} {'跌>20%':>10} {'盈亏比':>8} {'综合分':>8}")
    print("-" * 100)
    for _, row in top_rr.iterrows():
        print(f"{row['name']:<45} {row['n']:>6} {row['drop20']*100:>9.1f}% {row['rr_ratio']:>8.2f} {row['score']:>8.3f}")


def analyze_post_breakout_timing(df: pd.DataFrame):
    """分析突破后不同时间窗口做空的胜率"""
    print(f"\n{'='*100}")
    print("⏱️ 突破后不同时间窗口做空的胜率对比")
    print(f"{'='*100}")

    base_df = df[df['oversold_bg']].copy()

    windows = [
        ('突破后第1天收益<0', base_df['post_r5d'] < 0),
        ('突破后第1天收益>0', base_df['post_r5d'] > 0),
        ('突破后5天内未创新高', base_df['post_max30'] < 0.05),
        ('突破后5天内创新高>5%', base_df['post_max30'] >= 0.05),
        ('突破后5天内已跌破MA24', base_df['post_r5d'] < 0),
        ('突破前5天涨幅>10%', base_df['pre_r5d'] > 0.10),
        ('突破前20天涨幅>20%', base_df['pre_r20d'] > 0.20),
    ]

    print(f"{'条件':<35} {'样本':>6} {'跌破MA24':>10} {'跌>10%':>10} {'跌>20%':>10} {'跌>30%':>10} {'盈亏比':>8}")
    print("-" * 100)
    for name, mask in windows:
        subset = base_df[mask]
        n = len(subset)
        if n < 50:
            continue
        below30 = (~subset['above30']).mean()
        drop10 = (subset['post_r30d'] < -0.10).mean()
        drop20 = (subset['post_r30d'] < -0.20).mean()
        drop30 = (subset['post_r30d'] < -0.30).mean()
        rr = abs(subset['post_min30'].median()) / subset['post_max30'].median() if subset['post_max30'].median() > 0 else 0
        print(f"{name:<35} {n:>6} {below30*100:>9.1f}% {drop10*100:>9.1f}% {drop20*100:>9.1f}% {drop30*100:>9.1f}% {rr:>8.2f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', choices=['spot', 'swap'], default='spot')
    args = parser.parse_args()

    df = run_data_collection(args.market)

    # 基础统计
    print(f"\n{'='*100}")
    print("基础统计（全部超跌背景案例）")
    print(f"{'='*100}")
    base = df[df['oversold_bg']]
    print(f"总样本: {len(base)}")
    print(f"30天后跌破MA24: {(~base['above30']).mean()*100:.1f}%")
    print(f"30天跌>10%: {(base['post_r30d'] < -0.10).mean()*100:.1f}%")
    print(f"30天跌>20%: {(base['post_r30d'] < -0.20).mean()*100:.1f}%")
    print(f"30天跌>30%: {(base['post_r30d'] < -0.30).mean()*100:.1f}%")

    strategies_df = grid_search(df)
    print_top_strategies(strategies_df)
    analyze_post_breakout_timing(df)
