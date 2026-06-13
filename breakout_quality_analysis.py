#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
MA24突破质量分析
目的：从历史数据中找出所有MA24突破案例，区分：
  - 假突破（站上后很快回落）
  - 真突破（站上后持续上涨）
对比两类的量价特征差异，为做空/做多决策提供依据
"""

import os
import sys
import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
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


@dataclass
class BreakoutCase:
    symbol: str
    breakout_date: datetime
    # 突破前特征
    price_at_breakout: float
    ma24_at_breakout: float
    ma6_at_breakout: float
    drawdown_at_breakout: float  # 从500日高点回撤
    weak_days_at_breakout: int   # 低于60周均线天数
    weak_ratio_at_breakout: float
    pre_return_5d: float
    pre_return_10d: float
    pre_return_20d: float
    pre_return_60d: float
    volume_ratio: float          # 突破日成交量 / 前20日均量
    quote_volume_ratio: float    # 突破日成交额 / 前20日均成交额
    volatility_20d: float        # 前20日收益率标准差
    days_below_ma24_before: int  # 突破前连续跌破MA24天数
    distance_to_ma24: float      # (价格-MA24)/MA24
    # 突破后表现
    post_return_5d: float
    post_return_10d: float
    post_return_20d: float
    post_return_30d: float
    post_return_60d: float
    post_max_return_30d: float   # 30天内最大涨幅（从突破价）
    post_min_return_30d: float   # 30天内最大跌幅
    post_max_return_60d: float
    post_min_return_60d: float
    above_ma24_30d: bool         # 30天后是否仍在MA24之上
    above_ma24_60d: bool
    # 分类
    is_oversold_background: bool  # 突破时是否处于超跌背景（回撤>50% 或 弱势>100天）


def analyze_symbol_breakouts(symbol: str, market_type: str) -> List[BreakoutCase]:
    """分析单个币历史上所有MA24向上突破的案例"""
    min_date = datetime(2018, 1, 1)
    df = fetch_daily_klines(symbol, market_type, min_date)
    if df.empty or len(df) < 400:
        return []

    # 计算指标
    df['ma6'] = df['close'].rolling(window=6, min_periods=6).mean()
    df['ma24'] = df['close'].rolling(window=24, min_periods=24).mean()
    df['weekly_ma60'] = calculate_weekly_ma(df, 60)
    df['rolling_high_500d'] = df['high'].rolling(window=500, min_periods=300).max()
    df['drawdown'] = (df['rolling_high_500d'] - df['close']) / df['rolling_high_500d']
    df['below_weekly_ma'] = df['close'] < df['weekly_ma60']
    df['above_ma24'] = df['close'] > df['ma24']

    # 成交量均值
    df['volume_ma20'] = df['volume'].rolling(window=20, min_periods=15).mean()
    df['quote_volume_ma20'] = df['quote_volume'].rolling(window=20, min_periods=15).mean()

    # 波动率
    df['return_1d'] = df['close'].pct_change()
    df['volatility_20d'] = df['return_1d'].rolling(window=20, min_periods=15).std()

    # 前N日收益
    df['pre_return_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['pre_return_10d'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['pre_return_20d'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    df['pre_return_60d'] = (df['close'] - df['close'].shift(60)) / df['close'].shift(60)

    # 弱势天数（滚动250天低于60周均线天数）
    df['weak_days_250'] = df['below_weekly_ma'].rolling(window=250, min_periods=200).sum()
    df['weak_ratio'] = df['weak_days_250'] / 250

    # 找突破点：昨天close<=ma24，今天close>ma24
    df['prev_above_ma24'] = df['above_ma24'].shift(1)
    df['is_breakout'] = (~df['prev_above_ma24'].fillna(True)) & df['above_ma24']

    # 突破前连续跌破MA24天数
    df['below_ma24'] = df['close'] <= df['ma24']
    df['below_ma24_group'] = (df['below_ma24'] != df['below_ma24'].shift()).cumsum()
    df['days_below_ma24'] = df.groupby('below_ma24_group')['below_ma24'].cumsum() * df['below_ma24']

    cases = []
    for idx in df[df['is_breakout']].index:
        i = df.index.get_loc(idx)
        if i < 60 or i >= len(df) - 60:
            continue

        row = df.iloc[i]
        # 过滤：要求有足够的数据质量
        if pd.isna(row['ma24']) or pd.isna(row['drawdown']) or pd.isna(row['volume_ma20']):
            continue
        if row['volume_ma20'] == 0:
            continue

        # 突破后收益
        future_prices = df['close'].iloc[i+1:min(i+61, len(df))]
        future_ma24 = df['ma24'].iloc[i+1:min(i+61, len(df))]
        if len(future_prices) < 30:
            continue

        post_ret_5d = (future_prices.iloc[4] - row['close']) / row['close'] if len(future_prices) >= 5 else np.nan
        post_ret_10d = (future_prices.iloc[9] - row['close']) / row['close'] if len(future_prices) >= 10 else np.nan
        post_ret_20d = (future_prices.iloc[19] - row['close']) / row['close'] if len(future_prices) >= 20 else np.nan
        post_ret_30d = (future_prices.iloc[29] - row['close']) / row['close'] if len(future_prices) >= 30 else np.nan
        post_ret_60d = (future_prices.iloc[59] - row['close']) / row['close'] if len(future_prices) >= 60 else np.nan

        future_30 = future_prices.iloc[:30]
        post_max_30d = ((future_30.max() - row['close']) / row['close']) if len(future_30) > 0 else np.nan
        post_min_30d = ((future_30.min() - row['close']) / row['close']) if len(future_30) > 0 else np.nan

        future_60 = future_prices.iloc[:60]
        post_max_60d = ((future_60.max() - row['close']) / row['close']) if len(future_60) > 0 else np.nan
        post_min_60d = ((future_60.min() - row['close']) / row['close']) if len(future_60) > 0 else np.nan

        above_30d = future_prices.iloc[29] > future_ma24.iloc[29] if len(future_prices) >= 30 else False
        above_60d = future_prices.iloc[59] > future_ma24.iloc[59] if len(future_prices) >= 60 else False

        is_oversold_bg = (row['drawdown'] >= 0.50) or (row['weak_days_250'] >= 100)

        cases.append(BreakoutCase(
            symbol=symbol,
            breakout_date=idx,
            price_at_breakout=row['close'],
            ma24_at_breakout=row['ma24'],
            ma6_at_breakout=row['ma6'],
            drawdown_at_breakout=row['drawdown'],
            weak_days_at_breakout=int(row['weak_days_250']) if not pd.isna(row['weak_days_250']) else 0,
            weak_ratio_at_breakout=row['weak_ratio'] if not pd.isna(row['weak_ratio']) else 0,
            pre_return_5d=row['pre_return_5d'] if not pd.isna(row['pre_return_5d']) else 0,
            pre_return_10d=row['pre_return_10d'] if not pd.isna(row['pre_return_10d']) else 0,
            pre_return_20d=row['pre_return_20d'] if not pd.isna(row['pre_return_20d']) else 0,
            pre_return_60d=row['pre_return_60d'] if not pd.isna(row['pre_return_60d']) else 0,
            volume_ratio=(row['volume'] / row['volume_ma20']) if row['volume_ma20'] > 0 else 1.0,
            quote_volume_ratio=(row['quote_volume'] / row['quote_volume_ma20']) if row['quote_volume_ma20'] > 0 else 1.0,
            volatility_20d=row['volatility_20d'] if not pd.isna(row['volatility_20d']) else 0,
            days_below_ma24_before=int(row['days_below_ma24']) if not pd.isna(row['days_below_ma24']) else 0,
            distance_to_ma24=(row['close'] - row['ma24']) / row['ma24'],
            post_return_5d=post_ret_5d,
            post_return_10d=post_ret_10d,
            post_return_20d=post_ret_20d,
            post_return_30d=post_ret_30d,
            post_return_60d=post_ret_60d,
            post_max_return_30d=post_max_30d,
            post_min_return_30d=post_min_30d,
            post_max_return_60d=post_max_60d,
            post_min_return_60d=post_min_60d,
            above_ma24_30d=above_30d,
            above_ma24_60d=above_60d,
            is_oversold_background=is_oversold_bg,
        ))

    return cases


def run_analysis(market_type: str = 'spot'):
    symbols = get_symbols(market_type)
    logger.info(f"开始分析 {market_type} 市场所有MA24突破案例，共 {len(symbols)} 个币...")

    all_cases: List[BreakoutCase] = []
    for i, symbol in enumerate(symbols, 1):
        try:
            cases = analyze_symbol_breakouts(symbol, market_type)
            all_cases.extend(cases)
            if i % 50 == 0:
                logger.info(f"进度: {i}/{len(symbols)}，已收集 {len(all_cases)} 个突破案例")
        except Exception as e:
            logger.warning(f"[{symbol}] 分析异常: {e}")

    logger.info(f"分析完成! 共收集 {len(all_cases)} 个MA24突破案例")
    return all_cases


def print_comparison_report(cases: List[BreakoutCase]):
    if not cases:
        print("❌ 没有收集到足够的突破案例")
        return

    df = pd.DataFrame([{
        'symbol': c.symbol,
        'breakout_date': c.breakout_date,
        'drawdown': c.drawdown_at_breakout,
        'weak_days': c.weak_days_at_breakout,
        'weak_ratio': c.weak_ratio_at_breakout,
        'pre_r5d': c.pre_return_5d,
        'pre_r20d': c.pre_return_20d,
        'pre_r60d': c.pre_return_60d,
        'vol_ratio': c.volume_ratio,
        'quote_vol_ratio': c.quote_volume_ratio,
        'volatility': c.volatility_20d,
        'days_below_ma24': c.days_below_ma24_before,
        'dist_ma24': c.distance_to_ma24,
        'post_r5d': c.post_return_5d,
        'post_r10d': c.post_return_10d,
        'post_r20d': c.post_return_20d,
        'post_r30d': c.post_return_30d,
        'post_r60d': c.post_return_60d,
        'post_max30': c.post_max_return_30d,
        'post_min30': c.post_min_return_30d,
        'post_max60': c.post_max_return_60d,
        'post_min60': c.post_min_return_60d,
        'above30': c.above_ma24_30d,
        'above60': c.above_ma24_60d,
        'oversold_bg': c.is_oversold_background,
    } for c in cases])

    # 过滤去掉异常值
    df = df[df['vol_ratio'] < 50]  # 去掉成交量异常放大的
    df = df[df['quote_vol_ratio'] < 50]
    df = df[df['volatility'] < 0.5]  # 去掉波动率异常的

    print(f"\n{'='*90}")
    print(f"MA24突破质量分析报告（共 {len(df)} 个有效案例）")
    print(f"{'='*90}")

    # 总体成功率
    success_30d = df['above30'].mean()
    success_60d = df['above60'].mean()
    print(f"\n📊 总体统计:")
    print(f"  30天后仍站上MA24: {success_30d*100:.1f}%")
    print(f"  60天后仍站上MA24: {success_60d*100:.1f}%")
    print(f"  突破后30天平均收益: {df['post_r30d'].mean()*100:.1f}%")
    print(f"  突破后30天最大涨幅中位数: {df['post_max30'].median()*100:.1f}%")
    print(f"  突破后30天最大跌幅中位数: {df['post_min30'].median()*100:.1f}%")

    # 按超跌背景分类
    oversold_df = df[df['oversold_bg']]
    normal_df = df[~df['oversold_bg']]

    print(f"\n{'='*90}")
    print(f"【分类A】超跌背景下的突破（回撤≥50% 或 弱势≥100天）: {len(oversold_df)} 个")
    print(f"{'='*90}")
    print(f"  30天成功率: {oversold_df['above30'].mean()*100:.1f}%")
    print(f"  60天成功率: {oversold_df['above60'].mean()*100:.1f}%")
    print(f"  30天平均收益: {oversold_df['post_r30d'].mean()*100:.1f}%")
    print(f"  30天最大涨幅中位数: {oversold_df['post_max30'].median()*100:.1f}%")
    print(f"  30天最大跌幅中位数: {oversold_df['post_min30'].median()*100:.1f}%")

    print(f"\n{'='*90}")
    print(f"【分类B】正常背景下的突破（非超跌）: {len(normal_df)} 个")
    print(f"{'='*90}")
    print(f"  30天成功率: {normal_df['above30'].mean()*100:.1f}%")
    print(f"  60天成功率: {normal_df['above60'].mean()*100:.1f}%")
    print(f"  30天平均收益: {normal_df['post_r30d'].mean()*100:.1f}%")
    print(f"  30天最大涨幅中位数: {normal_df['post_max30'].median()*100:.1f}%")
    print(f"  30天最大跌幅中位数: {normal_df['post_min30'].median()*100:.1f}%")

    # 核心分析：在超跌背景下，区分成功 vs 失败
    if len(oversold_df) > 50:
        analyze_success_vs_failure(oversold_df, "超跌背景")
    if len(normal_df) > 50:
        analyze_success_vs_failure(normal_df, "正常背景")


def analyze_success_vs_failure(sub_df: pd.DataFrame, context: str):
    success = sub_df[sub_df['above30']]
    failure = sub_df[~sub_df['above30']]

    if len(success) < 10 or len(failure) < 10:
        return

    print(f"\n{'='*90}")
    print(f"【{context}】成功突破 vs 假突破 的量价特征对比")
    print(f"{'='*90}")
    print(f"样本: 成功 {len(success)} 个, 失败 {len(failure)} 个")

    metrics = [
        ('突破时回撤', 'drawdown', lambda x: f"{x*100:.1f}%"),
        ('弱势天数', 'weak_days', lambda x: f"{x:.0f}天"),
        ('突破前5天涨幅', 'pre_r5d', lambda x: f"{x*100:.1f}%"),
        ('突破前20天涨幅', 'pre_r20d', lambda x: f"{x*100:.1f}%"),
        ('突破前60天涨幅', 'pre_r60d', lambda x: f"{x*100:.1f}%"),
        ('成交量放大倍数', 'vol_ratio', lambda x: f"{x:.2f}x"),
        ('成交额放大倍数', 'quote_vol_ratio', lambda x: f"{x:.2f}x"),
        ('前20天波动率', 'volatility', lambda x: f"{x*100:.1f}%"),
        ('突破前连续跌破MA24天数', 'days_below_ma24', lambda x: f"{x:.0f}天"),
        ('突破时距MA24乖离率', 'dist_ma24', lambda x: f"{x*100:.2f}%"),
        ('突破后5天收益', 'post_r5d', lambda x: f"{x*100:.1f}%"),
        ('突破后10天收益', 'post_r10d', lambda x: f"{x*100:.1f}%"),
        ('突破后30天最大涨幅', 'post_max30', lambda x: f"{x*100:.1f}%"),
        ('突破后30天最大跌幅', 'post_min30', lambda x: f"{x*100:.1f}%"),
    ]

    print(f"\n{'指标':<25} {'成功组(中位数)':>18} {'失败组(中位数)':>18} {'差异':>12}")
    print("-"*80)
    for name, col, fmt in metrics:
        s_med = success[col].median()
        f_med = failure[col].median()
        diff = s_med - f_med
        diff_str = f"+{diff:.3f}" if diff >= 0 else f"{diff:.3f}"
        print(f"{name:<25} {fmt(s_med):>18} {fmt(f_med):>18} {diff_str:>12}")

    # 极端成功案例：找出30天收益>50%的案例特征
    big_winners = sub_df[sub_df['post_r30d'] > 0.5]
    if len(big_winners) > 0:
        print(f"\n🏆 极端成功案例（30天涨>50%）: {len(big_winners)} 个")
        print(f"   平均回撤: {big_winners['drawdown'].median()*100:.1f}%")
        print(f"   平均成交量放大: {big_winners['vol_ratio'].median():.2f}x")
        print(f"   平均成交额放大: {big_winners['quote_vol_ratio'].median():.2f}x")
        print(f"   平均突破前5天涨幅: {big_winners['pre_r5d'].median()*100:.1f}%")
        print(f"   平均突破前20天涨幅: {big_winners['pre_r20d'].median()*100:.1f}%")
        print(f"   典型币: {', '.join(big_winners['symbol'].unique()[:10])}")

    # 做空友好的案例：突破后很快回落
    big_losers = sub_df[sub_df['post_r30d'] < -0.2]
    if len(big_losers) > 0:
        print(f"\n📉 做空友好案例（30天跌>20%）: {len(big_losers)} 个")
        print(f"   平均回撤: {big_losers['drawdown'].median()*100:.1f}%")
        print(f"   平均成交量放大: {big_losers['vol_ratio'].median():.2f}x")
        print(f"   平均成交额放大: {big_losers['quote_vol_ratio'].median():.2f}x")
        print(f"   平均突破前5天涨幅: {big_losers['pre_r5d'].median()*100:.1f}%")
        print(f"   平均突破前20天涨幅: {big_losers['pre_r20d'].median()*100:.1f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', choices=['spot', 'swap'], default='spot')
    args = parser.parse_args()

    cases = run_analysis(args.market)
    print_comparison_report(cases)
