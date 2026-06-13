#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
历史市场结构对比分析
对比当前（2026-06）与历史上几次大跌底部的市场状态
"""

import os
import sys
import logging
import psycopg2
import pandas as pd
import numpy as np

DATABASE_URL = os.getenv('DB_DSN', 'postgresql://postgres:12@127.0.0.1:5432/market_data')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def get_db_conn():
    return psycopg2.connect(DATABASE_URL)


def get_symbols_at_date(target_date: str, market_type: str = 'spot'):
    """获取某个日期之前有数据的币种列表"""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT symbol FROM binance_daily_klines 
                WHERE market_type = %s AND open_time <= %s
                  AND open_time >= %s
                ORDER BY symbol
                """,
                (market_type, target_date, '2017-01-01')
            )
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def fetch_data_before(symbol: str, market_type: str, target_date: str, min_days: int = 400):
    conn = get_db_conn()
    query = """
        SELECT open_time, open, high, low, close, volume, quote_volume
        FROM binance_daily_klines
        WHERE symbol = %s AND market_type = %s AND open_time <= %s
        ORDER BY open_time ASC
    """
    try:
        df = pd.read_sql(query, conn, params=(symbol, market_type, target_date))
        if not df.empty:
            df['open_time'] = pd.to_datetime(df['open_time'])
            df = df.set_index('open_time')
        if len(df) < min_days:
            return pd.DataFrame()
        return df
    finally:
        conn.close()


def analyze_market_state(target_date: str, label: str, market_type: str = 'spot'):
    """分析某一天的整个市场结构"""
    symbols = get_symbols_at_date(target_date, market_type)
    logger.info(f"[{label}] 共 {len(symbols)} 个币有数据")
    
    total = 0
    oversold_count = 0
    above_ma24_count = 0
    extreme_signals = 0
    high_signals = 0
    mid_signals = 0
    
    btc_dd = None
    eth_dd = None
    sol_dd = None
    
    for symbol in symbols:
        df = fetch_data_before(symbol, market_type, target_date)
        if df.empty or len(df) < 300:
            continue
        
        total += 1
        
        # 计算指标
        df['ma24'] = df['close'].rolling(window=24, min_periods=24).mean()
        weekly = df['close'].resample('W-FRI').last().dropna()
        weekly_ma = weekly.rolling(window=60, min_periods=60).mean()
        df['weekly_ma60'] = weekly_ma.reindex(df.index, method='ffill')
        df['rolling_high_500d'] = df['high'].rolling(window=500, min_periods=300).max()
        df['drawdown'] = (df['rolling_high_500d'] - df['close']) / df['rolling_high_500d']
        df['below_weekly_ma'] = df['close'] < df['weekly_ma60']
        df['weak_days_250'] = df['below_weekly_ma'].rolling(window=250, min_periods=200).sum()
        df['weak_ratio'] = df['weak_days_250'] / 250
        df['above_ma24'] = df['close'] > df['ma24']
        df['pre_return_20d'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
        df['pre_return_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['volume_ma20'] = df['volume'].rolling(window=20, min_periods=15).mean()
        df['vol_ratio'] = df['volume'] / df['volume_ma20']
        
        last_row = df.iloc[-1]
        
        dd = last_row['drawdown']
        weak_ratio = last_row['weak_ratio'] if not pd.isna(last_row['weak_ratio']) else 0
        pre_r20d = last_row['pre_return_20d'] if not pd.isna(last_row['pre_return_20d']) else 0
        pre_r5d = last_row['pre_return_5d'] if not pd.isna(last_row['pre_return_5d']) else 0
        vol = last_row['vol_ratio'] if not pd.isna(last_row['vol_ratio']) else 1.0
        above_ma24 = last_row['above_ma24']
        
        is_oversold = (dd >= 0.80) and (weak_ratio >= 0.80)
        if is_oversold:
            oversold_count += 1
        if above_ma24:
            above_ma24_count += 1
        
        # 信号计数（简化版）
        extreme_count = sum([
            pre_r20d > 0.30,
            pre_r5d > 0.10 and pre_r20d > 0.20,
            vol > 0.07 if not pd.isna(vol) else False,
            (last_row['weak_days_250'] > 200 and vol > 1.5) if not pd.isna(last_row['weak_days_250']) else False,
        ])
        if is_oversold and above_ma24:
            if extreme_count >= 3:
                extreme_signals += 1
            elif extreme_count >= 2:
                high_signals += 1
            elif extreme_count >= 1:
                mid_signals += 1
        
        # BTC/ETH/SOL 回撤
        if symbol == 'BTCUSDT':
            ath = df['high'].max()
            btc_dd = (ath - last_row['close']) / ath
        elif symbol == 'ETHUSDT':
            ath = df['high'].max()
            eth_dd = (ath - last_row['close']) / ath
        elif symbol == 'SOLUSDT':
            ath = df['high'].max()
            sol_dd = (ath - last_row['close']) / ath
    
    return {
        'label': label,
        'date': target_date,
        'total_symbols': total,
        'oversold_ratio': oversold_count / total if total > 0 else 0,
        'above_ma24_ratio': above_ma24_count / total if total > 0 else 0,
        'btc_drawdown': btc_dd,
        'eth_drawdown': eth_dd,
        'sol_drawdown': sol_dd,
        'extreme_signals': extreme_signals,
        'high_signals': high_signals,
        'mid_signals': mid_signals,
    }


def main():
    periods = [
        ('2018-12-15', '2018熊市底'),
        ('2020-03-12', '2020疫情底'),
        ('2021-07-20', '2021-519底'),
        ('2022-11-21', '2022-FTX底'),
        ('2025-04-08', '2025调整底'),
        ('2026-06-06', '当前'),
    ]
    
    results = []
    for date, label in periods:
        logger.info(f"分析 {label} ({date})...")
        result = analyze_market_state(date, label)
        results.append(result)
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*120)
    print("  历史市场结构对比分析")
    print("="*120)
    
    print(f"\n  {'时期':<15} {'日期':<12} {'币种数':>6} {'超跌占比':>8} {'站上MA24':>8} {'BTC回撤':>8} {'ETH回撤':>8} {'SOL回撤':>8} {'极端信号':>6}")
    print(f"  {'-'*15} {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for _, row in df.iterrows():
        print(f"  {row['label']:<15} {row['date']:<12} {row['total_symbols']:>6} {row['oversold_ratio']*100:>7.1f}% {row['above_ma24_ratio']*100:>7.1f}% {row['btc_drawdown']*100:>7.1f}% {row['eth_drawdown']*100:>7.1f}% {row['sol_drawdown']*100:>7.1f}% {row['extreme_signals']:>6}")
    
    print(f"\n{'='*120}")


if __name__ == '__main__':
    main()
