#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
扫描反弹力度强的山寨币
找出与 ALLOUSDT、BEATUSDT 具有相似特征的币种
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


def get_symbols(market_type: str):
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


def fetch_klines(symbol: str, market_type: str):
    conn = get_db_conn()
    query = """
        SELECT open_time, open, high, low, close, volume, quote_volume
        FROM binance_daily_klines
        WHERE symbol = %s AND market_type = %s
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


def analyze_symbol(symbol: str, market_type: str):
    df = fetch_klines(symbol, market_type)
    if df.empty or len(df) < 30:
        return None
    
    first_date = df.index[0]
    last_date = df.index[-1]
    days_since_listing = (last_date - first_date).days
    
    ath = df['high'].max()
    ath_date = df['high'].idxmax()
    low_since_ath_idx = df.loc[ath_date:]['low'].idxmin() if ath_date < df.index[-1] else df.index[-1]
    low_since_ath = df.loc[ath_date:]['low'].min()
    
    current_price = df['close'].iloc[-1]
    current_drawdown = (ath - current_price) / ath
    recovery_to_ath = current_price / ath
    
    # 从最低点反弹
    if low_since_ath > 0:
        rebound_from_low = (current_price - low_since_ath) / low_since_ath
    else:
        rebound_from_low = 0
    
    # 近期表现
    df['ma24'] = df['close'].rolling(window=24, min_periods=10).mean()
    df['ma24'] = df['ma24'].fillna(df['close'].expanding().mean())
    df['above_ma24'] = df['close'] > df['ma24']
    df['volume_ma20'] = df['volume'].rolling(window=20, min_periods=10).mean()
    
    last_row = df.iloc[-1]
    
    # 7天、14天、30天涨幅
    if len(df) >= 7:
        r7d = (last_row['close'] - df['close'].iloc[-8]) / df['close'].iloc[-8]
    else:
        r7d = np.nan
    if len(df) >= 14:
        r14d = (last_row['close'] - df['close'].iloc[-15]) / df['close'].iloc[-15]
    else:
        r14d = np.nan
    if len(df) >= 30:
        r30d = (last_row['close'] - df['close'].iloc[-31]) / df['close'].iloc[-31]
    else:
        r30d = np.nan
    
    # 成交量比
    vol_ratio = last_row['volume'] / last_row['volume_ma20'] if last_row['volume_ma20'] > 0 else 1.0
    avg_vol_7d = df['volume'].iloc[-7:].mean()
    avg_vol_30d = df['volume'].iloc[-30:].mean()
    vol_surge = avg_vol_7d / avg_vol_30d if avg_vol_30d > 0 else 1.0
    
    # 7天内最大单日涨幅
    last_7d_returns = df['close'].iloc[-7:].pct_change().dropna()
    max_daily_gain_7d = last_7d_returns.max() if len(last_7d_returns) > 0 else 0
    
    return {
        'symbol': symbol,
        'market_type': market_type,
        'first_date': first_date,
        'days_since_listing': days_since_listing,
        'ath': ath,
        'ath_date': ath_date,
        'low_since_ath': low_since_ath,
        'low_since_ath_date': low_since_ath_idx,
        'current_price': current_price,
        'current_drawdown': current_drawdown,
        'recovery_to_ath': recovery_to_ath,
        'rebound_from_low': rebound_from_low,
        'above_ma24': last_row['above_ma24'],
        'r7d': r7d,
        'r14d': r14d,
        'r30d': r30d,
        'vol_ratio': vol_ratio,
        'vol_surge': vol_surge,
        'max_daily_gain_7d': max_daily_gain_7d,
        'quote_vol_7d_avg': df['quote_volume'].iloc[-7:].mean(),
    }


def main():
    symbols = get_symbols('spot')
    logger.info(f"共 {len(symbols)} 个 spot 币")
    
    results = []
    for i, symbol in enumerate(symbols):
        if i % 50 == 0:
            logger.info(f"进度: {i}/{len(symbols)}")
        result = analyze_symbol(symbol, 'spot')
        if result:
            results.append(result)
    
    # 也处理 swap
    swap_symbols = get_symbols('swap')
    logger.info(f"共 {len(swap_symbols)} 个 swap 币")
    for i, symbol in enumerate(swap_symbols):
        if i % 50 == 0:
            logger.info(f"swap 进度: {i}/{len(swap_symbols)}")
        # 如果 spot 已经处理过，跳过
        if symbol in symbols:
            continue
        result = analyze_symbol(symbol, 'swap')
        if result:
            results.append(result)
    
    df = pd.DataFrame(results)
    if df.empty:
        print("无数据")
        return
    
    print("\n" + "="*120)
    print("  反弹力度强的币种扫描")
    print("="*120)
    
    # 先看 ALLO 和 BEAT 的特征
    allo_spot = df[(df['symbol'] == 'ALLOUSDT') & (df['market_type'] == 'spot')]
    beat_swap = df[(df['symbol'] == 'BEATUSDT') & (df['market_type'] == 'swap')]
    
    print("\n【参考标的特征】")
    if not allo_spot.empty:
        row = allo_spot.iloc[0]
        print(f"\n  ALLOUSDT (spot):")
        print(f"    上市日期: {row['first_date'].strftime('%Y-%m-%d')} (已上市 {row['days_since_listing']} 天)")
        print(f"    历史最高: {row['ath']:.4f} @ {row['ath_date'].strftime('%Y-%m-%d')}")
        print(f"    上市后期低点: {row['low_since_ath']:.4f} @ {row['low_since_ath_date'].strftime('%Y-%m-%d')}")
        print(f"    当前价格: {row['current_price']:.4f}")
        print(f"    当前回撤: {row['current_drawdown']*100:.1f}%")
        print(f"    恢复到 ATH: {row['recovery_to_ath']*100:.1f}%")
        print(f"    从低点反弹: {row['rebound_from_low']*100:.1f}%")
        print(f"    7天涨幅: {row['r7d']*100:.1f}%")
        print(f"    14天涨幅: {row['r14d']*100:.1f}%")
        print(f"    30天涨幅: {row['r30d']*100:.1f}%")
        print(f"    成交量放大(7d/30d): {row['vol_surge']:.2f}x")
        print(f"    7天最大单日涨幅: {row['max_daily_gain_7d']*100:.1f}%")
        print(f"    站上MA24: {'是' if row['above_ma24'] else '否'}")
    
    if not beat_swap.empty:
        row = beat_swap.iloc[0]
        print(f"\n  BEATUSDT (swap):")
        print(f"    上市日期: {row['first_date'].strftime('%Y-%m-%d')} (已上市 {row['days_since_listing']} 天)")
        print(f"    历史最高: {row['ath']:.4f} @ {row['ath_date'].strftime('%Y-%m-%d')}")
        print(f"    上市后期低点: {row['low_since_ath']:.4f} @ {row['low_since_ath_date'].strftime('%Y-%m-%d')}")
        print(f"    当前价格: {row['current_price']:.4f}")
        print(f"    当前回撤: {row['current_drawdown']*100:.1f}%")
        print(f"    恢复到 ATH: {row['recovery_to_ath']*100:.1f}%")
        print(f"    从低点反弹: {row['rebound_from_low']*100:.1f}%")
        print(f"    7天涨幅: {row['r7d']*100:.1f}%")
        print(f"    14天涨幅: {row['r14d']*100:.1f}%")
        print(f"    30天涨幅: {row['r30d']*100:.1f}%")
        print(f"    成交量放大(7d/30d): {row['vol_surge']:.2f}x")
        print(f"    7天最大单日涨幅: {row['max_daily_gain_7d']*100:.1f}%")
        print(f"    站上MA24: {'是' if row['above_ma24'] else '否'}")
    
    # 筛选相似币种
    print("\n" + "="*120)
    print("  筛选条件：与 ALLO/BEAT 相似")
    print("="*120)
    
    # 条件：
    # 1. 上市时间 < 1年（新币）
    # 2. 当前回撤 40-80%
    # 3. 7天涨幅 > 30%
    # 4. 成交量放大 > 1.5x
    # 5. 站上 MA24
    # 6. 从低点反弹 > 50%
    
    filtered = df[
        (df['days_since_listing'] <= 365) &
        (df['current_drawdown'] >= 0.40) &
        (df['current_drawdown'] <= 0.85) &
        (df['r7d'] >= 0.20) &
        (df['vol_surge'] >= 1.5) &
        (df['above_ma24'] == True) &
        (df['rebound_from_low'] >= 0.30)
    ].copy()
    
    # 按恢复程度和反弹力度排序
    filtered['score'] = (
        filtered['r7d'] * 0.3 +
        filtered['rebound_from_low'] * 0.3 +
        filtered['recovery_to_ath'] * 0.2 +
        (filtered['vol_surge'] - 1) * 0.1 +
        (1 - filtered['current_drawdown']) * 0.1
    )
    filtered = filtered.sort_values('score', ascending=False)
    
    print(f"\n  符合条件的币种: {len(filtered)} 个")
    print(f"\n  {'排名':<4} {'币种':<16} {'市场':<6} {'上市日期':<12} {'回撤':>7} {'恢复ATH':>8} {'从低反弹':>8} {'7天涨幅':>8} {'14天涨幅':>8} {'成交量放大':>10} {'站上MA24':>8}")
    print(f"  {'-'*4} {'-'*16} {'-'*6} {'-'*12} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    
    for rank, (_, row) in enumerate(filtered.head(30).iterrows(), 1):
        print(f"  {rank:<4} {row['symbol']:<16} {row['market_type']:<6} {row['first_date'].strftime('%Y-%m-%d')} {row['current_drawdown']*100:>6.1f}% {row['recovery_to_ath']*100:>7.1f}% {row['rebound_from_low']*100:>7.1f}% {row['r7d']*100:>7.1f}% {row['r14d']*100:>7.1f}% {row['vol_surge']:>9.2f}x {'是' if row['above_ma24'] else '否':>8}")
    
    # 分析共同特征
    print(f"\n{'='*120}")
    print("  筛选结果的共同特征分析")
    print(f"{'='*120}")
    
    if len(filtered) > 0:
        print(f"\n  样本数: {len(filtered)} 个")
        print(f"\n  上市时间分布:")
        filtered['listing_year'] = filtered['first_date'].dt.year
        print(filtered['listing_year'].value_counts().sort_index().to_string())
        
        print(f"\n  关键指标中位数:")
        print(f"    当前回撤: {filtered['current_drawdown'].median()*100:.1f}%")
        print(f"    恢复到 ATH: {filtered['recovery_to_ath'].median()*100:.1f}%")
        print(f"    从低点反弹: {filtered['rebound_from_low'].median()*100:.1f}%")
        print(f"    7天涨幅: {filtered['r7d'].median()*100:.1f}%")
        print(f"    14天涨幅: {filtered['r14d'].median()*100:.1f}%")
        print(f"    30天涨幅: {filtered['r30d'].median()*100:.1f}%")
        print(f"    成交量放大: {filtered['vol_surge'].median():.2f}x")
        print(f"    7天最大单日涨幅: {filtered['max_daily_gain_7d'].median()*100:.1f}%")
        print(f"    上市天数: {filtered['days_since_listing'].median():.0f} 天")
    
    # 放宽条件，看更多潜力币
    print(f"\n{'='*120}")
    print("  放宽条件：近期强势股（不一定新币）")
    print(f"{'='*120}")
    
    filtered2 = df[
        (df['current_drawdown'] >= 0.30) &
        (df['r7d'] >= 0.30) &
        (df['vol_surge'] >= 1.5) &
        (df['above_ma24'] == True) &
        (df['rebound_from_low'] >= 0.30)
    ].copy()
    filtered2['score'] = filtered2['r7d'] * 0.4 + filtered2['rebound_from_low'] * 0.3 + filtered2['vol_surge'] * 0.1
    filtered2 = filtered2.sort_values('score', ascending=False)
    
    print(f"\n  符合条件的币种: {len(filtered2)} 个")
    print(f"\n  {'排名':<4} {'币种':<16} {'市场':<6} {'上市日期':<12} {'回撤':>7} {'恢复ATH':>8} {'从低反弹':>8} {'7天涨幅':>8} {'14天涨幅':>8} {'成交量放大':>10}")
    print(f"  {'-'*4} {'-'*16} {'-'*6} {'-'*12} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    
    for rank, (_, row) in enumerate(filtered2.head(20).iterrows(), 1):
        print(f"  {rank:<4} {row['symbol']:<16} {row['market_type']:<6} {row['first_date'].strftime('%Y-%m-%d')} {row['current_drawdown']*100:>6.1f}% {row['recovery_to_ath']*100:>7.1f}% {row['rebound_from_low']*100:>7.1f}% {row['r7d']*100:>7.1f}% {row['r14d']*100:>7.1f}% {row['vol_surge']:>9.2f}x")
    
    print(f"\n{'='*120}")


if __name__ == '__main__':
    main()
