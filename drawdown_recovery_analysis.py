#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
历史大回撤（≥80%）事件分析
统计 BTC、ETH 和山寨币各自经历 ≥80% 回撤的次数、恢复情况、最终结局
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
        SELECT open_time, open, high, low, close, volume
        FROM binance_daily_klines
        WHERE symbol = %s AND market_type = %s AND open_time >= '2017-01-01'
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


def find_drawdown_events(df, threshold=0.80):
    """
    找出一个币历史上所有 ≥threshold 回撤的"峰值事件"。
    事件定义：drawdown 的局部最大值（峰值），间隔至少 30 天。
    """
    df = df.copy()
    df['rolling_high'] = df['high'].expanding().max()
    df['drawdown'] = (df['rolling_high'] - df['close']) / df['rolling_high']
    
    # 只保留 ≥threshold 的日期
    severe = df[df['drawdown'] >= threshold].copy()
    if severe.empty:
        return []
    
    # 找局部最大值：连续序列中的最深点
    events = []
    current_peak_dd = severe['drawdown'].iloc[0]
    current_peak_idx = severe.index[0]
    current_start = severe.index[0]
    
    for i in range(1, len(severe)):
        idx = severe.index[i]
        prev_idx = severe.index[i-1]
        dd = severe.loc[idx, 'drawdown']
        
        # 如果日期不连续（间隔 > 5 天），结束当前序列
        if (idx - prev_idx).days > 5:
            events.append({
                'start': current_start,
                'end': prev_idx,
                'max_dd_date': current_peak_idx,
                'max_dd': current_peak_dd,
                'price_at_max_dd': df.loc[current_peak_idx, 'close'],
            })
            current_start = idx
            current_peak_dd = dd
            current_peak_idx = idx
        else:
            # 连续日期，更新峰值
            if dd >= current_peak_dd:
                current_peak_dd = dd
                current_peak_idx = idx
    
    # 添加最后一个
    events.append({
        'start': current_start,
        'end': severe.index[-1],
        'max_dd_date': current_peak_idx,
        'max_dd': current_peak_dd,
        'price_at_max_dd': df.loc[current_peak_idx, 'close'],
    })
    
    # 去重：相邻事件间隔至少 30 天（保留更深的）
    filtered = []
    for ev in events:
        if not filtered:
            filtered.append(ev)
        else:
            last = filtered[-1]
            days_gap = (ev['max_dd_date'] - last['max_dd_date']).days
            if days_gap >= 30:
                filtered.append(ev)
            elif ev['max_dd'] > last['max_dd']:
                # 替换为更深的
                filtered[-1] = ev
    
    return filtered


def analyze_recovery(df, event, max_dd_date):
    """分析某次最大回撤点之后的恢复情况"""
    idx = df.index.get_loc(max_dd_date)
    price = df.loc[max_dd_date, 'close']
    results = {}

    horizons = [30, 60, 90, 180, 365]
    for h in horizons:
        if idx + h < len(df):
            fut_price = df['close'].iloc[idx + h]
            results[f'return_{h}d'] = (fut_price - price) / price
            results[f'recovered_{h}d'] = fut_price >= price
        else:
            results[f'return_{h}d'] = np.nan
            results[f'recovered_{h}d'] = np.nan

    # 是否历史新高
    if idx + 1 < len(df):
        future_high = df['high'].iloc[idx+1:].max()
        results['ever_new_high'] = future_high > df['high'].iloc[:idx+1].max()
        results['max_recovery_pct'] = (future_high - price) / price
    else:
        results['ever_new_high'] = False
        results['max_recovery_pct'] = np.nan

    # 最终状态（数据最后一天）
    last_price = df['close'].iloc[-1]
    results['final_return'] = (last_price - price) / price
    results['final_recovered'] = last_price >= price

    return results


def main():
    symbols = get_symbols('spot')
    logger.info(f"共 {len(symbols)} 个币")

    all_events = []

    for i, symbol in enumerate(symbols):
        if i % 50 == 0:
            logger.info(f"进度: {i}/{len(symbols)}")

        df = fetch_klines(symbol, 'spot')
        if df.empty or len(df) < 200:
            continue

        events = find_drawdown_events(df, threshold=0.80)
        for ev in events:
            recovery = analyze_recovery(df, ev, ev['max_dd_date'])
            ev.update(recovery)
            ev['symbol'] = symbol
            ev['data_end'] = df.index[-1]
            all_events.append(ev)

    df_events = pd.DataFrame(all_events)
    if df_events.empty:
        print("无数据")
        return

    # 分类
    btc_eth = df_events[df_events['symbol'].isin(['BTCUSDT', 'ETHUSDT'])]
    altcoins = df_events[~df_events['symbol'].isin(['BTCUSDT', 'ETHUSDT'])]

    print("\n" + "="*100)
    print("  历史 ≥80% 回撤事件全景分析")
    print("="*100)

    # ========== BTC/ETH ==========
    print(f"\n【一、比特币 & 以太坊】")
    print(f"  经历 ≥80% 回撤事件总数: {len(btc_eth)} 次")
    for _, row in btc_eth.iterrows():
        print(f"\n  {row['symbol']:<10} 最大回撤日: {row['max_dd_date'].strftime('%Y-%m-%d')}")
        print(f"             最大回撤幅度: {row['max_dd']*100:.1f}%")
        print(f"             进入回撤区间: {row['start'].strftime('%Y-%m-%d')} → {row['end'].strftime('%Y-%m-%d')}")
        print(f"             30天回报: {row.get('return_30d', np.nan)*100:.1f}%" if not pd.isna(row.get('return_30d')) else "             30天回报: N/A")
        print(f"             90天回报: {row.get('return_90d', np.nan)*100:.1f}%" if not pd.isna(row.get('return_90d')) else "             90天回报: N/A")
        print(f"             1年回报: {row.get('return_365d', np.nan)*100:.1f}%" if not pd.isna(row.get('return_365d')) else "             1年回报: N/A")
        print(f"             是否曾创新高: {'✅ 是' if row.get('ever_new_high') else '❌ 否'}")
        print(f"             最大反弹幅度: {row.get('max_recovery_pct', np.nan)*100:.1f}%" if not pd.isna(row.get('max_recovery_pct')) else "             最大反弹幅度: N/A")
        print(f"             截至最新数据回报: {row.get('final_return', np.nan)*100:.1f}%" if not pd.isna(row.get('final_return')) else "             截至最新数据回报: N/A")

    # ========== 山寨币整体统计 ==========
    print(f"\n{'='*100}")
    print("【二、山寨币整体统计】")
    print(f"{'='*100}")
    print(f"  涉及山寨币数量: {altcoins['symbol'].nunique()} 个")
    print(f"  ≥80% 回撤事件总数: {len(altcoins)} 次")
    print(f"  平均每币经历次数: {len(altcoins) / altcoins['symbol'].nunique():.1f} 次")

    # 按币统计次数
    coin_counts = altcoins.groupby('symbol').size().reset_index(name='event_count')
    print(f"\n  经历次数分布:")
    print(f"    1 次: {(coin_counts['event_count'] == 1).sum()} 个币")
    print(f"    2 次: {(coin_counts['event_count'] == 2).sum()} 个币")
    print(f"    3 次: {(coin_counts['event_count'] >= 3).sum()} 个币")

    # 恢复情况
    print(f"\n  恢复情况（从最大回撤点起算）:")
    for h in [30, 60, 90, 180, 365]:
        col = f'return_{h}d'
        valid = altcoins[col].dropna()
        if len(valid) > 0:
            recovered = (valid > 0).mean()
            print(f"    {h}天 内价格回升: {recovered*100:.1f}% 的事件  (中位数回报: {valid.median()*100:.1f}%)")

    # 是否曾创新高
    valid_new_high = altcoins['ever_new_high'].dropna()
    if len(valid_new_high) > 0:
        print(f"\n  是否曾创历史新高:")
        print(f"    曾创新高: {(valid_new_high == True).sum()} 次 ({(valid_new_high == True).mean()*100:.1f}%)")
        print(f"    从未创新高: {(valid_new_high == False).sum()} 次 ({(valid_new_high == False).mean()*100:.1f}%)")

    # 最终状态
    valid_final = altcoins['final_return'].dropna()
    if len(valid_final) > 0:
        print(f"\n  截至最新数据的最终状态（相对最大回撤点）:")
        print(f"    仍低于回撤点: {(valid_final < 0).sum()} 次 ({(valid_final < 0).mean()*100:.1f}%)")
        print(f"    已恢复至回撤点以上: {(valid_final > 0).sum()} 次 ({(valid_final > 0).mean()*100:.1f}%)")
        print(f"    中位数最终回报: {valid_final.median()*100:.1f}%")

    # 按最大回撤幅度分桶
    print(f"\n  最大回撤幅度分布:")
    altcoins['dd_bucket'] = pd.cut(altcoins['max_dd'], bins=[0.80, 0.90, 0.95, 0.99, 1.0], labels=['80-90%', '90-95%', '95-99%', '99%+'])
    for bucket, group in altcoins.groupby('dd_bucket', observed=True):
        if len(group) > 0:
            recovery_rate = group['ever_new_high'].dropna().mean() if 'ever_new_high' in group else 0
            print(f"    {bucket}: {len(group)} 次, 曾创新高率: {recovery_rate*100:.1f}%")

    # 最严重的案例
    print(f"\n{'='*100}")
    print("【三、最惨烈的回撤案例 Top 20】")
    print(f"{'='*100}")
    worst = altcoins.nlargest(20, 'max_dd')[['symbol', 'max_dd_date', 'max_dd', 'return_30d', 'return_90d', 'ever_new_high', 'final_return']]
    for _, row in worst.iterrows():
        nh = '✅' if row.get('ever_new_high') else '❌'
        print(f"  {row['symbol']:<14} {row['max_dd_date'].strftime('%Y-%m-%d')}  回撤:{row['max_dd']*100:>6.1f}%  30d:{row.get('return_30d', np.nan)*100:>+7.1f}%  90d:{row.get('return_90d', np.nan)*100:>+7.1f}%  新高:{nh}  最终:{row.get('final_return', np.nan)*100:>+7.1f}%")

    # 成功恢复的案例
    print(f"\n{'='*100}")
    print("【四、成功创历史新高的山寨币案例】")
    print(f"{'='*100}")
    recovered = altcoins[altcoins['ever_new_high'] == True].sort_values('max_dd', ascending=False)
    print(f"  共 {len(recovered)} 个事件来自 {recovered['symbol'].nunique()} 个不同币种")
    for _, row in recovered.head(15).iterrows():
        print(f"  {row['symbol']:<14} {row['max_dd_date'].strftime('%Y-%m-%d')}  回撤:{row['max_dd']*100:>5.1f}%  最大反弹:{row.get('max_recovery_pct', np.nan)*100:>+7.1f}%")

    # 典型币种的多次回撤
    print(f"\n{'='*100}")
    print("【五、典型币种多次回撤记录】")
    print(f"{'='*100}")
    multi_event_coins = coin_counts[coin_counts['event_count'] >= 2]['symbol'].tolist()
    for symbol in multi_event_coins[:10]:
        coin_events = altcoins[altcoins['symbol'] == symbol].sort_values('max_dd_date')
        print(f"\n  {symbol} ({len(coin_events)} 次):")
        for _, row in coin_events.iterrows():
            nh = '✅新高' if row.get('ever_new_high') else '❌未新高'
            print(f"    {row['max_dd_date'].strftime('%Y-%m-%d')}  回撤:{row['max_dd']*100:>5.1f}%  最终:{row.get('final_return', np.nan)*100:>+7.1f}%  {nh}")

    print(f"\n{'='*100}")


if __name__ == '__main__':
    main()
