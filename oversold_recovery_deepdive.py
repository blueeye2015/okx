#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
深度研究：80%+ 回撤后的恢复特征
1. 创新高的案例有什么共同特征？
2. 没创新高的案例，最大反弹到历史最高的多少？
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
    """找所有 >=threshold 回撤的峰值事件，间隔至少 30 天"""
    df = df.copy()
    df['rolling_high'] = df['high'].expanding().max()
    df['drawdown'] = (df['rolling_high'] - df['close']) / df['rolling_high']
    severe = df[df['drawdown'] >= threshold].copy()
    if severe.empty:
        return []
    
    events = []
    current_peak_dd = severe['drawdown'].iloc[0]
    current_peak_idx = severe.index[0]
    current_start = severe.index[0]
    
    for i in range(1, len(severe)):
        idx = severe.index[i]
        prev_idx = severe.index[i-1]
        dd = severe.loc[idx, 'drawdown']
        if (idx - prev_idx).days > 5:
            events.append({
                'start': current_start,
                'end': prev_idx,
                'max_dd_date': current_peak_idx,
                'max_dd': current_peak_dd,
                'price_at_max_dd': df.loc[current_peak_idx, 'close'],
                'ath_at_max_dd': df.loc[current_peak_idx, 'rolling_high'],
            })
            current_start = idx
            current_peak_dd = dd
            current_peak_idx = idx
        else:
            if dd >= current_peak_dd:
                current_peak_dd = dd
                current_peak_idx = idx
    
    events.append({
        'start': current_start,
        'end': severe.index[-1],
        'max_dd_date': current_peak_idx,
        'max_dd': current_peak_dd,
        'price_at_max_dd': df.loc[current_peak_idx, 'close'],
        'ath_at_max_dd': df.loc[current_peak_idx, 'rolling_high'],
    })
    
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
                filtered[-1] = ev
    return filtered


def analyze_event(df, event):
    """分析单个回撤事件的恢复特征"""
    idx = df.index.get_loc(event['max_dd_date'])
    if idx >= len(df) - 10:
        return None
    
    # 后续数据
    future = df.iloc[idx+1:]
    if len(future) < 30:
        return None
    
    # 恢复信号：连续3天站上MA24
    future = future.copy()
    future['ma24'] = future['close'].rolling(window=24, min_periods=24).mean()
    future['above_ma24'] = future['close'] > future['ma24']
    future['recovery_signal'] = (
        future['above_ma24'] & 
        future['above_ma24'].shift(1).fillna(False) & 
        future['above_ma24'].shift(2).fillna(False)
    )
    
    signal_days = future[future['recovery_signal']]
    if signal_days.empty:
        return None  # 还没有恢复信号
    
    signal_idx = df.index.get_loc(signal_days.index[0])
    signal_row = df.iloc[signal_idx]
    
    # 信号日特征
    signal_date = signal_days.index[0]
    signal_price = signal_row['close']
    
    # 计算 signal 日的 pre_r20d, vol_ratio 等
    if signal_idx < 30:
        return None
    
    pre_r20d = (signal_price - df['close'].iloc[signal_idx-20]) / df['close'].iloc[signal_idx-20]
    vol_20d = df['volume'].iloc[signal_idx-20:signal_idx].mean()
    vol_ratio = signal_row['volume'] / vol_20d if vol_20d > 0 else 1.0
    
    # 后续最高点和是否创新高
    future_after_signal = df.iloc[signal_idx+1:]
    if future_after_signal.empty:
        return None
    
    future_high = future_after_signal['high'].max()
    future_high_idx = future_after_signal['high'].idxmax()
    ath_at_signal = df['high'].iloc[:signal_idx+1].max()
    
    ever_new_high = future_high > ath_at_signal
    max_recovery_pct = (future_high - event['price_at_max_dd']) / event['price_at_max_dd']
    recovery_to_ath_pct = future_high / ath_at_signal  # 反弹高点相对历史最高的比例
    
    # 信号日到最高点的天数
    days_to_high = (future_high_idx - signal_date).days if not pd.isna(future_high_idx) else np.nan
    
    return {
        'symbol': event.get('symbol', ''),
        'max_dd_date': event['max_dd_date'],
        'max_dd': event['max_dd'],
        'price_at_max_dd': event['price_at_max_dd'],
        'signal_date': signal_date,
        'signal_price': signal_price,
        'pre_r20d': pre_r20d,
        'vol_ratio': vol_ratio,
        'ath_at_signal': ath_at_signal,
        'future_high': future_high,
        'ever_new_high': ever_new_high,
        'max_recovery_pct': max_recovery_pct,
        'recovery_to_ath_pct': recovery_to_ath_pct,
        'days_to_high': days_to_high,
        'signal_drawdown': (ath_at_signal - signal_price) / ath_at_signal,
    }


def main():
    symbols = get_symbols('spot')
    logger.info(f"共 {len(symbols)} 个币")
    
    all_events = []
    
    for i, symbol in enumerate(symbols):
        if i % 50 == 0:
            logger.info(f"进度: {i}/{len(symbols)}")
        
        df = fetch_klines(symbol, 'spot')
        if df.empty or len(df) < 400:
            continue
        
        events = find_drawdown_events(df, threshold=0.80)
        for ev in events:
            ev['symbol'] = symbol
            result = analyze_event(df, ev)
            if result:
                all_events.append(result)
    
    df_events = pd.DataFrame(all_events)
    if df_events.empty:
        print("无数据")
        return
    
    print("\n" + "="*100)
    print("  80%+ 回撤后恢复深度研究")
    print("="*100)
    
    print(f"\n【样本总数】{len(df_events)} 个有恢复信号的回撤事件")
    
    new_high = df_events[df_events['ever_new_high'] == True]
    no_new_high = df_events[df_events['ever_new_high'] == False]
    
    print(f"\n  最终创历史新高: {len(new_high)} 个 ({len(new_high)/len(df_events)*100:.1f}%)")
    print(f"  未创新高: {len(no_new_high)} 个 ({len(no_new_high)/len(df_events)*100:.1f}%)")
    
    # 一、特征对比
    print(f"\n{'='*100}")
    print("  一、创新高组 vs 未创新高组 特征对比")
    print(f"{'='*100}")
    
    features = [
        ('最大回撤幅度', 'max_dd'),
        ('恢复信号日回撤', 'signal_drawdown'),
        ('信号日pre_r20d', 'pre_r20d'),
        ('信号日成交量比', 'vol_ratio'),
        ('从最低点到信号日涨幅', 'max_recovery_pct'),
        ('信号日到最高点天数', 'days_to_high'),
    ]
    
    print(f"\n  {'特征':<25} {'创新高组中位数':>15} {'未创新高组中位数':>18} {'差异':>10}")
    print(f"  {'-'*25} {'-'*15} {'-'*18} {'-'*10}")
    for label, col in features:
        if col in df_events.columns:
            nh_median = new_high[col].median() if len(new_high) > 0 else np.nan
            nnh_median = no_new_high[col].median() if len(no_new_high) > 0 else np.nan
            diff = nh_median - nnh_median if not (pd.isna(nh_median) or pd.isna(nnh_median)) else np.nan
            print(f"  {label:<25} {nh_median*100:>14.1f}% {nnh_median*100:>17.1f}% {diff*100:>+9.1f}%")
    
    # 二、未创新高组的反弹高度
    print(f"\n{'='*100}")
    print("  二、未创新高组：反弹到历史最高的多少？")
    print(f"{'='*100}")
    
    if len(no_new_high) > 0:
        print(f"\n  未创新高组反弹高度分布（相对历史最高价的最高点）:")
        print(f"    平均: {no_new_high['recovery_to_ath_pct'].mean()*100:.1f}%")
        print(f"    中位数: {no_new_high['recovery_to_ath_pct'].median()*100:.1f}%")
        print(f"    10%分位: {no_new_high['recovery_to_ath_pct'].quantile(0.10)*100:.1f}%")
        print(f"    25%分位: {no_new_high['recovery_to_ath_pct'].quantile(0.25)*100:.1f}%")
        print(f"    50%分位: {no_new_high['recovery_to_ath_pct'].quantile(0.50)*100:.1f}%")
        print(f"    75%分位: {no_new_high['recovery_to_ath_pct'].quantile(0.75)*100:.1f}%")
        print(f"    90%分位: {no_new_high['recovery_to_ath_pct'].quantile(0.90)*100:.1f}%")
        
        # 按最大回撤分桶
        print(f"\n  按最大回撤分桶看反弹高度（中位数）:")
        df_events['dd_bucket'] = pd.cut(df_events['max_dd'], bins=[0.80, 0.90, 0.95, 0.99, 1.0], labels=['80-90%', '90-95%', '95-99%', '99%+'])
        for bucket, group in df_events.groupby('dd_bucket', observed=True):
            nh = group[group['ever_new_high'] == True]
            nnh = group[group['ever_new_high'] == False]
            if len(nnh) > 0:
                print(f"    {bucket}: 未创新高组反弹到 ATH 的 {nnh['recovery_to_ath_pct'].median()*100:.1f}% (样本{len(nnh)})")
            if len(nh) > 0:
                print(f"    {bucket}: 创新高组反弹倍数中位数 {nh['max_recovery_pct'].median()*100:.0f}% (样本{len(nh)})")
    
    # 三、创新高组案例
    print(f"\n{'='*100}")
    print("  三、成功创历史新高的典型案例")
    print(f"{'='*100}")
    
    if len(new_high) > 0:
        top = new_high.nlargest(15, 'max_recovery_pct')[['symbol', 'max_dd_date', 'max_dd', 'signal_date', 'max_recovery_pct', 'days_to_high']]
        for _, row in top.iterrows():
            print(f"  {row['symbol']:<14} 最深回撤:{row['max_dd']*100:>5.1f}%  信号日:{row['signal_date'].strftime('%Y-%m-%d')}  反弹倍数:{row['max_recovery_pct']*100:>7.0f}%  到高点天数:{row['days_to_high']:>4.0f}")
    
    # 四、未创新高但反弹不错的案例
    print(f"\n{'='*100}")
    print("  四、未创新高但反弹接近 ATH 的案例（反弹到 ATH 80%+）")
    print(f"{'='*100}")
    
    close_to_ath = no_new_high[no_new_high['recovery_to_ath_pct'] >= 0.80].sort_values('recovery_to_ath_pct', ascending=False)
    print(f"  共 {len(close_to_ath)} 个案例")
    if len(close_to_ath) > 0:
        for _, row in close_to_ath.head(15).iterrows():
            print(f"  {row['symbol']:<14} 最深回撤:{row['max_dd']*100:>5.1f}%  反弹到 ATH:{row['recovery_to_ath_pct']*100:>6.1f}%  信号日:{row['signal_date'].strftime('%Y-%m-%d')}")
    
    # 五、反弹最弱的案例
    print(f"\n{'='*100}")
    print("  五、反弹最弱的案例（反弹到 ATH 不足 10%）")
    print(f"{'='*100}")
    
    weak = no_new_high[no_new_high['recovery_to_ath_pct'] < 0.10].sort_values('recovery_to_ath_pct')
    print(f"  共 {len(weak)} 个案例")
    if len(weak) > 0:
        for _, row in weak.head(15).iterrows():
            print(f"  {row['symbol']:<14} 最深回撤:{row['max_dd']*100:>5.1f}%  反弹到 ATH:{row['recovery_to_ath_pct']*100:>6.1f}%  信号日:{row['signal_date'].strftime('%Y-%m-%d')}")
    
    print(f"\n{'='*100}")


if __name__ == '__main__':
    main()
