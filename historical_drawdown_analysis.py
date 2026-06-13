#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
历史极端回撤分析
研究：
1. BTC/ETH/SOL 历史上出现当前级别回撤的次数和持续时间
2. 全市场"超跌状态"（跌80%+弱势250天）的历史频率
3. 当前这种"二线币平均跌86%"的极端状态在历史上出现过几次
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


def fetch_symbol_history(symbol: str, market_type: str = 'spot') -> pd.DataFrame:
    conn = get_db_conn()
    query = """
        SELECT open_time, open, high, low, close, volume
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


def analyze_major_coin_drawdowns(symbol: str, name: str) -> pd.DataFrame:
    """分析单个主流币的历史回撤"""
    df = fetch_symbol_history(symbol)
    if df.empty or len(df) < 500:
        logger.warning(f"{symbol} 数据不足")
        return pd.DataFrame()

    # 滚动500日高点
    df['rolling_high_500d'] = df['high'].rolling(window=500, min_periods=200).max()
    df['drawdown'] = (df['rolling_high_500d'] - df['close']) / df['rolling_high_500d']

    # 60周均线 ≈ 300日均线
    df['ma300'] = df['close'].rolling(window=300, min_periods=250).mean()
    df['below_ma300'] = df['close'] < df['ma300']

    # 滚动250天弱势天数
    df['weak_days_250'] = df['below_ma300'].rolling(window=250, min_periods=200).sum()
    df['weak_ratio'] = df['weak_days_250'] / 250

    df['symbol'] = name
    return df


def find_drawdown_periods(df: pd.DataFrame, threshold: float = 0.40) -> List[Dict]:
    """找出回撤超过threshold的连续区间"""
    df = df.copy()
    df['extreme'] = df['drawdown'] >= threshold
    df['group'] = (df['extreme'] != df['extreme'].shift()).cumsum()

    periods = []
    for g, group in df[df['extreme']].groupby('group'):
        periods.append({
            'start': group.index.min(),
            'end': group.index.max(),
            'duration_days': (group.index.max() - group.index.min()).days + 1,
            'max_drawdown': group['drawdown'].max(),
            'avg_drawdown': group['drawdown'].mean(),
            'end_price': group['close'].iloc[-1],
        })
    return periods


def monthly_oversold_snapshot(market_type: str = 'spot', sample_months: List[str] = None):
    """
    按月做全市场快照：统计当月第一个交易日有多少币满足超跌条件
    """
    conn = get_db_conn()
    # 获取所有symbol
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT symbol FROM binance_daily_klines WHERE market_type = %s ORDER BY symbol", (market_type,))
    symbols = [row[0] for row in cur.fetchall()]
    cur.close()

    if sample_months is None:
        # 自动选取每年1月、4月、7月、10月的第一天
        sample_months = []
        for year in range(2018, 2027):
            for month in [1, 4, 7, 10]:
                sample_months.append(f"{year}-{month:02d}-01")

    results = []
    for date_str in sample_months:
        snapshot_date = pd.Timestamp(date_str)
        # 找该日期或之前的最近交易日
        cur = conn.cursor()
        cur.execute(
            "SELECT MAX(open_time)::date FROM binance_daily_klines WHERE market_type = %s AND open_time <= %s",
            (market_type, snapshot_date)
        )
        actual_date = cur.fetchone()[0]
        cur.close()
        if actual_date is None:
            continue

        actual_ts = pd.Timestamp(actual_date)
        min_history_date = actual_ts - pd.Timedelta(days=900)

        oversold_count = 0
        total_valid = 0
        mega_oversold = 0
        large_oversold = 0
        small_oversold = 0
        mega_total = 0
        large_total = 0
        small_total = 0

        mega_caps = {'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
                     'ADAUSDT', 'TRXUSDT', 'AVAXUSDT', 'TONUSDT', 'SHIBUSDT', 'DOTUSDT',
                     'LINKUSDT', 'NEARUSDT', 'MATICUSDT'}
        large_caps = {'LTCUSDT', 'BCHUSDT', 'UNIUSDT', 'ATOMUSDT', 'ETCUSDT', 'FILUSDT',
                      'APTUSDT', 'IMXUSDT', 'OPUSDT', 'ARBUSDT', 'SUIUSDT', 'SEIUSDT',
                      'INJUSDT', 'GRTUSDT', 'STXUSDT', 'RUNEUSDT', 'ALGOUSDT', 'VETUSDT',
                      'ICPUSDT', 'MANAUSDT', 'SANDUSDT', 'AXSUSDT', 'THETAUSDT', 'FTMUSDT',
                      'EGLDUSDT', 'XTZUSDT', 'EOSUSDT', 'AAVEUSDT', 'FLOWUSDT', 'NEOUSDT',
                      'QNTUSDT', 'MKRUSDT', 'GALAUSDT', 'CHZUSDT', 'COMPUSDT', 'CRVUSDT',
                      'LDOUSDT', 'SNXUSDT', 'ENJUSDT', 'ZECUSDT', 'DASHUSDT', 'XMRUSDT',
                      'KSMUSDT', 'YFIUSDT', 'ZILUSDT', 'WAVESUSDT', 'CELOUSDT', 'ROSEUSDT',
                      'MINAUSDT', 'KAVAUSDT', 'SKLUSDT', 'RVNUSDT', 'SUSHIUSDT', '1INCHUSDT',
                      'BATUSDT', 'STORJUSDT', 'ANKRUSDT', 'CHRUSDT', 'COTIUSDT', 'DGBUSDT',
                      'IOSTUSDT', 'TFUELUSDT', 'ONEUSDT', 'ONGUSDT', 'WINUSDT', 'DENTUSDT',
                      'HOTUSDT', 'SCUSDT', 'IOTXUSDT', 'BLZUSDT', 'NKNUSDT', 'RSRUSDT',
                      'BANDUSDT', 'RLCUSDT', 'OCEANUSDT', 'SFPUSDT', 'CVCUSDT', 'STMXUSDT',
                      'MDTUSDT', 'DODOUSDT', 'PONDUSDT', 'ALICEUSDT', 'FARMUSDT', 'BALUSDT',
                      'PERPUSDT', 'TRBUSDT', 'BELUSDT', 'FLMUSDT', 'HARDUSDT', 'TOMOUSDT',
                      'DIAUSDT', 'REEFUSDT', 'AKROUSDT', 'SUNUSDT', 'NBSUSDT', 'LITUSDT',
                      'PSGUSDT', 'JUVUSDT', 'ASRUSDT', 'ATMUSDT', 'ACMUSDT', 'BARUSDT',
                      'OGUSDT', 'CITYUSDT', 'PORTOUSDT', 'LAZIOUSDT', 'SANTOSUSDT', 'ALPINEUSDT',
                      'TUSDT', 'PROMUSDT', 'QIUSDT', 'API3USDT', 'CTKUSDT', 'LPTUSDT',
                      'AUDIOUSDT', 'RAYUSDT', 'C98USDT', 'DYDXUSDT', 'ENSUSDT', 'PEOPLEUSDT',
                      'JOEUSDT', 'MASKUSDT', 'ETHWUSDT', 'ASTRUSDT', 'PHBUSDT', 'GLMRUSDT',
                      'ACHUSDT', 'MAGICUSDT', 'HIGHUSDT', 'LOKAUSDT', 'SCRTUSDT', 'WOOUSDT',
                      'KNCUSDT', 'STGUSDT', 'CVXUSDT', 'FXSUSDT', 'HOOKUSDT', 'EDUUSDT',
                      'MAVUSDT', 'PENDLEUSDT', 'ARKMUSDT', 'WLDUSDT', 'CYBERUSDT', 'TIAUSDT',
                      'SUIUSDT', 'MEMEUSDT', 'ORDIUSDT', 'BLURUSDT', 'JTOUSDT', 'ACEUSDT',
                      'NFPUSDT', 'AIUSDT', 'XAIUSDT', 'MANTAUSDT', 'ALTUSDT', 'JUPUSDT',
                      'PYTHUSDT', 'DYMUSDT', 'PIXELUSDT', 'STRKUSDT', 'PORTALUSDT', 'WUSDT',
                      'SAGAUSDT', 'TAOUSDT', 'ENAUSDT', 'WIFUSDT', 'BOMEUSDT', 'ETHFIUSDT',
                      'METISUSDT', 'AEVOUSDT', 'BBUSDT', 'NOTUSDT', 'IOUSDT', 'ZKUSDT',
                      'LISTAUSDT', 'ZROUSDT', 'RENDERUSDT', 'MEWUSDT', 'BONKUSDT', 'FLOKIUSDT',
                      'PEPEUSDT', 'OMUSDT', 'POLUSDT'}

        for symbol in symbols:
            # 拉取该symbol到snapshot日期的数据
            cur = conn.cursor()
            cur.execute(
                """SELECT open_time::date, high, close FROM binance_daily_klines
                   WHERE symbol = %s AND market_type = %s AND open_time >= %s AND open_time <= %s
                   ORDER BY open_time ASC""",
                (symbol, market_type, min_history_date, actual_ts + pd.Timedelta(days=1))
            )
            rows = cur.fetchall()
            cur.close()

            if len(rows) < 400:
                continue

            df_sym = pd.DataFrame(rows, columns=['date', 'high', 'close'])
            df_sym['date'] = pd.to_datetime(df_sym['date'])
            df_sym = df_sym.set_index('date')

            # 计算500日高点回撤
            recent_500 = df_sym.tail(500)
            if len(recent_500) < 400:
                continue

            hist_high = recent_500['high'].max()
            current_price = df_sym['close'].iloc[-1]
            drawdown = (hist_high - current_price) / hist_high

            # 计算60周均线（300日）弱势天数
            df_sym['ma300'] = df_sym['close'].rolling(window=300, min_periods=250).mean()
            bear_window = df_sym.tail(250).dropna(subset=['ma300'])
            if len(bear_window) < 200:
                continue

            below_ma_count = (bear_window['close'] < bear_window['ma300']).sum()
            below_ratio = below_ma_count / len(bear_window)

            total_valid += 1
            is_oversold = (drawdown >= 0.80) and (below_ratio >= 0.80)
            if is_oversold:
                oversold_count += 1

            if symbol in mega_caps:
                mega_total += 1
                if is_oversold: mega_oversold += 1
            elif symbol in large_caps:
                large_total += 1
                if is_oversold: large_oversold += 1
            else:
                small_total += 1
                if is_oversold: small_oversold += 1

        results.append({
            'date': actual_date,
            'total_valid': total_valid,
            'oversold_count': oversold_count,
            'oversold_ratio': oversold_count / total_valid if total_valid > 0 else 0,
            'mega_ratio': mega_oversold / mega_total if mega_total > 0 else 0,
            'large_ratio': large_oversold / large_total if large_total > 0 else 0,
            'small_ratio': small_oversold / small_total if small_total > 0 else 0,
        })
        ratio_str = f"{oversold_count/total_valid*100:.1f}%" if total_valid > 0 else "N/A"
        logger.info(f"{actual_date}: 有效{total_valid}个, 超跌{oversold_count}个 ({ratio_str})")

    conn.close()
    return pd.DataFrame(results)


def print_major_coin_report(btc_df: pd.DataFrame, eth_df: pd.DataFrame, sol_df: pd.DataFrame):
    print("\n" + "="*90)
    print("主流币历史极端回撤报告")
    print("="*90)

    for df, name in [(btc_df, 'BTC'), (eth_df, 'ETH'), (sol_df, 'SOL')]:
        if df.empty:
            continue
        print(f"\n{'='*90}")
        print(f"【{name}】")
        print(f"{'='*90}")
        current_dd = df['drawdown'].iloc[-1] if not df.empty else 0
        print(f"当前回撤: {current_dd*100:.1f}%")

        for threshold in [0.40, 0.50, 0.60, 0.70, 0.80]:
            periods = find_drawdown_periods(df, threshold)
            if periods:
                print(f"\n  回撤≥{threshold*100:.0f}% 的历史时期: {len(periods)} 次")
                for i, p in enumerate(periods[-5:], 1):  # 只显示最近5次
                    print(f"    {i}. {p['start'].strftime('%Y-%m-%d')} ~ {p['end'].strftime('%Y-%m-%d')} "
                          f"(持续{p['duration_days']}天, 最大回撤{p['max_drawdown']*100:.1f}%)")
            else:
                print(f"\n  回撤≥{threshold*100:.0f}%: 历史上从未出现")


def print_market_history_report(history_df: pd.DataFrame):
    print("\n" + "="*90)
    print("全市场超跌状态历史频率（每季度快照）")
    print("="*90)
    print(f"{'日期':<12} {'有效币数':>8} {'超跌数':>8} {'总占比':>8} {'大币占比':>8} {'二线占比':>8} {'小币占比':>8}")
    print("-"*90)

    for _, row in history_df.iterrows():
        print(f"{row['date']:<12} {row['total_valid']:>8} {row['oversold_count']:>8} "
              f"{row['oversold_ratio']*100:>7.1f}% {row['mega_ratio']*100:>7.1f}% "
              f"{row['large_ratio']*100:>7.1f}% {row['small_ratio']*100:>7.1f}%")

    # 找出历史极端时期
    print(f"\n{'='*90}")
    print("历史极端时期（全市场超跌占比最高的时期）")
    print(f"{'='*90}")
    top_extreme = history_df.nlargest(10, 'oversold_ratio')
    for _, row in top_extreme.iterrows():
        print(f"  {row['date']}: 全市场 {row['oversold_ratio']*100:.1f}% 的币满足超跌 "
              f"(二线{row['large_ratio']*100:.1f}%, 小币{row['small_ratio']*100:.1f}%)")

    # 当前水平定位
    current = history_df.iloc[-1]
    all_ratios = history_df['oversold_ratio'].sort_values(ascending=False).reset_index(drop=True)
    current_rank = (all_ratios > current['oversold_ratio']).sum() + 1

    print(f"\n{'='*90}")
    print("当前市场定位")
    print(f"{'='*90}")
    print(f"  当前日期: {current['date']}")
    print(f"  全市场超跌占比: {current['oversold_ratio']*100:.1f}%")
    print(f"  在历史 {len(history_df)} 个季度快照中排名: 第 {current_rank} 极端")
    print(f"  二线大币超跌占比: {current['large_ratio']*100:.1f}%")
    print(f"  小币超跌占比: {current['small_ratio']*100:.1f}%")

    if current_rank <= 3:
        print(f"  ⚠️ 这是历史上最极端的熊市状态之一")
    elif current_rank <= 10:
        print(f"  ⚠️ 这是历史上较为极端的熊市状态")
    else:
        print(f"  当前处于中等偏极端水平")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', choices=['spot', 'swap'], default='spot')
    args = parser.parse_args()

    # 1. 主流币回撤历史
    logger.info("分析 BTC/ETH/SOL 历史回撤...")
    btc_df = analyze_major_coin_drawdowns('BTCUSDT', 'BTC')
    eth_df = analyze_major_coin_drawdowns('ETHUSDT', 'ETH')
    sol_df = analyze_major_coin_drawdowns('SOLUSDT', 'SOL')
    print_major_coin_report(btc_df, eth_df, sol_df)

    # 2. 全市场历史快照（关键日期）
    # 选取历史上已知的熊市底部附近 + 每季度
    key_dates = []
    for year in range(2018, 2027):
        for month in [1, 4, 7, 10]:
            key_dates.append(f"{year}-{month:02d}-01")

    # 额外增加已知的极端日期
    extra_dates = [
        '2018-12-15',  # 2018熊市底
        '2019-01-01',
        '2020-03-15',  # 312暴跌
        '2021-07-20',  # 519后低点
        '2022-06-18',  # LUNA崩盘后
        '2022-11-10',  # FTX崩盘
        '2022-12-15',  # 2022熊市底
    ]
    key_dates = sorted(list(set(key_dates + extra_dates)))

    logger.info("开始全市场历史快照分析...")
    history_df = monthly_oversold_snapshot(args.market, key_dates)
    print_market_history_report(history_df)
