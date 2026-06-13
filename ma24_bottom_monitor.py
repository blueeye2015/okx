#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
MA24 底部突破实时监控系统

用法:
  python ma24_bottom_monitor.py --market swap,spot
  python ma24_bottom_monitor.py --market swap --telegram

环境变量:
  DB_DSN: PostgreSQL 连接串 (默认 postgresql://postgres:12@127.0.0.1:5432/market_data)
  TELEGRAM_BOT_TOKEN: Telegram Bot Token
  TELEGRAM_CHAT_ID: Telegram Chat ID
"""

import os
import sys
import json
import logging
import argparse
import urllib.request
import urllib.parse
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

import ma24_bottom_test_study as study

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('ma24_monitor')

DB_DSN = os.getenv('DB_DSN', 'postgresql://postgres:12@127.0.0.1:5432/market_data')
LARGE_CAPS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
              'ADAUSDT', 'TRXUSDT', 'LINKUSDT', 'AVAXUSDT', 'SUIUSDT', 'TONUSDT']


def get_db_conn():
    return psycopg2.connect(DB_DSN)


def init_tables():
    """初始化监控表"""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ma24_bottom_monitor (
                    id SERIAL PRIMARY KEY,
                    scan_date DATE NOT NULL,
                    scan_time TIMESTAMP NOT NULL DEFAULT NOW(),
                    market_type VARCHAR(10) NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    price NUMERIC,
                    ma24 NUMERIC,
                    dist_ma24 NUMERIC,
                    volume_ratio NUMERIC,
                    below_ratio NUMERIC,
                    cross_up_count_30d INT,
                    days_since_last_cross INT,
                    window_drawdown NUMERIC,
                    recent_drawdown NUMERIC,
                    signal_grade VARCHAR(10),
                    stage VARCHAR(50),
                    is_new_signal BOOLEAN DEFAULT FALSE,
                    UNIQUE(scan_date, market_type, symbol, signal_grade)
                );
                CREATE INDEX IF NOT EXISTS idx_ma24_monitor_date ON ma24_bottom_monitor(scan_date);
                CREATE INDEX IF NOT EXISTS idx_ma24_monitor_grade ON ma24_bottom_monitor(signal_grade);
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ma24_market_breadth (
                    id SERIAL PRIMARY KEY,
                    scan_date DATE NOT NULL,
                    market_type VARCHAR(10) NOT NULL,
                    total_symbols INT,
                    above_ma24_count INT,
                    above_ma24_ratio NUMERIC,
                    deep_correction_count INT,
                    deep_correction_ratio NUMERIC,
                    median_drawdown NUMERIC,
                    btc_stage VARCHAR(50),
                    eth_stage VARCHAR(50),
                    UNIQUE(scan_date, market_type)
                );
            """)
        conn.commit()
        logger.info("数据库表初始化完成")
    finally:
        conn.close()


def analyze_symbol(symbol: str, market_type: str) -> Optional[Dict]:
    """分析单个币的当前状态"""
    df = study.fetch_klines(symbol, market_type)
    if len(df) < 60:
        return None

    df['ma24'] = df['close'].rolling(window=24, min_periods=20).mean()
    df['above_ma24'] = df['close'] > df['ma24']
    df['cross_up'] = (~df['above_ma24'].shift(1).fillna(False)) & df['above_ma24']
    df['cross_down'] = df['above_ma24'].shift(1).fillna(False) & (~df['above_ma24'])
    df['dist_ma24'] = (df['close'] - df['ma24']) / df['ma24']
    df['volume_ma20'] = df['quote_volume'].rolling(window=20, min_periods=15).mean()
    df['volume_ratio'] = df['quote_volume'] / df['volume_ma20']
    df = df.dropna(subset=['ma24', 'volume_ma20'])
    if len(df) < 31:
        return None

    latest = df.iloc[-1]
    window = df.iloc[-31:-1]

    below_ratio = (~window['above_ma24']).sum() / len(window)
    cross_up_count_30d = window['cross_up'].sum()
    window_drawdown = (window['low'].min() - window['high'].max()) / window['high'].max()
    recent_below = (~window['above_ma24'].iloc[-5:]).any()

    cross_ups = df[df['cross_up']]
    days_since_last_cross = -1 if cross_ups.empty else (df.index[-1] - cross_ups.index[-1]).days

    recent_high = df.iloc[-30:]['high'].max()
    recent_low = df.iloc[-30:]['low'].min()
    recent_drawdown = (recent_low - recent_high) / recent_high

    above = latest['above_ma24']
    dist = latest['dist_ma24']
    vr = latest['volume_ratio']
    is_cross_up_today = latest['cross_up']

    # 阶段判断
    if above:
        if recent_below and below_ratio >= 0.5 and cross_up_count_30d >= 2:
            stage = 'signal_first_breakout' if is_cross_up_today else 'signal_continuation'
        else:
            stage = 'above_ma24'
    else:
        if cross_up_count_30d >= 2 and days_since_last_cross <= 10 and dist > -0.05:
            stage = 'post_breakout_pullback'
        elif recent_drawdown <= -0.30:
            stage = 'deep_correction'
        elif below_ratio >= 0.6:
            stage = 'bottoming_no_signal'
        else:
            stage = 'downtrend'

    return {
        'symbol': symbol,
        'market_type': market_type,
        'date': df.index[-1],
        'price': latest['close'],
        'ma24': latest['ma24'],
        'dist_ma24': dist,
        'above_ma24': above,
        'volume_ratio': vr,
        'below_ratio': below_ratio,
        'cross_up_count_30d': cross_up_count_30d,
        'days_since_last_cross': days_since_last_cross,
        'window_drawdown': window_drawdown,
        'recent_drawdown': recent_drawdown,
        'stage': stage,
    }


def scan_market(market_type: str) -> pd.DataFrame:
    """扫描整个市场"""
    symbols = study.get_symbols(market_type)
    rows = []
    for i, sym in enumerate(symbols, 1):
        try:
            row = analyze_symbol(sym, market_type)
            if row:
                rows.append(row)
            if i % 100 == 0:
                logger.info(f"[{market_type}] 已扫描 {i}/{len(symbols)}")
        except Exception as e:
            logger.debug(f"[{sym}] 分析失败: {e}")
    return pd.DataFrame(rows)


def classify_signals(df: pd.DataFrame) -> pd.DataFrame:
    """给每个币种打上信号等级"""
    def grade(row):
        stage = row['stage']
        vr = row['volume_ratio']
        dist = row['dist_ma24']
        wd = row['window_drawdown']

        if stage == 'signal_first_breakout':
            if vr > 1.5 and dist < 0.03 and wd < -0.20:
                return 'A'
            elif vr > 1.0 and dist < 0.05:
                return 'B'
            else:
                return 'C'
        elif stage == 'signal_continuation':
            if 1.5 <= vr <= 5.0 and dist < 0.05:
                return 'B'
            else:
                return 'C'
        elif stage == 'post_breakout_pullback':
            if vr > 1.2 and -0.05 < dist < 0.0:
                return 'C'
            else:
                return 'D'
        elif row['above_ma24']:
            return 'D'
        else:
            return None

    df['signal_grade'] = df.apply(grade, axis=1)
    return df


def compare_with_yesterday(df: pd.DataFrame, market_type: str, scan_date: date) -> pd.DataFrame:
    """对比昨日信号，标记新出现/持续/消失"""
    conn = get_db_conn()
    try:
        yesterday = scan_date - timedelta(days=1)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, signal_grade, stage
                FROM ma24_bottom_monitor
                WHERE scan_date = %s AND market_type = %s AND signal_grade IN ('A','B','C')
            """, (yesterday, market_type))
            yesterday_signals = {(r[0], r[1]): r[2] for r in cur.fetchall()}
    finally:
        conn.close()

    def is_new(row):
        if pd.isna(row['signal_grade']) or row['signal_grade'] not in ['A', 'B', 'C']:
            return False
        return (row['symbol'], row['signal_grade']) not in yesterday_signals

    df['is_new_signal'] = df.apply(is_new, axis=1)
    return df


def save_signals(df: pd.DataFrame, market_type: str, scan_date: date):
    """保存信号到数据库"""
    if df.empty:
        return

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            rows = []
            for _, r in df.iterrows():
                rows.append((
                    scan_date, datetime.utcnow(), market_type, r['symbol'],
                    float(r['price']), float(r['ma24']), float(r['dist_ma24']),
                    float(r['volume_ratio']), float(r['below_ratio']),
                    int(r['cross_up_count_30d']), int(r['days_since_last_cross']),
                    float(r['window_drawdown']), float(r['recent_drawdown']),
                    r['signal_grade'] if pd.notna(r['signal_grade']) else None,
                    r['stage'], bool(r['is_new_signal'])
                ))
            execute_values(cur, """
                INSERT INTO ma24_bottom_monitor (
                    scan_date, scan_time, market_type, symbol, price, ma24,
                    dist_ma24, volume_ratio, below_ratio, cross_up_count_30d,
                    days_since_last_cross, window_drawdown, recent_drawdown,
                    signal_grade, stage, is_new_signal
                ) VALUES %s
                ON CONFLICT (scan_date, market_type, symbol, signal_grade)
                DO UPDATE SET
                    scan_time = EXCLUDED.scan_time,
                    price = EXCLUDED.price,
                    dist_ma24 = EXCLUDED.dist_ma24,
                    volume_ratio = EXCLUDED.volume_ratio,
                    stage = EXCLUDED.stage,
                    is_new_signal = EXCLUDED.is_new_signal
            """, rows)
        conn.commit()
        logger.info(f"[{market_type}] 已保存 {len(rows)} 条记录")
    finally:
        conn.close()


def save_breadth(df: pd.DataFrame, market_type: str, scan_date: date, large_cap_stages: Dict):
    """保存市场宽度指标"""
    total = len(df)
    above = int(df['above_ma24'].sum())
    deep = int((df['recent_drawdown'] <= -0.30).sum())
    median_dd = float(df['recent_drawdown'].median())

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ma24_market_breadth (
                    scan_date, market_type, total_symbols, above_ma24_count,
                    above_ma24_ratio, deep_correction_count, deep_correction_ratio,
                    median_drawdown, btc_stage, eth_stage
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (scan_date, market_type) DO UPDATE SET
                    total_symbols = EXCLUDED.total_symbols,
                    above_ma24_count = EXCLUDED.above_ma24_count,
                    above_ma24_ratio = EXCLUDED.above_ma24_ratio,
                    deep_correction_count = EXCLUDED.deep_correction_count,
                    deep_correction_ratio = EXCLUDED.deep_correction_ratio,
                    median_drawdown = EXCLUDED.median_drawdown,
                    btc_stage = EXCLUDED.btc_stage,
                    eth_stage = EXCLUDED.eth_stage
            """, (
                scan_date, market_type, total, above,
                above / total if total else 0, deep, deep / total if total else 0,
                median_dd, large_cap_stages.get('BTCUSDT'), large_cap_stages.get('ETHUSDT')
            ))
        conn.commit()
    finally:
        conn.close()


def send_telegram(message: str):
    """发送 Telegram 通知"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        logger.warning("未配置 TELEGRAM_BOT_TOKEN/CHAT_ID，跳过推送")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown',
        'disable_web_page_preview': 'true'
    }).encode()
    try:
        req = urllib.request.Request(url, data=data, method='POST')
        urllib.request.urlopen(req, timeout=20)
        logger.info("Telegram 推送完成")
    except Exception as e:
        logger.error(f"Telegram 推送失败: {e}")


def build_alert_message(scan_date: date, results: Dict) -> str:
    """构建 Telegram 推送消息"""
    lines = [f"📊 *MA24 底部监控 {scan_date}*", ""]

    for mt in results:
        df = results[mt]['df']
        if df.empty:
            continue

        above_ratio = df['above_ma24'].mean() * 100
        deep_ratio = (df['recent_drawdown'] <= -0.30).mean() * 100
        lines.append(f"*{mt.upper()}* 在MA24上方 {above_ratio:.1f}% | 深调 {deep_ratio:.1f}%")

        a_signals = df[(df['signal_grade'] == 'A') & (df['is_new_signal'])]
        if not a_signals.empty:
            lines.append("🔴 *A 级新信号*")
            for _, r in a_signals.iterrows():
                lines.append(
                    f"  `${r['symbol']}` 价格 {r['price']:.6g} | "
                    f"vr {r['volume_ratio']:.2f} | dist {r['dist_ma24']*100:.2f}% | "
                    f"回撤 {r['recent_drawdown']*100:.1f}%"
                )
        else:
            lines.append("🟡 无 A 级新信号")
        lines.append("")

    return "\n".join(lines)


def print_report(scan_date: date, results: Dict):
    """打印终端报告"""
    print(f"\n{'='*80}")
    print(f"  MA24 底部突破监控报告  {scan_date}")
    print(f"{'='*80}")

    for mt in results:
        df = results[mt]['df']
        print(f"\n[{mt.upper()}]")
        print(f"  总币种: {len(df)}")
        print(f"  在MA24上方: {df['above_ma24'].sum()} ({df['above_ma24'].mean()*100:.1f}%)")
        print(f"  近30天回撤>30%: {(df['recent_drawdown'] <= -0.30).sum()} ({(df['recent_drawdown'] <= -0.30).mean()*100:.1f}%)")
        print(f"  中位数回撤: {df['recent_drawdown'].median()*100:.1f}%")

        print(f"\n  阶段分布:")
        print(df['stage'].value_counts().to_string().replace('\n', '\n    '))

        print(f"\n  A 级新信号:")
        a_new = df[(df['signal_grade'] == 'A') & (df['is_new_signal'])]
        if a_new.empty:
            print("    无")
        else:
            print(a_new[['symbol','price','dist_ma24','volume_ratio','window_drawdown','recent_drawdown']].to_string(index=False))

        print(f"\n  B 级新信号:")
        b_new = df[(df['signal_grade'] == 'B') & (df['is_new_signal'])]
        if b_new.empty:
            print("    无")
        else:
            print(b_new[['symbol','price','dist_ma24','volume_ratio','window_drawdown','recent_drawdown']].to_string(index=False))

        print(f"\n  大市值状态:")
        lc = df[df['symbol'].isin(LARGE_CAPS)][['symbol','dist_ma24','recent_drawdown','stage']].sort_values('recent_drawdown')
        if not lc.empty:
            print(lc.to_string(index=False))

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='MA24 底部突破实时监控')
    parser.add_argument('--market', type=str, default='swap,spot', help='市场类型，逗号分隔')
    parser.add_argument('--telegram', action='store_true', help='发送 Telegram 推送')
    parser.add_argument('--no-save', action='store_true', help='不写入数据库')
    args = parser.parse_args()

    markets = [m.strip() for m in args.market.split(',')]
    scan_date = datetime.utcnow().date()

    if not args.no_save:
        init_tables()

    results = {}
    large_cap_stages_all = {}

    for mt in markets:
        logger.info(f"开始扫描 {mt}...")
        df = scan_market(mt)
        if df.empty:
            logger.warning(f"[{mt}] 无有效数据")
            continue

        df = classify_signals(df)
        if not args.no_save:
            df = compare_with_yesterday(df, mt, scan_date)
        else:
            df['is_new_signal'] = True

        # 提取大市值状态
        lc = df[df['symbol'].isin(LARGE_CAPS)]
        large_cap_stages = dict(zip(lc['symbol'], lc['stage']))
        large_cap_stages_all[mt] = large_cap_stages

        if not args.no_save:
            save_signals(df, mt, scan_date)
            save_breadth(df, mt, scan_date, large_cap_stages)

        results[mt] = {
            'df': df,
            'large_cap_stages': large_cap_stages
        }

    print_report(scan_date, results)

    if args.telegram:
        msg = build_alert_message(scan_date, results)
        send_telegram(msg)


if __name__ == '__main__':
    main()
