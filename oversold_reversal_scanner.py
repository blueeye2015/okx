#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
超跌反弹扫描器 (Oversold Reversal Scanner)
运行环境: /data/anaconda3/envs/okx_api/bin/python3

逻辑:
1. 跌了足够久、足够深:
   - 从历史最高价回撤 >= drawdown_threshold (默认 80%)
   - 长期低于 60周均线 (约300日均线) 超过 long_bear_days (默认250日≈1年)
2. 突然反转:
   - 当日收盘价上穿 short_ma (默认6日) 和 medium_ma (默认24日)
   - 即今天收盘 > 6MA 且 > 24MA，但昨天不满足

数据源: PostgreSQL binance_daily_klines 表
"""

import os
import sys
import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv('/data/okx/.env')

# ==========================================
# 配置区
# ==========================================
DATABASE_URL = os.getenv('DB_DSN', 'postgresql://postgres:12@127.0.0.1:5432/market_data')

# 筛选参数 (可自行调整)
CONFIG = {
    'market_type': 'spot',          # 'spot' 或 'swap'
    'drawdown_threshold': 0.80,     # 从历史高点回撤 >= 80%
    'long_bear_days': 250,          # 长期弱势天数 (约250日=1年)
    'weekly_ma_period': 60,         # 周线均线周期 (60周 ≈ 300日)
    'short_ma': 6,                  # 短期均线 (日线)
    'medium_ma': 24,                # 中期均线 (日线)
    'lookback_days': 500,           # 分析回看天数 (至少覆盖 long_bear_days + ma_period)
    'min_history_days': 400,        # 最小历史数据天数要求
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """扫描结果"""
    symbol: str
    market_type: str
    current_price: float
    historical_high: float
    drawdown: float          # 回撤幅度 0.85 = 85%
    below_weekly_ma_days: int  # 低于60周均线天数
    short_ma_value: float
    medium_ma_value: float
    signal_date: datetime
    
    def __str__(self):
        return (f"[{self.symbol}] 信号日:{self.signal_date.date()} "
                f"价格:{self.current_price:.4f} "
                f"高点:{self.historical_high:.4f} "
                f"回撤:{self.drawdown*100:.1f}% "
                f"弱势:{self.below_weekly_ma_days}天 "
                f"6MA:{self.short_ma_value:.4f} 24MA:{self.medium_ma_value:.4f}")


def get_db_conn():
    return psycopg2.connect(DATABASE_URL)


def get_symbols(market_type: str) -> List[str]:
    """获取指定市场的所有交易对"""
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
    """从数据库读取日K线数据"""
    conn = get_db_conn()
    query = """
        SELECT open_time, open, high, low, close, volume
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
    """
    基于日K线计算周K线的 N 周均线
    方法: 将日K线 resample 为周K线(周五收盘), 计算MA, 再 forward-fill 回日线
    """
    # resample为周K线 (以周五为周结束)
    weekly = df_daily['close'].resample('W-FRI').last().dropna()
    weekly_ma = weekly.rolling(window=weekly_period, min_periods=weekly_period).mean()
    
    # 将周均线重新对齐到日线 (forward fill)
    weekly_ma_daily = weekly_ma.reindex(df_daily.index, method='ffill')
    return weekly_ma_daily


def scan_symbol(symbol: str, market_type: str, cfg: dict) -> Optional[ScanResult]:
    """
    扫描单个交易对是否符合超跌反弹条件
    """
    # 1. 计算需要拉取的数据起始日期
    # 需要足够的历史数据来计算回撤、长期均线、日线均线
    buffer_days = max(cfg['long_bear_days'], cfg['weekly_ma_period'] * 7, cfg['medium_ma'] * 2)
    min_date = datetime.now() - timedelta(days=cfg['lookback_days'] + buffer_days)
    
    df = fetch_daily_klines(symbol, market_type, min_date)
    
    if df.empty or len(df) < cfg['min_history_days']:
        return None
    
    # 2. 计算日线均线
    df['ma6'] = df['close'].rolling(window=cfg['short_ma'], min_periods=cfg['short_ma']).mean()
    df['ma24'] = df['close'].rolling(window=cfg['medium_ma'], min_periods=cfg['medium_ma']).mean()
    
    # 3. 计算周线均线 (60周 ≈ 300日)
    df['weekly_ma60'] = calculate_weekly_ma(df, cfg['weekly_ma_period'])
    
    # 4. 计算历史最高价及回撤 (用最近 lookback_days 内的高点)
    recent_df = df.tail(cfg['lookback_days'] + cfg['medium_ma'])
    if len(recent_df) < cfg['lookback_days']:
        return None
    
    historical_high = recent_df['high'].max()
    
    # 5. 检查长期弱势: 在过去 long_bear_days 中，低于60周均线的天数占比
    bear_window = recent_df.tail(cfg['long_bear_days'])
    bear_window = bear_window.dropna(subset=['weekly_ma60'])
    if len(bear_window) < cfg['long_bear_days'] * 0.8:
        return None
    
    below_ma_count = (bear_window['close'] < bear_window['weekly_ma60']).sum()
    below_ma_ratio = below_ma_count / len(bear_window)
    
    # 要求大部分时间在60周均线下方 (>=80%)
    if below_ma_ratio < 0.80:
        return None
    
    # 6. 检查深度回撤
    current_price = recent_df['close'].iloc[-1]
    drawdown = (historical_high - current_price) / historical_high
    
    if drawdown < cfg['drawdown_threshold']:
        return None
    
    # 7. 检查反转信号: 当天收盘价超过6MA和24MA
    today = recent_df.iloc[-1]
    
    # 今天收盘 > 6MA 且 > 24MA
    today_above_ma = (today['close'] > today['ma6']) and (today['close'] > today['ma24'])
    
    if not today_above_ma:
        return None
    
    return ScanResult(
        symbol=symbol,
        market_type=market_type,
        current_price=current_price,
        historical_high=historical_high,
        drawdown=drawdown,
        below_weekly_ma_days=below_ma_count,
        short_ma_value=today['ma6'],
        medium_ma_value=today['ma24'],
        signal_date=today.name
    )


def run_scan(cfg: dict = None) -> List[ScanResult]:
    """执行全市场扫描"""
    if cfg is None:
        cfg = CONFIG
    
    symbols = get_symbols(cfg['market_type'])
    logger.info(f"开始扫描 {cfg['market_type']} 市场，共 {len(symbols)} 个交易对...")
    logger.info(f"筛选条件: 回撤>={cfg['drawdown_threshold']*100:.0f}%, "
                f"低于{cfg['weekly_ma_period']}周均线>={cfg['long_bear_days']}天, "
                f"突破{cfg['short_ma']}MA+{cfg['medium_ma']}MA")
    
    results = []
    for i, symbol in enumerate(symbols, 1):
        try:
            result = scan_symbol(symbol, cfg['market_type'], cfg)
            if result:
                results.append(result)
                logger.info(f"🎯 {result}")
            if i % 50 == 0:
                logger.info(f"进度: {i}/{len(symbols)}，已发现 {len(results)} 个信号")
        except Exception as e:
            logger.warning(f"[{symbol}] 扫描异常: {e}")
    
    logger.info(f"=" * 60)
    logger.info(f"扫描完成! 共检查 {len(symbols)} 个交易对，发现 {len(results)} 个信号")
    return results


def print_summary(results: List[ScanResult]):
    """打印结果摘要"""
    if not results:
        print("\n❌ 未发现符合条件的交易对")
        return
    
    print(f"\n{'='*80}")
    print(f"发现 {len(results)} 个超跌反弹信号 ({CONFIG['market_type']})")
    print(f"{'='*80}")
    print(f"{'Symbol':<15} {'Price':>12} {'High':>12} {'Drawdown':>10} {'WeakDays':>10} {'6MA':>12} {'24MA':>12} {'Date':>12}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x.drawdown, reverse=True):
        print(f"{r.symbol:<15} {r.current_price:>12.4f} {r.historical_high:>12.4f} "
              f"{r.drawdown*100:>9.1f}% {r.below_weekly_ma_days:>10} {r.short_ma_value:>12.4f} "
              f"{r.medium_ma_value:>12.4f} {r.signal_date.strftime('%Y-%m-%d'):>12}")
    print("=" * 80)


# ==========================================
# 命令行入口
# ==========================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='超跌反弹扫描器')
    parser.add_argument('--market', choices=['spot', 'swap'], default='spot',
                        help='市场类型 (默认: spot)')
    parser.add_argument('--drawdown', type=float, default=0.80,
                        help='回撤阈值，如 0.80 表示跌80%以上 (默认: 0.80)')
    parser.add_argument('--weak-days', type=int, default=250,
                        help='低于长期均线天数 (默认: 250 ≈ 1年)')
    parser.add_argument('--short-ma', type=int, default=6,
                        help='短期均线周期 (默认: 6日)')
    parser.add_argument('--medium-ma', type=int, default=24,
                        help='中期均线周期 (默认: 24日)')
    
    args = parser.parse_args()
    
    CONFIG['market_type'] = args.market
    CONFIG['drawdown_threshold'] = args.drawdown
    CONFIG['long_bear_days'] = args.weak_days
    CONFIG['short_ma'] = args.short_ma
    CONFIG['medium_ma'] = args.medium_ma
    
    results = run_scan(CONFIG)
    print_summary(results)
