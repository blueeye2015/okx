#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
市场分层分析：大币 vs 小币的超跌反弹特征对比
验证假设：大币和小币之间是否存在跷跷板效应？
"""

import os
import sys
import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv('/data/okx/.env')

DATABASE_URL = os.getenv('DB_DSN', 'postgresql://postgres:12@127.0.0.1:5432/market_data')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ==========================================
# 手动定义大币种（按市场共识的市值/影响力分层）
# ==========================================
LARGE_CAPS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'TRXUSDT', 'MATICUSDT',
    'SHIBUSDT', 'LTCUSDT', 'BCHUSDT', 'UNIUSDT', 'ATOMUSDT', 'ETCUSDT',
    'FILUSDT', 'NEARUSDT', 'APTUSDT', 'IMXUSDT', 'OPUSDT', 'ARBUSDT',
    'SUIUSDT', 'SEIUSDT', 'INJUSDT', 'GRTUSDT', 'STXUSDT', 'RUNEUSDT',
    'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'MANAUSDT', 'SANDUSDT', 'AXSUSDT',
    'THETAUSDT', 'FTMUSDT', 'EGLDUSDT', 'XTZUSDT', 'EOSUSDT', 'AAVEUSDT',
    'FLOWUSDT', 'NEOUSDT', 'QNTUSDT', 'MKRUSDT', 'GALAUSDT', 'CHZUSDT',
    'COMPUSDT', 'CRVUSDT', 'LDOUSDT', 'SNXUSDT', 'ENJUSDT', 'ZECUSDT',
    'DASHUSDT', 'XMRUSDT', 'KSMUSDT', 'YFIUSDT', 'ZILUSDT', 'ONTUSDT',
    'IOTAUSDT', 'WAVESUSDT', 'CELOUSDT', 'ROSEUSDT', 'MINAUSDT', 'KAVAUSDT',
    'SKLUSDT', 'RVNUSDT', 'SUSHIUSDT', '1INCHUSDT', 'BATUSDT', 'STORJUSDT',
    'ANKRUSDT', 'GLMRUSDT', 'CHRUSDT', 'COTIUSDT', 'DGBUSDT', 'IOSTUSDT',
    'TFUELUSDT', 'ONEUSDT', 'ONGUSDT', 'WINUSDT', 'DENTUSDT', 'HOTUSDT',
    'SCUSDT', 'IOTXUSDT', 'BLZUSDT', 'NKNUSDT', 'RSRUSDT', 'BANDUSDT',
    'RLCUSDT', 'OCEANUSDT', 'SFPUSDT', 'CVCUSDT', 'STMXUSDT', 'MDTUSDT',
    'DODOUSDT', 'PONDUSDT', 'ALICEUSDT', 'FARMUSDT', 'BALUSDT', 'PERPUSDT',
    'TRBUSDT', 'BELUSDT', 'FLMUSDT', 'HARDUSDT', 'TOMOUSDT', 'DIAUSDT',
    'REEFUSDT', 'AKROUSDT', 'SUNUSDT', 'NBSUSDT', 'LITUSDT', 'PSGUSDT',
    'JUVUSDT', 'ASRUSDT', 'ATMUSDT', 'ACMUSDT', 'BARUSDT', 'OGUSDT',
    'CITYUSDT', 'PORTOUSDT', 'LAZIOUSDT', 'SANTOSUSDT', 'ALPINEUSDT',
    'TUSDT', 'PROMUSDT', 'QIUSDT', 'API3USDT', 'CTKUSDT', 'LPTUSDT',
    'AUDIOUSDT', 'RAYUSDT', 'C98USDT', 'DYDXUSDT', 'ENSUSDT', 'PEOPLEUSDT',
    'JOEUSDT', 'MASKUSDT', 'ETHWUSDT', 'ASTRUSDT', 'PHBUSDT', 'GLMRUSDT',
    'ACHUSDT', 'IMXUSDT', 'MAGICUSDT', 'HIGHUSDT', 'LOKAUSDT', 'SCRTUSDT',
    'API3USDT', 'WOOUSDT', 'KNCUSDT', 'STGUSDT', 'CVXUSDT', 'FXSUSDT',
    'HOOKUSDT', 'EDUUSDT', 'MAVUSDT', 'PENDLEUSDT', 'ARKMUSDT', 'WLDUSDT',
    'SEIUSDT', 'CYBERUSDT', 'TIAUSDT', 'SUIUSDT', 'MEMEUSDT', 'ORDIUSDT',
    'BLURUSDT', 'JTOUSDT', '1000SATSUSDT', 'ACEUSDT', 'NFPUSDT', 'AIUSDT',
    'XAIUSDT', 'MANTAUSDT', 'ALTUSDT', 'JUPUSDT', 'PYTHUSDT', 'DYMUSDT',
    'PIXELUSDT', 'STRKUSDT', 'PORTALUSDT', 'WUSDT', 'SAGAUSDT', 'TAOUSDT',
    'ENAUSDT', 'WIFUSDT', 'BOMEUSDT', 'ETHFIUSDT', 'METISUSDT', 'AEVOUSDT',
    'BBUSDT', 'NOTUSDT', 'IOUSDT', 'ZKUSDT', 'LISTAUSDT', 'ZROUSDT',
    'RENDERUSDT', 'MEWUSDT', 'BONKUSDT', 'FLOKIUSDT', 'PEPEUSDT', 'SHIBUSDT',
    'DOGEUSDT', 'TRXUSDT', 'TONUSDT', 'NEARUSDT', 'INJUSDT', 'FETUSDT',
    'RNDRUSDT', 'ARBUSDT', 'OPUSDT', 'APTUSDT', 'STRKUSDT', 'SEIUSDT',
    'SUIUSDT', 'TIAUSDT', 'WLDUSDT', 'PYTHUSDT', 'JUPUSDT', 'ORDIUSDT',
    'STXUSDT', 'IMXUSDT', 'GRTUSDT', 'RUNEUSDT', 'FLOWUSDT', 'ROSEUSDT',
    'KAVAUSDT', 'MINAUSDT', 'LDOUSDT', 'SNXUSDT', 'CRVUSDT', '1INCHUSDT',
    'SUSHIUSDT', 'COMPUSDT', 'MKRUSDT', 'YFIUSDT', 'AAVEUSDT', 'UNIUSDT',
    'LINKUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT', 'ATOMUSDT', 'ETCUSDT',
    'LTCUSDT', 'BCHUSDT', 'ADAUSDT', 'XRPUSDT', 'BNBUSDT', 'SOLUSDT',
    'ETHUSDT', 'BTCUSDT', 'OMUSDT', 'POLUSDT'
]

# 去重
LARGE_CAPS = list(dict.fromkeys(LARGE_CAPS))

# 更精简的核心大币定义（真正的一线大币）
MEGA_CAPS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
             'ADAUSDT', 'TRXUSDT', 'AVAXUSDT', 'TONUSDT', 'SHIBUSDT', 'DOTUSDT',
             'LINKUSDT', 'NEARUSDT', 'MATICUSDT']


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
class TierAnalysis:
    symbol: str
    market_type: str
    tier: str  # 'mega', 'large', 'small'
    current_price: float
    historical_high: float
    drawdown: float
    below_weekly_ma_ratio: float
    below_weekly_ma_days: int
    current_above_ma6: bool
    current_above_ma24: bool
    days_above_ma24_250d: int
    days_below_ma24_250d: int
    ever_above_then_below: bool
    ever_below_then_above: bool
    last_cross_to_above_date: Optional[datetime]
    last_cross_to_below_date: Optional[datetime]
    avg_volume_250d: float
    avg_quote_volume_250d: float
    is_oversold: bool  # 是否满足超跌条件


def analyze_symbol(symbol: str, market_type: str, tier: str, cfg: dict) -> Optional[TierAnalysis]:
    buffer_days = max(cfg['long_bear_days'], cfg['weekly_ma_period'] * 7, cfg['medium_ma'] * 2)
    min_date = datetime.now() - timedelta(days=cfg['lookback_days'] + buffer_days)

    df = fetch_daily_klines(symbol, market_type, min_date)
    if df.empty or len(df) < cfg['min_history_days']:
        return None

    df['ma6'] = df['close'].rolling(window=cfg['short_ma'], min_periods=cfg['short_ma']).mean()
    df['ma24'] = df['close'].rolling(window=cfg['medium_ma'], min_periods=cfg['medium_ma']).mean()
    df['weekly_ma60'] = calculate_weekly_ma(df, cfg['weekly_ma_period'])

    recent_df = df.tail(cfg['lookback_days'] + cfg['medium_ma'])
    if len(recent_df) < cfg['lookback_days']:
        return None

    historical_high = recent_df['high'].max()
    current_price = recent_df['close'].iloc[-1]
    drawdown = (historical_high - current_price) / historical_high

    # 长期弱势
    bear_window = recent_df.tail(cfg['long_bear_days']).dropna(subset=['weekly_ma60'])
    below_ma_count = 0
    below_ma_ratio = 0.0
    is_oversold = False
    if len(bear_window) >= cfg['long_bear_days'] * 0.8:
        below_ma_count = (bear_window['close'] < bear_window['weekly_ma60']).sum()
        below_ma_ratio = below_ma_count / len(bear_window)
        is_oversold = (below_ma_ratio >= 0.80) and (drawdown >= cfg['drawdown_threshold'])

    # MA24穿越分析
    analysis_window = recent_df.tail(cfg['long_bear_days']).copy().dropna(subset=['ma24'])
    current_above_ma6 = recent_df['close'].iloc[-1] > recent_df['ma6'].iloc[-1] if not pd.isna(recent_df['ma6'].iloc[-1]) else False
    current_above_ma24 = recent_df['close'].iloc[-1] > recent_df['ma24'].iloc[-1] if not pd.isna(recent_df['ma24'].iloc[-1]) else False

    days_above = 0
    days_below = 0
    ever_above_then_below = False
    ever_below_then_above = False
    last_above_date = None
    last_below_date = None

    if len(analysis_window) >= 50:
        analysis_window['above_ma24'] = analysis_window['close'] > analysis_window['ma24']
        days_above = int(analysis_window['above_ma24'].sum())
        days_below = int((~analysis_window['above_ma24']).sum())

        for i in range(1, len(analysis_window)):
            prev = analysis_window['above_ma24'].iloc[i-1]
            curr = analysis_window['above_ma24'].iloc[i]
            curr_date = analysis_window.index[i]
            if not prev and curr:
                ever_below_then_above = True
                last_above_date = curr_date
            elif prev and not curr:
                ever_above_then_below = True
                last_below_date = curr_date

    # 成交量
    vol_window = df.tail(250)
    avg_volume = vol_window['volume'].mean() if 'volume' in vol_window.columns else 0
    avg_quote_vol = vol_window['quote_volume'].mean() if 'quote_volume' in vol_window.columns else 0

    return TierAnalysis(
        symbol=symbol, market_type=market_type, tier=tier,
        current_price=current_price, historical_high=historical_high,
        drawdown=drawdown, below_weekly_ma_ratio=below_ma_ratio,
        below_weekly_ma_days=below_ma_count,
        current_above_ma6=current_above_ma6, current_above_ma24=current_above_ma24,
        days_above_ma24_250d=days_above, days_below_ma24_250d=days_below,
        ever_above_then_below=ever_above_then_below, ever_below_then_above=ever_below_then_above,
        last_cross_to_above_date=last_above_date, last_cross_to_below_date=last_below_date,
        avg_volume_250d=avg_volume, avg_quote_volume_250d=avg_quote_vol,
        is_oversold=is_oversold
    )


def run_tier_analysis(market_type: str = 'spot'):
    cfg = {
        'market_type': market_type,
        'drawdown_threshold': 0.80,
        'long_bear_days': 250,
        'weekly_ma_period': 60,
        'short_ma': 6,
        'medium_ma': 24,
        'lookback_days': 500,
        'min_history_days': 400,
    }

    symbols = get_symbols(market_type)
    logger.info(f"开始分层分析 {market_type} 市场，共 {len(symbols)} 个交易对...")

    results: List[TierAnalysis] = []
    for i, symbol in enumerate(symbols, 1):
        if symbol in MEGA_CAPS:
            tier = 'mega'
        elif symbol in LARGE_CAPS:
            tier = 'large'
        else:
            tier = 'small'

        try:
            res = analyze_symbol(symbol, market_type, tier, cfg)
            if res:
                results.append(res)
        except Exception as e:
            logger.warning(f"[{symbol}] 分析异常: {e}")

        if i % 100 == 0:
            logger.info(f"进度: {i}/{len(symbols)}")

    return results


def print_tier_report(results: List[TierAnalysis], market_type: str):
    mega = [r for r in results if r.tier == 'mega']
    large = [r for r in results if r.tier == 'large']
    small = [r for r in results if r.tier == 'small']

    def tier_stats(items: List[TierAnalysis], name: str):
        if not items:
            return
        total = len(items)
        oversold = [r for r in items if r.is_oversold]
        above_ma24 = [r for r in items if r.current_above_ma24]
        above_ma6 = [r for r in items if r.current_above_ma6]
        above_both = [r for r in items if r.current_above_ma6 and r.current_above_ma24]
        fake_breakout = [r for r in items if r.is_oversold and r.ever_above_then_below and not r.current_above_ma24]
        real_reversal = [r for r in items if r.is_oversold and r.current_above_ma24]
        avg_dd = np.mean([r.drawdown for r in items]) * 100
        avg_dd_oversold = np.mean([r.drawdown for r in oversold]) * 100 if oversold else 0
        avg_days_above = np.mean([r.days_above_ma24_250d for r in items])

        print(f"\n{'='*80}")
        print(f"【{name}】共 {total} 个")
        print(f"{'='*80}")
        print(f"  平均回撤: {avg_dd:.1f}%")
        print(f"  满足超跌条件(跌80%+弱势250天): {len(oversold)} 个 ({len(oversold)/total*100:.1f}%)")
        if oversold:
            print(f"    └─ 平均回撤: {avg_dd_oversold:.1f}%")
        print(f"  当前站上MA6: {len(above_ma6)} 个 ({len(above_ma6)/total*100:.1f}%)")
        print(f"  当前站上MA24: {len(above_ma24)} 个 ({len(above_ma24)/total*100:.1f}%)")
        print(f"  当前同时站上MA6+MA24: {len(above_both)} 个 ({len(above_both)/total*100:.1f}%)")
        print(f"  250天内平均站上天数: {avg_days_above:.1f} 天")
        if oversold:
            print(f"  【超跌币中】假突破(站上后又跌破): {len(fake_breakout)} 个 ({len(fake_breakout)/len(oversold)*100:.1f}%)")
            print(f"  【超跌币中】当前站上MA24(含反转): {len(real_reversal)} 个 ({len(real_reversal)/len(oversold)*100:.1f}%)")

        # 列出具体币名
        if above_both:
            print(f"  当前突破MA6+MA24的币: {', '.join([r.symbol.replace('USDT','') for r in above_both])}")

    print(f"\n{'#'*80}")
    print(f"# 市场分层分析报告 ({market_type})")
    print(f"# 数据日期: {results[0].last_cross_to_above_date.strftime('%Y-%m-%d') if results and results[0].last_cross_to_above_date else '最新'}")
    print(f"{'#'*80}")

    tier_stats(mega, "一线大币 MEGA-CAP (BTC/ETH/SOL/XRP/BNB/DOGE/ADA/TRX/AVAX等)")
    tier_stats(large, "二线大币 LARGE-CAP (市值中上，非山寨)")
    tier_stats(small, "小币 SMALL-CAP (山寨币)")

    # 跨层对比
    print(f"\n{'='*80}")
    print(f"【跨层对比总结】")
    print(f"{'='*80}")

    all_oversold = [r for r in results if r.is_oversold]
    mega_oversold = [r for r in mega if r.is_oversold]
    small_oversold = [r for r in small if r.is_oversold]

    mega_above = [r for r in mega if r.current_above_ma24]
    small_above = [r for r in small if r.current_above_ma24]

    print(f"\n1. 超跌比例对比:")
    print(f"   一线大币: {len(mega_oversold)}/{len(mega)} = {len(mega_oversold)/len(mega)*100:.1f}% 满足超跌")
    print(f"   二线大币: {len([r for r in large if r.is_oversold])}/{len(large)} = {len([r for r in large if r.is_oversold])/len(large)*100:.1f}% 满足超跌")
    print(f"   小币种:   {len(small_oversold)}/{len(small)} = {len(small_oversold)/len(small)*100:.1f}% 满足超跌")

    print(f"\n2. 当前站上MA24比例对比:")
    print(f"   一线大币: {len(mega_above)}/{len(mega)} = {len(mega_above)/len(mega)*100:.1f}%")
    print(f"   二线大币: {len([r for r in large if r.current_above_ma24])}/{len(large)} = {len([r for r in large if r.current_above_ma24])/len(large)*100:.1f}%")
    print(f"   小币种:   {len(small_above)}/{len(small)} = {len(small_above)/len(small)*100:.1f}%")

    print(f"\n3. 跷跷板效应观察:")
    if mega_above and not small_above:
        print(f"   ⚠️ 大币在涨，小币全跌 — 资金明显向大币集中")
    elif small_above and not mega_above:
        print(f"   ⚠️ 小币在涨，大币全跌 — 资金明显向小币轮动")
    elif len(mega_above)/len(mega) > len(small_above)/len(small):
        print(f"   📊 大币站上MA24的比例({len(mega_above)/len(mega)*100:.1f}%) > 小币({len(small_above)/len(small)*100:.1f}%) — 大币相对强势")
    else:
        print(f"   📊 小币站上MA24的比例({len(small_above)/len(small)*100:.1f}%) > 大币({len(mega_above)/len(mega)*100:.1f}%) — 小币相对活跃")

    print(f"\n4. 大币详细状态:")
    for r in sorted(mega, key=lambda x: x.drawdown, reverse=True):
        status = "📈站上MA24" if r.current_above_ma24 else "📉跌破MA24"
        oversold_flag = " [超跌]" if r.is_oversold else ""
        print(f"   {r.symbol.replace('USDT',''):<6} 回撤{r.drawdown*100:>5.1f}% | {status}{oversold_flag}")

    print(f"\n5. 小币反弹个数 vs 大币:")
    small_reversal = [r for r in small if r.is_oversold and r.current_above_ma24]
    mega_reversal = [r for r in mega if r.is_oversold and r.current_above_ma24]
    print(f"   小币中超跌且站上MA24: {len(small_reversal)} 个")
    print(f"   大币中超跌且站上MA24: {len(mega_reversal)} 个")
    if len(small_reversal) > len(mega_reversal) * 5:
        print(f"   ✅ 验证假设：小币反弹个数远多于大币，可能存在跷跷板效应")
    else:
        print(f"   ❌ 当前数据不支持明显的跷跷板效应")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='市场分层分析')
    parser.add_argument('--market', choices=['spot', 'swap'], default='spot')
    args = parser.parse_args()

    results = run_tier_analysis(args.market)
    print_tier_report(results, args.market)
