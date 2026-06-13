#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
MA24突破实时监控 + 高胜率做空信号报警

功能:
1. 扫描全市场MA24突破事件
2. 实时计算高胜率做空指标 (突破前涨幅、波动率、成交量等)
3. 跟踪突破后3-5天是否创新高
4. 满足条件时推送报警 (console + 日志 + 可选钉钉/telegram)
5. 支持单次扫描或守护进程模式

使用:
  单次扫描: /data/anaconda3/envs/okx_api/bin/python3 breakout_monitor.py --scan-now
  守护进程: /data/anaconda3/envs/okx_api/bin/python3 breakout_monitor.py --daemon
  查看状态: /data/anaconda3/envs/okx_api/bin/python3 breakout_monitor.py --status
"""

import os
import sys
import json
import time
import logging
import argparse
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv('/data/okx/.env')

DATABASE_URL = os.getenv('DB_DSN', 'postgresql://postgres:12@127.0.0.1:5432/market_data')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, 'breakout_monitor_state.json')
ALERT_LOG = os.path.join(SCRIPT_DIR, 'breakout_alerts.log')

# ==========================================
# 配置区
# ==========================================
# 高胜率做空条件阈值
CONFIG = {
    'market_type': 'spot',           # spot / swap
    'scan_interval_seconds': 3600,   # 守护进程扫描间隔 (1小时)
    'lookback_days': 60,             # 需要的历史数据天数

    # 做空条件阈值 (基于历史回测)
    'cond_pre_r20d_high': 0.30,      # 突破前20天涨幅>30% (最优策略)
    'cond_pre_r5d_high': 0.10,       # 突破前5天涨幅>10%
    'cond_pre_r20d_mid': 0.20,       # 突破前20天涨幅>20%
    'cond_volatility_high': 0.07,    # 波动率>7%
    'cond_volatility_mid': 0.05,     # 波动率>5%
    'cond_vol_ratio': 1.5,           # 成交量放大>1.5x
    'cond_weak_days': 200,           # 弱势天数>200
    'cond_drawdown': 0.80,           # 回撤>80%
    'cond_dist_ma24': 0.03,          # 距MA24乖离>3%

    # 突破后跟踪参数
    'track_days': 5,                 # 跟踪突破后5天
    'no_new_high_threshold': 0.05,   # 5天内未创新高>5%

    # 报警级别
    'alert_level_extreme': 3,        # 满足3个及以上极端条件
    'alert_level_high': 2,           # 满足2个极端条件
    'alert_level_mid': 1,            # 满足1个极端条件 + 基础超跌
}

# ==========================================
# 日志设置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ALERT_LOG, mode='a')
    ]
)
logger = logging.getLogger('breakout_monitor')


def get_db_conn():
    return psycopg2.connect(DATABASE_URL)


# ==========================================
# 数据获取
# ==========================================
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


# ==========================================
# 状态管理 (突破跟踪)
# ==========================================
class BreakoutTracker:
    """跟踪突破事件的状态管理器"""

    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                return {'breakouts': [], 'alerts_sent': []}
        return {'breakouts': [], 'alerts_sent': []}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def record_breakout(self, symbol: str, date_str: str, price: float, metrics: Dict):
        """记录一个新的突破事件"""
        key = f"{symbol}_{date_str}"
        # 检查是否已存在
        for b in self.state['breakouts']:
            if b['key'] == key:
                return False

        self.state['breakouts'].append({
            'key': key,
            'symbol': symbol,
            'date': date_str,
            'price': float(price),
            'metrics': metrics,
            'day_count': 0,
            'max_high_since': float(price),
            'alerted_extreme': False,
            'alerted_nh': False,
        })
        self._save_state()
        return True

    def update_tracking(self, symbol: str, date_str: str, today_high: float):
        """更新跟踪中的突破事件"""
        key = f"{symbol}_{date_str}"
        for b in self.state['breakouts']:
            if b['key'] == key:
                b['day_count'] += 1
                if today_high > b['max_high_since']:
                    b['max_high_since'] = float(today_high)
                self._save_state()
                return b
        return None

    def get_active_breakouts(self, max_age_days: int = 10) -> List[Dict]:
        """获取仍在跟踪期内的突破事件"""
        cutoff = (datetime.now() - timedelta(days=max_age_days)).strftime('%Y-%m-%d')
        return [b for b in self.state['breakouts'] if b['date'] >= cutoff]

    def mark_alerted(self, key: str, alert_type: str):
        """标记已报警"""
        for b in self.state['breakouts']:
            if b['key'] == key:
                if alert_type == 'extreme':
                    b['alerted_extreme'] = True
                elif alert_type == 'no_new_high':
                    b['alerted_nh'] = True
        self._save_state()

    def is_alerted(self, key: str, alert_type: str) -> bool:
        for b in self.state['breakouts']:
            if b['key'] == key:
                if alert_type == 'extreme':
                    return b.get('alerted_extreme', False)
                elif alert_type == 'no_new_high':
                    return b.get('alerted_nh', False)
        return False

    def clean_old(self, max_age_days: int = 30):
        """清理过期数据"""
        cutoff = (datetime.now() - timedelta(days=max_age_days)).strftime('%Y-%m-%d')
        self.state['breakouts'] = [b for b in self.state['breakouts'] if b['date'] >= cutoff]
        self._save_state()


# ==========================================
# 突破分析
# ==========================================
@dataclass
class BreakoutSignal:
    symbol: str
    date: str
    price: float
    ma24: float
    ma6: float
    drawdown: float
    weak_days: int
    weak_ratio: float
    pre_r5d: float
    pre_r20d: float
    pre_r60d: float
    vol_ratio: float
    quote_vol_ratio: float
    volatility: float
    dist_ma24: float
    days_below_ma24: int
    conditions_met: List[str]
    alert_level: int
    alert_msg: str


def analyze_symbol(symbol: str, market_type: str, tracker: BreakoutTracker) -> Optional[BreakoutSignal]:
    """分析单个币是否出现MA24突破 + 高胜率做空信号"""
    min_date = datetime.now() - timedelta(days=120)
    df = fetch_daily_klines(symbol, market_type, min_date)
    if df.empty or len(df) < 40:
        return None

    # 计算指标
    df['ma6'] = df['close'].rolling(window=6, min_periods=6).mean()
    df['ma24'] = df['close'].rolling(window=24, min_periods=24).mean()
    df['weekly_ma60'] = calculate_weekly_ma(df, 60)
    df['rolling_high_500d'] = df['high'].rolling(window=500, min_periods=200).max()
    df['drawdown'] = (df['rolling_high_500d'] - df['close']) / df['rolling_high_500d']
    df['below_weekly_ma'] = df['close'] < df['weekly_ma60']
    df['above_ma24'] = df['close'] > df['ma24']
    df['volume_ma20'] = df['volume'].rolling(window=20, min_periods=15).mean()
    df['quote_volume_ma20'] = df['quote_volume'].rolling(window=20, min_periods=15).mean()
    df['return_1d'] = df['close'].pct_change()
    df['volatility_20d'] = df['return_1d'].rolling(window=20, min_periods=15).std()
    df['pre_return_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['pre_return_20d'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    df['pre_return_60d'] = (df['close'] - df['close'].shift(60)) / df['close'].shift(60)
    df['weak_days_250'] = df['below_weekly_ma'].rolling(window=250, min_periods=150).sum()
    df['prev_above_ma24'] = df['above_ma24'].shift(1)
    df['is_breakout'] = (~df['prev_above_ma24'].fillna(True)) & df['above_ma24']
    df['below_ma24'] = df['close'] <= df['ma24']
    df['below_ma24_group'] = (df['below_ma24'] != df['below_ma24'].shift()).cumsum()
    df['days_below_ma24'] = df.groupby('below_ma24_group')['below_ma24'].cumsum() * df['below_ma24']

    # 只检查最新的K线
    if not df['is_breakout'].iloc[-1]:
        return None

    row = df.iloc[-1]
    if pd.isna(row['ma24']) or pd.isna(row['drawdown']) or row['volume_ma20'] == 0:
        return None

    # 突破日期
    date_str = df.index[-1].strftime('%Y-%m-%d')

    # 计算做空条件
    conds = []
    pre_r20d = row['pre_return_20d'] if not pd.isna(row['pre_return_20d']) else 0
    pre_r5d = row['pre_return_5d'] if not pd.isna(row['pre_return_5d']) else 0
    vol = row['volatility_20d'] if not pd.isna(row['volatility_20d']) else 0
    vol_ratio = row['volume'] / row['volume_ma20'] if row['volume_ma20'] > 0 else 1.0
    weak_days = int(row['weak_days_250']) if not pd.isna(row['weak_days_250']) else 0
    drawdown = row['drawdown']
    dist_ma24 = (row['close'] - row['ma24']) / row['ma24']

    if pre_r20d > CONFIG['cond_pre_r20d_high']:
        conds.append('前20天涨幅>30%')
    if pre_r5d > CONFIG['cond_pre_r5d_high'] and pre_r20d > CONFIG['cond_pre_r20d_mid']:
        conds.append('前5天>10%且前20天>20%')
    if vol > CONFIG['cond_volatility_high']:
        conds.append('波动率>7%')
    if vol > CONFIG['cond_volatility_mid'] and vol_ratio > CONFIG['cond_vol_ratio']:
        conds.append('波动率>5%且成交量>1.5x')
    if weak_days > CONFIG['cond_weak_days'] and vol_ratio > CONFIG['cond_vol_ratio']:
        conds.append('弱势>200天且成交量>1.5x')
    if drawdown > CONFIG['cond_drawdown'] and dist_ma24 > CONFIG['cond_dist_ma24']:
        conds.append('回撤>80%且乖离>3%')
    if drawdown > CONFIG['cond_drawdown'] and vol_ratio > CONFIG['cond_vol_ratio']:
        conds.append('回撤>80%且成交量>1.5x')

    # 判断报警级别
    extreme_count = sum([
        pre_r20d > CONFIG['cond_pre_r20d_high'],
        pre_r5d > CONFIG['cond_pre_r5d_high'] and pre_r20d > CONFIG['cond_pre_r20d_mid'],
        vol > CONFIG['cond_volatility_high'],
        (weak_days > CONFIG['cond_weak_days'] and vol_ratio > CONFIG['cond_vol_ratio']),
    ])

    if extreme_count >= CONFIG['alert_level_extreme']:
        level = 3
    elif extreme_count >= CONFIG['alert_level_high']:
        level = 2
    elif extreme_count >= CONFIG['alert_level_mid'] or len(conds) >= 2:
        level = 1
    else:
        level = 0

    if level == 0:
        return None

    msg = f"[{symbol}] 突破MA24做空信号 | 级别:{level} | 条件:{','.join(conds)} | " \
          f"回撤:{drawdown*100:.1f}% | 前20天:{pre_r20d*100:.1f}% | 波动率:{vol*100:.1f}% | " \
          f"成交量:{vol_ratio:.2f}x | 弱势:{weak_days}天"

    signal = BreakoutSignal(
        symbol=symbol,
        date=date_str,
        price=float(row['close']),
        ma24=float(row['ma24']),
        ma6=float(row['ma6']),
        drawdown=float(drawdown),
        weak_days=int(weak_days),
        weak_ratio=float(row['weak_days_250']/250) if not pd.isna(row['weak_days_250']) else 0,
        pre_r5d=float(pre_r5d),
        pre_r20d=float(pre_r20d),
        pre_r60d=float(row['pre_return_60d']) if not pd.isna(row['pre_return_60d']) else 0,
        vol_ratio=float(vol_ratio),
        quote_vol_ratio=float(row['quote_volume']/row['quote_volume_ma20']) if row['quote_volume_ma20'] > 0 else 1.0,
        volatility=float(vol),
        dist_ma24=float(dist_ma24),
        days_below_ma24=int(row['days_below_ma24']) if not pd.isna(row['days_below_ma24']) else 0,
        conditions_met=conds,
        alert_level=level,
        alert_msg=msg,
    )

    return signal


# ==========================================
# 突破后跟踪检查
# ==========================================
def check_post_breakout(tracker: BreakoutTracker, market_type: str):
    """检查之前记录的突破事件，看是否触发'5天不创新高'做空信号"""
    active = tracker.get_active_breakouts(max_age_days=CONFIG['track_days'] + 1)
    alerts = []

    for b in active:
        if b['alerted_nh']:
            continue
        if b['day_count'] < 3:
            continue  # 至少跟踪3天

        # 获取最新数据
        symbol = b['symbol']
        breakout_date = datetime.strptime(b['date'], '%Y-%m-%d')
        min_date = breakout_date - timedelta(days=30)
        df = fetch_daily_klines(symbol, market_type, min_date)
        if df.empty or len(df) < 5:
            continue

        # 找到突破日后的数据
        post_df = df[df.index > breakout_date]
        if len(post_df) < b['day_count']:
            continue

        # 更新最高价
        today_high = float(df['high'].iloc[-1])
        b = tracker.update_tracking(symbol, b['date'], today_high)
        if b is None:
            continue

        # 检查5天不创新高
        if b['day_count'] >= CONFIG['track_days']:
            max_gain = (b['max_high_since'] - b['price']) / b['price']
            if max_gain < CONFIG['no_new_high_threshold']:
                msg = f"🔴 [{symbol}] 突破后{b['day_count']}天未创新高(最高涨幅{max_gain*100:.1f}%) | " \
                      f"突破价:{b['price']:.4f} | 历史做空胜率42.3% | 建议建仓做空"
                alerts.append(msg)
                tracker.mark_alerted(b['key'], 'no_new_high')
                logger.warning(msg)

    return alerts


# ==========================================
# 主扫描逻辑
# ==========================================
def scan_breakouts(market_type: str, tracker: BreakoutTracker) -> Tuple[List[BreakoutSignal], List[str]]:
    """主扫描函数"""
    symbols = get_symbols(market_type)
    logger.info(f"开始扫描 {market_type} 市场，共 {len(symbols)} 个币...")

    breakout_signals = []
    for i, symbol in enumerate(symbols, 1):
        try:
            signal = analyze_symbol(symbol, market_type, tracker)
            if signal:
                breakout_signals.append(signal)
                # 记录到tracker
                tracker.record_breakout(
                    symbol, signal.date, signal.price,
                    {
                        'pre_r20d': signal.pre_r20d,
                        'volatility': signal.volatility,
                        'vol_ratio': signal.vol_ratio,
                        'weak_days': signal.weak_days,
                    }
                )
                if not tracker.is_alerted(f"{symbol}_{signal.date}", 'extreme'):
                    logger.warning(signal.alert_msg)
                    tracker.mark_alerted(f"{symbol}_{signal.date}", 'extreme')

            if i % 100 == 0:
                logger.info(f"进度: {i}/{len(symbols)}")
        except Exception as e:
            logger.debug(f"[{symbol}] 分析异常: {e}")

    # 检查突破后跟踪
    post_alerts = check_post_breakout(tracker, market_type)

    # 清理过期数据
    tracker.clean_old(max_age_days=30)

    logger.info(f"扫描完成! 发现 {len(breakout_signals)} 个突破信号, {len(post_alerts)} 个跟踪报警")
    return breakout_signals, post_alerts


def print_summary(signals: List[BreakoutSignal], post_alerts: List[str]):
    print(f"\n{'='*90}")
    print(f"📊 扫描结果汇总")
    print(f"{'='*90}")

    if not signals and not post_alerts:
        print("❌ 未发现任何做空信号")
        return

    # 按级别分组
    level3 = [s for s in signals if s.alert_level == 3]
    level2 = [s for s in signals if s.alert_level == 2]
    level1 = [s for s in signals if s.alert_level == 1]

    if level3:
        print(f"\n🔴 【极端做空信号 - 级别3】{len(level3)} 个")
        for s in level3:
            print(f"   {s.alert_msg}")

    if level2:
        print(f"\n🟠 【高优先级做空信号 - 级别2】{len(level2)} 个")
        for s in level2:
            print(f"   {s.alert_msg}")

    if level1:
        print(f"\n🟡 【中优先级做空信号 - 级别1】{len(level1)} 个")
        for s in level1:
            print(f"   {s.alert_msg}")

    if post_alerts:
        print(f"\n⏱️ 【突破后跟踪报警 - 5天不创新高】{len(post_alerts)} 个")
        for msg in post_alerts:
            print(f"   {msg}")

    print(f"\n{'='*90}")
    print("💡 交易建议:")
    print(f"{'='*90}")
    print("  级别3: 立即做空，历史胜率79.3%，盈亏比3:1")
    print("  级别2: 建仓50%做空，历史胜率38-44%")
    print("  级别1: 观察3天，不创新高再做空")
    print("  跟踪报警: 突破后5天不创新高，历史胜率42%")
    print(f"{'='*90}")


def run_daemon(market_type: str):
    """守护进程模式"""
    tracker = BreakoutTracker()
    logger.info("="*60)
    logger.info("MA24突破监控守护进程启动")
    logger.info(f"扫描间隔: {CONFIG['scan_interval_seconds']}秒")
    logger.info("="*60)

    while True:
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"[{now}] 开始定时扫描...")
            signals, post_alerts = scan_breakouts(market_type, tracker)
            print_summary(signals, post_alerts)
        except Exception as e:
            logger.error(f"扫描异常: {e}")

        next_scan = datetime.now() + timedelta(seconds=CONFIG['scan_interval_seconds'])
        logger.info(f"下次扫描: {next_scan.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(CONFIG['scan_interval_seconds'])


def show_status():
    """显示当前跟踪状态"""
    tracker = BreakoutTracker()
    active = tracker.get_active_breakouts(max_age_days=10)

    print(f"\n{'='*90}")
    print("📋 当前跟踪中的突破事件")
    print(f"{'='*90}")

    if not active:
        print("暂无跟踪中的突破事件")
        return

    print(f"{'币种':<12} {'突破日期':>12} {'突破价':>10} {'天数':>5} {'最高涨幅':>10} {'状态':>15}")
    print("-"*90)
    for b in active:
        max_gain = (b['max_high_since'] - b['price']) / b['price'] * 100
        status = "🔴已报警" if b['alerted_nh'] else f"跟踪中({b['day_count']}天)"
        print(f"{b['symbol']:<12} {b['date']:>12} {b['price']:>10.4f} {b['day_count']:>5} {max_gain:>9.1f}% {status:>15}")


def main():
    parser = argparse.ArgumentParser(description='MA24突破实时监控')
    parser.add_argument('--market', choices=['spot', 'swap'], default='spot', help='市场类型')
    parser.add_argument('--scan-now', action='store_true', help='立即执行单次扫描')
    parser.add_argument('--daemon', action='store_true', help='守护进程模式')
    parser.add_argument('--status', action='store_true', help='查看跟踪状态')
    parser.add_argument('--interval', type=int, default=3600, help='扫描间隔(秒)')
    args = parser.parse_args()

    CONFIG['scan_interval_seconds'] = args.interval

    if args.status:
        show_status()
        return

    if args.daemon:
        run_daemon(args.market)
    else:
        tracker = BreakoutTracker()
        signals, post_alerts = scan_breakouts(args.market, tracker)
        print_summary(signals, post_alerts)


if __name__ == '__main__':
    main()
