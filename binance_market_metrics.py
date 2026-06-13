#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
币安市场深度指标采集器（日线）

合约指标：
- 大户账户数多空比
- 大户持仓量多空比
- 全球多空人数比
- 主动买卖量（Taker Buy/Sell）
- 持仓量
- 资金费率
- 基差率（mark/index）

现货指标：
- 大单净流入（按阈值筛选 aggTrades）

用法：
  python binance_market_metrics.py --market swap --symbol BTCUSDT --days 30
  python binance_market_metrics.py --market spot --symbol BTCUSDT --days 7 --whale-threshold 100000
  python binance_market_metrics.py --market all --days 7
"""

import os
import sys
import time
import logging
import argparse
import requests
import psycopg2
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('market_metrics')

DB_DSN = os.getenv('DB_DSN', 'postgresql://postgres:12@127.0.0.1:5432/market_data')
PROXY = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}

BASE_FAPI = 'https://fapi.binance.com'
BASE_SPOT = 'https://api.binance.com'

# 币安 futures/data 接口最多返回 500 条，按日线算就是 500 天
MAX_DAYS = 500


def get_db_conn():
    return psycopg2.connect(DB_DSN)


def init_database():
    """初始化数据表"""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS binance_futures_metrics (
                    symbol VARCHAR(30) NOT NULL,
                    metric_date DATE NOT NULL,
                    long_short_ratio_accounts FLOAT,
                    long_account_ratio FLOAT,
                    long_short_ratio_positions FLOAT,
                    long_position_ratio FLOAT,
                    global_long_short_ratio FLOAT,
                    global_long_account_ratio FLOAT,
                    taker_buy_sell_ratio FLOAT,
                    taker_buy_volume FLOAT,
                    taker_sell_volume FLOAT,
                    open_interest FLOAT,
                    funding_rate FLOAT,
                    mark_price FLOAT,
                    index_price FLOAT,
                    basis_rate FLOAT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (symbol, metric_date)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS binance_spot_flow (
                    symbol VARCHAR(30) NOT NULL,
                    flow_date DATE NOT NULL,
                    mode VARCHAR(20) NOT NULL DEFAULT 'klines',
                    threshold FLOAT DEFAULT NULL,
                    buy_volume FLOAT,
                    sell_volume FLOAT,
                    net_flow FLOAT,
                    buy_count INT,
                    sell_count INT,
                    total_volume FLOAT,
                    whale_volume_ratio FLOAT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (symbol, flow_date, mode, threshold)
                )
            """)
        conn.commit()
        logger.info("数据库表初始化完成")
    finally:
        conn.close()


def binance_request(base_url: str, endpoint: str, params: Dict = None) -> List[Dict]:
    """带重试和代理的请求"""
    url = f"{base_url}{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, proxies=PROXY, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"请求失败 ({attempt+1}/3): {url} {e}")
            time.sleep(1)
    raise Exception(f"请求最终失败: {url}")


def get_swap_symbols() -> List[str]:
    """获取所有 USDT 永续合约交易对"""
    data = binance_request(BASE_FAPI, '/fapi/v1/exchangeInfo')
    symbols = []
    for s in data['symbols']:
        if s['status'] == 'TRADING' and s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT':
            symbols.append(s['symbol'])
    return sorted(symbols)


def get_spot_symbols() -> List[str]:
    """获取所有 USDT 现货交易对"""
    data = binance_request(BASE_SPOT, '/api/v3/exchangeInfo')
    symbols = []
    for s in data['symbols']:
        if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT':
            symbols.append(s['symbol'])
    return sorted(symbols)


def fetch_futures_data_ratio(endpoint: str, symbol: str, period: str = '1d', limit: int = 30) -> List[Dict]:
    """获取 futures/data 类指标"""
    return binance_request(
        BASE_FAPI,
        f'/futures/data/{endpoint}',
        {'symbol': symbol, 'period': period, 'limit': limit}
    )


def fetch_open_interest_hist(symbol: str, limit: int = 30) -> List[Dict]:
    """获取持仓量历史"""
    return binance_request(
        BASE_FAPI,
        '/futures/data/openInterestHist',
        {'symbol': symbol, 'period': '1d', 'limit': limit}
    )


def fetch_funding_rate(symbol: str, limit: int = 30) -> List[Dict]:
    """获取资金费率"""
    return binance_request(
        BASE_FAPI,
        '/fapi/v1/fundingRate',
        {'symbol': symbol, 'limit': limit}
    )


def fetch_mark_price_klines(symbol: str, limit: int = 30) -> List[List]:
    """获取标记价格K线"""
    return binance_request(
        BASE_FAPI,
        '/fapi/v1/markPriceKlines',
        {'symbol': symbol, 'interval': '1d', 'limit': limit}
    )


def fetch_index_price_klines(symbol: str, limit: int = 30) -> List[List]:
    """获取指数价格K线"""
    return binance_request(
        BASE_FAPI,
        '/fapi/v1/indexPriceKlines',
        {'pair': symbol, 'interval': '1d', 'limit': limit}
    )


def collect_futures_metrics(symbol: str, days: int = 30) -> pd.DataFrame:
    """采集某个合约币的所有日线指标"""
    logger.info(f"[{symbol}] 采集合约指标...")
    limit = min(days, MAX_DAYS)

    # 1. 大户账户数多空比
    acc_ratio = fetch_futures_data_ratio('topLongShortAccountRatio', symbol, '1d', limit)
    df_acc = pd.DataFrame(acc_ratio)
    if not df_acc.empty:
        df_acc['metric_date'] = pd.to_datetime(df_acc['timestamp'], unit='ms').dt.date
        df_acc = df_acc.rename(columns={
            'longShortRatio': 'long_short_ratio_accounts',
            'longAccount': 'long_account_ratio'
        })

    # 2. 大户持仓量多空比
    pos_ratio = fetch_futures_data_ratio('topLongShortPositionRatio', symbol, '1d', limit)
    df_pos = pd.DataFrame(pos_ratio)
    if not df_pos.empty:
        df_pos['metric_date'] = pd.to_datetime(df_pos['timestamp'], unit='ms').dt.date
        df_pos = df_pos.rename(columns={
            'longShortRatio': 'long_short_ratio_positions',
            'longAccount': 'long_position_ratio'
        })

    # 3. 全球多空人数比
    global_ratio = fetch_futures_data_ratio('globalLongShortAccountRatio', symbol, '1d', limit)
    df_global = pd.DataFrame(global_ratio)
    if not df_global.empty:
        df_global['metric_date'] = pd.to_datetime(df_global['timestamp'], unit='ms').dt.date
        df_global = df_global.rename(columns={
            'longShortRatio': 'global_long_short_ratio',
            'longAccount': 'global_long_account_ratio'
        })

    # 4. 主动买卖量
    taker = fetch_futures_data_ratio('takerlongshortRatio', symbol, '1d', limit)
    df_taker = pd.DataFrame(taker)
    if not df_taker.empty:
        df_taker['metric_date'] = pd.to_datetime(df_taker['timestamp'], unit='ms').dt.date
        df_taker = df_taker.rename(columns={
            'buySellRatio': 'taker_buy_sell_ratio',
            'buyVol': 'taker_buy_volume',
            'sellVol': 'taker_sell_volume'
        })

    # 5. 持仓量历史
    oi_hist = fetch_open_interest_hist(symbol, limit)
    df_oi = pd.DataFrame(oi_hist)
    if not df_oi.empty:
        df_oi['metric_date'] = pd.to_datetime(df_oi['timestamp'], unit='ms').dt.date
        df_oi = df_oi.rename(columns={'sumOpenInterest': 'open_interest'})

    # 6. 资金费率（按日期聚合，取平均）
    funding = fetch_funding_rate(symbol, limit * 3)  # 一天可能多次结算
    df_funding = pd.DataFrame(funding)
    if not df_funding.empty:
        df_funding['metric_date'] = pd.to_datetime(df_funding['fundingTime'], unit='ms').dt.date
        df_funding['funding_rate'] = df_funding['fundingRate'].astype(float)
        df_funding = df_funding.groupby('metric_date')['funding_rate'].mean().reset_index()

    # 7. 基差率 = (标记价格 - 指数价格) / 指数价格
    mark_klines = fetch_mark_price_klines(symbol, limit)
    index_klines = fetch_index_price_klines(symbol, limit)
    df_basis = pd.DataFrame()
    if mark_klines and index_klines:
        df_mark = pd.DataFrame(mark_klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df_index = pd.DataFrame(index_klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df_mark['metric_date'] = pd.to_datetime(df_mark['open_time'], unit='ms').dt.date
        df_index['metric_date'] = pd.to_datetime(df_index['open_time'], unit='ms').dt.date
        df_basis = pd.merge(
            df_mark[['metric_date', 'close']].rename(columns={'close': 'mark_price'}),
            df_index[['metric_date', 'close']].rename(columns={'close': 'index_price'}),
            on='metric_date'
        )
        df_basis['mark_price'] = df_basis['mark_price'].astype(float)
        df_basis['index_price'] = df_basis['index_price'].astype(float)
        df_basis['basis_rate'] = (df_basis['mark_price'] - df_basis['index_price']) / df_basis['index_price']

    # 合并所有数据
    dfs = [df_acc, df_pos, df_global, df_taker, df_oi, df_funding, df_basis]
    result = None
    for df in dfs:
        if df is None or df.empty:
            continue
        df = df[['metric_date'] + [c for c in df.columns if c != 'metric_date' and c not in [
            'symbol', 'shortAccount', 'timestamp', 'fundingTime', 'open_time', 'close_time',
            'open', 'high', 'low', 'close', 'volume'
        ]]]
        if result is None:
            result = df
        else:
            result = pd.merge(result, df, on='metric_date', how='outer')

    if result is None or result.empty:
        return pd.DataFrame()

    result = result.sort_values('metric_date')
    result['symbol'] = symbol
    return result


def fetch_agg_trades_hour(symbol: str, start_ms: int, end_ms: int) -> List[Dict]:
    """获取一个小时的聚合成交（币安限制startTime/endTime区间不超过1小时）"""
    all_trades = []
    current_start = start_ms
    while current_start < end_ms:
        params = {
            'symbol': symbol,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': 1000
        }
        trades = binance_request(BASE_SPOT, '/api/v3/aggTrades', params)
        if not trades:
            break

        all_trades.extend(trades)
        last_trade = trades[-1]
        # 如果最后一条已经到 end_ms 或返回不足1000条，结束
        if last_trade['T'] >= end_ms or len(trades) < 1000:
            break
        # 下一次从最后一条之后开始
        current_start = last_trade['T'] + 1
    return all_trades


def fetch_agg_trades_day(symbol: str, day: date) -> pd.DataFrame:
    """获取某一天的全部聚合成交（按小时拆分请求）"""
    day_start = datetime.combine(day, datetime.min.time())
    all_trades = []

    for hour in range(24):
        hour_start = day_start + timedelta(hours=hour)
        hour_end = hour_start + timedelta(hours=1)
        start_ms = int(hour_start.timestamp() * 1000)
        end_ms = int(hour_end.timestamp() * 1000) - 1

        trades = fetch_agg_trades_hour(symbol, start_ms, end_ms)
        all_trades.extend(trades)

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    df['price'] = df['p'].astype(float)
    df['qty'] = df['q'].astype(float)
    df['amount'] = df['price'] * df['qty']
    df['is_buyer_maker'] = df['m'].astype(bool)
    # m=True 表示买方是 maker（即主动卖出），m=False 表示买方是 taker（主动买入）
    df['is_active_buy'] = ~df['is_buyer_maker']
    return df


def fetch_spot_klines(symbol: str, interval: str = '1d', limit: int = 30) -> pd.DataFrame:
    """获取现货K线"""
    data = binance_request(
        BASE_SPOT,
        '/api/v3/klines',
        {'symbol': symbol, 'interval': interval, 'limit': limit}
    )
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    if not df.empty:
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('open_time')
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']:
            df[col] = df[col].astype(float)
    return df


def calculate_spot_flow_klines(symbol: str, days: int = 30) -> pd.DataFrame:
    """从K线计算现货主动资金流（快速模式）"""
    logger.info(f"[{symbol}] 从K线计算现货主动资金流...")
    df = fetch_spot_klines(symbol, '1d', days)
    if df.empty:
        return pd.DataFrame()

    df['buy_volume'] = df['taker_buy_quote']
    df['sell_volume'] = df['quote_volume'] - df['taker_buy_quote']
    df['net_flow'] = df['buy_volume'] - df['sell_volume']
    df['symbol'] = symbol
    df = df.reset_index()
    df['flow_date'] = df['open_time'].dt.date
    df['mode'] = 'klines'
    df['threshold'] = 0  # klines模式无阈值，用0占位
    df['total_volume'] = df['quote_volume']
    return df[['symbol', 'flow_date', 'mode', 'threshold', 'buy_volume', 'sell_volume', 'net_flow', 'total_volume']]


def calculate_whale_flow_aggtrades(symbol: str, day: date, threshold: float = 100000.0) -> Optional[Dict]:
    """计算某一天的大单净流入（精确但很慢）"""
    logger.info(f"[{symbol}] 计算 {day} 现货大单流入 (阈值 ${threshold:,.0f})...")
    df = fetch_agg_trades_day(symbol, day)
    if df.empty:
        return None

    whale = df[df['amount'] >= threshold]
    if whale.empty:
        return None

    buy_volume = whale[whale['is_active_buy']]['amount'].sum()
    sell_volume = whale[~whale['is_active_buy']]['amount'].sum()

    return {
        'symbol': symbol,
        'flow_date': day,
        'mode': 'aggtrades',
        'threshold': threshold,
        'buy_volume': float(buy_volume),
        'sell_volume': float(sell_volume),
        'net_flow': float(buy_volume - sell_volume),
        'buy_count': int(whale[whale['is_active_buy']].shape[0]),
        'sell_count': int(whale[~whale['is_active_buy']].shape[0]),
        'total_volume': float(df['amount'].sum()),
        'whale_volume_ratio': float(whale['amount'].sum() / df['amount'].sum()) if df['amount'].sum() > 0 else 0,
    }


def save_futures_metrics(df: pd.DataFrame):
    """保存合约指标到数据库"""
    if df.empty:
        return 0

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            def to_val(x):
                if pd.isna(x):
                    return None
                return x
            for _, row in df.iterrows():
                cur.execute("""
                    INSERT INTO binance_futures_metrics (
                        symbol, metric_date, long_short_ratio_accounts, long_account_ratio,
                        long_short_ratio_positions, long_position_ratio,
                        global_long_short_ratio, global_long_account_ratio,
                        taker_buy_sell_ratio, taker_buy_volume, taker_sell_volume,
                        open_interest, funding_rate, mark_price, index_price, basis_rate, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (symbol, metric_date) DO UPDATE SET
                        long_short_ratio_accounts = EXCLUDED.long_short_ratio_accounts,
                        long_account_ratio = EXCLUDED.long_account_ratio,
                        long_short_ratio_positions = EXCLUDED.long_short_ratio_positions,
                        long_position_ratio = EXCLUDED.long_position_ratio,
                        global_long_short_ratio = EXCLUDED.global_long_short_ratio,
                        global_long_account_ratio = EXCLUDED.global_long_account_ratio,
                        taker_buy_sell_ratio = EXCLUDED.taker_buy_sell_ratio,
                        taker_buy_volume = EXCLUDED.taker_buy_volume,
                        taker_sell_volume = EXCLUDED.taker_sell_volume,
                        open_interest = EXCLUDED.open_interest,
                        funding_rate = EXCLUDED.funding_rate,
                        mark_price = EXCLUDED.mark_price,
                        index_price = EXCLUDED.index_price,
                        basis_rate = EXCLUDED.basis_rate,
                        updated_at = NOW()
                """, (
                    to_val(row.get('symbol')), to_val(row.get('metric_date')),
                    to_val(row.get('long_short_ratio_accounts')), to_val(row.get('long_account_ratio')),
                    to_val(row.get('long_short_ratio_positions')), to_val(row.get('long_position_ratio')),
                    to_val(row.get('global_long_short_ratio')), to_val(row.get('global_long_account_ratio')),
                    to_val(row.get('taker_buy_sell_ratio')), to_val(row.get('taker_buy_volume')), to_val(row.get('taker_sell_volume')),
                    to_val(row.get('open_interest')), to_val(row.get('funding_rate')),
                    to_val(row.get('mark_price')), to_val(row.get('index_price')), to_val(row.get('basis_rate'))
                ))
        conn.commit()
        return len(df)
    finally:
        conn.close()


def save_spot_flow(df: pd.DataFrame):
    """保存现货资金流到数据库"""
    if df.empty:
        return 0

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute("""
                    INSERT INTO binance_spot_flow (
                        symbol, flow_date, mode, threshold, buy_volume, sell_volume,
                        net_flow, buy_count, sell_count, total_volume, whale_volume_ratio, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (symbol, flow_date, mode, threshold) DO UPDATE SET
                        buy_volume = EXCLUDED.buy_volume,
                        sell_volume = EXCLUDED.sell_volume,
                        net_flow = EXCLUDED.net_flow,
                        buy_count = EXCLUDED.buy_count,
                        sell_count = EXCLUDED.sell_count,
                        total_volume = EXCLUDED.total_volume,
                        whale_volume_ratio = EXCLUDED.whale_volume_ratio,
                        updated_at = NOW()
                """, (
                    row.get('symbol'), row.get('flow_date'), row.get('mode'), row.get('threshold'),
                    row.get('buy_volume'), row.get('sell_volume'), row.get('net_flow'),
                    row.get('buy_count'), row.get('sell_count'), row.get('total_volume'),
                    row.get('whale_volume_ratio')
                ))
        conn.commit()
        return len(df)
    finally:
        conn.close()


def show_recent_futures(symbol: str, days: int = 7):
    """展示最近几天的合约指标"""
    conn = get_db_conn()
    try:
        df = pd.read_sql("""
            SELECT * FROM binance_futures_metrics
            WHERE symbol = %s
            ORDER BY metric_date DESC
            LIMIT %s
        """, conn, params=(symbol, days))
        if df.empty:
            print(f"\n[{symbol}] 无数据\n")
            return

        print(f"\n{'='*120}")
        print(f"  [{symbol}] 最近 {len(df)} 天合约指标")
        print(f"{'='*120}")
        print(df.to_string(index=False))
        print(f"{'='*120}\n")
    finally:
        conn.close()


def show_recent_spot(symbol: str, days: int = 7):
    """展示最近几天的现货资金流"""
    conn = get_db_conn()
    try:
        df = pd.read_sql("""
            SELECT * FROM binance_spot_flow
            WHERE symbol = %s
            ORDER BY flow_date DESC
            LIMIT %s
        """, conn, params=(symbol, days))
        if df.empty:
            print(f"\n[{symbol}] 无现货资金流数据\n")
            return
        print(f"\n{'='*100}")
        print(f"  [{symbol}] 最近 {len(df)} 天现货资金流")
        print(f"{'='*100}")
        print(df.to_string(index=False))
        print(f"{'='*100}\n")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='币安市场深度指标采集器（日线）')
    parser.add_argument('--market', type=str, default='swap', choices=['spot', 'swap', 'all'])
    parser.add_argument('--symbol', type=str, help='指定交易对，为空则全市场')
    parser.add_argument('--days', type=int, default=30, help='回溯天数（默认30，最大500）')
    parser.add_argument('--whale-threshold', type=float, default=100000, help='现货精确大单阈值 USD（默认10万，仅aggtrades模式）')
    parser.add_argument('--spot-mode', type=str, default='klines', choices=['klines', 'aggtrades'],
                        help='现货资金流计算模式：klines(快速主动买卖量) 或 aggtrades(精确大单但极慢)')
    parser.add_argument('--no-save', action='store_true', help='只展示不保存')
    args = parser.parse_args()

    init_database()

    days = min(args.days, MAX_DAYS)

    if args.market in ('swap', 'all'):
        symbols = [args.symbol] if args.symbol else get_swap_symbols()
        logger.info(f"开始采集 {len(symbols)} 个合约交易对，回溯 {days} 天")
        for sym in symbols:
            try:
                df = collect_futures_metrics(sym, days)
                if not df.empty:
                    if not args.no_save:
                        saved = save_futures_metrics(df)
                        logger.info(f"[{sym}] 保存 {saved} 条合约指标")
                    else:
                        logger.info(f"[{sym}] 采集到 {len(df)} 条合约指标（未保存）")
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"[{sym}] 合约指标采集失败: {e}")

    if args.market in ('spot', 'all'):
        symbols = [args.symbol] if args.symbol else get_spot_symbols()
        logger.info(f"开始采集 {len(symbols)} 个现货交易对资金流，模式={args.spot_mode}，回溯 {days} 天")
        for sym in symbols:
            try:
                if args.spot_mode == 'klines':
                    df = calculate_spot_flow_klines(sym, days)
                    if not df.empty:
                        if not args.no_save:
                            saved = save_spot_flow(df)
                            logger.info(f"[{sym}] 保存 {saved} 条现货资金流")
                        else:
                            logger.info(f"[{sym}] 计算 {len(df)} 条现货资金流（未保存）")
                elif args.spot_mode == 'aggtrades':
                    for d in range(days):
                        day = date.today() - timedelta(days=d)
                        record = calculate_whale_flow_aggtrades(sym, day, args.whale_threshold)
                        if record:
                            # 包装成 DataFrame 保存
                            df = pd.DataFrame([record])
                            if not args.no_save:
                                save_spot_flow(df)
                                logger.info(f"[{sym}] {day} 大单净流入: ${record['net_flow']:,.0f}")
                            else:
                                logger.info(f"[{sym}] {day} 大单净流入: ${record['net_flow']:,.0f}（未保存）")
                        time.sleep(0.2)
            except Exception as e:
                logger.error(f"[{sym}] 现货资金流计算失败: {e}")

    if args.symbol:
        show_recent_futures(args.symbol, days)
        show_recent_spot(args.symbol, days)


if __name__ == '__main__':
    main()
