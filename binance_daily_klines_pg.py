#!/data/anaconda3/envs/okx_api/bin/python3
# -*- coding: utf-8 -*-
"""
币安日K线获取程序
运行环境: /data/anaconda3/envs/okx_api/bin/python3

功能：
1. 获取币安所有 USDT 现货交易对的日K线
2. 获取币安所有 USDT 永续合约的日K线
3. 增量更新，存入 PostgreSQL
4. 支持每日定时运行

币安API文档：
- 现货 K线: https://api.binance.com/api/v3/klines
- 合约 K线: https://fapi.binance.com/fapi/v1/klines
- 日K线 interval = 1d
"""

import os
import sys
import time
import logging
import requests
import psycopg2
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('/data/okx/.env')

# ==========================================
# 配置区
# ==========================================
PROXY = {
    'http': os.getenv('HTTP_PROXY', 'http://127.0.0.1:7890'),
    'https': os.getenv('HTTPS_PROXY', 'http://127.0.0.1:7890'),
}

# PostgreSQL 连接串 (优先从环境变量读取)
DATABASE_URL = os.getenv('DB_DSN', 'postgresql://postgres:12@127.0.0.1:5432/market_data')

# 币安 API 基础地址
SPOT_BASE_URL = 'https://api.binance.com'
SWAP_BASE_URL = 'https://fapi.binance.com'

# 获取K线的起始日期（数据库为空时从此日期开始）
DEFAULT_START_DATE = '2017-01-01'

# 每次请求获取的最大条数 (币安上限 1000)
LIMIT = 1000

# API 请求间隔（秒），避免触发频率限制
REQUEST_DELAY = 0.15

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('binance_daily_klines.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# PostgreSQL 数据库操作
# ==========================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS binance_daily_klines (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(30)  NOT NULL,
    market_type     VARCHAR(10)  NOT NULL,  -- 'spot' 或 'swap'
    open_time       TIMESTAMP    NOT NULL,
    open            NUMERIC(30, 12),
    high            NUMERIC(30, 12),
    low             NUMERIC(30, 12),
    close           NUMERIC(30, 12),
    volume          NUMERIC(30, 12),
    quote_volume    NUMERIC(30, 12),
    trades_count    INTEGER,
    taker_buy_base  NUMERIC(30, 12),
    taker_buy_quote NUMERIC(30, 12),
    close_time      TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (symbol, market_type, open_time)
);

-- 为常用查询创建索引
CREATE INDEX IF NOT EXISTS idx_binance_klines_symbol_market 
    ON binance_daily_klines (symbol, market_type);
CREATE INDEX IF NOT EXISTS idx_binance_klines_open_time 
    ON binance_daily_klines (open_time);
CREATE INDEX IF NOT EXISTS idx_binance_klines_symbol_time 
    ON binance_daily_klines (symbol, market_type, open_time);
"""

UPSERT_SQL = """
INSERT INTO binance_daily_klines 
    (symbol, market_type, open_time, open, high, low, close, 
     volume, quote_volume, trades_count, taker_buy_base, taker_buy_quote, close_time)
VALUES %s
ON CONFLICT (symbol, market_type, open_time) DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume,
    quote_volume = EXCLUDED.quote_volume,
    trades_count = EXCLUDED.trades_count,
    taker_buy_base = EXCLUDED.taker_buy_base,
    taker_buy_quote = EXCLUDED.taker_buy_quote,
    close_time = EXCLUDED.close_time,
    updated_at = NOW();
"""


def get_db_conn():
    """获取数据库连接"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise


def init_database():
    """初始化数据库表结构"""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        logger.info("数据库表初始化完成")
    except Exception as e:
        conn.rollback()
        logger.error(f"初始化数据库表失败: {e}")
        raise
    finally:
        conn.close()


def get_latest_open_time(symbol: str, market_type: str) -> Optional[int]:
    """
    获取某交易对在数据库中的最新开盘时间（毫秒时间戳）
    返回 None 表示该交易对尚无数据
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MAX(open_time) 
                FROM binance_daily_klines 
                WHERE symbol = %s AND market_type = %s
                """,
                (symbol, market_type)
            )
            result = cur.fetchone()
            if result and result[0]:
                # 返回毫秒时间戳，并加1天（获取下一天开始）
                latest_dt = result[0]
                next_dt = latest_dt + timedelta(days=1)
                return int(next_dt.timestamp() * 1000)
            return None
    finally:
        conn.close()


def get_stats() -> Dict:
    """获取数据库统计信息"""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM binance_daily_klines")
            total_rows = cur.fetchone()[0]
            
            cur.execute("""
                SELECT market_type, COUNT(DISTINCT symbol), COUNT(*) 
                FROM binance_daily_klines 
                GROUP BY market_type
            """)
            market_stats = cur.fetchall()
            
            return {
                'total_rows': total_rows,
                'market_stats': market_stats
            }
    finally:
        conn.close()


# ==========================================
# 币安 API 请求
# ==========================================

def binance_request(base_url: str, endpoint: str, params: dict = None, max_retries: int = 3) -> dict:
    """发送币安 API 请求，带重试机制"""
    url = f"{base_url}{endpoint}"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, proxies=PROXY, timeout=15)
            if response.status_code == 429:
                logger.warning(f"触发频率限制，等待 60 秒后重试...")
                time.sleep(60)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                raise
    return {}


def get_spot_usdt_symbols() -> List[str]:
    """获取所有 USDT 现货交易对"""
    logger.info("正在获取现货 USDT 交易对列表...")
    data = binance_request(SPOT_BASE_URL, '/api/v3/exchangeInfo')
    symbols = []
    for s in data.get('symbols', []):
        if (s.get('status') == 'TRADING' and
            s.get('quoteAsset') == 'USDT' and
            s.get('isSpotTradingAllowed')):
            symbols.append(s['symbol'])
    logger.info(f"现货 USDT 交易对: {len(symbols)} 个")
    return sorted(symbols)


def get_swap_usdt_symbols() -> List[str]:
    """获取所有 USDT 永续合约交易对"""
    logger.info("正在获取合约 USDT 交易对列表...")
    data = binance_request(SWAP_BASE_URL, '/fapi/v1/exchangeInfo')
    symbols = []
    for s in data.get('symbols', []):
        if (s.get('status') == 'TRADING' and
            s.get('quoteAsset') == 'USDT' and
            s.get('contractType') == 'PERPETUAL'):  # 只取永续合约
            symbols.append(s['symbol'])
    logger.info(f"合约 USDT 永续交易对: {len(symbols)} 个")
    return sorted(symbols)


def fetch_klines(symbol: str, market_type: str, start_time_ms: int) -> List[List]:
    """
    获取日K线数据
    market_type: 'spot' 或 'swap'
    返回币安原始K线数据列表
    """
    base_url = SPOT_BASE_URL if market_type == 'spot' else SWAP_BASE_URL
    endpoint = '/api/v3/klines' if market_type == 'spot' else '/fapi/v1/klines'
    
    all_klines = []
    current_start = start_time_ms
    
    while True:
        params = {
            'symbol': symbol,
            'interval': '1d',
            'startTime': current_start,
            'limit': LIMIT
        }
        
        try:
            data = binance_request(base_url, endpoint, params)
        except Exception as e:
            logger.error(f"[{symbol}/{market_type}] 获取K线失败: {e}")
            break
        
        if not data or len(data) == 0:
            break
        
        all_klines.extend(data)
        
        # 币安返回的数据中最后一条的开盘时间
        last_open_time = int(data[-1][0])
        
        # 如果返回的数据少于 LIMIT，说明已经没有更多数据了
        if len(data) < LIMIT:
            break
        
        # 下一次请求从最后一条的收盘时间+1毫秒开始
        current_start = int(data[-1][6]) + 1
        
        # 如果已经获取到今天的数据，停止
        now_ms = int(time.time() * 1000)
        if last_open_time >= now_ms - 86400000:  # 24小时前
            break
        
        time.sleep(REQUEST_DELAY)
    
    return all_klines


# ==========================================
# 数据处理与存储
# ==========================================

def klines_to_db_rows(symbol: str, market_type: str, klines: List[List]) -> List[Tuple]:
    """将币安K线数据转换为数据库行元组"""
    rows = []
    for k in klines:
        # 币安K线格式:
        # [openTime, open, high, low, close, volume, closeTime, 
        #  quoteVolume, numTrades, takerBuyBase, takerBuyQuote, ignore]
        rows.append((
            symbol,
            market_type,
            datetime.fromtimestamp(int(k[0]) / 1000),  # open_time
            float(k[1]),   # open
            float(k[2]),   # high
            float(k[3]),   # low
            float(k[4]),   # close
            float(k[5]),   # volume
            float(k[7]),   # quote_volume
            int(k[8]),     # trades_count
            float(k[9]),   # taker_buy_base
            float(k[10]),  # taker_buy_quote
            datetime.fromtimestamp(int(k[6]) / 1000),  # close_time
        ))
    return rows


def batch_insert_rows(rows: List[Tuple], batch_size: int = 5000):
    """批量插入数据到PostgreSQL"""
    if not rows:
        return 0
    
    from psycopg2.extras import execute_values
    
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                execute_values(cur, UPSERT_SQL, batch)
                conn.commit()
                logger.debug(f"已写入 {len(batch)} 条记录")
        return len(rows)
    except Exception as e:
        conn.rollback()
        logger.error(f"批量插入失败: {e}")
        raise
    finally:
        conn.close()


# ==========================================
# 核心任务
# ==========================================

def process_symbol(symbol: str, market_type: str) -> int:
    """
    处理单个交易对：获取增量日K线并入库
    返回插入/更新的记录数
    """
    # 1. 确定起始时间
    latest_ts = get_latest_open_time(symbol, market_type)
    if latest_ts:
        start_ts = latest_ts
        logger.info(f"[{symbol}/{market_type}] 从数据库最新日期+1天开始: {datetime.fromtimestamp(start_ts/1000)}")
    else:
        start_ts = int(datetime.strptime(DEFAULT_START_DATE, "%Y-%m-%d").timestamp() * 1000)
        logger.info(f"[{symbol}/{market_type}] 数据库无数据，从 {DEFAULT_START_DATE} 开始")
    
    # 2. 如果起始时间 >= 明天开始(UTC)，跳过
    # Binance 日线在 UTC 00:00 收盘，只要 start_ts < 明天开始就应该获取
    now = datetime.utcnow()
    tomorrow_start = datetime(now.year, now.month, now.day) + timedelta(days=1)
    if datetime.utcfromtimestamp(start_ts / 1000) >= tomorrow_start:
        logger.info(f"[{symbol}/{market_type}] 数据已是最新，跳过")
        return 0
    
    # 3. 获取K线
    try:
        klines = fetch_klines(symbol, market_type, start_ts)
    except Exception as e:
        logger.error(f"[{symbol}/{market_type}] 获取K线异常: {e}")
        return 0
    
    if not klines:
        logger.info(f"[{symbol}/{market_type}] 无新数据")
        return 0
    
    # 4. 转换并入库
    rows = klines_to_db_rows(symbol, market_type, klines)
    inserted = batch_insert_rows(rows)
    
    logger.info(f"[{symbol}/{market_type}] 成功写入 {inserted} 条日K线")
    return inserted


def run_daily_update(market_types: List[str] = None, symbols_filter: List[str] = None):
    """
    执行每日增量更新
    market_types: ['spot', 'swap'] 或子集
    symbols_filter: 如果指定，只更新这些交易对
    """
    if market_types is None:
        market_types = ['spot', 'swap']
    
    init_database()
    
    total_inserted = 0
    total_symbols = 0
    
    for mtype in market_types:
        if mtype == 'spot':
            symbols = symbols_filter if symbols_filter else get_spot_usdt_symbols()
        elif mtype == 'swap':
            symbols = symbols_filter if symbols_filter else get_swap_usdt_symbols()
        else:
            logger.warning(f"未知市场类型: {mtype}")
            continue
        
        logger.info(f"开始处理 {mtype} 市场，共 {len(symbols)} 个交易对")
        
        for i, sym in enumerate(symbols, 1):
            try:
                count = process_symbol(sym, mtype)
                total_inserted += count
                total_symbols += 1
                
                if i % 10 == 0:
                    logger.info(f"[{mtype}] 进度: {i}/{len(symbols)}")
                    
            except Exception as e:
                logger.error(f"处理 {sym}/{mtype} 时出错: {e}")
                continue
            
            time.sleep(REQUEST_DELAY)
    
    logger.info(f"=" * 50)
    logger.info(f"更新完成! 共处理 {total_symbols} 个交易对，写入 {total_inserted} 条记录")
    
    stats = get_stats()
    logger.info(f"数据库统计: 总记录数={stats['total_rows']}")
    for row in stats['market_stats']:
        logger.info(f"  - {row[0]}: {row[1]} 个交易对, {row[2]} 条记录")


# ==========================================
# 定时任务模式
# ==========================================

def run_daemon(run_time: str = "00:05"):
    """
    守护进程模式：每天固定时间执行一次全量更新
    run_time: 执行时间，格式 HH:MM (24小时制)
    """
    target_hour, target_minute = map(int, run_time.split(':'))
    logger.info(f"守护进程启动，每日 {run_time} 执行全量更新...")
    
    while True:
        now = datetime.now()
        target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        
        # 如果今天的时间已过，设置为明天
        if target <= now:
            target += timedelta(days=1)
        
        wait_seconds = (target - now).total_seconds()
        logger.info(f"下次执行时间: {target}，等待 {wait_seconds/3600:.1f} 小时")
        time.sleep(wait_seconds)
        
        try:
            run_daily_update(market_types=['spot', 'swap'])
        except Exception as e:
            logger.error(f"定时任务执行失败: {e}")
            # 失败后等待10分钟再试，避免无限快速重试
            time.sleep(600)


# ==========================================
# 命令行入口
# ==========================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='币安日K线获取工具')
    parser.add_argument('--market', choices=['spot', 'swap', 'all'], default='all',
                        help='要获取的市场类型: spot(现货), swap(合约), all(全部)')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='指定交易对列表，如 BTCUSDT ETHUSDT')
    parser.add_argument('--init-only', action='store_true',
                        help='仅初始化数据库表，不获取数据')
    parser.add_argument('--proxy', default=None,
                        help='HTTP代理地址，如 http://127.0.0.1:7890')
    parser.add_argument('--daemon', action='store_true',
                        help='守护进程模式，每日自动执行')
    parser.add_argument('--run-at', default='00:05',
                        help='守护进程每日执行时间 (默认 00:05)')
    
    args = parser.parse_args()
    
    if args.proxy:
        PROXY = {'http': args.proxy, 'https': args.proxy}
    
    if args.init_only:
        init_database()
        sys.exit(0)
    
    if args.daemon:
        run_daemon(run_time=args.run_at)
        sys.exit(0)
    
    if args.market == 'all':
        markets = ['spot', 'swap']
    else:
        markets = [args.market]
    
    run_daily_update(market_types=markets, symbols_filter=args.symbols)
