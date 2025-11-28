#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REST 快照 + 增量 Stream 完整订单簿管理
与 trade 脚本保持相同：ClickHouse 配置 / 代理 / 日志 / 重连
"""
import json
import asyncio
import platform
import aiohttp
import ssl
import logging
from datetime import datetime
from typing import Dict, List, Optional
import clickhouse_connect
import time
# 文件顶部
from collections import deque
WRITE_BATCH = 500   # 条数阈值
WRITE_SEC   = 10     # 时间阈值

# ----------------  环境补丁  ----------------
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------  ClickHouse  ----------------
CLICKHOUSE = dict(
    host='localhost', port=8123, database='marketdata',
    username='default', password='12'
)

# ----------------  REST / Stream  ----------------
REST_BASE   = "https://api.binance.com"
STREAM_URL  = "wss://stream.binance.com:9443/ws"
PROXY       = "http://127.0.0.1:7890"                # http://127.0.0.1:7890
SYMBOLS     = ['btcusdt']
DEPTH_SPEED = '@1000ms'          # 或 '' 表示 1000 ms
SNAPSHOT_LIMIT = 200          # 1000 档

# ========================================================================
class OrderBook:
    """单个币种的内存镜像"""
    __slots__ = (
        'symbol', 'bids', 'asks', 'last_update_id', 'synced'  # 新增 synced
    )

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[float, float] = {}   # price -> qty
        self.asks: Dict[float, float] = {}
        self.last_update_id = 0
        self.synced = False   # 新增

    def apply_diff(self, first: int, final: int, bids: List[List], asks: List[List]):
        """apply 一条增量，返回是否连续"""
        if first != self.last_update_id + 1:
            return False
        # 买单
        for item in bids:
            px, qty = item[0], item[1]
            self.bids[float(px)] = float(qty)
            if float(qty) == 0:
                self.bids.pop(float(px), None)
        # 卖单
        for item in asks:
            px, qty = item[0], item[1]
            self.asks[float(px)] = float(qty)
            if float(qty) == 0:
                self.asks.pop(float(px), None)
        self.last_update_id = final
        return True

    def top5(self):
        """打印盘口 5 档，方便肉眼核对"""
        bid5 = sorted(self.bids.items(), reverse=True)[:5]
        ask5 = sorted(self.asks.items())[:5]
        logger.info("%s TOP-5 bids: %s", self.symbol.upper(), bid5)
        logger.info("%s TOP-5 asks: %s", self.symbol.upper(), ask5)


# ========================================================================
class DepthFullCollector:
    def __init__(self):
        self.client = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.books: Dict[str, OrderBook] = {}
        self.setup_db()
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
        self._buf = deque()          # 缓存
        self._last_flush = time.time()

    # ----------------  建库建表  ----------------
    def setup_db(self):
        tmp = clickhouse_connect.get_client(**{**CLICKHOUSE, 'database': 'default'})
        tmp.command(f"CREATE DATABASE IF NOT EXISTS {CLICKHOUSE['database']}")
        tmp.close()
        self.client = clickhouse_connect.get_client(**CLICKHOUSE)
        self.client.command('''
            CREATE TABLE IF NOT EXISTS depth (
                event_time        DateTime,
                symbol            String,
                side              String,
                price             Float64,
                quantity          Float64,
                first_update_id   UInt64,
                final_update_id   UInt64
            ) ENGINE = MergeTree()
              ORDER BY (symbol, side, event_time)
        ''')
        logger.info("ClickHouse 就绪")

    # ----------------  REST 快照  ----------------
    async def fetch_snapshot(self, symbol: str) -> OrderBook:
        url = f"{REST_BASE}/api/v3/depth"
        params = {'symbol': symbol.upper(), 'limit': SNAPSHOT_LIMIT}
        async with self.session.get(url, params=params, proxy=PROXY) as resp:
            resp.raise_for_status()
            data = await resp.json()
        book = OrderBook(symbol)
        book.last_update_id = data['lastUpdateId']
        for px, qty in data['bids']:
            book.bids[float(px)] = float(qty)
        for px, qty in data['asks']:
            book.asks[float(px)] = float(qty)
        logger.info("%s 快照完成 lastUpdateId=%s", symbol.upper(), book.last_update_id)
        ##book.top5()
        # 快照落表 depth_snapshot
        rows = [(datetime.utcnow(), symbol.upper(), 'bid', float(px), float(qty), book.last_update_id)
                for px, qty in data['bids']] + [(datetime.utcnow(), symbol.upper(), 'ask', float(px), float(qty), book.last_update_id)
                for px, qty in data['asks']]
        self.client.insert('depth_snapshot', rows,
                        column_names=['snapshot_time','symbol','side','price','quantity','update_id'])
        return book

    # ----------------  写入变动  ----------------
    def insert_diff(self, symbol, side, price, qty, first_u, final_u, ts):
        try:
            self._buf.append([ts, symbol.upper(), side, price, qty, first_u, final_u])
            # 触批：条数 or 时间
            if len(self._buf) >= WRITE_BATCH or time.time() - self._last_flush >= WRITE_SEC:
                self.flush()
        except Exception as e:
            logger.error("insert diff error: %s", e)

    def flush(self):
        if not self._buf:
            return
        try:
            self.client.insert('depth', list(self._buf),
                            column_names=['event_time','symbol','side','price','quantity','first_update_id','final_update_id'])
        except Exception as e:
            logger.error("批量写入失败: %s", e)
        self._buf.clear()
        self._last_flush = time.time()
    # ----------------  apply 增量 ----------------
    def apply_diff_and_persist(self, data: dict):
        symbol = data['s'].lower()
        book = self.books[symbol]
        first, final = data['U'], data['u']
        ts = datetime.fromtimestamp(data['E']/1000)

        # 1. 还未对齐 (synced = False)
        if not book.synced:
            # 1a. 丢弃快照之前的旧帧
            if final <= book.last_update_id:
                logger.debug("%s 丢弃旧帧 (pre-sync) final=%s <= last_id=%s", 
                             symbol.upper(), final, book.last_update_id)
                return

            # 1b. 寻找同步点 (根据 Binance 官方文档)
            # The first processed event should have U <= lastUpdateId+1 AND u >= lastUpdateId+1.
            if first <= book.last_update_id + 1 and final >= book.last_update_id + 1:
                logger.info("%s 找到同步点! first=%s, final=%s, last_id=%s", 
                            symbol.upper(), first, final, book.last_update_id)
                book.synced = True
                # 找到同步点，继续往下执行 apply 和落库
            
            # 1c. 出现跳号 (Gap)，我们错过了同步点
            elif first > book.last_update_id + 1:
                logger.warning("%s 序号跳号 (pre-sync gap)，重拉快照. first=%s, last_id=%s", 
                             symbol.upper(), first, book.last_update_id)
                asyncio.create_task(self.resync_one(symbol))
                return # 丢弃此帧，等待 resync
            
            # 1d. 还没到同步点 (first < last_update_id + 1)
            else:
                return # 丢弃此帧，继续等待

        # --- 到这里，要么 book.synced 刚被设为 True, 要么本来就是 True ---

        # 2. 已经对齐 (synced = True)，做连续性检查
        # (我们必须处理 book.synced 刚被设为 True 的情况)
        
        # 2a. 检查跳号
        if first != book.last_update_id + 1:
            # 这种情况可能在 1b 刚同步时发生 (first < last_update_id + 1),
            # 此时我们只 apply 变动，但不落库，因为这些变动已包含在快照中
            if first <= book.last_update_id:
                # 这是同步帧的一部分，仅在内存中 apply
                book.apply_diff(first, final, data['b'], data['a']) 
                logger.debug("%s Apply 同步帧的旧部分: first=%s, last_id=%s", 
                             symbol.upper(), first, book.last_update_id)
                return # 不落库
            
            # 这是真正的跳号
            logger.warning("%s 序号跳号 (post-sync)，重拉快照. first=%s, last_id+1=%s", 
                         symbol.upper(), first, book.last_update_id + 1)
            asyncio.create_task(self.resync_one(symbol))
            return

        # 3. 连续的帧 (first == book.last_update_id + 1)，正常 apply + 落库
        ok = book.apply_diff(first, final, data['b'], data['a'])
        
        if not ok: # 理论上不应该，因为我们已检查过
            logger.error("%s apply_diff 内部返回 False，重拉. %s vs %s", 
                         symbol.upper(), first, book.last_update_id + 1)
            asyncio.create_task(self.resync_one(symbol))
            return
        
        # 把变动落库
        for item in data['b']:
            px, qty = item[0], item[1]
            self.insert_diff(symbol, 'bid', float(px), float(qty), first, final, ts)
        for item in data['a']:
            px, qty = item[0], item[1]
            self.insert_diff(symbol, 'ask', float(px), float(qty), first, final, ts)

    # ----------------  重拉快照  ----------------
    async def resync_one(self, symbol: str):
        await asyncio.sleep(1)
        self.books[symbol] = await self.fetch_snapshot(symbol)

    # ----------------  WebSocket 连接  ----------------
    async def stream_loop(self):
        streams = [f"{s.lower()}@depth{DEPTH_SPEED}" for s in SYMBOLS]
        sub = {"method": "SUBSCRIBE", "params": streams, "id": 1}
        sslctx = ssl.create_default_context()
        sslctx.check_hostname = False
        sslctx.verify_mode = ssl.CERT_NONE
        conn_kw = {'ssl': sslctx}
        if PROXY:
            conn_kw['proxy'] = PROXY

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.ws_connect(STREAM_URL, **conn_kw) as ws:
                        await ws.send_str(json.dumps(sub))
                        logger.info("订阅已发送 %s", streams)
                        async for msg in ws:
                            ##logger.info("原始帧>>> %s", msg.data)
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                break
                            data = json.loads(msg.data)
                            if data.get('e') == 'depthUpdate':
                                self.apply_diff_and_persist(data)
                            elif data.get('result') is None and 'id' in data:
                                logger.info("订阅成功 %s", data)
                            else:
                                logger.debug("忽略帧 %s", data)
                except Exception as e:
                    logger.error("Stream 异常 %s，5 秒后重连", e)
                    await asyncio.sleep(5)

    # ----------------  主流程  ----------------
    async def run(self):
        # 1. 先拉快照
        for sym in SYMBOLS:
            self.books[sym.lower()] = await self.fetch_snapshot(sym)
        # 2. 启动增量流
        await self.stream_loop()

    def close(self):
        if self.client:
            self.client.close()
        if self.session:
            asyncio.create_task(self.session.close())
        logger.info("Collector 已关闭")


# ========================================================================
async def main():
    col = DepthFullCollector()
    try:
        await col.run()
    finally:
        col.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("用户中断")