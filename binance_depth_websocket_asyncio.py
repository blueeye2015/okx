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
from collections import deque
import functools  # [!! 修复 1 !!] 导入 functools

WRITE_BATCH = 500   # 条数阈值
WRITE_SEC   = 10      # 时间阈值

# ----------------  环境补丁  ----------------
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------  ClickHouse  ----------------
CLICKHOUSE = dict(
    host='localhost', port=8123, database='marketdata',
    username='default', password='12'
)

# ----------------  REST / Stream  ----------------
REST_BASE   = "https://api.binance.com"
STREAM_URL  = "wss://stream.binance.com:9443/ws"
PROXY       = "http://127.0.0.1:7890"            # http://127.0.0.1:7890
SYMBOLS     = ['btcusdt', 'ethusdt'] 
DEPTH_SPEED = '@100ms' 
SNAPSHOT_LIMIT = 200            

# ========================================================================
class OrderBook:
    # ... (OrderBook 类的代码没有变化，保持原样) ...
    """单个币种的内存镜像"""
    __slots__ = (
        'symbol', 'bids', 'asks', 'last_update_id', 'synced'
    )

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[float, float] = {}   # price -> qty
        self.asks: Dict[float, float] = {}
        self.last_update_id = 0
        self.synced = False
        #self._last_snapshot_save = time.time()
        #self._SNAPSHOT_INTERVAL = 60 # 假设每 60 秒存一次快照

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
        # [!! 关键修复 !!] 添加一个浏览器 User-Agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
        }
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20),headers=headers)
        self.loop = asyncio.get_event_loop() 
        
        # [!! 修复 2 !!] 创建一个数据库访问锁
        self._db_lock = asyncio.Lock()
        
        self._write_queue = asyncio.Queue(maxsize=10000) 
        
        self._buf = deque()        # 缓存
        self._last_flush = time.time()
        self._last_snapshot_save = time.time()
        self._SNAPSHOT_INTERVAL = 60

    # ----------------  建库建表  ----------------
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
        self.client.command('''
             CREATE TABLE IF NOT EXISTS depth_snapshot (
                snapshot_time     DateTime,
                symbol            String,
                side              String,
                price             Float64,
                quantity          Float64,
                update_id         UInt64
             ) ENGINE = MergeTree()
               ORDER BY (symbol, side, snapshot_time)
         ''')
        logger.info("ClickHouse 就绪")

    # ----------------  REST 快照  ----------------
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
        
        rows = [(datetime.utcnow(), symbol.upper(), 'bid', float(px), float(qty), book.last_update_id)
                for px, qty in data['bids']] + [(datetime.utcnow(), symbol.upper(), 'ask', float(px), float(qty), book.last_update_id)
                for px, qty in data['asks']]
        
        try:
            logger.info(f"正在将 {symbol} 快照写入数据库...")

            # [!! 修复 3a !!] (functools.partial)
            insert_call = functools.partial(
                self.client.insert,
                'depth_snapshot',
                rows,
                column_names=[
                    'snapshot_time', 'symbol', 'side', 
                    'price', 'quantity', 'update_id'
                ]
            )
            
            # [!! 修复 3b !!] (asyncio.Lock)
            async with self._db_lock:
                await self.loop.run_in_executor(None, insert_call)
            
            logger.info(f"{symbol} 快照写入完成")
        except Exception as e:
            logger.error(f"{symbol} 快照写入失败: {e}", exc_info=True) 
            
        return book

    # ----------------  写入变动  ----------------
    async def insert_diff(self, symbol, side, price, qty, first_u, final_u, ts):
        try:
            self._buf.append([ts, symbol.upper(), side, price, qty, first_u, final_u])
            if len(self._buf) >= WRITE_BATCH or time.time() - self._last_flush >= WRITE_SEC:
                await self.flush()
        except Exception as e:
            logger.error("insert diff error: %s", e)

    # ----------------  [!! 关键修复 !!]  ----------------
    async def flush(self):
        if not self._buf:
            return
        
        rows_to_insert = list(self._buf)
        self._buf.clear()
        self._last_flush = time.time()
        
        try:
            # [!! 修复 3a !!] (functools.partial)
            insert_call = functools.partial(
                self.client.insert, 
                'depth',            
                rows_to_insert,     
                column_names=[      
                    'event_time', 'symbol', 'side', 'price', 
                    'quantity', 'first_update_id', 'final_update_id'
                ]
            )
            
            # [!! 修复 3b !!] (asyncio.Lock)
            async with self._db_lock:
                await self.loop.run_in_executor(None, insert_call)
            
            logger.debug(f"成功批量写入 {len(rows_to_insert)} 条记录")
        except Exception as e:
            logger.error(f"批量写入失败 ({len(rows_to_insert)} 条): {e}", exc_info=True)

    # ----------------  apply 增量 ----------------
    # (这整个函数 'apply_diff_and_persist' 保持不变, 已经是 async)
    async def apply_diff_and_persist(self, data: dict):
        symbol = data['s'].lower()
        book = self.books.get(symbol) 
        if not book:
            logger.warning(f"收到了未知 symbol {symbol} 的数据，丢弃")
            return
        
        first, final = data['U'], data['u']
        ts = datetime.fromtimestamp(data['E']/1000)

        # 1. 还未对齐 (synced = False)
        if not book.synced:
            if final <= book.last_update_id:
                logger.debug("%s 丢弃旧帧 (pre-sync) final=%s <= last_id=%s", 
                             symbol.upper(), final, book.last_update_id)
                return

            if first <= book.last_update_id + 1 and final >= book.last_update_id + 1:
                logger.info("%s 找到同步点! first=%s, final=%s, last_id=%s", 
                            symbol.upper(), first, final, book.last_update_id)
                book.synced = True
            
            elif first > book.last_update_id + 1:
                logger.warning("%s 序号跳号 (pre-sync gap)，重拉快照. first=%s, last_id=%s", 
                               symbol.upper(), first, book.last_update_id)
                asyncio.create_task(self.resync_one(symbol))
                return 
            
            else:
                return 

        # 2. 已经对齐 (synced = True)，做连续性检查
        if first != book.last_update_id + 1:
            if first <= book.last_update_id:
                book.apply_diff(first, final, data['b'], data['a']) 
                logger.debug("%s Apply 同步帧的旧部分: first=%s, last_id=%s", 
                             symbol.upper(), first, book.last_update_id)
                return 
            
            logger.warning("%s 序号跳号 (post-sync)，重拉快照. first=%s, last_id+1=%s", 
                           symbol.upper(), first, book.last_update_id + 1)
            asyncio.create_task(self.resync_one(symbol))
            return

        # 3. 连续的帧 (first == book.last_update_id + 1)，正常 apply + 落库
        ok = book.apply_diff(first, final, data['b'], data['a'])
        
        if not ok: 
            logger.error("%s apply_diff 内部返回 False，重拉. %s vs %s", 
                         symbol.upper(), first, book.last_update_id + 1)
            asyncio.create_task(self.resync_one(symbol))
            return
        
        for item in data['b']:
            px, qty = item[0], item[1]
            await self.insert_diff(symbol, 'bid', float(px), float(qty), first, final, ts)
        for item in data['a']:
            px, qty = item[0], item[1]
            await self.insert_diff(symbol, 'ask', float(px), float(qty), first, final, ts)

    # ----------------  重拉快照  ----------------
    async def resync_one(self, symbol: str):
        # (这个函数 'resync_one' 保持不变)
        logger.info(f"{symbol.upper()} 开始重新同步...")
        await asyncio.sleep(1) 
        try:
            self.books[symbol] = await self.fetch_snapshot(symbol)
            logger.info(f"{symbol.upper()} 重新同步完成")
        except Exception as e:
            logger.error(f"{symbol.upper()} 重新同步失败: {e}", exc_info=True)

    # ----------------  WebSocket 连接 (Reader)  ----------------
    async def stream_loop(self):
        # (这个函数 'stream_loop' 保持不变)
        """[!! 这是 Reader 任务 !!]"""
        streams = [f"{s.lower()}@depth{DEPTH_SPEED}" for s in SYMBOLS]
        sub = {"method": "SUBSCRIBE", "params": streams, "id": 1}
        sslctx = ssl.create_default_context()
        sslctx.check_hostname = False
        sslctx.verify_mode = ssl.CERT_NONE
        conn_kw = {'ssl': sslctx}
        if PROXY:
            conn_kw['proxy'] = PROXY

        # [!! 关键修复 !!] 添加 User-Agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            while True:
                try:
                    async with session.ws_connect(STREAM_URL, **conn_kw) as ws:
                        await ws.send_str(json.dumps(sub))
                        logger.info("订阅已发送 %s", streams)
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                break
                            data = json.loads(msg.data)
                            if data.get('e') == 'depthUpdate':
                                await self._write_queue.put(data)
                            elif data.get('result') is None and 'id' in data:
                                logger.info("订阅成功 %s", data)
                            else:
                                logger.debug("忽略帧 %s", data)
                except Exception as e:
                    logger.error("Stream 异常 %s，5 秒后重连", e, exc_info=True)
                    await asyncio.sleep(5)

    # ----------------  写入者任务  ----------------
    async def writer_loop(self):
        # (这个函数 'writer_loop' 保持不变)
        """[!! 这是 Writer 任务 !!]"""
        logger.info("Writer 写入循环已启动.")
        while True:
            try:
                data = await asyncio.wait_for(self._write_queue.get(), timeout=WRITE_SEC)
                await self.apply_diff_and_persist(data)
                self._write_queue.task_done()
                
            except asyncio.TimeoutError:
                if self._buf:
                    logger.info(f"Writer 因超时 ({WRITE_SEC}s) 触发 flush...")
                    await self.flush()

                    # [!! 新增逻辑 !!] 检查是否需要保存完整快照
            except Exception as e:
                logger.error(f"Writer 循环异常: {e}", exc_info=True)
                await asyncio.sleep(1) 
            
            now = time.time()
            if now - self._last_snapshot_save >= self._SNAPSHOT_INTERVAL:
                await self.save_full_books_snapshot()
                self._last_snapshot_save = now

    async def save_full_books_snapshot(self):
        """
        将内存中所有 'synced' 的 order books 存为一个新快照
        """
        logger.info("正在执行定期完整快照保存...")
        snapshot_time = datetime.utcnow()
        rows_to_insert = []
        
        # [!! 关键 !!]
        # 我们需要安全地访问 self.books，但 writer_loop 是单线程处理
        # apply_diff_and_persist，所以在这里短时间访问是安全的。
        # 如果担心，可以增加一个 self._book_lock
        for symbol, book in self.books.items():
            if not book.synced:
                continue # 只保存已同步的
            
            update_id = book.last_update_id
            
            # 收集 Bids
            for px, qty in book.bids.items():
                rows_to_insert.append(
                    (snapshot_time, symbol.upper(), 'bid', px, qty, update_id)
                )
            # 收集 Asks
            for px, qty in book.asks.items():
                rows_to_insert.append(
                    (snapshot_time, symbol.upper(), 'ask', px, qty, update_id)
                )

        if not rows_to_insert:
            logger.info("没有已同步的 books，跳过快照保存")
            return

        try:
            # 使用你已有的异步写入逻辑
            insert_call = functools.partial(
                self.client.insert,
                'depth_snapshot',
                rows_to_insert,
                column_names=[
                    'snapshot_time', 'symbol', 'side', 
                    'price', 'quantity', 'update_id'
                ]
            )
            
            async with self._db_lock:
                await self.loop.run_in_executor(None, insert_call)
            
            logger.info(f"成功保存 {len(rows_to_insert)} 条定期快照数据")
                
        except Exception as e:
            logger.error(f"定期快照写入失败: {e}", exc_info=True)

    # ----------------  主流程  ----------------
    async def run(self):
        # (这个函数 'run' 保持不变)
        snapshot_tasks = [self.fetch_snapshot(sym) for sym in SYMBOLS]
        results = await asyncio.gather(*snapshot_tasks, return_exceptions=True) 
        
        for i, res in enumerate(results):
            sym = SYMBOLS[i].lower()
            if isinstance(res, Exception):
                logger.error(f"启动时拉取 {sym} 快照失败: {res}", exc_info=True)
            elif res:
                self.books[res.symbol.lower()] = res

        if not self.books:
             logger.error("所有快照均拉取失败，程序退出。")
             return

        writer_task = asyncio.create_task(self.writer_loop())
        reader_task = asyncio.create_task(self.stream_loop())

        try:
            await asyncio.gather(reader_task, writer_task)
        except Exception as e:
            logger.error(f"主循环出错: {e}", exc_info=True)
        finally:
            logger.info("主循环停止，开始清理...")
            reader_task.cancel()
            writer_task.cancel()
            await asyncio.gather(reader_task, writer_task, return_exceptions=True)

    # ----------------  清理函数  ----------------
    async def close(self):
        # (这个函数 'close' 保持不变)
        if self.client:
            if self._buf:
                logger.info("正在关闭... Flush 最后的数据...")
                try:
                    await self.flush() 
                except Exception as e:
                    logger.error(f"关闭时 flush 失败: {e}", exc_info=True)
            self.client.close()
        if self.session and not self.session.closed:
            await self.session.close() 
        logger.info("Collector 已关闭")


# ========================================================================
async def main():
    # (这个函数 'main' 保持不变)
    col = DepthFullCollector()
    try:
        await col.run()
    finally:
        await col.close()


if __name__ == '__main__':
    # (这个函数 'main' 保持不变)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("用户中断")