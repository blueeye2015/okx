#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[程序 A] 纯增量数据存档器
功能：只通过 WebSocket 接收 depthUpdate，直接存入 depth 表。
特点：不请求 REST API，不维护 OrderBook，永不触发 418 错误。
"""
import json
import asyncio
import platform
import aiohttp
import ssl
import logging
import clickhouse_connect
import time
from datetime import datetime
from collections import deque

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
STREAM_URL = "wss://stream.binance.com:9443/ws"
SYMBOLS = ['btcusdt', 'ethusdt']
DEPTH_SPEED = '@100ms'
PROXY = "http://127.0.0.1:7890"  # 如果不需要代理设为 None
WRITE_BATCH = 1000  # 批量写入阈值
WRITE_SEC = 5       # 时间阈值

# --- 日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StreamArchiver")

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class StreamArchiver:
    def __init__(self):
        self.client = clickhouse_connect.get_client(**CLICKHOUSE)
        self._buf = deque()
        self._last_flush = time.time()
        
        # 简单的 User-Agent
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
        }

    async def flush(self):
        if not self._buf: return
        
        # 提取数据用于写入
        data_to_write = list(self._buf)
        self._buf.clear()
        self._last_flush = time.time()
        
        try:
            # 使用 run_in_executor 避免阻塞 Event Loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: self.client.insert(
                'depth',
                data_to_write,
                column_names=['event_time', 'symbol', 'side', 'price', 'quantity', 'first_update_id', 'final_update_id']
            ))
            logger.info(f"✅ 已存档增量数据: {len(data_to_write)} 条")
        except Exception as e:
            logger.error(f"❌ 写入失败: {e}")

    async def buffer_data(self, data):
        """解析 WebSocket JSON 并放入缓存"""
        try:
            symbol = data['s'].upper()
            first_id = data['U']
            final_id = data['u']
            ts = datetime.fromtimestamp(data['E'] / 1000)
            
            # 处理 Bids
            for px, qty in data['b']:
                self._buf.append((ts, symbol, 'bid', float(px), float(qty), first_id, final_id))
            
            # 处理 Asks
            for px, qty in data['a']:
                self._buf.append((ts, symbol, 'ask', float(px), float(qty), first_id, final_id))

            # 检查是否需要写入
            if len(self._buf) >= WRITE_BATCH or time.time() - self._last_flush >= WRITE_SEC:
                await self.flush()
                
        except Exception as e:
            logger.error(f"解析数据错误: {e}")

    async def start(self):
        streams = [f"{s.lower()}@depth{DEPTH_SPEED}" for s in SYMBOLS]
        subscribe_msg = {"method": "SUBSCRIBE", "params": streams, "id": 1}
        
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        
        conn_kwargs = {'ssl': ssl_ctx}
        if PROXY: conn_kwargs['proxy'] = PROXY

        async with aiohttp.ClientSession(headers=self.headers) as session:
            while True:
                try:
                    logger.info(f"🔌 连接 WebSocket: {streams}")
                    async with session.ws_connect(STREAM_URL, **conn_kwargs) as ws:
                        await ws.send_str(json.dumps(subscribe_msg))
                        
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                if data.get('e') == 'depthUpdate':
                                    await self.buffer_data(data)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                break
                                
                except Exception as e:
                    logger.error(f"⚠️ 连接断开: {e}，3秒后重连...")
                    await asyncio.sleep(3)

if __name__ == '__main__':
    archiver = StreamArchiver()
    try:
        asyncio.run(archiver.start())
    except KeyboardInterrupt:
        pass