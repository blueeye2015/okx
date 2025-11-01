#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OKX 完整深度轮询 → ClickHouse
限速：10次/2s，单 IP
"""
import asyncio
import json
import logging
import time
import datetime as dt
from decimal import Decimal
from typing import List, Tuple

import aiohttp
import clickhouse_connect

# ---------------- 配置 ----------------
CLICKHOUSE = dict(
    host='localhost', port=8123, database='marketdata',
    username='default', password='12'
)
PROXY = "http://127.0.0.1:7890"          # 不需要就写 None
SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]  # OKX 格式
SZ = 100                                   # 档位数量
BATCH_SIZE = 100
MAX_FLUSH_INTERVAL = 5

# OKX REST
REST_URL = "https://www.okx.com/api/v5/market/books-full"
LIMIT_QPS = 10 / 2                        # 官方限速
# -------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

buffer: List[Tuple] = []
last_flush = time.time()


# ---------------- 工具 ----------------
def get_client():
    cli = clickhouse_connect.get_client(**CLICKHOUSE)
    logger.info("Connected to ClickHouse: %s", cli.url)
    return cli


def ensure_table(client: clickhouse_connect.driver.Client):
    client.command("""
        CREATE TABLE IF NOT EXISTS okx_depth_full (
            ts          DateTime64(3, 'Asia/Shanghai'),
            symbol      String,
            side        String,
            price       Float64,
            size        Float64,
            order_count UInt32
        ) ENGINE = MergeTree()
        ORDER BY (symbol, side, ts)
    """)


async def flush(client):
    global buffer, last_flush
    if not buffer:
        return
    client.insert(
        'okx_depth_full',
        buffer,
        column_names=['ts', 'symbol', 'side', 'price', 'size', 'order_count']
    )
    logger.info("Flushed %d rows", len(buffer))
    buffer.clear()
    last_flush = time.time()


# ---------------- 拉取 + 解析 ----------------
async def fetch_one(session: aiohttp.ClientSession, symbol: str):
    params = {"instId": symbol, "sz": str(SZ)}
    async with session.get(REST_URL, params=params, proxy=PROXY) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"{resp.status} {text}")
        return await resp.json()


def parse(symbol: str, raw: dict) -> List[Tuple]:
    if raw.get("code") != "0":
        raise RuntimeError(raw)
    data = raw["data"][0]
    ts = int(data["ts"])
    rows = []
    for side, key in (("bid", "bids"), ("ask", "asks")):
        for lvl in data[key]:
            price, size, cnt = map(Decimal, lvl)
            rows.append((
                dt.datetime.fromtimestamp(ts / 1000),
                symbol,
                side,
                float(price),
                float(size),
                int(cnt)
            ))
    return rows


# ---------------- 轮询主循环 ----------------
async def poll(client):
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            t0 = time.time()
            for sym in SYMBOLS:
                try:
                    raw = await fetch_one(session, sym)
                    buffer.extend(parse(sym, raw))
                except Exception as e:
                    logger.error("Failed to fetch %s: %s", sym, e)

                # 限速：10次/2s
                await asyncio.sleep(1 / LIMIT_QPS)

            # flush 策略同 1m_klines.py
            if len(buffer) >= BATCH_SIZE or time.time() - last_flush >= MAX_FLUSH_INTERVAL:
                await flush(client)


# ---------------- 入口 ----------------
async def main():
    client = get_client()
    ensure_table(client)
    try:
        await poll(client)
    except KeyboardInterrupt:
        logger.info("Exit by user")
    finally:
        await flush(client)
        client.close()


if __name__ == "__main__":
    asyncio.run(main())