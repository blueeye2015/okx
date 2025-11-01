import json
import asyncio
import platform
import aiohttp
import ssl
from datetime import datetime
import logging
from typing import Dict, Any
import clickhouse_connect  # 只用这个驱动

# Windows 特殊处理
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------  ClickHouse 配置（HTTP 端口 8123） -----------
CLICKHOUSE = dict(
    host='localhost',
    port=8123,
    database='marketdata',
    username='default',
    password='12'
)

# WebSocket 配置
WEBSOCKET_URL = "wss://stream.binance.com:9443/ws"
PROXY = "http://127.0.0.1:7890"  # 不需要就留空 ""
SYMBOLS = ["ethusdt", "btcusdt", "uxlinkusdt",
           "pnutusdt", "meusdt", "penguusdt", "moveusdt"]


class BinanceTradeCollector:
    def __init__(self):
        self.client = None
        self.setup_database()

    # ----------  建库建表  ----------
    def setup_database(self):
        try:
            # 先连 default 库，确保目标库存在
            tmp = clickhouse_connect.get_client(**{**CLICKHOUSE, 'database': 'default'})
            tmp.command(f"CREATE DATABASE IF NOT EXISTS {CLICKHOUSE['database']}")
            tmp.close()

            # 再连目标库
            self.client = clickhouse_connect.get_client(**CLICKHOUSE)
            self.client.command('''
                CREATE TABLE IF NOT EXISTS trades (
                    event_type        String,
                    event_time        DateTime,
                    symbol            String,
                    trade_id          UInt64,
                    price             Float64,
                    quantity          Float64,
                    buyer_order_maker UInt8,
                    trade_time        DateTime
                ) ENGINE = MergeTree()
                  ORDER BY (symbol, trade_time)
            ''')
            logger.info("ClickHouse 连接成功并创建表完成")
        except Exception as e:
            logger.error(f"数据库设置错误: {e}")
            raise

    # ----------  插入数据  ----------
    def insert_trade(self, trade_data: Dict[Any, Any]):
        try:
            symbol = trade_data['s']          # BTCUSDT
            formatted_symbol = f"{symbol[:-4]}-{symbol[-4:]}"  # BTC-USDT
            self.client.insert(
                "trades",
                [
                    [
                        trade_data['e'],
                        datetime.fromtimestamp(trade_data['E'] / 1000),
                        formatted_symbol.upper(),
                        trade_data['t'],
                        float(trade_data['p']),
                        float(trade_data['q']),
                        int(trade_data['m']),
                        datetime.fromtimestamp(trade_data['T'] / 1000)
                    ]
                ],
                column_names=['event_type', 'event_time', 'symbol',
                              'trade_id', 'price', 'quantity',
                              'buyer_order_maker', 'trade_time']
            )
        except Exception as e:
            logger.error(f"插入数据错误: {e}")

    # ----------  WebSocket 主循环  ----------
    async def handle_websocket(self):
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol}@trade" for symbol in SYMBOLS],
            "id": 1
        }
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        conn_kwargs = {
            'ssl': ssl_context,
            'timeout': aiohttp.ClientTimeout(total=60)
        }
        if PROXY:
            conn_kwargs['proxy'] = PROXY

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.ws_connect(WEBSOCKET_URL, **conn_kwargs) as ws:
                        await ws.send_str(json.dumps(subscribe_message))
                        logger.info("已发送订阅请求")
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                if 'result' in data:     # 订阅回执
                                    continue
                                self.insert_trade(data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED,
                                              aiohttp.WSMsgType.ERROR):
                                logger.warning("WebSocket 连接异常，5 秒后重连")
                                break
                except Exception as e:
                    logger.error(f"WebSocket 异常: {e}")
                    await asyncio.sleep(5)

    def close(self):
        if self.client:
            self.client.close()
        logger.info("ClickHouse 连接已关闭")


async def main():
    collector = BinanceTradeCollector()
    try:
        await collector.handle_websocket()
    finally:
        collector.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("程序被用户中断")