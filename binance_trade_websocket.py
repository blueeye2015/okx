import json
import asyncio
import platform
import aiohttp
import psycopg2
from datetime import datetime
import logging
from typing import Dict, Any
import ssl

# Windows系统特殊处理
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    'dbname': 'market_data',
    'user': 'postgres',
    'password': '12',
    'host': 'localhost',
    'port': '5432'
}

# WebSocket配置
WEBSOCKET_URL = "wss://stream.binance.com:9443/ws"
PROXY = "http://127.0.0.1:7890"  # 如果需要代理，设置为 "http://your_proxy_host:your_proxy_port"

# 要订阅的交易对
SYMBOLS = ["ethusdt", "btcusdt","uxlinkusdt","pnutusdt","meusdt","penguusdt","moveusdt"]  # 可以添加更多交易对

class BinanceTradeCollector:
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.setup_database()

    def setup_database(self):
        """设置数据库连接和表结构"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor()
            
            # 创建trades表
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    event_type VARCHAR(50),
                    event_time TIMESTAMP,
                    symbol VARCHAR(20),
                    trade_id BIGINT,
                    price DECIMAL,
                    quantity DECIMAL,
                    buyer_order_maker BOOLEAN,
                    trade_time TIMESTAMP,
                    PRIMARY KEY (symbol, trade_id)
                )
            """)
            self.conn.commit()
            logger.info("数据库连接成功并创建表完成")
        except Exception as e:
            logger.error(f"数据库设置错误: {e}")
            raise

    def insert_trade(self, trade_data: Dict[Any, Any]):
        """将交易数据插入数据库"""
        try:
            # 转换 symbol 格式
            symbol = trade_data['s']  # 例如 "BTCUSDT"
            # 在 USDT 前添加连字符
            formatted_symbol = f"{symbol[:-4]}-{symbol[-4:]}"  # "BTC-USDT"
            self.cursor.execute("""
                INSERT INTO trades (
                    event_type, event_time, symbol, trade_id, 
                    price, quantity, buyer_order_maker, trade_time
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, trade_id) DO NOTHING
            """, (
                trade_data['e'],
                datetime.fromtimestamp(trade_data['E'] / 1000),
                formatted_symbol.upper(),
                trade_data['t'],
                float(trade_data['p']),
                float(trade_data['q']),
                trade_data['m'],
                datetime.fromtimestamp(trade_data['T'] / 1000)
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"插入数据错误: {e}")
            self.conn.rollback()

    async def handle_websocket(self):
        """处理WebSocket连接和数据接收"""
        # 构建订阅消息
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol}@trade" for symbol in SYMBOLS],
            "id": 1
        }

        # 创建 SSL 上下文
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # 设置连接参数
        conn_kwargs = {
            'ssl': ssl_context,
            'timeout': aiohttp.ClientTimeout(total=60)
        }
        
        if PROXY:
            conn_kwargs['proxy'] = PROXY

        # 创建 aiohttp session
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.ws_connect(WEBSOCKET_URL, **conn_kwargs) as websocket:
                        # 发送订阅消息
                        await websocket.send_str(json.dumps(subscribe_message))
                        logger.info("已发送订阅请求")

                        async for msg in websocket:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                
                                # 跳过订阅确认消息
                                if 'result' in data:
                                    continue
                                    
                                self.insert_trade(data)
                                #logger.info(f"已处理交易数据: {data['s']} - {data['t']}")
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                logger.warning("WebSocket连接关闭")
                                break
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f"WebSocket错误: {msg.data}")
                                break

                except aiohttp.ClientError as e:
                    logger.error(f"WebSocket连接错误: {e}")
                    await asyncio.sleep(5)  # 等待5秒后重试
                except Exception as e:
                    logger.error(f"处理数据时发生错误: {e}")
                    await asyncio.sleep(5)  # 等待5秒后重试

    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("数据库连接已关闭")

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
    except Exception as e:
        logger.error(f"程序异常退出: {e}")