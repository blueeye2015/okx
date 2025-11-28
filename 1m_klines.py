import asyncio
import json
import datetime
import logging
import time
import aiohttp
from websockets import connect, ConnectionClosed
import clickhouse_connect

# --- 配置区 ---
# OKX WebSocket URL (公共频道)
# 实盘: wss://ws.okx.com:8443/ws/v5/public
# 模拟盘: wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999
OKX_URL = "wss://ws.okx.com:8443/ws/v5/business"

# 代理配置 (如果不需要，请设为 None)
PROXY = 'http://127.0.0.1:7890'
# PROXY = None

# ClickHouse数据库配置
CH_HOST = 'localhost'
CH_PORT = 8123
CH_DATABASE = 'marketdata'


# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 数据缓冲配置 (批量插入以提高性能)
BATCH_SIZE = 500  # 积攒100条数据后批量插入
MAX_FLUSH_INTERVAL = 10  # 或者每隔5秒插入一次，以防数据量少时长时间不插入
data_buffer = []
last_flush_time = time.time()

# --- 核心功能函数 ---

def load_symbols(filename="/data/okx/symbol.txt"):
    """从文件中加载要订阅的交易对列表"""
    try:
        with open(filename, 'r') as f:
            # 读取每一行，去除空白，并过滤掉空行
            symbols = [line.strip() for line in f if line.strip()]
            if not symbols:
                logging.warning(f"'{filename}' 文件为空或不存在, 未加载任何交易对。")
                return []
            logging.info(f"成功从 '{filename}' 加载 {len(symbols)} 个交易对: {symbols}")
            return symbols
    except FileNotFoundError:
        logging.error(f"错误: '{filename}' 文件未找到。请创建该文件并填入交易对。")
        return []

def get_clickhouse_client():
    """创建并返回一个ClickHouse客户端连接"""
    try:
        # 在这里添加了 username 和 password 参数
        client = clickhouse_connect.get_client(
            host=CH_HOST, 
            port=CH_PORT, 
            database=CH_DATABASE,
            username='default',
            password='12'  # 您的密码
        )
        logging.info(f"成功以 'default' 用户连接到ClickHouse数据库 (host: {CH_HOST})")
        return client
    except Exception as e:
        # 如果密码错误，通常会在这里抛出异常
        logging.error(f"连接ClickHouse失败: {e}")
        logging.error("请检查您的ClickHouse主机、端口、用户名和密码是否正确。")
        raise

async def flush_data_to_clickhouse(client):
    """将缓冲区的数据批量插入到ClickHouse"""
    global data_buffer, last_flush_time
    if not data_buffer:
        return

    try:
        # 字段顺序必须与CREATE TABLE语句中的列完全对应
        column_names = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        client.insert('okx_klines_1m', data_buffer, column_names=column_names)
        logging.info(f"成功将 {len(data_buffer)} 条K线数据插入ClickHouse。")
        data_buffer.clear()
        last_flush_time = time.time()
    except Exception as e:
        logging.error(f"插入数据到ClickHouse时发生错误: {e}")


def process_message(msg):
    """处理从WebSocket接收到的消息，并将其格式化后放入缓冲区"""
    global data_buffer
    try:
        data = json.loads(msg)
        
        if 'event' in data:
            logging.info(f"收到事件消息: {data}")
            return

        if 'arg' in data and data['arg']['channel'] == 'candle1m' and 'data' in data:
            symbol = data['arg']['instId']
            for kline in data['data']:
                # kline 格式: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
                # confirm 字段在索引 8 的位置
                is_confirmed = kline[8]

                # --- 核心改动：只处理已确认的K线 ---
                if is_confirmed == '1':
                    formatted_kline = [
                        datetime.datetime.fromtimestamp(int(kline[0]) / 1000), # 时间戳
                        symbol,              # 交易对
                        float(kline[1]),     # 开盘价
                        float(kline[2]),     # 最高价
                        float(kline[3]),     # 最低价
                        float(kline[4]),     # 收盘价
                        float(kline[5])      # 交易量
                    ]
                    data_buffer.append(formatted_kline)
                    # 可以在日志中看到我们只处理了确认的数据
                    # logging.info(f"已处理 {symbol} 的确认K线: {formatted_kline}")
                
    except json.JSONDecodeError:
        if msg == 'pong':
            logging.info("收到Pong心跳回应。")
        else:
            logging.warning(f"无法解析JSON消息: {msg}")
    except Exception as e:
        logging.error(f"处理消息时发生未知错误: {e}, 消息内容: {msg}")

async def websocket_client(url, symbols, db_client, proxy=None):
    """主WebSocket客户端，负责连接、订阅、接收数据和重连"""
    if not symbols:
        logging.error("没有要订阅的交易对，程序退出。")
        return
        
    # 根据symbols列表构建订阅请求
    channels = [{"channel": "candle1m", "instId": s} for s in symbols]
    subscribe_request = json.dumps({"op": "subscribe", "args": channels})

    while True:
        try:
            logging.info("尝试连接到WebSocket服务器...")
            async with connect(url, ping_interval=20, ping_timeout=10) as ws:
                logging.info("WebSocket连接成功!")
                
                # 发送订阅请求
                await ws.send(subscribe_request)
                logging.info(f"已发送订阅请求: {subscribe_request}")

                # 持续接收消息
                while True:
                    message = await ws.recv()
                    process_message(message)
                    
                    # 检查是否需要刷新缓冲区
                    if len(data_buffer) >= BATCH_SIZE or (time.time() - last_flush_time > MAX_FLUSH_INTERVAL):
                        await flush_data_to_clickhouse(db_client)

        except (ConnectionClosed, ConnectionRefusedError, asyncio.TimeoutError) as e:
            logging.warning(f"WebSocket连接断开: {e}. 5秒后将自动重连...")
        except Exception as e:
            logging.error(f"发生未知连接错误: {e}. 5秒后将自动重连...")
        
        # 发生任何异常后，等待一段时间再重连
        await asyncio.sleep(5)


# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 加载交易对
    target_symbols = load_symbols()
    
    # 2. 连接数据库
    clickhouse_client = get_clickhouse_client()
    
    # 3. 启动异步WebSocket客户端
    if target_symbols and clickhouse_client:
        try:
            asyncio.run(websocket_client(OKX_URL, target_symbols, clickhouse_client, PROXY))
        except KeyboardInterrupt:
            logging.info("程序被手动中断。")
        finally:
            # 程序退出前，确保缓冲区最后的数据也被写入
            asyncio.run(flush_data_to_clickhouse(clickhouse_client))
            clickhouse_client.close()
            logging.info("数据库连接已关闭。")