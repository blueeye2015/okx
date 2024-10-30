import asyncio
import json
import requests
import datetime
import time
import aiohttp
from websockets import connect, ConnectionClosed
from websockets.client import WebSocketClientProtocol
from ticker_to_influxdb import TickerData,InfluxDBManager
from queue import Queue, Empty
from threading import Thread

INFLUX_CONFIG = {
    "url": "http://localhost:8086",
    "token": "SHKUNhCDk25b9mBZD9cQnTd5JI8Bwj6t8tQctQZKvomLSI6W5fZacdwgwQtc89HFmPbUqsNk3bUFBbl4urjddw==",
    "org": "marketdata",
    "bucket": "history_trades"
}

class DatabaseWriter:
    def __init__(self, config, buffer_size=1000):
        self.db_manager = InfluxDBManager(**config)
        self.queue = Queue(maxsize=buffer_size)
        self.running = True
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def write(self, ticker):
        """将数据放入队列"""
        self.queue.put(ticker)

    def _worker(self):
        """后台工作线程，批量写入数据"""
        batch = []
        batch_size = 5  # 每批处理的数据量
        batch_timeout = 1  # 最大等待时间（秒）

        while self.running:
            try:
                # 收集批量数据
                start_time = time.time()
                while len(batch) < batch_size and time.time() - start_time < batch_timeout:
                    try:
                        ticker = self.queue.get(timeout=0.1)
                        batch.append(ticker)
                    except Empty:  # 使用导入的 Empty 异常
                        break

                # 如果有数据则批量写入
                if batch:
                    try:
                        for ticker in batch:
                            self.db_manager.write_data(ticker)
                        batch = []
                    except Exception as e:
                        print(f"Error writing to database: {e}")

            except Exception as e:
                print(f"Worker thread error: {e}")

    def close(self):
        """关闭数据库连接"""
        self.running = False
        self.worker_thread.join()
        self.db_manager.close()
        
async def create_connection_with_proxy(uri, proxy):
    try:
        session = aiohttp.ClientSession()
        ws = await session.ws_connect(uri, proxy=proxy)
        return ws, session
    except Exception as e:
        print(f"连接失败: {e}")
        raise

    
def get_timestamp():
    now = datetime.datetime.now()
    t = now.isoformat("T", "milliseconds")
    return t + "Z"


def get_server_time():
    url = "https://www.okex.com/api/v5/public/time"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['data'][0]['ts']
    else:
        return ""


def get_local_timestamp():
    return int(time.time())


async def subscribe_without_login(url, channels, proxy=None):
    # 创建数据库写入器（全局单例）
    db_writer = DatabaseWriter(INFLUX_CONFIG)
    
    while True:
        ws = None
        session = None
        try:
            if proxy:
                ws, session = await create_connection_with_proxy(url, proxy)
            else:
                ws = await connect(url)
                
            
            sub_param = {"op": "subscribe", "args": channels}
            sub_str = json.dumps(sub_param)
            if isinstance(ws, aiohttp.ClientWebSocketResponse):
                await ws.send_str(sub_str)
            else:
                await ws.send(sub_str)
            print(f"send: {sub_str}")


            while True:
                try:
                    if isinstance(ws, aiohttp.ClientWebSocketResponse):
                        res = await asyncio.wait_for(ws.receive(), timeout=25)
                        if res.type == aiohttp.WSMsgType.TEXT:
                            res_data = res.data
                        elif res.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            raise ConnectionClosed(1000, "Connection closed")
                    else:
                        res_data = await asyncio.wait_for(ws.recv(), timeout=25)

                except asyncio.TimeoutError:
                    try:
                        if isinstance(ws, aiohttp.ClientWebSocketResponse):
                            await ws.send_str('ping')
                            res = await ws.receive()
                            if res.type == aiohttp.WSMsgType.TEXT:
                                print(res.data)
                        else:
                            await ws.send('ping')
                            res_data = await ws.recv()
                            print(res_data)
                        continue
                    except Exception as e:
                        print(f"Ping失败，连接可能已关闭: {e}")
                        break
                except Exception as e:
                    print(f"接收消息时发生错误: {e}")
                    break

                print(f"{get_timestamp()} {res_data}")
                ticker_data = json.loads(res_data)
                
                # 创建数据库管理器并写入数据
                if 'data' in ticker_data:  # 只有在成功解析时才写入
                    #如果是行情数据则存入influxdb
                    ticker = TickerData.from_json(ticker_data)
                    db_writer.write(ticker)
                res = eval(res[1])
                if 'event' in res:
                    continue
                
                                     
        except Exception as e:
            print(f"连接断开，正在重连…… 错误: {e}")
        finally:
            if ws:
                await ws.close()
            if session:
                await session.close()
            await asyncio.sleep(5)  # 等待5秒后重试

if __name__ == "__main__":
    proxies = 'http://127.0.0.1:7890'
    url = "wss://ws.okx.com:8443/ws/v5/public"
    channels = [{"channel": "tickers", "instId": "BTC-USDT"}]

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(subscribe_without_login(url, channels, proxies))
    finally:
        loop.close()