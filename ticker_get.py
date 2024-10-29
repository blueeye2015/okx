import asyncio
import websockets
import json
import requests
import hmac
import base64
import zlib
import datetime
import time
import aiohttp
from websockets import connect, ConnectionClosed
from websockets.client import WebSocketClientProtocol
from ticker_to_influxdb import TickerData,InfluxDBManager


INFLUX_CONFIG = {
    "url": "http://localhost:8086",
    "token": "your-token",
    "org": "marketdata",
    "bucket": "ticker"
}

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
    l = []
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
                #如果是行情数据则存入influxdb
                
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


proxies = 'http://127.0.0.1:7890'
# WebSocket公共频道
# 实盘
url = "wss://ws.okx.com:8443/ws/v5/public"
# 模拟盘
# url = "wss://ws.okex.com:8443/ws/v5/public?brokerId=9999"

# WebSocket私有频道
# 实盘
# url = "wss://ws.okex.com:8443/ws/v5/private"
# 模拟盘
# url = "wss://ws.okex.com:8443/ws/v5/private?brokerId=9999"

'''
公共频道
:param channel: 频道名
:param instType: 产品类型
:param instId: 产品ID
:param uly: 合约标的指数

'''

# 行情频道
channels = [{"channel": "tickers", "instId": "BTC-USDT"}]

'''
私有频道
:param channel: 频道名
:param ccy: 币种
:param instType: 产品类型
:param uly: 合约标的指数
:param instId: 产品ID

'''





loop = asyncio.get_event_loop()

# 公共频道 不需要登录（行情，持仓总量，K线，标记价格，深度，资金费率等）
loop.run_until_complete(subscribe_without_login(url, channels,proxies))

loop.close()