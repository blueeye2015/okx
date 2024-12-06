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
from config.influxdb.bookdata import OrderBookInfluxWriter



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


def login_params(timestamp, api_key, passphrase, secret_key):
    message = timestamp + 'GET' + '/users/self/verify'

    mac = hmac.new(bytes(secret_key, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
    d = mac.digest()
    sign = base64.b64encode(d)

    login_param = {"op": "login", "args": [{"apiKey": api_key,
                                            "passphrase": passphrase,
                                            "timestamp": timestamp,
                                            "sign": sign.decode("utf-8")}]}
    login_str = json.dumps(login_param)
    return login_str



# subscribe channels un_need login
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
                
                                     
        except Exception as e:
            print(f"连接断开，正在重连…… 错误: {e}")
        finally:
            if ws:
                await ws.close()
            if session:
                await session.close()
            await asyncio.sleep(5)  # 等待5秒后重试

if __name__ == "__main__":
    # 使用示例
    proxies = 'http://127.0.0.1:7890'
    url = "wss://ws.okx.com:8443/ws/v5/public"
    channels = [
        {"channel": "books5", "instId": "CETUS-USDT"}
    ]

    # 运行
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(subscribe_without_login(url, channels, proxies))
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        loop.close()