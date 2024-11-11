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
from influxdb_client import InfluxDBClient
from bookdata import OrderBookInfluxWriter

 # 创建InfluxDB客户端
client = InfluxDBClient(
        url = "http://localhost:8086",
        token = "aEOL_HtGEGsMLtPp1KI3y0VA5L8MxRSAOWDXZh2TektZoizfZ5pGKPPLjSnBVxo9HQjND865WKN8ehFY-e7HOA==",
        org = "marketdata"
    )

    # 创建writer实例
writer = OrderBookInfluxWriter(
        client = client,
        org = "marketdata",
        bucket ="history_trades"
    )

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


def partial(res):
    data_obj = res['data'][0]
    bids = data_obj['bids']
    asks = data_obj['asks']
    instrument_id = res['arg']['instId']
    # print('全量数据bids为：' + str(bids))
    # print('档数为：' + str(len(bids)))
    # print('全量数据asks为：' + str(asks))
    # print('档数为：' + str(len(asks)))
    return bids, asks, instrument_id


def update_bids(res, bids_p):
    # 获取增量bids数据
    bids_u = res['data'][0]['bids']
    # print('增量数据bids为：' + str(bids_u))
    # print('档数为：' + str(len(bids_u)))
    # bids合并
    for i in bids_u:
        bid_price = i[0]
        for j in bids_p:
            if bid_price == j[0]:
                if i[1] == '0':
                    bids_p.remove(j)
                    break
                else:
                    del j[1]
                    j.insert(1, i[1])
                    break
        else:
            if i[1] != "0":
                bids_p.append(i)
    else:
        bids_p.sort(key=lambda price: sort_num(price[0]), reverse=True)
        # print('合并后的bids为：' + str(bids_p) + '，档数为：' + str(len(bids_p)))
    return bids_p


def update_asks(res, asks_p):
    # 获取增量asks数据
    asks_u = res['data'][0]['asks']
    # print('增量数据asks为：' + str(asks_u))
    # print('档数为：' + str(len(asks_u)))
    # asks合并
    for i in asks_u:
        ask_price = i[0]
        for j in asks_p:
            if ask_price == j[0]:
                if i[1] == '0':
                    asks_p.remove(j)
                    break
                else:
                    del j[1]
                    j.insert(1, i[1])
                    break
        else:
            if i[1] != "0":
                asks_p.append(i)
    else:
        asks_p.sort(key=lambda price: sort_num(price[0]))
        # print('合并后的asks为：' + str(asks_p) + '，档数为：' + str(len(asks_p)))
    return asks_p


def sort_num(n):
    if n.isdigit():
        return int(n)
    else:
        return float(n)


def check(bids, asks):
    # 获取bid档str
    bids_l = []
    bid_l = []
    count_bid = 1
    while count_bid <= 25:
        if count_bid > len(bids):
            break
        bids_l.append(bids[count_bid-1])
        count_bid += 1
    for j in bids_l:
        str_bid = ':'.join(j[0 : 2])
        bid_l.append(str_bid)
    # 获取ask档str
    asks_l = []
    ask_l = []
    count_ask = 1
    while count_ask <= 25:
        if count_ask > len(asks):
            break
        asks_l.append(asks[count_ask-1])
        count_ask += 1
    for k in asks_l:
        str_ask = ':'.join(k[0 : 2])
        ask_l.append(str_ask)
    # 拼接str
    num = ''
    if len(bid_l) == len(ask_l):
        for m in range(len(bid_l)):
            num += bid_l[m] + ':' + ask_l[m] + ':'
    elif len(bid_l) > len(ask_l):
        # bid档比ask档多
        for n in range(len(ask_l)):
            num += bid_l[n] + ':' + ask_l[n] + ':'
        for l in range(len(ask_l), len(bid_l)):
            num += bid_l[l] + ':'
    elif len(bid_l) < len(ask_l):
        # ask档比bid档多
        for n in range(len(bid_l)):
            num += bid_l[n] + ':' + ask_l[n] + ':'
        for l in range(len(bid_l), len(ask_l)):
            num += ask_l[l] + ':'

    new_num = num[:-1]
    int_checksum = zlib.crc32(new_num.encode())
    fina = change(int_checksum)
    return fina

# unsubscribe channels
async def unsubscribe_without_login(url, channels, proxy=None):
    if proxy:
        ws = await create_connection_with_proxy(url, proxy)
    else:
        ws = await connect(url)
        
    async with ws:
        # unsubscribe
        sub_param = {"op": "unsubscribe", "args": channels}
        sub_str = json.dumps(sub_param)
        await ws.send(sub_str)
        print(f"send: {sub_str}")

        res = await ws.recv()
        print(f"recv: {res}")


def change(num_old):
    num = pow(2, 31) - 1
    if num_old > num:
        out = num_old - num * 2 - 2
    else:
        out = num_old
    return out

def create_orderbook_message(bids: list, asks: list, inst_id: str , ts: str ) -> dict:
    """
    将bid和ask列表转换为标准的订单簿消息格式
    
    Args:
        bids: bid列表
        asks: ask列表
        inst_id: 交易对ID，默认为"BTC-USDT"
        ts: 时间戳，如果为None则使用当前时间
    
    Returns:
        dict: 标准格式的订单簿消息
    """
    
    # 格式化bid和ask列表
    def format_level(level):
        # 如果输入已经是完整格式
        if len(level) >= 4:
            return [str(level[0]), str(level[1]), str(level[2]), str(level[3])]
        # 如果只有价格和数量
        elif len(level) == 2:
            return [str(level[0]), str(level[1]), "0", "1"]
        # 如果是其他格式，保持原样并补充缺失字段
        else:
            return [str(level[0]), str(level[1]), "0", "1"]
    
    formatted_bids = [format_level(bid) for bid in bids]
    formatted_asks = [format_level(ask) for ask in asks]
    
    return {
        "arg": {
            "channel": "books",
            "instId": inst_id
        },
        "action": "snapshot",
        "data": [{
            "asks": formatted_asks,
            "bids": formatted_bids,
            "ts": str(ts)           
        }]
    }



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

                #print(f"{get_timestamp()} {res_data}")
                res = eval(res[1])
                if 'event' in res:
                    continue
                for i in res['arg']:
                    if 'books' in res['arg'][i] and 'books5' not in res['arg'][i]:
                        # 订阅频道是深度频道
                        if res['action'] == 'snapshot':
                            for m in l:
                                if res['arg']['instId'] == m['instrument_id']:
                                    l.remove(m)
                            # 获取首次全量深度数据
                            bids_p, asks_p, instrument_id = partial(res)
                            d = {}
                            d['instrument_id'] = instrument_id
                            d['bids_p'] = bids_p
                            d['asks_p'] = asks_p
                            l.append(d)

                            # 校验checksum
                            checksum = res['data'][0]['checksum']
                            # print('推送数据的checksum为：' + str(checksum))
                            check_num = check(bids_p, asks_p)
                            # print('校验后的checksum为：' + str(check_num))
                            if check_num == checksum:
                                print("校验结果为：True")
                                bookdata = json.loads(res_data)
                                writer.process_order_book(bookdata)
                            else:
                                print("校验结果为：False，正在重新订阅……")

                                # 取消订阅
                                await unsubscribe_without_login(url, channels, proxy)
                                # 发送订阅
                                if proxy:
                                    new_ws = await create_connection_with_proxy(url, proxy)
                                else:
                                    new_ws = await connect(url)
                                    sub_param = {"op": "subscribe", "args": channels}
                                    sub_str = json.dumps(sub_param)
                                    await ws.send(sub_str)
                                    print(f"send: {sub_str}")

                        elif res['action'] == 'update':
                            for j in l:
                                if res['arg']['instId'] == j['instrument_id']:
                                    # 获取全量数据
                                    bids_p = j['bids_p']
                                    asks_p = j['asks_p']
                                    # 获取合并后数据
                                    bids_p = update_bids(res, bids_p)
                                    asks_p = update_asks(res, asks_p)

                                    # 校验checksum
                                    checksum = res['data'][0]['checksum']
                                    # print('推送数据的checksum为：' + str(checksum))
                                    check_num = check(bids_p, asks_p)
                                    # print('校验后的checksum为：' + str(check_num))
                                    if check_num == checksum:
                                        print("校验结果为：True")
                                        bookdata=create_orderbook_message(bids_p,asks_p,res['arg']['instId'],res['data'][0]['ts'])
                                        writer.process_order_book(bookdata)
                                    else:
                                        print("校验结果为：False，正在重新订阅……")

                                        # 取消订阅
                                        await unsubscribe_without_login(url, channels, proxy)
                                        # 发送订阅
                                        if proxy:
                                            new_ws = await create_connection_with_proxy(url, proxy)
                                        else:
                                            new_ws = await connect(url)
                                        async with new_ws:
                                            sub_param = {"op": "subscribe", "args": channels}
                                            sub_str = json.dumps(sub_param)
                                            await new_ws.send(sub_str)
                                            print(f"send: {sub_str}") 
                                     
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
        {"channel": "books", "instId": "HMSTR-USDT"}
    ]

    # 运行
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(subscribe_without_login(url, channels, proxies))
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        loop.close()