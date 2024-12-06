import asyncio
import json
import aiohttp
from dataclasses import dataclass
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta
import pytz

tz = pytz.timezone('Asia/Shanghai')  # GMT+8
# InfluxDB配置
INFLUX_CONFIG = {
    "url": "http://localhost:8086",
    "token": "aEOL_HtGEGsMLtPp1KI3y0VA5L8MxRSAOWDXZh2TektZoizfZ5pGKPPLjSnBVxo9HQjND865WKN8ehFY-e7HOA==",
    "org": "marketdata",
    "bucket": "history_trades"
}

@dataclass
class TickerData:
    inst_id: str
    inst_type: str
    timestamp: int
    channel: str
    last: float
    last_sz: float
    ask_px: float
    ask_sz: float
    bid_px: float
    bid_sz: float
    open_24h: float
    high_24h: float
    low_24h: float
    sod_utc0: float
    sod_utc8: float
    vol_ccy_24h: float
    vol_24h: float
    
    @classmethod
    def from_json(cls, json_data: dict) -> 'TickerData':
        """从JSON数据创建TickerData对象"""
        try:
            data = json_data['data'][0]  # 获取data数组的第一个元素
            arg = json_data['arg']       # 获取arg对象
            ts_int = int(data['ts'])
            trade_time = datetime.fromtimestamp(ts_int / 1000, tz=tz)
            return cls(
                inst_id=data['instId'],
                inst_type=data['instType'],
                timestamp=trade_time,
                channel=arg['channel'],
                last=float(data['last']),
                last_sz=float(data['lastSz']),
                ask_px=float(data['askPx']),
                ask_sz=float(data['askSz']),
                bid_px=float(data['bidPx']),
                bid_sz=float(data['bidSz']),
                open_24h=float(data['open24h']),
                high_24h=float(data['high24h']),
                low_24h=float(data['low24h']),
                sod_utc0=float(data['sodUtc0']),
                sod_utc8=float(data['sodUtc8']),
                vol_ccy_24h=float(data['volCcy24h']),
                vol_24h=float(data['vol24h'])
            )
        except Exception as e:
            print(f"Error creating TickerData from JSON: {e}")
            print(f"JSON data: {json_data}")
            raise

class TickerWriter:
    def __init__(self, config):
        """初始化InfluxDB连接"""
        self.client = InfluxDBClient(
            url=config["url"],
            token=config["token"],
            org=config["org"]
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.bucket = config["bucket"]
        self.org = config["org"]

    def write_ticker(self, ticker: TickerData) -> None:
        """写入单条Ticker数据"""
        try:
            point = (Point("tickers")
                    .tag("instId", ticker.inst_id)
                    .tag("instType", ticker.inst_type)
                    .tag("channel", ticker.channel)
                    .field("last", ticker.last)
                    .field("lastSz", ticker.last_sz)
                    .field("askPx", ticker.ask_px)
                    .field("askSz", ticker.ask_sz)
                    .field("bidPx", ticker.bid_px)
                    .field("bidSz", ticker.bid_sz)
                    .field("open24h", ticker.open_24h)
                    .field("high24h", ticker.high_24h)
                    .field("low24h", ticker.low_24h)
                    .field("sodUtc0", ticker.sod_utc0)
                    .field("sodUtc8", ticker.sod_utc8)
                    .field("volCcy24h", ticker.vol_ccy_24h)
                    .field("vol24h", ticker.vol_24h)
                    .time(ticker.timestamp))

            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            print(f"Successfully wrote ticker data for {ticker.inst_id} at {datetime.now()}")
        except Exception as e:
            print(f"Error writing ticker data: {e}")
            print(f"Problematic data: {ticker}")

    def close(self):
        """关闭数据库连接"""
        if self.client:
            self.client.close()

async def create_connection_with_proxy(url, proxy):
    """创建带代理的WebSocket连接"""
    session = aiohttp.ClientSession()
    ws = await session.ws_connect(url, proxy=proxy)
    return ws, session

def get_timestamp():
    """获取当前时间戳"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

async def subscribe_without_login(url, channels, proxy=None):
    # 创建数据库写入器
    db_writer = TickerWriter(INFLUX_CONFIG)
    
    while True:
        ws = None
        session = None
        try:
            if proxy:
                ws, session = await create_connection_with_proxy(url, proxy)
            else:
                session = aiohttp.ClientSession()
                ws = await session.ws_connect(url)
            
            # 发送订阅请求
            sub_param = {"op": "subscribe", "args": channels}
            await ws.send_str(json.dumps(sub_param))
            print(f"Subscribed to channels: {channels}")

            while True:
                try:
                    # 接收消息
                    msg = await asyncio.wait_for(ws.receive(), timeout=25)
                    
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = msg.data
                        #print(f"{get_timestamp()} Received: {data}")
                        
                        # 解析数据
                        ticker_data = json.loads(data)
                        
                        # 只处理包含'data'的消息（行情数据）
                        if 'data' in ticker_data:
                            try:
                                # 转换为TickerData对象并存储
                                ticker = TickerData.from_json(ticker_data)
                                db_writer.write_ticker(ticker)
                            except Exception as e:
                                print(f"Error processing ticker data: {e}")
                        
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        raise ConnectionError("WebSocket connection closed")
                        
                except asyncio.TimeoutError:
                    # 发送ping保持连接
                    try:
                        await ws.send_str('ping')
                        pong = await ws.receive()
                        print(f"Ping response: {pong.data}")
                        continue
                    except Exception as e:
                        print(f"Ping failed: {e}")
                        break
                        
                except Exception as e:
                    print(f"Error receiving message: {e}")
                    break
                    
        except Exception as e:
            print(f"Connection error, reconnecting... Error: {e}")
            
        finally:
            if ws:
                await ws.close()
            if session:
                await session.close()
            await asyncio.sleep(5)  # 重连前等待5秒

if __name__ == "__main__":
    # 使用示例
    proxies = 'http://127.0.0.1:7890'
    url = "wss://ws.okx.com:8443/ws/v5/public"
    channels = [{"channel": "tickers", "instId": "GRASS-USDT-SWAP"}]

    # 运行
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(subscribe_without_login(url, channels, proxies))
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        loop.close()
