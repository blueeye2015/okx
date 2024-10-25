import okex.Market_api as Market
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from websocket import WebSocketApp
import json
import threading
import time
import sys
import signal

# InfluxDB 配置
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "SHKUNhCDk25b9mBZD9cQnTd5JI8Bwj6t8tQctQZKvomLSI6W5fZacdwgwQtc89HFmPbUqsNk3bUFBbl4urjddw=="
INFLUXDB_ORG = "marketdata"
INFLUXDB_BUCKET = "history_trades"

# OKEx API 配置
API_KEY = ""
SECRET_KEY = ""
PASSPHRASE = ""
FLAG = '0'  # 实盘

# WebSocket 配置
WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

# 要订阅的产品ID
INST_ID = "BTC-USDT"

# 全局变量用于控制WebSocket循环
running = True

# 初始化 InfluxDB 客户端
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

def on_message(ws, message):
    global running
    try:
        data = json.loads(message)
        
        if 'data' in data:
            for item in data['data']:
                point = (
                    Point("tickers")
                    .tag("instId", item['instId'])
                    .field("last", float(item['last']))
                    .field("lastSz", float(item['lastSz']))
                    .field("askPx", float(item['askPx']))
                    .field("askSz", float(item['askSz']))
                    .field("bidPx", float(item['bidPx']))
                    .field("bidSz", float(item['bidSz']))
                    .field("open24h", float(item['open24h']))
                    .field("high24h", float(item['high24h']))
                    .field("low24h", float(item['low24h']))
                    .field("volCcy24h", float(item['volCcy24h']))
                    .field("vol24h", float(item['vol24h']))
                    .time(int(item['ts']))
                )
                
                write_api.write(bucket=INFLUXDB_BUCKET, record=point)
                print(f"Data written to InfluxDB: {item['instId']} at {item['ts']}")
        elif 'event' in data and data['event'] == 'error':
            print(f"WebSocket error: {data}")
            running = False
    except Exception as e:
        print(f"Error processing message: {e}")
        running = False

def on_error(ws, error):
    global running
    print(f"WebSocket error: {error}")
    running = False

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket connection closed: {close_status_code} - {close_msg}")

def on_open(ws):
    subscribe_message = {
        "op": "subscribe",
        "args": [
            {
                "channel": "tickers",
                "instId": INST_ID
            }
        ]
    }
    ws.send(json.dumps(subscribe_message))
    print(f"Subscribed to tickers for {INST_ID}")

def run_websocket():
    global running
    while running:
        try:
            ws = WebSocketApp(WS_URL,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
            
            ws.run_forever()
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            time.sleep(5)  # 等待5秒后重试

def signal_handler(signum, frame):
    global running
    print("Interrupt received, stopping...")
    running = False

def main():
    global running
    # 初始化 OKEx API
    marketAPI = Market.MarketAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)

    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动 WebSocket 连接
    websocket_thread = threading.Thread(target=run_websocket)
    websocket_thread.start()

    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping...")
    finally:
        running = False
        websocket_thread.join(timeout=5)
        client.close()
        print("Script stopped.")

if __name__ == "__main__":
    main()
