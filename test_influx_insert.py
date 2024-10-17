from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import time

url = "http://localhost:8086"
token = "SHKUNhCDk25b9mBZD9cQnTd5JI8Bwj6t8tQctQZKvomLSI6W5fZacdwgwQtc89HFmPbUqsNk3bUFBbl4urjddw=="
org = "marketdata"
bucket = "history_trades"
current_time_ms = int(time.time() * 1e9)
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

try:
    point = Point("test_trade").tag("instId", "BTC-USDT").tag("side", "buy").field("sz", 67392).field("sz",0.008)
    write_api.write(bucket=bucket, record=point)
    print("测试数据点已成功写入")
except Exception as e:
    print("写入错误:", str(e))
    if hasattr(e, 'response'):
        print("响应状态码:", e.response.status_code)
        print("响应内容:", e.response.text)

client.close()
