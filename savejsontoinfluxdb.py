from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json
from datetime import datetime, timezone
import pytz

url = "http://localhost:8086"
token = "SHKUNhCDk25b9mBZD9cQnTd5JI8Bwj6t8tQctQZKvomLSI6W5fZacdwgwQtc89HFmPbUqsNk3bUFBbl4urjddw=="
org = "marketdata"
bucket = "history_trades"

tz = pytz.timezone('Asia/Shanghai')  # GMT+8
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

try:
    with open('D:\okx\history_trade.json', 'r') as file:
        json_data = json.load(file)

    for trade in json_data['data']:
        ts_int = int(trade['ts'])
    
        # 假设 ts 是毫秒级时间戳
        trade_time = datetime.fromtimestamp(ts_int / 1000, tz=tz)
        point = Point("test") \
            .tag("instId", trade['instId']) \
            .tag("side", trade['side']) \
            .tag("tradeId", trade['tradeId']) \
            .field("sz", float(trade['sz'])) \
            .field("px", float(trade['px'])) \
            .time(trade_time)  # 使用当前时间而不是 trade['ts']

        try:
            write_api.write(bucket=bucket, record=point)
            print(f"Successfully wrote point: {point}")
            print(f"Writing point: {point.to_line_protocol()}")

        except Exception as e:
            print(f"Error writing point: {e}")

    print("All data has been processed")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    client.close()
