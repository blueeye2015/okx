from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json

url = "http://localhost:8086"
token = "SHKUNhCDk25b9mBZD9cQnTd5JI8Bwj6t8tQctQZKvomLSI6W5fZacdwgwQtc89HFmPbUqsNk3bUFBbl4urjddw=="
org = "marketdata"
bucket = "history_trades"

client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

try:
    with open('D:\okx\history_trade.json', 'r') as file:
        json_data = json.load(file)

    for trade in json_data['data']:
        point = Point("trades") \
            .tag("instId", trade['instId']) \
            .tag("side", trade['side']) \
            .field("sz", float(trade['sz'])) \
            .field("px", float(trade['px'])) \
            .field("tradeId", trade['tradeId']) \
            .time(int(trade['ts']))

        try:
            write_api.write(bucket=bucket, record=point)
            print(f"Successfully wrote point: {point}")
        except Exception as e:
            print(f"Error writing point: {e}")

    print("All data has been processed")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    client.close()
