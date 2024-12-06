from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS


tz = pytz.timezone('Asia/Shanghai')  # GMT+8

class OrderBookInfluxWriter:
    def __init__(self, client: InfluxDBClient, bucket: str, org: str):
        """
        初始化OrderBookInfluxWriter
        
        Args:
            client: InfluxDB客户端实例
            bucket: 存储的bucket名称
            org: 组织名称
        """
        self.client = client
        self.bucket = bucket
        self.org = org
        self.write_api = self.client.write_api()
        
        
    def process_order_book(self, data: dict) -> None:
        """
        处理订单簿数据并写入InfluxDB
        
        Args:
            data: 原始订单簿数据
        """
        try:
            # 解析基础数据
            inst_id = data['arg']['instId']
            book_data = data['data'][0]
            timestamp = int(book_data['ts'])  # 确保时间戳是整数
            
            
            points = []
            
            # 处理asks数据
            counter = 0
            for i,ask in enumerate(book_data['asks'],1):
                price, amount, _, _ = ask
                point = (Point("asks")
                        .tag("instId", inst_id)
                        .field("price", float(price))
                        .field("amount", float(amount))
                        .tag("level", i)
                        .time(datetime.fromtimestamp(timestamp / 1000, tz=tz)))
                points.append(point)
                
                
            
            # 处理bids数据
            counter = 0 #counter置零
            for i, bid in enumerate(book_data['bids'], 1):
                price, amount, _, _ = bid
                point = (Point("bids")
                        .tag("instId", inst_id)
                        .field("price", float(price))
                        .field("amount", float(amount))
                        .tag("level", i)
                        .time(datetime.fromtimestamp(timestamp / 1000, tz=tz)))
                points.append(point)
                
                 
            
            # 批量写入数据
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
            
        except Exception as e:
            print(f"Error processing order book data: {e}")
            raise

    def close(self):
        """关闭写入API"""
        self.write_api.close()

