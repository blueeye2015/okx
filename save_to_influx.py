from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta
import pytz
import time

class Books:
    def __init__(self):
        self.bids_p = {}  # 保存当前全量买单数据
        self.asks_p = {}  # 保存当前全量卖单数据
        self.inst_id = None

