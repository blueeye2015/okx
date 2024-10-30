from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

@dataclass
class BaseMarketData:
    inst_id: str
    timestamp: int
    channel: str
    inst_type: str 

    def to_influx_point(self) -> Point:
        """转换为InfluxDB的Point对象"""
        point = Point(self.get_measurement())
        
        # 添加通用标签
        for tag_key, tag_value in self.get_tags().items():
            point = point.tag(tag_key, tag_value)
        
        # 添加字段
        for field_key, field_value in self.get_fields().items():
            point = point.field(field_key, field_value)
        
        # 设置时间戳
        point = point.time(self.timestamp)
        
        return point

    @abstractmethod
    def get_measurement(self) -> str:
        """获取measurement名称"""
        return "tickers"
        pass

    def get_tags(self) -> Dict[str, str]:
        """获取标签字段"""
        return {
            "instId": self.inst_id,
            "instType": self.inst_type,
            "channel": self.channel
        }

    @abstractmethod
    def get_fields(self) -> Dict[str, float]:
        """获取数值字段"""
        pass

@dataclass
class TickerData(BaseMarketData):
    inst_type: str
    inst_id: str
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
    
    def get_measurement(self) -> str:
        return "tickers"
    
    @classmethod
    def from_json(cls, json_data: Dict) -> 'TickerData':
        """从JSON数据创建TickerData对象"""
        data = json_data['data'][0]  # 获取data数组的第一个元素
        arg = json_data['arg']       # 获取arg对象
        return cls(
            # BaseMarketData的必需参数
            inst_type=data['instType'],
            inst_id=data['instId'],
            timestamp=int(data['ts']),
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


class InfluxDBManager:
    def __init__(self, url: str, token: str, org: str, bucket: str):
        """
        初始化InfluxDB管理器
        
        Args:
            url: InfluxDB服务器地址
            token: 认证令牌
            org: 组织名称
            bucket: 数据桶名称
        """
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.bucket = bucket
        self.org = org
        
    def write_data(self, data: BaseMarketData) -> None:
        """
        写入单条数据
        
        Args:
            data: 市场数据对象
        """
        try:
            point = data.to_influx_point()
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
        except Exception as e:
            print(f"Error writing data: {e}")
            
    def write_batch(self, data_list: List[BaseMarketData]) -> None:
        """
        批量写入数据
        
        Args:
            data_list: 市场数据对象列表
        """
        try:
            points = [data.to_influx_point() for data in data_list]
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
        except Exception as e:
            print(f"Error writing batch data: {e}")
    
    def close(self):
        """关闭数据库连接"""
        self.client.close()

