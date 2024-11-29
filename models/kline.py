from dataclasses import dataclass
from datetime import datetime

@dataclass
class Kline:
    """K线数据类"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @classmethod
    def from_exchange_data(cls, symbol: str, data: list) -> 'Kline':
        """
        从交易所数据创建Kline对象
        data格式: [timestamp, open, high, low, close, volume]
        """
        return cls(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(data[0] / 1000),  # 转换毫秒时间戳
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5])
        )
        
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }