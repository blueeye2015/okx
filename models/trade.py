from dataclasses import dataclass
from datetime import datetime

@dataclass
class trade:
    """成交数据类"""
    symbol: str
    tradeId: str
    px: float
    sz: float
    side: str
    timestamp: datetime
    
    @classmethod
    def from_exchange_data(cls, symbol: str, data: list) -> 'trade':
        """
        从交易所数据创建Kline对象
        data格式: [instId, tradeId, px, sz, side, timestamp]
        """
        return cls(
            symbol=data[0],
            timestamp=datetime.fromtimestamp(data[5] / 1000),  # 转换毫秒时间戳
            tradeId=data[1],
            px=float(data[2]),
            sz=float(data[3]),
            side=float(data[4])
        )
        
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'tradeId': self.tradeId,
            'px': self.px,
            'sz': self.sz,
            'side': self.side
        }