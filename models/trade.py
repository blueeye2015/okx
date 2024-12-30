from dataclasses import dataclass
from datetime import datetime
from typing import Union, Dict, List

@dataclass
class Trade:
    """成交数据类"""
    trade_id: str             # 成交ID（主键）
    symbol: str               # 交易对
    event_type: str          # 事件类型
    event_time: datetime     # 事件时间
    price: float             # 成交价格
    quantity: float          # 成交数量
    buyer_order_maker: bool  # 是否是买方挂单
    trade_time: datetime     # 成交时间

    @classmethod
    def from_exchange_data(cls, data: List) -> 'Trade':
        """
        从现有交易所数据创建Trade对象
        data格式: [instId, tradeId, px, sz, side, timestamp]
        """
        timestamp = datetime.fromtimestamp(data[5] / 1000)  # 转换毫秒时间戳
        return cls(
            trade_id=int(data[1]),
            symbol=data[0],
            event_type='trade',  # 固定为trade类型
            event_time=timestamp,  # 使用相同的时间
            price=float(data[2]),
            quantity=float(data[3]),
            buyer_order_maker=True if data[4].lower() == 'buy' else False,  # 根据side转换
            trade_time=timestamp
        )

    @classmethod
    def from_binance_ws(cls, data: Dict) -> 'Trade':
        """
        从Binance WebSocket数据创建Trade对象
        
        Args:
            data: Binance WebSocket推送的交易数据
            示例:
            {
                "e": "trade",        // 事件类型
                "E": 123456789,      // 事件时间
                "s": "BNBBTC",       // 交易对
                "t": 12345,          // 交易ID
                "p": "0.001",        // 价格
                "q": "100",          // 数量
                "b": 88,             // 买方订单ID
                "a": 50,             // 卖方订单ID
                "T": 123456785,      // 交易时间
                "m": true,           // 是否是买方挂单
                "M": true            // 是否忽略
            }
        """
        return cls(
            trade_id=int(data['t']),
            symbol=data['s'],
            event_type=data['e'],
            event_time=datetime.fromtimestamp(data['E'] / 1000),
            price=float(data['p']),
            quantity=float(data['q']),
            buyer_order_maker=data['m'],
            trade_time=datetime.fromtimestamp(data['T'] / 1000)
        )

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'event_type': self.event_type,
            'event_time': self.event_time,
            'price': self.price,
            'quantity': self.quantity,
            'buyer_order_maker': self.buyer_order_maker,
            'trade_time': self.trade_time
        }

    def __str__(self) -> str:
        """字符串表示"""
        return (f"Trade(trade_id={self.trade_id}, "
                f"symbol={self.symbol}, "
                f"price={self.price}, "
                f"quantity={self.quantity}, "
                f"trade_time={self.trade_time})")
