import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase
import enum

class Base(DeclarativeBase):
    pass

class KlineModel(Base):
    __tablename__ = 'klines'
    
    symbol = sa.Column(sa.String, primary_key=True)
    timestamp = sa.Column(sa.DateTime, primary_key=True)
    open = sa.Column(sa.Float)
    high = sa.Column(sa.Float)
    low = sa.Column(sa.Float)
    close = sa.Column(sa.Float)
    volume = sa.Column(sa.Float)
    created_at = sa.Column(sa.DateTime, server_default=sa.func.now())


class TradeModel(Base):
    """交易数据模型"""
    __tablename__ = 'trades'
    
    # 主键字段
    trade_id = sa.Column(sa.BigInteger, primary_key=True, comment='交易ID')
    
    # 交易基本信息
    symbol = sa.Column(sa.String, nullable=False, comment='交易对符号')
    event_type = sa.Column(sa.String, nullable=False, comment='事件类型')
    event_time = sa.Column(sa.DateTime, nullable=False, comment='事件时间')
    price = sa.Column(sa.Float, nullable=False, comment='成交价格')
    quantity = sa.Column(sa.Float, nullable=False, comment='成交数量')
    buyer_order_maker = sa.Column(sa.Boolean, nullable=False, comment='是否是买方挂单')
    trade_time = sa.Column(sa.DateTime, nullable=False, comment='交易时间')
    
    
    __table_args__ = (
        # 添加索引以优化查询性能
        sa.Index('ix_trades_symbol', 'symbol'),
        sa.Index('ix_trades_trade_time', 'trade_time'),
        sa.Index('ix_trades_symbol_trade_time', 'symbol', 'trade_time'),
        {'comment': '币安交易数据表'}
    )