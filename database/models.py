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

class TrendType(enum.Enum):
    UP = "up"
    DOWN = "down"
    SHOCK = "shock"
    
class TradeSignal(Base):
    __tablename__ = 'trade_signals'
    
    id = sa.Column(sa.Integer, primary_key=True)
    symbol = sa.Column(sa.String(20), nullable=False)
    side = sa.Column(sa.String(4), nullable=False)  # buy/sell
    vol_beishu = sa.Column(sa.Float, nullable=False)  # 量比
    vol_eq_forward_minutes = sa.Column(sa.Integer, nullable=False)  # 相当于前多少分钟的总量
    trend = sa.Column(enum(TrendType), nullable=False)
    created_at = sa.Column(sa.DateTime, nullable=False)