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

