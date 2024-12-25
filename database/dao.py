from abc import ABC, abstractmethod
from typing import List, Optional
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.future import select
from database.models import KlineModel,TradeModel
from models.kline import Kline
from models.trade import trade
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import time
from functools import wraps
import asyncio
from sqlalchemy.sql import text

def async_timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        task_id = id(asyncio.current_task())
        logging.info(f"Starting task {func.__name__} (ID: {task_id})")
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start
            logging.info(f"Task {func.__name__} (ID: {task_id}) completed in {elapsed:.2f} seconds")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logging.error(f"Task {func.__name__} (ID: {task_id}) failed after {elapsed:.2f} seconds: {str(e)}")
            raise
    return wrapper


class BaseDAO(ABC):
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    @abstractmethod
    async def create_table(self): pass
    
    @abstractmethod
    async def insert(self, data): pass
    
    @abstractmethod
    async def query(self, **kwargs): pass

class KlineDAO(BaseDAO):
    async def create_table(self):
        pass
    
    #@async_timer
    async def insert(self, kline: Kline):
        """插入单条数据"""
        async with self.db_manager.get_session() as session:
            try:
                kline_model = KlineModel(
                    symbol=kline.symbol,
                    timestamp=kline.timestamp,
                    open=kline.open,
                    high=kline.high,
                    low=kline.low,
                    close=kline.close,
                    volume=kline.volume
                )
                
                stmt = insert(KlineModel).values(
                    vars(kline_model)
                ).on_conflict_do_update(
                    index_elements=['symbol', 'timestamp'],
                    set_={
                        'open': kline_model.open,
                        'high': kline_model.high,
                        'low': kline_model.low,
                        'close': kline_model.close,
                        'volume': kline_model.volume
                    }
                )
                
                await session.execute(stmt)
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
            finally:
                await session.close()
    
    #@async_timer
    async def save_klines(self, kline_models: List[Kline]):
        if not kline_models:
            return
        async with self.db_manager.get_session() as session:
            try:
                # 使用批量插入
                values = [{
                    'symbol': model.symbol,
                    'timestamp': model.timestamp,
                    'open': model.open,
                    'high': model.high,
                    'low': model.low,
                    'close': model.close,
                    'volume': model.volume
                } for model in kline_models]
                
                await session.execute(
                    text("""
                    INSERT INTO klines_5m (symbol, timestamp, open, high, low, close, volume)
                    VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)
                    ON CONFLICT (symbol, timestamp) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                    """),
                    values
                )
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
    
    #@async_timer
    async def get_latest_kline(self, symbol: str) -> Optional[Kline]:
        """获取指定交易对的最新K线数据（同步方式）"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = select(KlineModel).filter(
                    KlineModel.symbol == symbol
                ).order_by(
                    KlineModel.timestamp.desc()
                ).limit(1)
                
                result = await session.execute(stmt)
                row = result.scalar_one_or_none()
                
                if row:
                    return Kline(
                        symbol=row.symbol,
                        timestamp=row.timestamp,
                        open=row.open,
                        high=row.high,
                        low=row.low,
                        close=row.close,
                        volume=row.volume
                    )
                return None
                
            except Exception as e:
                logging.error(f"获取最新K线数据失败: {e}")
                raise
           
    #@async_timer
    async def query(self, symbol: str = None, 
              start_time: datetime = None, 
              end_time: datetime = None) -> List[Kline]:
        """查询数据"""
        async with self.db_manager.get_session() as session:
            query = select(KlineModel)
            
            
            if symbol:
                query = query.filter(KlineModel.symbol == symbol)
            if start_time:
                query = query.filter(KlineModel.timestamp >= start_time)
            if end_time:
                query = query.filter(KlineModel.timestamp <= end_time)
                
            query = query.order_by(KlineModel.timestamp)
            
            return [
                Kline(
                    symbol=row.symbol,
                    timestamp=row.timestamp,
                    open=row.open,
                    high=row.high,
                    low=row.low,
                    close=row.close,
                    volume=row.volume
                ) for row in query.all()
            ]

class TradeDAO(BaseDAO):
    async def create_table(self):
        pass
    
    #@async_timer
    async def insert(self, trade: trade):
        """插入单条数据"""
        async with self.db_manager.get_session() as session:
            try:
                trade_model = TradeModel(
                    symbol=trade.symbol,
                    timestamp=trade.timestamp,
                    tradeId=trade.tradeId,
                    px=trade.px,
                    sz=trade.sz,
                    side=trade.side
                )
                
                stmt = insert(TradeModel).values(
                    vars(trade_model)
                ).on_conflict_do_update(
                    index_elements=['symbol', 'timestamp'],
                    set_={
                        'tradeId': trade_model.tradeId,
                        'px': trade_model.px,
                        'sz': trade_model.sz,
                        'side': trade_model.side
                    }
                )
                
                await session.execute(stmt)
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
            finally:
                await session.close()        

    #@async_timer
    async def save_trade(self, trade_models: List[trade]):
        if not trade_models:
            return
        async with self.db_manager.get_session() as session:
            try:
                # 使用批量插入
                values = [{
                    'symbol': model.symbol,
                    'timestamp': model.timestamp,
                    'tradeId': model.tradeId,
                    'px': model.px,
                    'sz': model.sz,
                    'side': model.side
                } for model in trade_models]
                
                await session.execute(
                    text("""
                    INSERT INTO trade_data (symbol, timestamp, tradeId, px, sz, side)
                    VALUES (:symbol, :timestamp, :tradeId, :px, :sz, :side) 
                    ON CONFLICT (symbol, tradeId, timestamp) DO UPDATE SET
                        px = EXCLUDED.px,
                        sz = EXCLUDED.sz                                     
                    """),
                    values
                )
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
            
    #@async_timer
    async def query(self, symbol: str = None, 
              start_time: datetime = None, 
              end_time: datetime = None) -> List[trade]:
        """查询数据"""
        async with self.db_manager.get_session() as session:
            query = select(TradeModel)
            
            
            if symbol:
                query = query.filter(TradeModel.symbol == symbol)
            if start_time:
                query = query.filter(TradeModel.timestamp >= start_time)
            if end_time:
                query = query.filter(TradeModel.timestamp <= end_time)
                
            query = query.order_by(TradeModel.timestamp)
            
            return [
                trade(
                    symbol=row.symbol,
                    timestamp=row.timestamp,
                    side=row.side,
                    px=row.px,
                    sz=row.sz,
                    tradeId=row.tradeId
                ) for row in query.all()
            ]