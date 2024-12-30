from abc import ABC, abstractmethod
from typing import List, Optional
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.future import select
from database.models import KlineModel,TradeModel
from models.kline import Kline
from models.trade import Trade
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
    async def insert(self, data): pass
    
    @abstractmethod
    async def query(self, **kwargs): pass

class KlineDAO(BaseDAO):
        
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
        
    async def insert(self, trade: Trade):
        """插入单条交易数据"""
        if not isinstance(trade, Trade):
            raise ValueError(f"Expected Trade object, got {type(trade)}")
            
        async with self.db_manager.get_session() as session:
            try:
                # 直接创建字典
                trade_data = {
                    'trade_id': int(trade.trade_id),  # 确保是字符串
                    'symbol': str(trade.symbol),
                    'event_type': str(trade.event_type),
                    'event_time': trade.event_time,
                    'price': float(trade.price),
                    'quantity': float(trade.quantity),
                    'buyer_order_maker': bool(trade.buyer_order_maker),
                    'trade_time': trade.trade_time
                }
                
                stmt = insert(TradeModel).values(trade_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol','trade_id'],
                    set_=trade_data
                )
                
                await session.execute(stmt)
                await session.commit()
            except Exception as e:
                await session.rollback()
                logging.error(f"Insert trade error: {str(e)}, trade data: {trade_data}")
                raise e
            finally:
                await session.close()       

    async def save_trades(self, trades: List[Trade]):
        """批量保存交易数据"""
        if not trades:
            return
            
        if not isinstance(trades, (list, tuple)):
            raise ValueError(f"Expected list of Trade objects, got {type(trades)}")
            
        async with self.db_manager.get_session() as session:
            try:
                # 添加类型检查和数据转换
                values = []
                for trade in trades:
                    if not isinstance(trade, Trade):
                        raise ValueError(f"Expected Trade object, got {type(trade)}")
                        
                    values.append({
                        'trade_id': int(trade.trade_id),
                        'symbol': str(trade.symbol),
                        'event_type': str(trade.event_type),
                        'event_time': trade.event_time,
                        'price': float(trade.price),
                        'quantity': float(trade.quantity),
                        'buyer_order_maker': bool(trade.buyer_order_maker),
                        'trade_time': trade.trade_time
                    })
                
                # 使用参数绑定的方式执行SQL
                stmt = text("""
                    INSERT INTO trades (
                        trade_id, symbol, event_type, event_time, price,
                        quantity, buyer_order_maker, trade_time
                    )
                    VALUES (
                        :trade_id, :symbol, :event_type, :event_time, :price,
                        :quantity, :buyer_order_maker, :trade_time
                    )
                    ON CONFLICT (symbol, trade_id) DO UPDATE SET                        
                        event_type = EXCLUDED.event_type,
                        event_time = EXCLUDED.event_time,
                        price = EXCLUDED.price,
                        quantity = EXCLUDED.quantity,
                        buyer_order_maker = EXCLUDED.buyer_order_maker,
                        trade_time = EXCLUDED.trade_time
                """)
                
                # 打印调试信息
                logging.debug(f"Executing batch insert with {len(values)} trades")
                
                await session.execute(stmt, values)
                await session.commit()
                
            except Exception as e:
                await session.rollback()
                logging.error(f"Save trades error: {str(e)}")
                # 打印更详细的错误信息
                if values:
                    logging.error(f"First trade in batch: {values[0]}")
                raise e
            
    async def get_latest_trade(self, symbol: str) -> Optional[Trade]:
        """获取指定交易对的最新交易数据"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = select(TradeModel).filter(
                    TradeModel.symbol == symbol
                ).order_by(
                    TradeModel.trade_time.desc()  # 使用trade_time替代timestamp
                ).limit(1)
                
                result = await session.execute(stmt)
                row = result.scalar_one_or_none()
                
                if row:
                    return Trade(
                        trade_id=row.trade_id,
                        symbol=row.symbol,
                        event_type=row.event_type,
                        event_time=row.event_time,
                        price=row.price,
                        quantity=row.quantity,
                        buyer_order_maker=row.buyer_order_maker,
                        trade_time=row.trade_time
                    )
                return None
                
            except Exception as e:
                logging.error(f"获取最新交易数据失败: {e}")
                raise

    async def query(self, symbol: str = None,
              start_time: datetime = None,
              end_time: datetime = None) -> List[Trade]:
        """查询交易数据"""
        async with self.db_manager.get_session() as session:
            query = select(TradeModel)
            
            if symbol:
                query = query.filter(TradeModel.symbol == symbol)
            if start_time:
                query = query.filter(TradeModel.trade_time >= start_time)
            if end_time:
                query = query.filter(TradeModel.trade_time <= end_time)
                
            query = query.order_by(TradeModel.trade_time)
            
            result = await session.execute(query)
            rows = result.scalars().all()
            
            return [
                Trade(
                    trade_id=row.trade_id,
                    symbol=row.symbol,
                    event_type=row.event_type,
                    event_time=row.event_time,
                    price=row.price,
                    quantity=row.quantity,
                    buyer_order_maker=row.buyer_order_maker,
                    trade_time=row.trade_time
                ) for row in rows
            ]
