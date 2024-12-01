from abc import ABC, abstractmethod
from typing import List, Optional
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.future import select
from database.models import KlineModel
from models.kline import Kline
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

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

    async def save_klines(self, kline_models: List[Kline]):
        async with self.db_manager.get_session() as session:
            try:
                stmt = insert(KlineModel)
                values = [vars(model) for model in kline_models]
                
                stmt = stmt.values(values).on_conflict_do_update(
                    index_elements=['symbol', 'timestamp'],
                    set_={
                        'open': stmt.excluded.open,
                        'high': stmt.excluded.high,
                        'low': stmt.excluded.low,
                        'close': stmt.excluded.close,
                        'volume': stmt.excluded.volume
                    }
                )
                
                await session.execute(stmt)
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
            finally:
                await session.close()

    async def get_latest_kline(self, symbol: str) -> Optional[Kline]:
        """获取指定交易对的最新K线数据（同步方式）"""
        async with self.db_manager.get_session() as session:
            query = select(KlineModel).filter(
                KlineModel.symbol == symbol
            ).order_by(
                KlineModel.timestamp.desc()
            ).limit(1)

            result = await session.execute(query)
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
        