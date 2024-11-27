import os
import time
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

# 1. 配置管理
@dataclass
class DBConfig:
    db_path: str
    read_only: bool = False
    
class Config:
    DB_PATH = "D:/docker/duckdb/market_data.duckdb"
    SYMBOLS = ["BTC-USDT", "ETH-USDT"]  # 交易对列表
    INTERVAL = "1m"  # K线间隔
    BATCH_SIZE = 1000  # 批量插入大小

# 2. 数据模型
@dataclass
class Kline:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
# 3. 数据库连接管理
class DatabaseManager:
    _instance = None
    _conn = None
    
    def __new__(cls, config: DBConfig):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.config = config
        return cls._instance
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(
                self.config.db_path,
                read_only=self.config.read_only
            )
        return self._conn
    
    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
            
    def __enter__(self):
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# 4. 数据访问层基类
class BaseDAO(ABC):
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    @abstractmethod
    def create_table(self):
        pass
    
    @abstractmethod
    def insert(self, data):
        pass
    
    @abstractmethod
    def query(self, **kwargs):
        pass

# 修改 KlineDAO 类，实现所有抽象方法
class KlineDAO(BaseDAO):
    def create_table(self):
        with self.db_manager as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS klines (
                    symbol VARCHAR,
                    timestamp TIMESTAMP,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            
            # 创建索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_klines_symbol_timestamp 
                ON klines(symbol, timestamp)
            """)
    
    def insert(self, kline: Kline):
        """实现单条数据插入方法"""
        with self.db_manager as conn:
            conn.execute("""
                INSERT INTO klines (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """, [kline.symbol, kline.timestamp, kline.open, kline.high, 
                 kline.low, kline.close, kline.volume])
    
    def query(self, symbol: str = None, 
              start_time: datetime = None, 
              end_time: datetime = None) -> List[Kline]:
        """实现查询方法"""
        conditions = []
        params = []
        
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
            
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
            
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
            
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self.db_manager as conn:
            df = conn.execute(f"""
                SELECT symbol, timestamp, open, high, low, close, volume
                FROM klines
                WHERE {where_clause}
                ORDER BY timestamp
            """, params).df()
            
            return [Kline(**row) for row in df.to_dict('records')]
    
    # 保留原有的便利方法
    def insert_batch(self, klines: List[Kline]):
        df = pd.DataFrame([vars(k) for k in klines])
        with self.db_manager as conn:
            conn.execute("""
                INSERT INTO klines (symbol, timestamp, open, high, low, close, volume)
                SELECT symbol, timestamp, open, high, low, close, volume FROM df
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """)
    
    def get_latest_kline(self, symbol: str) -> Optional[Kline]:
        with self.db_manager as conn:
            result = conn.execute("""
                SELECT symbol, timestamp, open, high, low, close, volume
                FROM klines
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, [symbol]).fetchone()
            
            if result:
                return Kline(*result)
            return None
    
    def get_klines(self, symbol: str, 
                   start_time: datetime, 
                   end_time: datetime) -> List[Kline]:
        return self.query(symbol, start_time, end_time)

# 6. 业务逻辑层
class MarketDataService:
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = DatabaseManager(DBConfig(config.DB_PATH))
        self.kline_dao = KlineDAO(self.db_manager)
        self.kline_dao.create_table()
        
    def fetch_klines(self, symbol: str, start_time: datetime) -> List[Kline]:
        """
        从交易所获取K线数据
        这里需要实现具体的数据获取逻辑
        """
        # 示例数据，实际应用中需要替换为真实的API调用
        klines = []
        current_time = start_time
        
        for i in range(10):  # 示例生成10个K线
            kline = Kline(
                symbol=symbol,
                timestamp=current_time,
                open=100 + i,
                high=102 + i,
                low=98 + i,
                close=101 + i,
                volume=1000 + i
            )
            klines.append(kline)
            current_time += timedelta(minutes=1)
            
        return klines
    
    def update_market_data(self):
        """更新所有交易对的市场数据"""
        for symbol in self.config.SYMBOLS:
            try:
                # 获取最新的K线数据
                latest_kline = self.kline_dao.get_latest_kline(symbol)
                start_time = datetime.now() - timedelta(minutes=1000)
                
                if latest_kline:
                    start_time = latest_kline.timestamp + timedelta(minutes=1)
                
                # 获取新数据
                new_klines = self.fetch_klines(symbol, start_time)
                
                # 批量保存
                if new_klines:
                    self.kline_dao.insert_batch(new_klines)
                    logging.info(f"Updated {len(new_klines)} klines for {symbol}")
                    
            except Exception as e:
                logging.error(f"Error updating {symbol}: {str(e)}")

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建服务实例
    config = Config()
    market_service = MarketDataService(config)
    
    # 测试数据访问
    symbol = "BTC-USDT"
    kline = Kline(
        symbol=symbol,
        timestamp=datetime.now(),
        open=50000.0,
        high=51000.0,
        low=49000.0,
        close=50500.0,
        volume=1000.0
    )
    
    # 测试单条插入
    market_service.kline_dao.insert(kline)
    
    # 测试查询
    start_time = datetime.now() - timedelta(days=1)
    end_time = datetime.now()
    klines = market_service.kline_dao.query(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time
    )
    
    logging.info(f"Found {len(klines)} klines for {symbol}")
    
    # 运行数据更新循环
    while True:
        try:
            market_service.update_market_data()
            time.sleep(60)  # 每分钟更新一次
            
        except KeyboardInterrupt:
            logging.info("Stopping market data service...")
            break
            
        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            time.sleep(5)  # 发生错误时等待5秒后重试

if __name__ == "__main__":
    main()

