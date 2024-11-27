import os
import time
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import ccxt
import requests

# 1. 添加交易所基类
class ExchangeBase:
    _instance = None
    _exchange = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExchangeBase, cls).__new__(cls)
        return cls._instance
    
    @property
    def exchange(self) -> ccxt.Exchange:
        
        proxies = {
            'http': 'http://127.0.0.1:7890',  # 根据您的实际代理地址修改
            'https': 'http://127.0.0.1:7890'  # 根据您的实际代理地址修改
            }
        if self._exchange is None:
            self._exchange = ccxt.okx({
            'apiKey': 'ba7f444f-e83e-4dd1-8507-bf8dd9033cbc',
            'secret': 'D5474EF76B0A7397BFD26B9656006480',
            'password': 'TgTB+pJoM!d20F',
            'enableRateLimit': True,
            'proxies': proxies,  # 添加代理设置
            'timeout': 30000,    # 设置超时时间为30秒
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
            })
        try:
            self.exchange.load_markets()
            logging.info("交易所连接成功")
        except Exception as e:
            logging.info(f"交易所连接失败: {str(e)}")
            raise e
        return self._exchange
    
    def close(self):
        if self._exchange:
            self._exchange = None

# 1. 配置管理
@dataclass
class DBConfig:
    db_path: str
    read_only: bool = False
    
class Config:
    DB_PATH = "D:/duckdb/market_data.duckdb"
    INTERVAL = "1m"  # K线间隔
    BATCH_SIZE = 1000  # 批量插入大小

    def __init__(self):
            self.market_analyzer = MarketAnalyzer()
            self.update_symbols()
        
    def update_symbols(self):
            self.SYMBOLS = self.market_analyzer.get_valid_symbols()
            logging.info(f"更新交易对列表，共 {len(self.SYMBOLS)} 个")

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
class ExchangeBase:
    _instance = None
    _exchange = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExchangeBase, cls).__new__(cls)
        return cls._instance
    
    @property
    def exchange(self) -> ccxt.Exchange:
        # 代理设置
        proxies = {
            'http': 'http://127.0.0.1:7890',  # 根据您的实际代理地址修改
            'https': 'http://127.0.0.1:7890'  # 根据您的实际代理地址修改
        }
        
        # 初始化交易所API（这里以binance为例）
        if self._exchange is None:
            self._exchange = ccxt.okx({
            'apiKey': 'ba7f444f-e83e-4dd1-8507-bf8dd9033cbc',
            'secret': 'D5474EF76B0A7397BFD26B9656006480',
            'password': 'TgTB+pJoM!d20F',
            'enableRateLimit': True,
            'proxies': proxies,  # 添加代理设置
            'timeout': 30000,    # 设置超时时间为30秒
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
            })
        return self._exchange
    
    def close(self):
        if self._exchange:
            self._exchange = None

class MarketDataService(ExchangeBase):
    def __init__(self, config: Config):
        super().__init__() 
        self.config = config
        self.db_manager = DatabaseManager(DBConfig(config.DB_PATH))
        self.kline_dao = KlineDAO(self.db_manager)
        self.kline_dao.create_table()
        
    def fetch_klines(self, symbol: str, start_time: datetime) -> List[Kline]:
        """
        从交易所获取K线数据
        """           
            # 将时间转换为毫秒时间戳
        since = int(start_time.timestamp() * 1000)
            
            # 获取K线数据
        ohlcv = self.exchange.fetch_ohlcv(
            symbol,  # 将 BTC-USDT 转换为 BTC/USDT
            timeframe='1m',
            since=since,
            limit=1000
        )
            
        # 转换为 Kline 对象列表
        klines = []
        for data in ohlcv:
            timestamp = datetime.fromtimestamp(data[0] / 1000)  # 转换毫秒时间戳为datetime
            kline = Kline(
                symbol=symbol,
                timestamp=timestamp,
                open=float(data[1]),
                high=float(data[2]),
                low=float(data[3]),
                close=float(data[4]),
                volume=float(data[5])
            )
            klines.append(kline)
                
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
                    logging.info(f"更新了 {symbol} 的 {len(new_klines)} 条K线数据")
                    
                # 添加适当的延迟以遵守API限制
                time.sleep(1)  # 根据实际需要调整延迟时间
                    
            except Exception as e:
                logging.error(f"更新 {symbol} 时出错: {str(e)}")

class MarketAnalyzer(ExchangeBase):
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 3600  # 缓存1小时   
        self.proxies = {
            'http': 'http://127.0.0.1:7890',  # 根据你的实际代理地址修改
            'https': 'http://127.0.0.1:7890'  # 根据你的实际代理地址修改
        } 
                    
    def get_market_cap_data(self) -> Dict[str, float]:
        """
        从 CoinGecko 获取市值数据
        返回格式: {'BTC': 800000000000, 'ETH': 200000000000, ...}
        """
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 250,  # 获取前250个币种
                'page': 1,
                'sparkline': False
            }
            response = requests.get(url,proxies=self.proxies, params=params)
            data = response.json()
            
            return {
                item['symbol'].upper(): {
                    'market_cap': item['market_cap'],
                    'first_listed': item.get('genesis_date')  # 上市日期
                }
                for item in data if item['market_cap'] is not None
            }
        except Exception as e:
            logging.error(f"获取市值数据失败: {str(e)}")
            return {}

    def get_valid_symbols(self, min_market_cap: float = 20000000, min_age_months: int = 1) -> List[str]:
        """
        获取符合条件的交易对
        :param min_market_cap: 最小市值（美元）
        :param min_age_months: 最小上市月数
        :return: 符合条件的交易对列表
        """
        try:
            # 获取交易所支持的所有交易对
            markets = self.exchange.load_markets()
            
            # 获取市值数据
            market_cap_data = self.get_market_cap_data()
            
            # 当前时间
            current_time = datetime.now()
            min_list_date = current_time - timedelta(days=30 * min_age_months)
            
            valid_symbols = []
            
            for symbol, market in markets.items():
                try:
                    # 只考虑USDT交易对
                    if not symbol.endswith('/USDT'):
                        continue
                        
                    base_currency = market['base']  # 基础货币 (例如 BTC, ETH)
                    
                    # 检查是否有市值数据
                    if base_currency not in market_cap_data:
                        continue
                        
                    market_info = market_cap_data[base_currency]
                    
                    # 检查市值
                    if market_info['market_cap'] < min_market_cap:
                        continue
                        
                    # 检查上市时间
                    if market_info['first_listed']:
                        list_date = datetime.strptime(market_info['first_listed'], '%Y-%m-%d')
                        if list_date > min_list_date:
                            continue
                            
                    # 将交易所格式转换为我们的格式 (BTC/USDT -> BTC-USDT)
                    formatted_symbol = symbol.replace('/', '-')
                    valid_symbols.append(formatted_symbol)
                    
                except Exception as e:
                    logging.warning(f"处理交易对 {symbol} 时出错: {str(e)}")
                    continue
            
            logging.info(f"找到 {len(valid_symbols)} 个符合条件的交易对")
            return valid_symbols
            
        except Exception as e:
            logging.error(f"获取有效交易对时出错: {str(e)}")
            return []
        

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    config = Config()
    market_service = MarketDataService(config)
    
    while True:
        try:
            # 每天更新一次交易对列表
            if datetime.now().hour == 0 and datetime.now().minute == 0:
                config.update_symbols()
            
            market_service.update_market_data()
            time.sleep(60)
            
        except KeyboardInterrupt:
            logging.info("程序正在退出...")
            break
        except Exception as e:
            logging.error(f"发生错误: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    main()

